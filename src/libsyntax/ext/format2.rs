// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use codemap::Span;
use ext::base::*;
use ext::base;
use ext::build::AstBuilder;
use parse::token;
use ptr::P;

use std::collections::HashMap;

/// Parses the arguments from the given list of tokens, returning None
/// if there's a parse error so we can continue parsing other format!
/// expressions.
///
/// If parsing succeeds, the return value is:
///
///     Some((fmtstr, unnamed arguments, ordering of named arguments,
///           named arguments))
fn parse_args(ecx: &ExtCtxt, sp: Span, tts: &[ast::TokenTree])
              -> Option<(P<ast::Expr>, Vec<P<ast::Expr>>, Vec<String>,
                         HashMap<String, P<ast::Expr>>)> {
    let mut args = Vec::new();
    let mut names = HashMap::<String, P<ast::Expr>>::new();
    let mut order = Vec::new();

    let mut p = ecx.new_parser_from_tts(tts);

    if p.token == token::Eof {
        ecx.span_err(sp, "requires at least a format string argument");
        return None;
    }
    let fmtstr = p.parse_expr();
    let mut named = false;
    while p.token != token::Eof {
        if !panictry!(p.eat(&token::Comma)) {
            ecx.span_err(sp, "expected token: `,`");
            return None;
        }
        if p.token == token::Eof { break } // accept trailing commas
        if named || (p.token.is_ident() && p.look_ahead(1, |t| *t == token::Eq)) {
            named = true;
            let ident = match p.token {
                token::Ident(i, _) => {
                    panictry!(p.bump());
                    i
                }
                _ if named => {
                    ecx.span_err(p.span,
                                 "expected ident, positional arguments \
                                 cannot follow named arguments");
                    return None;
                }
                _ => {
                    ecx.span_err(p.span,
                                 &format!("expected ident for named argument, found `{}`",
                                         p.this_token_to_string()));
                    return None;
                }
            };
            let interned_name = token::get_ident(ident);
            let name = &interned_name[..];

            panictry!(p.expect(&token::Eq));
            let e = p.parse_expr();
            match names.get(name) {
                None => {}
                Some(prev) => {
                    ecx.span_err(e.span,
                                 &format!("duplicate argument named `{}`",
                                         name));
                    ecx.parse_sess.span_diagnostic.span_note(prev.span, "previously here");
                    continue
                }
            }
            order.push(name.to_string());
            names.insert(name.to_string(), e);
        } else {
            args.push(p.parse_expr());
        }
    }
    Some((fmtstr, args, order, names))
}

pub fn expand_format_args<'cx>(ecx: &'cx mut ExtCtxt, sp: Span,
                               tts: &[ast::TokenTree])
                               -> Box<base::MacResult+'cx> {
    match parse_args(ecx, sp, tts) {
        Some((efmt, args, order, names)) => {
            MacEager::expr(expand_preparsed_format_args(ecx, sp, efmt,
                                                      args, order, names))
        }
        None => DummyResult::expr(sp)
    }
}

/// Take the various parts of `format_args!(efmt, args..., name=names...)`
/// and construct the appropriate formatting expression.
fn expand_preparsed_format_args(ecx: &mut ExtCtxt, sp: Span,
                                efmt: P<ast::Expr>,
                                args: Vec<P<ast::Expr>>,
                                name_ordering: Vec<String>,
                                names: HashMap<String, P<ast::Expr>>)
                                -> P<ast::Expr> {
    // FIXME: Should we use efmt.span or sp? format.rs is inconsistent.
    let fmtsp = efmt.span;
    match parser::parse_format(ecx, fmtsp, efmt, args, name_ordering, names) {
        None => DummyResult::raw_expr(sp),
        Some(format) => trans::trans_format(ecx, fmtsp, format)
    }
}

mod parser {
    use self::ArgumentType::*;
    use self::Position::*;

    use ast;
    use codemap::Span;
    use ext::base::*;
    use ext::build::AstBuilder;
    use fmt_macros;
    use parse::token;
    use ptr::P;

    use std::collections::HashMap;
    use std::iter::FromIterator;

    pub struct Format {
        /// If the input expression is a width/precision value, its expected
        /// type will be usize. Otherwise, its expected type will be &T, where
        /// T implements a format trait.
        pub inputs: Vec<P<ast::Expr>>,
        pub outputs: Vec<Output>,
        pub trailing: token::InternedString,
    }

    pub struct Output {
        /// Prefix string literal.
        pub literal: token::InternedString,
        /// Trait name, such as Debug or Display.
        pub trait_: &'static str,
        /// Index of the input.
        pub input: usize,
        /// Non-default output parameters.
        pub params: Option<OutputParams>,
    }

    #[derive(PartialEq)]
    pub struct OutputParams {
        pub fill: char,
        pub align: fmt_macros::Alignment,
        pub flags: u32,
        pub width: Count,
        pub precision: Count,
    }

    #[derive(Copy, Clone, PartialEq)]
    pub enum Count {
        Implied,
        Is(usize),
        /// The count is specified by the integral input of the given index.
        Input(usize),
    }

    #[derive(PartialEq)]
    enum ArgumentType {
        Known(String),
        Unsigned
    }

    enum Position {
        Exact(usize),
        Named(String),
    }

    struct Context<'a, 'b:'a> {
        ecx: &'a mut ExtCtxt<'b>,
        fmtsp: Span,

        /// Positional input argument expressions.
        args: Vec<P<ast::Expr>>,
        /// Named input arguments. Note that we keep a side-array of the
        /// ordering of the named arguments found to be sure that we can
        /// translate them in the same order that they were declared in.
        names: HashMap<String, P<ast::Expr>>,
        name_ordering: Vec<String>,
        /// A map from each named argument to its index within the sequence of
        /// all arguments (including positional).
        name_indices: HashMap<String, usize>,

        /// Types of arguments we've found so far.
        arg_types: Vec<Option<ArgumentType>>,
        name_types: HashMap<String, ArgumentType>,

        /// The latest consecutive literal strings, or empty if there weren't any.
        literal: String,

        /// Accumulation of `Output` values collected so far.
        outputs: Vec<Output>,

        /// Updated as arguments are consumed or methods are entered
        next_arg: usize,
    }

    pub fn parse_format(ecx: &mut ExtCtxt, fmtsp: Span,
                        efmt: P<ast::Expr>,
                        args: Vec<P<ast::Expr>>,
                        name_ordering: Vec<String>,
                        names: HashMap<String, P<ast::Expr>>)
                        -> Option<Format> {
        let arg_types: Vec<_> = (0..args.len()).map(|_| None).collect();
        let name_indices: HashMap<_, _> = HashMap::from_iter(
            name_ordering.clone().into_iter().enumerate().map(|(i, x)| (x, args.len() + i)));
        let mut cx = Context {
            ecx: ecx,
            fmtsp: fmtsp,
            args: args,
            arg_types: arg_types,
            names: names,
            name_types: HashMap::new(),
            name_ordering: name_ordering,
            name_indices: name_indices,
            literal: String::new(),
            outputs: Vec::new(),
            next_arg: 0,
        };
        let fmt = match expr_to_string(cx.ecx,
                                       efmt,
                                       "format argument must be a string literal.") {
            Some((fmt, _)) => fmt,
            None => return None
        };

        let mut parser = fmt_macros::Parser::new(&fmt);

        loop {
            match parser.next() {
                Some(piece) => {
                    if parser.errors.len() > 0 { break }
                    cx.parse_piece(&piece);
                }
                None => break
            }
        }
        if !parser.errors.is_empty() {
            cx.ecx.span_err(cx.fmtsp, &format!("invalid format string: {}",
                                               parser.errors.remove(0)));
            return None;
        }

        // Make sure that all arguments were used and all arguments have types.
        for (i, ty) in cx.arg_types.iter().enumerate() {
            if ty.is_none() {
                cx.ecx.span_err(cx.args[i].span, "argument never used");
            }
        }
        for (name, e) in &cx.names {
            if !cx.name_types.contains_key(name) {
                cx.ecx.span_err(e.span, "named argument never used");
            }
        }

        Some(cx.into_format())
    }

    impl<'a, 'b> Context<'a, 'b> {
        /// Verifies one piece of a parse string. All errors are not emitted as
        /// fatal so we can continue giving errors about this and possibly
        /// other format strings.
        fn parse_piece(&mut self, p: &fmt_macros::Piece) {
            match *p {
                fmt_macros::String(s) => {
                    self.literal.push_str(s);
                }
                fmt_macros::NextArgument(ref arg) => {
                    let literal = token::intern_and_get_ident(&self.literal);
                    self.literal.clear();

                    // width/precision first, if they have implicit positional
                    // parameters it makes more sense to consume them first.
                    let width = self.verify_count(arg.format.width);
                    let precision = self.verify_count(arg.format.precision);

                    // argument second, if it's an implicit positional
                    // parameter it's written second, so it should come after
                    // width/precision.
                    let pos = match arg.position {
                        fmt_macros::ArgumentNext => {
                            let i = self.next_arg;
                            self.next_arg += 1;
                            Exact(i)
                        }
                        fmt_macros::ArgumentIs(i) => Exact(i),
                        fmt_macros::ArgumentNamed(s) => Named(s.to_string()),
                    };

                    let ty = Known(arg.format.ty.to_string());
                    let input = self.verify_arg_type(pos, ty);
                    let trait_ = self.lookup_arg_trait(arg.format.ty);

                    let default_params = OutputParams {
                        fill: ' ',
                        align: fmt_macros::AlignUnknown,
                        flags: 0,
                        width: Count::Implied,
                        precision: Count::Implied,
                    };

                    let params = OutputParams {
                        fill: arg.format.fill.unwrap_or(default_params.fill),
                        align: arg.format.align,
                        flags: arg.format.flags,
                        // FIXME: Can we harmonize the order of width and
                        // precision everywhere?
                        width: width,
                        precision: precision,
                    };

                    let opt_params = if default_params != params { Some(params) } else { None };

                    match input {
                        None => {},
                        Some(input_index) => {
                            self.outputs.push(Output {
                                literal: literal,
                                trait_: trait_,
                                input: input_index,
                                params: opt_params,
                            });
                        }
                    }
                }
            }
        }

        fn verify_count(&mut self, c: fmt_macros::Count) -> Count {
            fn verify_input(ctx: &mut Context, arg: Position) -> Count {
                match ctx.verify_arg_type(arg, Unsigned) {
                    None => Count::Implied,
                    Some(i) => Count::Input(i)
                }
            }
            match c {
                fmt_macros::CountImplied => Count::Implied,
                fmt_macros::CountIs(i) => Count::Is(i),
                fmt_macros::CountIsParam(i) => {
                    verify_input(self, Exact(i))
                }
                fmt_macros::CountIsName(s) => {
                    verify_input(self, Named(s.to_string()))
                }
                fmt_macros::CountIsNextParam => {
                    let next_arg = self.next_arg;
                    let ret = verify_input(self, Exact(next_arg));
                    self.next_arg += 1;
                    ret
                }
            }
        }

        fn describe_num_args(&self) -> String {
            match self.args.len() {
                0 => "no arguments given".to_string(),
                1 => "there is 1 argument".to_string(),
                x => format!("there are {} arguments", x),
            }
        }

        /// Verifies that the argument position is valid and that the argument
        /// type is consistent with existing information for the argument.
        /// Returns either a valid input index or None.
        fn verify_arg_type(&mut self, arg: Position, ty: ArgumentType) -> Option<usize> {
            match arg {
                Exact(arg) => {
                    if self.args.len() <= arg {
                        let msg = format!("invalid reference to argument `{}` ({})",
                                          arg, self.describe_num_args());

                        self.ecx.span_err(self.fmtsp, &msg[..]);
                        return None;
                    }
                    {
                        let arg_type = match self.arg_types[arg] {
                            None => None,
                            Some(ref x) => Some(x)
                        };
                        self.verify_same(self.args[arg].span, &ty, arg_type);
                    }
                    if self.arg_types[arg].is_none() {
                        self.arg_types[arg] = Some(ty);
                    }
                    Some(arg)
                }

                Named(name) => {
                    let span = match self.names.get(&name) {
                        Some(e) => e.span,
                        None => {
                            let msg = format!("there is no argument named `{}`", name);
                            self.ecx.span_err(self.fmtsp, &msg[..]);
                            return None;
                        }
                    };
                    self.verify_same(span, &ty, self.name_types.get(&name));
                    if !self.name_types.contains_key(&name) {
                        self.name_types.insert(name.clone(), ty);
                    }
                    Some(*self.name_indices.get(&name).unwrap())
                }
            }
        }

        /// When we're keeping track of the types that are declared for certain
        /// arguments, we assume that `None` means we haven't seen this argument
        /// yet, `Some(None)` means that we've seen the argument, but no format was
        /// specified, and `Some(Some(x))` means that the argument was declared to
        /// have type `x`.
        ///
        /// Obviously `Some(Some(x)) != Some(Some(y))`, but we consider it true
        /// that: `Some(None) == Some(Some(x))`
        fn verify_same(&self,
                       sp: Span,
                       ty: &ArgumentType,
                       before: Option<&ArgumentType>) {
            let cur = match before {
                None => return,
                Some(t) => t,
            };
            if *ty == *cur {
                return
            }
            match (cur, ty) {
                (&Known(ref cur), &Known(ref ty)) => {
                    self.ecx.span_err(sp,
                                      &format!("argument redeclared with type `{}` when \
                                               it was previously `{}`",
                                              *ty,
                                              *cur));
                }
                (&Known(ref cur), _) => {
                    self.ecx.span_err(sp,
                                      &format!("argument used to format with `{}` was \
                                               attempted to not be used for formatting",
                                               *cur));
                }
                (_, &Known(ref ty)) => {
                    self.ecx.span_err(sp,
                                      &format!("argument previously used as a format \
                                               argument attempted to be used as `{}`",
                                               *ty));
                }
                (_, _) => {
                    self.ecx.span_err(sp, "argument declared with multiple formats");
                }
            }
        }

        fn lookup_arg_trait(&self, tyname: &str) -> &'static str {
            match tyname {
                ""  => "Display",
                "?" => "Debug",
                "e" => "LowerExp",
                "E" => "UpperExp",
                "o" => "Octal",
                "p" => "Pointer",
                "b" => "Binary",
                "x" => "LowerHex",
                "X" => "UpperHex",
                _ => {
                    self.ecx.span_err(self.fmtsp,
                                      &format!("unknown format trait `{}`",
                                               tyname));
                    "Dummy"
                }
            }
        }

        fn into_format(self) -> Format {
            let pos_args = self.args.into_iter();
            let mut names = self.names;
            let named_args = self.name_ordering.iter().map(|n| names.remove(n).unwrap());
            let all_args: Vec<_> = pos_args.chain(named_args).collect();
            Format {
                inputs: all_args,
                outputs: self.outputs,
                trailing: token::intern_and_get_ident(&self.literal),
            }
        }
    }
}

mod trans {
    use ast;
    use codemap::Span;
    use ext::base::*;
    use ext::build::AstBuilder;
    use ext::format2::parser::{Format, Output, OutputParams, Count};
    use fmt_macros;
    use parse::token;
    use ptr::P;

    use std::collections::HashSet;
    use std::default::Default;

    const SPEC_IDENT: &'static str = "__spec";

    /// Consume the Format and return the result of format_args!().
    pub fn trans_format(ecx: &ExtCtxt, fmtsp: Span, fmt: Format) -> P<ast::Expr> {
        // FIXME: Sigh. "span" means two different things. Come up with better names...
        let macsp = ecx.call_site();
        let spans = identify_spans(fmt.inputs.len(), &fmt.outputs);
        let input_expr_spans = fmt.inputs.iter().map(|e| e.span).collect();

        // Outer match expression.
        let mut heads = Vec::new();
        let mut pats = Vec::new();

        // Add the `ArgumentsSpec`.
        heads.push(trans_arguments_spec(ecx, macsp, fmtsp, &input_expr_spans,
                                        &fmt.outputs, fmt.trailing));
        pats.push(ecx.pat_ident(macsp, ecx.ident_of(SPEC_IDENT)));

        // Bind all arguments to `__arg{}` and generate `FormattedArg` values.
        let mut inputs = fmt.inputs.into_iter().enumerate();
        for span in spans.iter() {
            match span.nested_match {
                None => {
                    for _ in 0..span.length {
                        let (input_index, input) = inputs.next().expect("no arg for simple span");
                        bind_input_expr(ecx, &mut heads, &mut pats, input_index, input, false);
                    }
                }
                Some(ref nested_match) => {
                    trans_nested_match(ecx, macsp, &input_expr_spans,
                                       &mut heads, &mut pats,
                                       inputs.by_ref().take(span.length), &fmt.outputs,
                                       nested_match);
                }
            }
        }

        // Generate the `rt::v2::ArgumentsBuf` value.
        let arguments_buf = trans_arguments_buf(ecx, macsp, &input_expr_spans, &fmt.outputs);

        // Finalize the outer match and convert it to `Arguments`.
        let head = ecx.expr_tuple(macsp, heads);
        let pat = ecx.pat_tuple(macsp, pats);
        let arm = ecx.arm(macsp, vec![pat], arguments_buf);
        let match_ = ecx.expr_match(macsp, head, vec![arm]);
        ecx.expr_method_call(macsp, match_, ecx.ident_of("to_arguments"), vec![])
    }

    #[derive(Default)]
    struct InputSpan {
        length: usize,
        nested_match: Option<NestedMatch>,
    }

    #[derive(Default)]
    struct NestedMatch {
        /// An input within the nested match can still be used in a simple
        /// output. In that case, it needs to escape from the nested match,
        /// which will slightly bloat the generated code.
        escaped_inputs: Vec<usize>,
        /// A list of input indices for inputs used as output counts.
        count_inputs: HashSet<usize>,
        /// Indices of complex outputs.
        complex_outputs: Vec<usize>,
    }

    fn identify_spans(input_count: usize, outputs: &Vec<Output>) -> Vec<InputSpan> {
        // Identify any inputs used in complex outputs. They must be matched
        // with a nested match expression. We must evaluate inputs in their
        // declared order.
        let mut input_in_cluster: Vec<_> = (0..input_count).map(|_| false).collect();
        for output in outputs.iter() {
            match output.params {
                None => {},
                Some(ref params) => {
                    let mut input_indices = vec![output.input];
                    match params.width {
                        Count::Input(i) => { input_indices.push(i) }
                        _ => {}
                    }
                    match params.precision {
                        Count::Input(i) => { input_indices.push(i) }
                        _ => {}
                    }
                    let min = *input_indices.iter().min().expect("no min for span");
                    let max = *input_indices.iter().max().expect("no max for span");
                    // NB: This loop is O(n*m) in the number of inputs and outputs.
                    for i in (min..max + 1) {
                        input_in_cluster[i] = true;
                    }
                }
            }
        }

        // Group the inputs into spans.
        let mut spans = Vec::new();

        {
            let mut push_span = |span: InputSpan| {
                if span.length > 0 {
                    spans.push(span);
                }
            };

            let mut cur_span: InputSpan = Default::default();
            for input_index in 0..input_count {
                // Ensure the current span is of the right kind.
                if input_in_cluster[input_index] != cur_span.nested_match.is_some() {
                    push_span(cur_span);
                    cur_span = Default::default();
                    if input_in_cluster[input_index] {
                        cur_span.nested_match = Some(Default::default());
                    }
                }
                // Add this item to the span.
                cur_span.length += 1;
                if input_in_cluster[input_index] {
                    let cur_nested_match = cur_span.nested_match.as_mut().unwrap();
                    let mut input_used_in_simple = false;
                    let mut input_used_in_count = false;
                    // NB: This loop is O(n*m) in the number of inputs and outputs.
                    for (output_index, output) in outputs.iter().enumerate() {
                        match output.params {
                            Some(ref params) => {
                                if output.input == input_index {
                                    cur_nested_match.complex_outputs.push(output_index);
                                }
                                if params.width == Count::Input(input_index) ||
                                        params.precision == Count::Input(input_index) {
                                    input_used_in_count = true;
                                }
                            }
                            None => {
                                if output.input == input_index {
                                    input_used_in_simple = true;
                                }

                            }
                        }
                    }
                    if input_used_in_simple {
                        cur_nested_match.escaped_inputs.push(input_index);
                    }
                    if input_used_in_count {
                        cur_nested_match.count_inputs.insert(input_index);
                    }
                }
            }
            push_span(cur_span);
        }

        spans
    }

    /// Generate an expression of type `&rt::v2::ArgumentsSpec`.
    ///
    /// NB: To accommodate arbitrarily many outputs with the current design, it
    /// will be necessary to bundle the outputs up into multiple fixed-size
    /// groups. Each bundle will have its own `rt::v2::ArgumentsSpec` and
    /// Arguments structs, but input matching should be unaffected.
    ///
    /// FIXME: trans_format_params_spec returns its argument by-value. This is
    /// by-ref. Fix the inconsistency.
    fn trans_arguments_spec(ecx: &ExtCtxt, macsp: Span, fmtsp: Span,
                            input_expr_spans: &Vec<Span>,
                            outputs: &Vec<Output>,
                            trailing: token::InternedString)
                            -> P<ast::Expr> {
        let mut count = rtpath(ecx, &format!("Count{}", outputs.len()));
        count.push(ecx.ident_of("Value"));
        let count = ecx.expr_path(ecx.path_global(macsp, count));

        let args: Vec<_> = outputs.iter().map(|output| {
            let in_sp = input_expr_spans[output.input];
            let fmt_path = vec![
                ecx.ident_of_std("core"),
                ecx.ident_of("fmt"),
                ecx.ident_of(output.trait_),
                ecx.ident_of("fmt"),
            ];
            let fmt_expr = ecx.expr_path(ecx.path_global(in_sp, fmt_path));
            let path = ecx.path_global(macsp, rtpath(ecx, "ArgumentVTable"));
            let vtable = ecx.expr_struct(macsp, path, vec![
                ecx.field_imm(macsp, ecx.ident_of("fmt"), fmt_expr),
            ]);
            let path = ecx.path_global(macsp, rtpath(ecx, "ArgumentsSpecItem"));
            let piece = ecx.expr_str(fmtsp, output.literal.clone());
            ecx.expr_struct(macsp, path, vec![
                ecx.field_imm(macsp, ecx.ident_of("piece"), piece),
                ecx.field_imm(macsp, ecx.ident_of("vtable"), ecx.expr_addr_of(macsp, vtable)),
            ])
        }).collect();

        let path = ecx.path_global(macsp, rtpath(ecx, "ArgumentsSpec"));
        let spec = ecx.expr_struct(macsp, path, vec![
            ecx.field_imm(macsp, ecx.ident_of("count"), count),
            ecx.field_imm(macsp, ecx.ident_of("trailing"), ecx.expr_str(fmtsp, trailing)),
            ecx.field_imm(macsp, ecx.ident_of("args"), ecx.expr_tuple(macsp, args)),
        ]);
        ecx.expr_addr_of(macsp, spec)
    }

    fn bind_input_expr(ecx: &ExtCtxt,
                       heads: &mut Vec<P<ast::Expr>>,
                       pats: &mut Vec<P<ast::Pat>>,
                       index: usize,
                       mut expr: P<ast::Expr>,
                       is_count: bool) {
        let sp = expr.span;
        if !is_count {
            expr = ecx.expr_addr_of(sp, expr);
        }
        heads.push(expr);
        pats.push(ecx.pat_ident(sp, ecx.ident_of(&format!("__arg{}", index))));
    }

    /// Translate the `NestedMatch`. Bind the inputs for the input span.
    /// Propagate both the inputs in `NestedMatch::escaped_inputs` and a
    /// `rt::v2::FormattedArg` for each output in
    /// `NestedMatch::complex_outputs`. Append the results to the outer match
    /// via `heads` and `pats`.
    fn trans_nested_match<I>(ecx: &ExtCtxt, macsp: Span,
                             input_expr_spans: &Vec<Span>,
                             heads: &mut Vec<P<ast::Expr>>,
                             pats: &mut Vec<P<ast::Pat>>,
                             inputs: I,
                             outputs: &Vec<Output>,
                             nested_match: &NestedMatch) where
        I: Iterator<Item=(usize, P<ast::Expr>)>
    {
        let mut inner_heads = Vec::new();
        let mut inner_pats = Vec::new();

        let mut inner_vals = Vec::new();
        let mut outer_pats = Vec::new();

        // Define each format spec and bind it to `__fps{}` in the inner match.
        for &output_index in nested_match.complex_outputs.iter() {
            let output = &outputs[output_index];
            let in_sp = input_expr_spans[output.input];
            let spec = trans_format_params_spec(ecx, macsp, output.params.as_ref()
                                .expect("no params for nested match output"));
            inner_heads.push(ecx.expr_addr_of(macsp, spec));
            inner_pats.push(ecx.pat_ident(in_sp, ecx.ident_of(&format!("__fps{}",
                                                                       output_index))));
        }

        // Bind each input into the inner match.
        for (input_index, input) in inputs {
            bind_input_expr(ecx, &mut inner_heads, &mut inner_pats,
                            input_index, input,
                            nested_match.count_inputs.contains(&input_index));
        }

        // Allow input arguments to escape.
        for &input_index in nested_match.escaped_inputs.iter() {
            let name = ecx.ident_of(&format!("__arg{}", input_index));
            let in_sp = input_expr_spans[input_index];
            inner_vals.push(ecx.expr_ident(in_sp, name));
            outer_pats.push(ecx.pat_ident(in_sp, name));
        }

        // Create each output `FormattedArg` value.
        for &output_index in nested_match.complex_outputs.iter() {
            let output = &outputs[output_index];
            let in_sp = input_expr_spans[output.input];
            let val = trans_formatted_arg(ecx, macsp, input_expr_spans, output_index, output);
            let pat = ecx.pat_ident_binding_mode(in_sp,
                                                 ecx.ident_of(&format!("__output{}",
                                                                       output_index)),
                                                 ast::BindByRef(ast::MutImmutable));
            inner_vals.push(val);
            outer_pats.push(pat);
        }

        // Glue all the pieces together.
        let inner_match = ecx.expr_match(macsp, ecx.expr_tuple(macsp, inner_heads), vec![
            ecx.arm(macsp,
                    vec![ecx.pat_tuple(macsp, inner_pats)],
                    ecx.expr_tuple(macsp, inner_vals))
        ]);
        let outer_pat = ecx.pat_tuple(macsp, outer_pats);

        // Improve code generation somewhat by matching against the address of
        // the match rather than the value directly. The inner match evaluates
        // to a tuple containing `FormattedArg` values, which are then bound by
        // reference to a `ArgumentsBuf` value. If any part of the tuple's
        // address escapes, LLVM preserves the entire tuple as-is (including
        // the addresses of the "escaped inputs"). If the nested match's tuple
        // were implanted by-value into the outer match's tuple, then the outer
        // tuple would also be preserved. We'd prefer that it be optimized
        // away.
        heads.push(ecx.expr_addr_of(macsp, inner_match));
        pats.push(ecx.pat(macsp, ast::PatRegion(outer_pat, ast::MutImmutable)));
    }

    /// Generate an expression of type `rt::v2::FormatParamsSpec`.
    fn trans_format_params_spec(ecx: &ExtCtxt, sp: Span, params: &OutputParams)
                                -> P<ast::Expr> {

        let fill = ecx.expr_lit(sp, ast::LitChar(params.fill));
        let flags = ecx.expr_u32(sp, params.flags);
        let align = |name| {
            let mut p = rtpath(ecx, "Alignment");
            p.push(ecx.ident_of(name));
            ecx.path_global(sp, p)
        };
        let align = match params.align {
            fmt_macros::AlignLeft => align("Left"),
            fmt_macros::AlignRight => align("Right"),
            fmt_macros::AlignCenter => align("Center"),
            fmt_macros::AlignUnknown => align("Unknown"),
        };
        let align = ecx.expr_path(align);
        let trans_count = |count| {
            let mut path = rtpath(ecx, "Count");
            path.push(ecx.ident_of(match count {
                Count::Implied => "Implied",
                Count::Is(_) | Count::Input(_) => "NextParam",
            }));
            ecx.expr_path(ecx.path_global(sp, path))
        };
        let precision = trans_count(params.precision);
        let width = trans_count(params.width);
        let path = ecx.path_global(sp, rtpath(ecx, "FormatParamsSpec"));
        ecx.expr_struct(sp, path, vec![
            ecx.field_imm(sp, ecx.ident_of("fill"), fill),
            ecx.field_imm(sp, ecx.ident_of("flags"), flags),
            ecx.field_imm(sp, ecx.ident_of("align"), align),
            ecx.field_imm(sp, ecx.ident_of("precision"), precision),
            ecx.field_imm(sp, ecx.ident_of("width"), width),
        ])
    }

    /// Generate an expression of type `rt::v2::FormattedArg`.
    fn trans_formatted_arg(ecx: &ExtCtxt, sp: Span,
                           input_expr_spans: &Vec<Span>,
                           output_index: usize,
                           output: &Output) -> P<ast::Expr> {
        let var_expr = |sp, name: &str| ecx.expr_ident(sp, ecx.ident_of(name));
        let count_expr = |count| {
            match count {
                Count::Implied => ecx.expr_usize(sp, 0),
                Count::Is(val) => ecx.expr_usize(sp, val),
                Count::Input(input_index) => {
                    var_expr(input_expr_spans[input_index],
                             &format!("__arg{}", input_index))
                }
            }
        };
        let arg = var_expr(sp, &format!("__arg{}", output.input));
        let spec = var_expr(sp, &format!("__fps{}", output_index));
        let params = output.params.as_ref().expect("no params for formatted arg");
        let precision = count_expr(params.precision);
        let width = count_expr(params.width);
        let path = ecx.path_global(sp, rtpath(ecx, "FormattedArg"));
        ecx.expr_struct(sp, path, vec![
            ecx.field_imm(sp, ecx.ident_of("arg"), arg),
            ecx.field_imm(sp, ecx.ident_of("spec"), spec),
            ecx.field_imm(sp, ecx.ident_of("precision"), precision),
            ecx.field_imm(sp, ecx.ident_of("width"), width),
        ])
    }

    /// Generate an expression of type `rt::v2::ArgumentsBuf`.
    fn trans_arguments_buf(ecx: &ExtCtxt, sp: Span,
                           input_expr_spans: &Vec<Span>,
                           outputs: &Vec<Output>)
                           -> P<ast::Expr> {
        let spec = ecx.expr_ident(sp, ecx.ident_of(SPEC_IDENT));

        let args: Vec<_> = outputs.iter().enumerate().map(|(i, output)| {
            let name = match output.params {
                None => format!("__arg{}", output.input),
                Some(_) => format!("__output{}", i)
            };
            ecx.expr_ident(input_expr_spans[output.input], ecx.ident_of(&name))
        }).collect();
        let args = ecx.expr_tuple(sp, args);

        let path = ecx.path_global(sp, rtpath(ecx, "ArgumentsBuf"));
        ecx.expr_struct(sp, path, vec![
            ecx.field_imm(sp, ecx.ident_of("spec"), spec),
            ecx.field_imm(sp, ecx.ident_of("args"), args),
        ])
    }

    fn rtpath(ecx: &ExtCtxt, s: &str) -> Vec<ast::Ident> {
        vec![ecx.ident_of_std("core"), ecx.ident_of("fmt"), ecx.ident_of("rt"),
             ecx.ident_of("v2"), ecx.ident_of(s)]
    }
}
