// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "core", reason = "internal to format_args!")]

//
// Argument vector dispatch
//

use fmt::{Formatter, Result};

pub struct ArgumentsBuf<'a, T: ArgumentsTuple<'a>+'a> {
    pub spec: &'a ArgumentsSpec<'a, T>,
    pub args: T,
}

pub struct ArgumentsSpec<'a, T: ArgumentsTuple<'a>> {
    pub count: <T as ArgumentsTuple<'a>>::Count,
    pub trailing: &'a str,
    pub args: <T as ArgumentsTuple<'a>>::SpecTuple,
}

pub trait ArgumentsTuple<'a> {
    type Count: 'a;
    type SpecTuple: 'a;
}

pub struct ArgumentsSpecItem<'a, T> {
    pub piece: &'a str,
    pub vtable: &'a ArgumentVTable<T>,
}

pub struct ArgumentVTable<T> {
    pub fmt: fn(&T, &mut Formatter) -> Result,
    // XXX: not implemented yet...
    // TODO: There's a strong possibility that enabling this field will result
    // in a duplicated diagnostic.
    //pub size_hint: fn(&T, &Formatter) -> usize,
}

macro_rules! args_peel {
    ($count:expr, { $($type_arg:ident,)* }, { }) => ();
    ($count:expr,
            { $($type_arg:ident,)* },
            { $count_type:ident, $next_type_arg:ident, $($rest:ident,)* }) => {
        args! {
            $count_type,
            ($count + 1),
            { $($type_arg,)* $next_type_arg, },
            { $($rest,)* }
        }
    }
}

macro_rules! args {
    ( $count_type:ident, $count:expr, { $($type_arg:ident,)* }, { $($rest:ident,)* } ) => {
        #[derive(Copy, Clone)]
        pub enum $count_type {
            Value = $count
        }
        impl<'a, $($type_arg:'a,)*> ArgumentsTuple<'a> for ($(&'a $type_arg,)*) {
            type Count = $count_type;
            type SpecTuple = ($(ArgumentsSpecItem<'a, $type_arg>,)*);
        }
        args_peel! { $count, { $($type_arg,)* }, { $($rest,)* } }
    }
}

// TODO: Rename CountXX and/or Count.
// XXX: The number of fmt::rt::v2::Count{} types should agree with the number
// of fmt::CountUniform enumerators.
args! {
    Count0, 0, {}, {
        Count1, T1,
        Count2, T2,
        Count3, T3,
        Count4, T4,
        Count5, T5,
        Count6, T6,
        Count7, T7,
        Count8, T8,
        Count9, T9,
        Count10, T10,
        Count11, T11,
        Count12, T12,
        Count13, T13,
        Count14, T14,
        Count15, T15,
        Count16, T16,
        Count17, T17,
        Count18, T18,
        Count19, T19,
        Count20, T20,
        Count21, T21,
        Count22, T22,
        Count23, T23,
        Count24, T24,
        Count25, T25,
        Count26, T26,
        Count27, T27,
        Count28, T28,
        Count29, T29,
        Count30, T30,
        Count31, T31,
        Count32, T32,
    }
}

//
// Detailed argument formatting options
//

pub struct FormattedArg<'a, T:'a> {
    pub arg: &'a T,
    pub spec: &'a FormatParamsSpec,
    pub precision: usize,
    pub width: usize,
}

#[derive(Copy, Clone)]
pub struct FormatParamsSpec {
    pub fill: char,
    pub flags: u32,
    pub align: Alignment,
    pub precision: Count,
    pub width: Count,
}

#[derive(Copy, Clone)]
pub enum Alignment {
    Left,
    Right,
    Center,
    Unknown,
}

// TODO: Rename CountXX and/or Count.
#[derive(Copy, Clone)]
pub enum Count {
    // TODO: Rename this enumerator.
    NextParam,
    Implied,
}
