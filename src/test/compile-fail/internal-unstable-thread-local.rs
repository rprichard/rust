// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:internal_unstable.rs

#![feature(rustc_attrs)]
#![allow(dead_code)]

extern crate internal_unstable;


thread_local!(static FOO: () = ());
thread_local!(static BAR: () = internal_unstable::unstable()); //~ WARN use of unstable

#[rustc_error]
fn main() {} //~ ERROR
