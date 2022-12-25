use core::fmt::Error;
use core::ops::{
    Range,
    RangeFull,
};
use std::time::Duration;

use duplicate::duplicate_item;

pub trait InitValue {
    fn init_value() -> Self;
}

#[duplicate_item(
    __primitive_type__                  __attr__                __impl_generics__;
    [ usize ]                           [ usize::MIN ]          [];
    [ u8 ]                              [ u8::MIN ]             [];
    [ u16 ]                             [ u16::MIN ]            [];
    [ u32 ]                             [ u32::MIN ]            [];
    [ u64 ]                             [ u64::MIN ]            [];
    [ u128 ]                            [ u128::MIN ]           [];
    [ isize ]                           [ 0_isize ]             [];
    [ i8 ]                              [ 0_i8 ]                [];
    [ i16 ]                             [ 0_i16 ]               [];
    [ i32 ]                             [ 0_i32 ]               [];
    [ i64 ]                             [ 0_i64 ]               [];
    [ i128 ]                            [ 0_i128 ]              [];
    [ f32 ]                             [ f32::NAN ]            [];
    [ f64 ]                             [ f64::NAN ]            [];
    [ String ]                          [ String::from("") ]    [];
    [ bool ]                            [ false ]               [];
    [ &str ]                            [ "" ]                  [];
    [ &mut str ]                        [ Self::default() ]     [];
    [ Duration ]                        [ Self::default() ]     [];
    [ Error ]                           [ Self::default() ]     [];
    [ Range<T> ]                        [ Self::default() ]     [ T:Default ];
    [ RangeFull ]                       [ Self::default() ]     [];
    [ Vec<T> ]                          [ Self::default() ]     [ T ];
    [ Option<T> ]                       [ None ]                [ T ];
)]
impl<__impl_generics__> InitValue for __primitive_type__ {
    fn init_value() -> Self {
        return __attr__;
    }
}
