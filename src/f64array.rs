use std::f64;
use std::ops::{
    Add,
    Sub,
    Mul,
    Div,
};
use std::cmp::Ordering;

use duplicate::duplicate_item;
use ndarray::{
    ArcArray,
    Array,
    ArrayBase,
    ArrayView,
    ArrayViewMut,
    Axis,
    Dimension,
    Ix1,
    Ix2,
    OwnedRepr,
    Zip,
};

use super::generic::{
    ArrayProxiedMethods,
};
use super::boolarray::{
    BoolArray,
};

pub type F64Array<D> = Array<f64, D>;
pub type F64Array1 = F64Array<Ix1>;
pub type F64Array2 = F64Array<Ix2>;
pub type F64ArcArray<D> = ArcArray<f64, D>;
pub type F64ArcArray1 = F64ArcArray<Ix1>;
pub type F64ArcArray2 = F64ArcArray<Ix2>;
pub type F64ArrayView<'a, D> = ArrayView<'a, f64, D>;
pub type F64ArrayViewMut<'a, D> = ArrayViewMut<'a, f64, D>;

pub type OptionF64Array<D> = Array<Option<f64>, D>;
pub type OptionF64Array1 = OptionF64Array<Ix1>;
pub type OptionF64Array2 = OptionF64Array<Ix2>;

#[allow(dead_code)]
pub type F64LatLng = F64Array1;
pub type F64LatLngView<'a> = F64ArrayView<'a, Ix1>;
pub type F64LatLngViewMut<'a> = F64ArrayViewMut<'a, Ix1>;
pub type F64LatLngArray = F64Array2;
pub type F64LatLngArcArray = F64ArcArray2;
#[allow(dead_code)]
pub type F64LatLngArrayView<'a> = F64ArrayView<'a, Ix2>;
#[allow(dead_code)]
pub type F64LatLngArrayViewMut<'a> = F64ArrayViewMut<'a, Ix2>;

/// Trait for Arrays supporting f64 operations.
/// 
/// While the elements of the base array is not required to be `f64`,
/// the returned arrays are locked to `f64` only.
pub trait ArrayWithF64Methods<D>: ArrayProxiedMethods
where
    D: Dimension
{
    fn abs(&self) -> F64Array<D>;
    fn abs_sub(&self, other: f64) -> F64Array<D>;
    fn acos(&self) -> F64Array<D>;
    fn acosh(&self) -> F64Array<D>;
    fn asin(&self) -> F64Array<D>;
    fn asinh(&self) -> F64Array<D>;
    fn atan(&self) -> F64Array<D>;

    // atan2 have 3 cases - other could be scalar or an array or an array view.
    // Because their implementation is completely different and there is no
    // type overloading in Rust, we split them into three, much like
    // powi and powf.
    fn atan2_f64(&self, other: f64) -> F64Array<D>;
    fn atan2_arr<'a>(&self, other: F64Array<D>) -> F64Array<D>;
    fn atan2_arrv<'b>(&self, other: F64ArrayView<'b, D>) -> F64Array<D>;

    fn atanh(&self) -> F64Array<D>;
    fn cbrt(&self) -> F64Array<D>;
    fn ceil(&self) -> F64Array<D>;
    fn copysign(&self, sign: f64) -> F64Array<D>;
    fn cos(&self) -> F64Array<D>;
    fn cosh(&self) -> F64Array<D>;
    fn div_euclid(&self, rhs: f64) -> F64Array<D>;
    fn exp(&self) -> F64Array<D>;
    fn exp2(&self) -> F64Array<D>;
    fn exp_m1(&self) -> F64Array<D>;
    fn floor(&self) -> F64Array<D>;
    fn fract(&self) -> F64Array<D>;
    fn hypot(&self, other: f64) -> F64Array<D>;
    fn ln(&self) -> F64Array<D>;
    fn ln_1p(&self) -> F64Array<D>;
    fn log(&self, base: f64) -> F64Array<D>;
    fn log10(&self) -> F64Array<D>;
    fn log2(&self) -> F64Array<D>;
    fn mul_add(&self, a: f64, b: f64) -> F64Array<D>;
    fn powf(&self, n: f64) -> F64Array<D>;
    fn powi(&self, n: i32) -> F64Array<D>;
    fn rem_euclid(&self, rhs: f64) -> F64Array<D>;
    fn round(&self) -> F64Array<D>;
    fn signum(&self) -> F64Array<D>;
    fn sin(&self) -> F64Array<D>;
    fn sin_cos(&self) -> ArrayBase<OwnedRepr<(f64,f64)>, D>;
    fn sinh(&self) -> F64Array<D>;
    fn sqrt(&self) -> F64Array<D>;
    fn tan(&self) -> F64Array<D>;
    fn tanh(&self) -> F64Array<D>;
    fn trunc(&self) -> F64Array<D>;
}


/// An additional trait that will be implemented twice, once for `f64`
/// and once for `F64Array<D>` so that `atan2` will be available as a
/// method on its own.
pub trait ArrayWithF64Atan2Methods<T, D> : ArrayWithF64Methods<D>
where   D: Dimension
{
    fn atan2(&self, other: T) -> F64Array<D>;
}

/// Implements all `f64` native (non-trait) methods for ArrayBase.
/// 
/// This `impl` is trait bound to `f64` Arrays only.
/// use duplicate::duplicate_item;
#[duplicate_item(
    __array_type__                      __impl_generics__;
    [ F64Array<D> ]                     [ D ];
    [ F64ArcArray<D> ]                  [ D ];
    [ F64ArrayView<'a, D> ]             [ 'a, D ];
    [ F64ArrayViewMut<'a, D> ]          [ 'a, D ];
)]
impl<__impl_generics__> ArrayWithF64Methods<D>
for __array_type__
where   D: Dimension {
    fn abs(&self) -> F64Array<D> {
        return ArrayBase::map(self,|num| num.abs());
    }
    fn abs_sub(&self, other: f64) -> F64Array<D> {
        return self.map(|num| (num - other).abs());
    }
    fn acos(&self) -> F64Array<D> {
        return self.map(|num| num.acos());
    }
    fn acosh(&self) -> F64Array<D> {
        return self.map(|num| num.acosh());
    }
    fn asin(&self) -> F64Array<D> {
        return self.map(|num| num.asin());
    }
    fn asinh(&self) -> F64Array<D> {
        return self.map(|num| num.asinh());
    }
    fn atan(&self) -> F64Array<D> {
        return self.map(|num| num.atan());
    }
    fn atan2_f64(&self, other: f64) -> F64Array<D> {
        return self.map(|num| num.atan2(other));
    }
    fn atan2_arr(&self, other: F64Array<D>) -> F64Array<D> {
        // From https://docs.rs/ndarray/latest/src/ndarray/impl_ops.rs.html#115.
        // Honestly not quite sure what this is doing but that's the only
        // elementwise operations between 2 arrays that work.

        // Because the trait bindings of this function dictates that the
        // 2 arrays have the same dimension, no checks are required.
        return Zip::from(self.view())
                    .and(other.view())
                    .map_collect(
                        move |x, y| f64::atan2(*x, *y)
                    );
    }
    fn atan2_arrv<'b>(&self, other: F64ArrayView<'b, D>) -> F64Array<D> {
        // From https://docs.rs/ndarray/latest/src/ndarray/impl_ops.rs.html#115.
        // Honestly not quite sure what this is doing but that's the only
        // elementwise operations between 2 arrays that work.

        // Because the trait bindings of this function dictates that the
        // 2 arrays have the same dimension, no checks are required.
        return Zip::from(self.view())
                    .and(other)
                    .map_collect(
                        move |x, y| f64::atan2(*x, *y)
                    );
    }
    fn atanh(&self) -> F64Array<D> {
        return self.map(|num| num.atanh());
    }
    fn cbrt(&self) -> F64Array<D> {
        return self.map(|num| num.cbrt());
    }
    fn ceil(&self) -> F64Array<D> {
        return self.map(|num| num.ceil());
    }
    fn copysign(&self, sign: f64) -> F64Array<D> {
        return self.map(|num| num.copysign(sign));
    }
    fn cos(&self) -> F64Array<D> {
        return self.map(|num| num.cos());
    }
    fn cosh(&self) -> F64Array<D> {
        return self.map(|num| num.cosh());
    }
    fn div_euclid(&self, rhs: f64) -> F64Array<D> {
        return self.map(|num| num.div_euclid(rhs));
    }
    fn exp(&self) -> F64Array<D> {
        return self.map(|num| num.exp());
    }
    fn exp2(&self) -> F64Array<D> {
        return self.map(|num| num.exp2());
    }
    fn exp_m1(&self) -> F64Array<D> {
        return self.map(|num| num.exp_m1());
    }
    fn floor(&self) -> F64Array<D> {
        return self.map(|num| num.floor());
    }
    fn fract(&self) -> F64Array<D> {
        return self.map(|num| num.fract());
    }
    fn hypot(&self, other: f64) -> F64Array<D> {
        return self.map(|num| num.hypot(other));
    }
    fn ln(&self) -> F64Array<D> {
        return self.map(|num| num.ln());
    }
    fn ln_1p(&self) -> F64Array<D> {
        return self.map(|num| num.ln_1p());
    }
    fn log(&self, base: f64) -> F64Array<D> {
        return self.map(|num| num.log(base));
    }
    fn log10(&self) -> F64Array<D> {
        return self.map(|num| num.log10());
    }
    fn log2(&self) -> F64Array<D> {
        return self.map(|num| num.log2());
    }
    fn mul_add(&self, a: f64, b: f64) -> F64Array<D> {
        return self.map(|num| num.mul_add(a, b));
    }
    fn powf(&self, n: f64) -> F64Array<D> {
        return self.map(|num| num.powf(n));
    }
    fn powi(&self, n: i32) -> F64Array<D> {
        return self.map(|num| num.powi(n));
    }
    fn rem_euclid(&self, rhs: f64) -> F64Array<D> {
        return self.map(|num| num.rem_euclid(rhs));
    }
    fn round(&self) -> F64Array<D> {
        return self.map(|num| num.round());
    }
    fn signum(&self) -> F64Array<D> {
        return self.map(|num| num.signum());
    }
    fn sin(&self) -> F64Array<D> {
        return self.map(|num| num.sin());
    }
    fn sin_cos(&self) -> ArrayBase<OwnedRepr<(f64,f64)>, D> {
        return self.map(|num| num.sin_cos());
    }
    fn sinh(&self) -> F64Array<D> {
        return self.map(|num| num.sinh());
    }
    fn sqrt(&self) -> F64Array<D> {
        return self.map(|num| num.sqrt());
    }
    fn tan(&self) -> F64Array<D> {
        return self.map(|num| num.tan());
    }
    fn tanh(&self) -> F64Array<D> {
        return self.map(|num| num.tanh());
    }
    fn trunc(&self) -> F64Array<D> {
        return self.map(|num| num.trunc());
    }
}

#[duplicate_item(
    __rhs_type__                        __func_name__;
    [ f64 ]                             [ atan2_f64 ];
    [ F64Array<D> ]                     [ atan2_arr ];
    [ F64ArrayView<'_, D> ]             [ atan2_arrv ];
)]
#[duplicate_item(
    __array_type__                      __impl_generics__;
    [ F64Array<D> ]                     [ D ];
    [ F64ArcArray<D> ]                  [ D ];
    [ F64ArrayView<'a, D> ]             [ 'a, D ];
    [ F64ArrayViewMut<'a, D> ]          [ 'a, D ];
)]
impl<__impl_generics__> ArrayWithF64Atan2Methods<__rhs_type__, D>
for __array_type__
where   D: Dimension {
    fn atan2(&self, other: __rhs_type__) -> F64Array<D> {
        return self.__func_name__(other);
    }
}

// =====================================================================================

/// PartialOrd comparison mimicking std::cmp::PartialOrd.
///
/// This does not however implement the actual trait, since that trait expects `bool`
/// as returns but we are expecting `BoolArray<d>` instead.
pub trait ArrayWithF64PartialOrd<D> : ArrayWithF64Methods<D>
where   D: Dimension
{
    fn partial_cmp(&self, other: &f64) -> Array<Option<Ordering>, D>;

    fn lt(&self, other: &f64) -> BoolArray<D>;
    fn le(&self, other: &f64) -> BoolArray<D>;
    fn gt(&self, other: &f64) -> BoolArray<D>;
    fn ge(&self, other: &f64) -> BoolArray<D>;
}

#[duplicate_item(
    __array_type__                      __impl_generics__;
    [ F64Array<D> ]                     [ D ];
    [ F64ArcArray<D> ]                  [ D ];
    [ F64ArrayView<'a, D> ]             [ 'a, D ];
    [ F64ArrayViewMut<'a, D> ]          [ 'a, D ];
)]
impl<__impl_generics__> ArrayWithF64PartialOrd<D>
for __array_type__
where   D: Dimension {
    fn partial_cmp(&self, other: &f64) -> Array<Option<Ordering>, D>{
        return self.map(
            |value| Some(
                match value {
                    value if value < other => Ordering::Less,
                    value if value > other => Ordering::Greater,
                    _ => Ordering::Equal,
                }
            )
        )
    }

    #[duplicate_item(
        PartialOrdMethod;
        [ lt ];
        [ le ];
        [ gt ];
        [ ge ];
    )]
    fn PartialOrdMethod(&self, other: &f64) -> BoolArray<D> {
        return self.map(
            |value| value.PartialOrdMethod(other)
        )
    }
}

// =====================================================================================

/// Operators between a 2-dimenion array and a 1-dimension one.
///
/// Normally, for an LHS array of (L, M), the RHS array of an arith operator is expected
/// to be of dimension (M).
/// This trait is for arrays to add, sub, mul or div in place a LHS array of dimension
/// (L) instead.
pub trait ArrayWithF64MappedOperators<T>:Add+Sub+Mul+Div+Sized+ArrayProxiedMethods {
    // Inplace operator.
    #[duplicate_item(
        __func_name__;
        [ add_assign_array1 ];
        [ sub_assign_array1 ];
        [ mul_assign_array1 ];
        [ div_assign_array1 ];
    )]
    fn __func_name__(
        &mut self,
        rhs: &T,
    ) -> &Self;
}

#[duplicate_item(
    __array_type__                      __impl_generics__;
    [ F64Array2 ]                       [ ];
    [ F64ArcArray2 ]                    [ ];
    // [ F64ArrayViewMut<'a, Ix2> ]        [ 'a ];
)]
#[duplicate_item(
    __rhs_type__;
    [ F64Array1 ];
    [ F64ArcArray1 ];
    [ F64ArrayView<'_, Ix1> ];
    [ F64ArrayViewMut<'_, Ix1> ];
)]
impl<__impl_generics__> ArrayWithF64MappedOperators<__rhs_type__> for __array_type__ {
    // Inplace operator.
    #[duplicate_item(
        __func_name__           __func__;
        [ add_assign_array1 ]   [ add ];
        [ sub_assign_array1 ]   [ sub ];
        [ mul_assign_array1 ]   [ mul ];
        [ div_assign_array1 ]   [ div ];
    )]
    fn __func_name__(
        &mut self,
        rhs: &__rhs_type__,
    ) -> &Self {
        Zip::from(self.rows_mut())
                      .and(rhs)
                      .for_each(
                        |mut s, o| {
                            s.assign(&s.__func__(*o))
                        }
                      );

        return self;
    }
}

// =====================================================================================

/// Additional Trait for Arrays containing angular f64 data.
/// 
pub trait ArrayWithF64AngularMethods<D> : ArrayWithF64Methods<D>
where
    D: Dimension
{
    fn to_rad(&self) -> F64Array<D>;
    fn to_dec(&self) -> F64Array<D>;
}

/// Implements `f64` angular conversion methods for ArrayBase.
/// 
/// This `impl` is trait bound to `f64` Arrays only.
/// use duplicate::duplicate_item;
#[duplicate_item(
    __array_type__                      __impl_generics__;
    [ F64Array<D> ]                     [ D ];
    [ F64ArcArray<D> ]                  [ D ];
    [ F64ArrayView<'a, D> ]             [ 'a, D ];
    [ F64ArrayViewMut<'a, D> ]          [ 'a, D ];
)]
impl<__impl_generics__> ArrayWithF64AngularMethods<D>
for __array_type__
where   D: Dimension {
    fn to_rad(&self) -> F64Array<D> {
        return self * f64::consts::PI / 180.;
    }

    fn to_dec(&self) -> F64Array<D> {
        return self * 180. / f64::consts::PI;
    }
}

// =====================================================================================

/// Additional Trait for Arrays containing latitude-longitude f64 data.
/// 
/// Only allow for arrays with 2 dimensions.
pub trait ArrayWithF64LatLngMethods : ArrayWithF64AngularMethods<Ix2>
{
    fn normalize(&mut self);
}

/// Implements `f64` latitude-longitude conversion methods for ArrayBase.
/// 
/// This `impl` is trait bound to `f64` Arrays only.
/// use duplicate::duplicate_item;
#[duplicate_item(
    __array_type__                      __impl_generics__;
    [ F64LatLngArray ]                  [ ];
    [ F64LatLngArcArray ]               [ ];
    // [ F64LatLngArrayView<'a> ]          [ 'a ];
    [ F64LatLngArrayViewMut<'a> ]          [ 'a ];
)]
impl<__impl_generics__> ArrayWithF64LatLngMethods
for __array_type__
{
    fn normalize(&mut self) {
        return self.axis_iter_mut(Axis(0)).for_each(
            | mut latlng | {
                let mut lat = latlng[0] % 360.;
                let mut lng = latlng[1] % 360.;

                if lat > 180. {lat -= 360.}
                if lat < -180. {lat += 360.}
                if lat > 90. {
                    lat = 180. - lat;
                    lng = lng + 180.;
                }
                if lat < -90. {
                    lat = -180. - lat;
                    lng = lng + 180.;
                }

                match lng {
                    value if value > 180. => lng -= 360.,
                    value if value < -180. => lng += 360.,
                    _ => {},
                };

                latlng[0] = lat;
                latlng[1] = lng;
            }
        );
    }
}