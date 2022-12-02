use duplicate::duplicate_item;
use ndarray::{
    ArcArray,
    Array,
    ArrayBase,
    ArrayView,
    Dimension,
    OwnedRepr,
};

pub type F64Array<D> = Array<f64, D>;
pub type F64ArcArray<D> = ArcArray<f64, D>;
pub type F64ArrayView<'a, D> = ArrayView<'a, f64, D>;


/// Trait for Arrays supporting f64 operations.
/// 
/// While the elements of the base array is not required to be `f64`,
/// the returned arrays are locked to `f64` only.
pub trait ArrayWithF64Methods<D> 
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
    fn atan2(&self, other: f64) -> F64Array<D>;
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

/// Implements all `f64` native (non-trait) methods for ArrayBase.
/// 
/// This `impl` is trait bound to `f64` Arrays only.
/// use duplicate::duplicate_item;
#[duplicate_item(
    ArrayType                           Generics;
    [ F64Array<D> ]                     [ D ];
    [ F64ArcArray<D> ]                  [ D ];
    [ F64ArrayView<'a, D> ]             [ 'a, D ];
)]
impl<Generics> ArrayWithF64Methods<D>
for ArrayType
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
    fn atan2(&self, other: f64) -> F64Array<D> {
        return self.map(|num| num.atan2(other));
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
