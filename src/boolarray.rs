use duplicate::duplicate_item;
use ndarray::{
    ArcArray,
    Array,
    Array1,
    ArrayView,
    ArrayViewMut,
    Dimension,
    Ix1,
    Ix2,
    // NdProducer,
    Zip,
};

use super::generic::{
    ArrayProxiedMethods,
};

pub type BoolArray<D> = Array<bool, D>;
pub type BoolArray1 = BoolArray<Ix1>;
pub type BoolArray2 = BoolArray<Ix2>;
pub type BoolArcArray<D> = ArcArray<bool, D>;
pub type BoolArcArray1 = BoolArcArray<Ix1>;
pub type BoolArcArray2 = BoolArcArray<Ix2>;
pub type BoolArrayView<'a, D> = ArrayView<'a, bool, D>;
pub type BoolArrayViewMut<'a, D> = ArrayViewMut<'a, bool, D>;

pub type OptionBoolArray<D> = Array<Option<bool>, D>;
pub type OptionBoolArray1 = OptionBoolArray<Ix1>;
pub type OptionBoolArray2 = OptionBoolArray<Ix2>;

pub trait ArrayWithBoolIterMethods<D>: ArrayProxiedMethods<D, bool>
where
    D: Dimension
{
    type Pattern;

    /// This is different from `Iterator::any` - this does not take a
    /// Function argument. `Iterator::any` is still accessible through
    /// self.iter().any()
    fn any(&self) -> bool;

    /// This is different from `Iterator::all` - this does not take a
    /// Function argument. `Iterator::all` is still accessible through
    /// self.iter().all()
    fn all(&self) -> bool;

    /// Count all the `true`s in an array
    fn count(&self) -> usize;

    /// Indices of `true` values in the array.
    /// Equivalent to `where` in numpy, which is a reserved token in
    /// Rust.
    fn indices(&self) -> Array1<Self::Pattern>;
}


#[duplicate_item(
    __array_type__                       __impl_generics__;
    [ BoolArray<D> ]                     [ D ];
    [ BoolArcArray<D> ]                  [ D ];
    [ BoolArrayView<'a, D> ]             [ 'a, D ];
    [ BoolArrayViewMut<'a, D> ]          [ 'a, D ];
)]
impl<__impl_generics__> ArrayWithBoolIterMethods<D>
for __array_type__
where   D: Dimension {
    type Pattern = <D as Dimension>::Pattern;

    /// Returns true if any of the elements are true. Short circuited.
    fn any(&self) -> bool{
        return {
            self
            .iter()
            .any(|&v| v)
        };
    }

    /// Returns false if any of the elements are false. Short circuited.
    fn all(&self) -> bool{
        return {
            self
            .iter()
            .all(|&v| v)
        };
    }

    /// Count all the `true`s in an array
    fn count(&self) -> usize {
        return {
            self
            .iter()
            .filter(|&v| *v)
            .map(|_| 1 as usize)
            .sum()
        };
    }

    /// Indices of `true` values in the array.
    /// Equivalent to `where` in numpy, which is a reserved token in
    /// Rust.
    fn indices(&self) -> Array1<Self::Pattern> {
        let indices: Vec<Self::Pattern> = {
            self
            .indexed_iter()
            .filter(|(_, &value)| value)
            .map(|(s, _)| s)
            .collect()
        };

        return {
            Array1::from_shape_vec(
                (indices.len(), ),
                indices
            )
            .unwrap()
        }
    }
}

// =====================================================================================

/// Using BoolArrays as masks for identical sized arrays.
pub trait ArrayWithBoolMaskMethods<D, A, F, T>: ArrayWithBoolIterMethods<D>
where
    D: Dimension
{
    fn mask_apply_inplace(
        &self,
        array: T,
        f: F,
    );
}

#[duplicate_item(
    __array_type__                       __impl_generics__;
    [ BoolArray<D> ]                     [ D, A ];
    [ BoolArcArray<D> ]                  [ D, A ];
    [ BoolArrayView<'a, D> ]             [ 'a, D, A ];
    [ BoolArrayViewMut<'a, D> ]          [ 'a, D, A ];
)]
#[duplicate_item(
    __rhs_type__;
    [ &mut Array<A, D> ];
    
    // Cannot use &mut ArcArray: it will break sharing
    // See https://docs.rs/ndarray/latest/ndarray/type.ArcArray.html
    // [ ArcArray<A, D> ];
)]
impl<__impl_generics__> ArrayWithBoolMaskMethods<D, A, &dyn Fn(&mut A), __rhs_type__>
for __array_type__
where   D: Dimension {
    /// Apply a function to another mutable array, using itself as a mask.
    /// 
    /// The function is only applied to ``array[S<D>]`` if array `self[S<D>]`
    /// is ``true``.
    /// 
    /// ``f`` should change the value in place; thus all changes are directly
    /// applied to the ``array`` in place as well.
    fn mask_apply_inplace(
        &self,
        array: __rhs_type__,
        f: &dyn Fn(&mut A),
    )
    {
        Zip::from(array)
            .and(self)
            .for_each(|v, mask| {
                if *mask {
                    f(v);
                }
            })
    }
}

// macro_rules! map_impl {
//     ($([$notlast:ident $($p:ident)*],)+) => {
//         $(
//             #[duplicate_item(
//                 __array_type__                       __impl_generics__;
//                 [ BoolArray<D> ]                     [ D, A ];
//                 [ BoolArcArray<D> ]                  [ D, A];
//                 [ BoolArrayView<'a, D> ]             [ 'a, D, A ];
//                 [ BoolArrayViewMut<'a, D> ]          [ 'a, D, A ];
//             )]
//             #[allow(non_snake_case)]
//             impl<__impl_generics__, $($p),*> ArrayWithBoolMaskMethods<D, A, &dyn Fn($($p::Item),*), Zip<($($p,)*), D>>
//             for __array_type__
//             where   D: Dimension,
//                     $($p: NdProducer<Dim=D> ,)*
//             {
//                 fn mask_apply_inplace(
//                     &self,
//                     array: Zip<($($p,)*), D>,
//                     f: &dyn Fn($($p::Item),*),
//                 ) {
//                     array
//                     .and(self)
//                     .for_each(| $($p),* , mask | {
//                         if *mask {
//                             f($($p),*);
//                         }
//                     })
//                 }
//             }
//         )+
//     }
// }

// map_impl! {
//     [true P1],
//     [true P1 P2],
//     [true P1 P2 P3],
//     [true P1 P2 P3 P4],
//     [true P1 P2 P3 P4 P5],
//     // [false P1 P2 P3 P4 P5 P6],
// }
