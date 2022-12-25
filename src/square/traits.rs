use duplicate::duplicate_item;

use ndarray::{
    ArrayBase,
    Array2,
    ArcArray2,
    ArrayView1,
    ArrayView2,
    ArrayViewMut2,
    Axis,
    Data,
    Ix2,
    s,
    Slice,
};
use super::func;

use super::super::generic::{
    ArrayProxiedMethods,
    // ArrayMutProxiedMethods,
};


pub trait SquareShapedArray<D, A, T>: ArrayProxiedMethods<D, A>
where   A: Clone
{
    fn from_mapped_array2_fn<F>(
        arr: &T,
        f: F,
        workers: usize,
    ) -> Self
    where   F: Fn(ArrayView1<'_, A>, ArrayView2<'_, A>)->Array2<A>;
}

#[duplicate_item(
    __array_type__                      __impl_generics__;
    [ Array2<A> ]                       [ A ];
    [ ArcArray2<A> ]                    [ A ];
    [ ArrayView2<'a, A> ]               [ 'a, A ];
    [ ArrayViewMut2<'a, A> ]            [ 'a, A ];
)]
impl<__impl_generics__> SquareShapedArray<Ix2, A, __array_type__> for Array2<A>
where A: Clone
{
    /// 2-dimensional array from applying a function between each pair of values in a 1-dimensional array.
    /// 
    /// This function avoids duplicating calculations by only calling the function only
    /// once for each identical pairs.
    /// 
    /// .. warning::
    ///     For this function to be usable, `f(a, b)` *MUST BE* the same as `f(b, a)`.
    /// 
    ///     This function will NOT calculate both - it will only calculate `f(a, b)` and
    ///     clone the result to mirrored position.
    fn from_mapped_array2_fn<F>(
        arr: &__array_type__,
        f: F,
        workers: usize,
    ) -> Self
    where   F: Fn(ArrayView1<'_, A>, ArrayView2<'_, A>)->Array2<A>
    {
        let len = arr.shape()[0];
        let shape = (len, len);

        let mut result = Array2::<A>::uninit(shape);

        func::trapizoid_slices_of_lower_half(len, workers)
             .iter()
             .for_each(
                | range | {
                    let range_factory = || range.clone();

                    range_factory().for_each(
                        | row | {
                            // Include (row, row) so that the diagonal will be calculated
                            let e_ref = Slice::from(..=row);
                            let s = arr.row(row);
                            let e = arr.slice_axis(Axis(0), e_ref);
                            
                            let to_ref = s![row, ..=row];
                            let mut to_slice = result.slice_mut(to_ref);

                            let calculated = f(s, e);

                            // calculated
                            // .move_into_uninit(&mut to_slice);
                        }
                    );
                }
             );
        
        // TODO Placeholder only
        return unsafe {
            result.assume_init()
        };
    }
}