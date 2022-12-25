use duplicate::duplicate_item;

use ndarray::{
    ArrayBase,
    Array,
    Array2,
    // ArcArray2,
    ArrayView,
    ArrayView1,
    ArrayViewMut,
    Axis,
    Data,
    DataMut,
    Dimension,
    // Ix,
    Ix2,
    RawData,
    Slice,
    SliceArg,
};

pub trait ArrayProxiedMethods<D, A> {
    fn to_owned(&self) -> Array<A, D>
    where A: Clone;
    fn view(&self) -> ArrayView<'_, A, D>;

    fn shape(&self) -> &[usize];

    // These methods are Ix2 only
    // fn row(&self, index: Ix) -> ArrayView1<'_, A>;
    // fn column(&self, index: Ix) -> ArrayView1<'_, A>;

    // fn slice<I>(&self, info: I) -> ArrayView<'_, A, I::OutDim>
    // where   I: SliceArg<D>,
    //         D: Dimension;

    fn slice_axis(&self, axis: Axis, indices: Slice) -> ArrayView<'_, A, D>
    where   D: Dimension;
}

pub trait ArrayMutProxiedMethods<D, A> {
    fn slice_mut<I>(&mut self, info: I) -> ArrayViewMut<'_, A, I::OutDim>
    where   I: SliceArg<D>,
            D: Dimension;
}

impl<S, D, A> ArrayProxiedMethods<D, A> for ArrayBase<S, D>
where   S:RawData<Elem=A>+Data,
        D:Dimension
{
    fn to_owned(&self) -> Array<A, D> 
    where A: Clone
    {
        return ArrayBase::<S, D>::to_owned(&self);
    }
    fn view(&self) -> ArrayView<'_, A, D> {
        return ArrayBase::<S, D>::view(&self);
    }

    fn shape(&self) -> &[usize] {
        return ArrayBase::<S, D>::shape(&self);
    }

    // These methods are Ix2 only
    // fn row(&self, index: Ix) -> ArrayView1<'_, A> {
    //     return ArrayBase::<S, D>::row(&self, index);
    // }

    // The trait `ArrayProxiedMethods` cannot be made into an object
    // ...because method `slice` has generic type parameters
    // fn column(&self, index: Ix) -> ArrayView1<'_, A> {
    //     return ArrayBase::<S, D>::column(&self, index);
    // }

    // fn slice<I>(&self, info: I) -> ArrayView<'_, A, I::OutDim>
    // where   I: SliceArg<D>,
    //         D: Dimension,
    // {
    //     return ArrayBase::<S, D>::slice(self, info);
    // }

    fn slice_axis(&self, axis: Axis, indices: Slice) -> ArrayView<'_, A, D>
    where   D: Dimension
    {
        return ArrayBase::<S, D>::slice_axis(self, axis, indices);
    }
}

impl<S, D, A> ArrayMutProxiedMethods<D, A> for ArrayBase<S, D>
where   S:RawData<Elem=A>+DataMut,
        D:Dimension
{
    fn slice_mut<I>(&mut self, info: I) -> ArrayViewMut<'_, A, I::OutDim>
    where   I: SliceArg<D>,
            D: Dimension
    {
        return ArrayBase::<S, D>::slice_mut(self, info);
    }
}


pub trait ArrayFromDuplicatedRows<A>:ArrayProxiedMethods<Ix2, A> {
    fn from_duplicated_rows(
        row:ArrayView1<'_, A>,
        count:usize,
    ) -> Self
    where A: Clone;
}

#[duplicate_item(
    __array_type__;
    [ Array2 ];
    // [ ArcArray2 ];
)]
impl<A> ArrayFromDuplicatedRows<A> for __array_type__<A> {
    fn from_duplicated_rows(
        row:ArrayView1<'_, A>,
        count:usize,
    ) -> __array_type__<A>
    where A: Clone {
        // Because there is nothing in the array, we can safely init it.
        let mut result = unsafe {
            __array_type__::<A>
            ::uninit((0, row.len()))
            .assume_init()
        };

        for _ in 0..count {
            drop(result.push_row(row));
        }

        return result;
    }
}