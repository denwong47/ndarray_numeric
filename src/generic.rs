use duplicate::duplicate_item;

use ndarray::{
    ArrayBase,
    Array2,
    // ArcArray2,
    ArrayView1,
    Dimension,
    RawData,
};

pub trait ArrayProxiedMethods {
    fn shape(&self) -> &[usize];
}

impl<S, D> ArrayProxiedMethods for ArrayBase<S, D>
where   S:RawData,
        D:Dimension
{
    fn shape(&self) -> &[usize] {
        return ArrayBase::<S, D>::shape(&self);
    }
}

pub trait ArrayFromDuplicatedRows<A>:ArrayProxiedMethods {
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