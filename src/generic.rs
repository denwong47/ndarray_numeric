use duplicate::duplicate_item;

use ndarray::{
    Array2,
    // ArcArray2,
    ArrayView1,
};

pub trait ArrayFromDuplicatedRows<A> {
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