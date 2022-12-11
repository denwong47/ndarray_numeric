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

pub trait ArrayWithBoolIterMethods<D>
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
    ArrayType                            Generics;
    [ BoolArray<D> ]                     [ D ];
    [ BoolArcArray<D> ]                  [ D ];
    [ BoolArrayView<'a, D> ]             [ 'a, D ];
    [ BoolArrayViewMut<'a, D> ]          [ 'a, D ];
)]
impl<Generics> ArrayWithBoolIterMethods<D>
for ArrayType
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
