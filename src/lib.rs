#[allow(unused_imports)]
use duplicate::duplicate_item;

mod f64array;
pub use f64array::{
    F64ArcArray,
    F64Array,
    ArrayWithF64Methods,
    ArrayWithF64AngularMethods,
    F64ArrayView,
};

#[cfg(test)]
#[duplicate_item(
    ArrayType       TestName;
    [ Array2 ]      [test_array2];
    [ ArcArray2 ]   [test_arcarray2];
)]
mod TestName {
    use std::f64::consts;
    use super::*;
    use ndarray::{ArrayType, s};

    #[test]
    fn test_f64array_2d() {
        let arr = F64Array::from_shape_fn(
            (3, 4),
            |x| ((x.0)*4 + (x.1)) as f64 * consts::PI / 180. * 10.
        );

        let ans = ArrayType::from_shape_vec(
            (3, 4),
            vec![
                0.0,                0.17453292519943295, 0.3490658503988659, 0.5235987755982988,
                0.6981317007977318, 0.8726646259971648,  1.0471975511965976, 1.2217304763960306,
                1.3962634015954636, 1.5707963267948966,  1.7453292519943295, 1.9198621771937625
            ]
        ).unwrap();

        assert!(arr==ans);
        
        let slice = ArrayType::from_shape_vec(
            (2, 2),
            vec![
                0.17364817766693033, 0.3420201433256687,
                0.766044443118978, 0.8660254037844386
            ]
        ).unwrap();

        assert!(&arr.slice(s![0..2, 1..3]).sin() == slice);
    }
}


#[cfg(test)]
mod test_readme {
    use super::f64array::{
        F64Array,
        ArrayWithF64Methods,
        ArrayWithF64AngularMethods,
    };

    static SHAPE:(usize, usize) = (3, 4);

    #[test]
    #[allow(unused_variables)]
    fn test_readme() {
        // Generate an array of degrees
        let degs = F64Array::from_shape_fn(
            SHAPE,
            |x| ((x.0)*SHAPE.1 + (x.1)) as f64 * 10.
        );

        // ndarrays already support simple arithematics out of the box
        let rads = degs.to_rad();

        // F64Array further allows `f64` native methods to be used on the array.
        let sin_values = rads.sin();
    }
}