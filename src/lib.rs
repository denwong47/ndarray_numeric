#[allow(unused_imports)]
use duplicate::duplicate_item;

mod f64array;
pub use f64array::{
    F64ArcArray,
    F64ArcArray1,
    F64ArcArray2,
    F64Array,
    F64Array1,
    F64Array2,
    F64ArrayView,
    F64ArrayViewMut,
    F64LatLng,
    F64LatLngView,
    F64LatLngViewMut,
    F64LatLngArcArray,
    F64LatLngArray,
    F64LatLngArrayView,
    F64LatLngArrayViewMut,
    
    ArrayWithF64Methods,
    ArrayWithF64PartialOrd,
    ArrayWithF64AngularMethods,
    ArrayWithF64LatLngMethods,
    
    OptionF64Array,
    OptionF64Array1,
    OptionF64Array2,
};

/// ====================================================================================
/// UNIT TESTS

#[cfg(test)]
#[duplicate_item(
    ArrayType       TestName;
    [ Array2 ]      [test_array2];
    [ ArcArray2 ]   [test_arcarray2];
)]
mod TestName {
    use std::cmp::Ordering;
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

        // Test PartialOrd
        let cmp_ge = ArrayType::from_shape_vec(
            (3, 4),
            vec![
                false,              false,               false,              false,
                false,              true,                true,               true, 
                true,               true,                true,               true
            ]
        ).unwrap();
        println!("{:?}", &arr.ge(&0.8726646259971648));
        assert!(&arr.ge(&0.87) == cmp_ge);

        let cmp_gt = ArrayType::from_shape_vec(
            (3, 4),
            vec![
                false,              false,               false,              false,
                false,              false,               true,               true, 
                true,               true,                true,               true
            ]
        ).unwrap();
        assert!(&arr.gt(&0.88) == cmp_gt);

        let cmp_partial = ArrayType::from_shape_vec(
            (3, 4),
            vec![
                Some(Ordering::Less), Some(Ordering::Less), Some(Ordering::Less), Some(Ordering::Less),
                Some(Ordering::Less), Some(Ordering::Equal), Some(Ordering::Greater), Some(Ordering::Greater),
                Some(Ordering::Greater), Some(Ordering::Greater), Some(Ordering::Greater), Some(Ordering::Greater),
            ]
        ).unwrap();
        assert!(&arr.partial_cmp(&0.8726646259971648) == cmp_partial);
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

#[cfg(test)]
mod test_latlng {
    use super::f64array::{
        F64Array,
        ArrayWithF64LatLngMethods,
    };

    static SHAPE:(usize, usize) = (18, 2);

    #[test]
    #[allow(unused_variables)]
    fn test_normalize() {
        // Generate an array of degrees
        let mut degs = F64Array::from_shape_fn(
            SHAPE,
            |x| ((x.0)*SHAPE.1 + (x.1)) as f64 * 40. - 180.
        );

        // Normalize the numbers in place.
        degs.normalize();
        
        // Check normalization results
        assert!(degs == F64Array::from_shape_vec(
            SHAPE,
            vec![
                0.0, 40.0,
                -80.0, 120.0,
                -20.0, 20.0,
                60.0, 100.0,
                40.0, 0.0,
                -40.0, 80.0,
                -60.0, -20.0,
                20.0, 60.0,
                80.0, -40.0,
                0.0, 40.0,
                -80.0, 120.0,
                -20.0, 20.0,
                60.0, 100.0,
                40.0, 0.0,
                -40.0, 80.0,
                -60.0, -20.0,
                20.0, 60.0,
                80.0, -40.0
            ]
        ).unwrap());
    }
}