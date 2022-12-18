#[allow(unused_imports)]
use duplicate::duplicate_item;

mod boolarray;
pub use boolarray::{
    BoolArcArray,
    BoolArcArray1,
    BoolArcArray2,
    BoolArray,
    BoolArray1,
    BoolArray2,
    BoolArrayView,
    BoolArrayViewMut,
    
    ArrayWithBoolIterMethods,
    ArrayWithBoolMaskMethods,

    OptionBoolArray,
    OptionBoolArray1,
    OptionBoolArray2,
};

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
    ArrayWithF64Atan2Methods,
    ArrayWithF64PartialOrd,
    ArrayWithF64MappedOperators,
    ArrayWithF64AngularMethods,
    ArrayWithF64LatLngMethods,
    
    OptionF64Array,
    OptionF64Array1,
    OptionF64Array2,
};

/// ====================================================================================
/// UNIT TESTS



#[cfg(test)]
mod test_boolarray {
    use super::*;

    use ndarray::prelude::*;
    
    
    #[duplicate_item(
        ArrayType           TestName                BaseType        ReshapeFunction                     Answer;
        [ BoolArray1 ]      [test_boolarray1]       [ Array ]       [ into_shape((100, )).unwrap() ]    [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97];
        [ BoolArray2 ]      [test_boolarray2]       [ Array ]       [ into_shape((20, 5)).unwrap() ]    [(0, 1), (0, 4), (1, 2), (2, 0), (2, 3), (3, 1), (3, 4), (4, 2), (5, 0), (5, 3), (6, 1), (6, 4), (7, 2), (8, 0), (8, 3), (9, 1), (9, 4), (10, 2), (11, 0), (11, 3), (12, 1), (12, 4), (13, 2), (14, 0), (14, 3), (15, 1), (15, 4), (16, 2), (17, 0), (17, 3), (18, 1), (18, 4), (19, 2)];
        [ BoolArcArray1 ]   [test_boolarcarray1]    [ ArcArray ]    [ reshape((100, )) ]                [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97];
        [ BoolArcArray2 ]   [test_boolarcarray2]    [ ArcArray ]    [ reshape((20, 5)) ]                [(0, 1), (0, 4), (1, 2), (2, 0), (2, 3), (3, 1), (3, 4), (4, 2), (5, 0), (5, 3), (6, 1), (6, 4), (7, 2), (8, 0), (8, 3), (9, 1), (9, 4), (10, 2), (11, 0), (11, 3), (12, 1), (12, 4), (13, 2), (14, 0), (14, 3), (15, 1), (15, 4), (16, 2), (17, 0), (17, 3), (18, 1), (18, 4), (19, 2)];
    )]
    #[test]
    fn TestName() {
        let arr = {
            BaseType::from_iter(0_usize..100_usize)
            .ReshapeFunction
            .mapv( |v| v%3 == 1)
        };

        assert!(arr.any() == true);
        assert!(arr.all() == false);
        assert!(arr.count() == (100 / 3));

        assert!(arr.indices() == array![Answer]);
    }

    #[duplicate_item(
        ArrayType           TestName                BaseType        ReshapeFunction;
        [ BoolArray1 ]      [test_boolmask1]       [ Array ]       [ into_shape((100, )).unwrap() ];
        [ BoolArray2 ]      [test_boolmask2]       [ Array ]       [ into_shape((20, 5)).unwrap() ];
    )]
    #[test]
    fn TestName() {
        let mut arr = {
            BaseType::from_iter(0_usize..100_usize)
            .ReshapeFunction
        };

        let answer = arr.map(|v| if *v%3 == 1 { v * 3 } else {*v} );
        let mask = arr.mapv( |v| v%3 == 1);
        
        mask.mask_apply_inplace(&mut arr, |v| *v *= 3);

        assert!(arr == answer);
    }
}

#[cfg(test)]
mod test_f64array {
    use std::cmp::Ordering;
    use std::f64::consts;
    use crate::f64array::ArrayWithF64MappedOperators;

    use super::*;
    use ndarray::prelude::*;
    use ndarray::{
        ArcArray2,
    };

    #[duplicate_item(
        ArrayType       TestName;
        [ Array2 ]      [test_f64array_2d];
        [ ArcArray2 ]   [test_f64arcarray_2d];
    )]
    #[test]
    fn TestName() {
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

    #[duplicate_item(
        ArrayType       TestName;
        [ Array2 ]      [test_f64array_opassign_2d];
        [ ArcArray2 ]   [test_f64arcarray_opassign_2d];
    )]
    #[test]
    fn TestName() {
        let mut lhs = F64Array::from_shape_fn(
            (5, 2),
            |(x, y)| (x+y*10) as f64
        );

        let rhs = F64Array::range(-3., 2., 1.);

        // Remember that these operations are done in place.
        // So the following operations cumulates.

        assert_eq!(
            &lhs.add_assign_array1(&rhs),
            &arr2(&[[-3.0, 7.0],
                    [-1.0, 9.0],
                    [1.0, 11.0],
                    [3.0, 13.0],
                    [5.0, 15.0]])
        );

        assert_eq!(
            &lhs.sub_assign_array1(&rhs),
            &arr2(&[[0.0, 10.0],
                    [1.0, 11.0],
                    [2.0, 12.0],
                    [3.0, 13.0],
                    [4.0, 14.0]])
        );

        assert_eq!(
            &lhs.mul_assign_array1(&rhs),
            &arr2(&[[0.0,   -30.0],
                    [-2.0,  -22.0],
                    [-2.0,  -12.0],
                    [0.0,     0.0],
                    [4.0,    14.0]])
        );

        assert_eq!(
            &lhs.div_assign_array1(&F64Array1::from_elem((5,), 0.1)),
            &arr2(&[[0.0,    -300.0],
                    [-20.0,  -220.0],
                    [-20.0,  -120.0],
                    [0.0,       0.0],
                    [40.0,   140.0]])
        );
        
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