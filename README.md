### Read Me for
# ndarray_numeric

Extends `ndarray::ArrayBase` with `f64` elements to use `f64` methods, vectorised
across each item of the array:

```rust
use std;
use f64array::F64Array;

let shape = (3, 4);

// Generate an array of degrees
let degs = F64Array::from_shape_fn(
    shape,
    |x| ((x.0)*shape.1 + (x.1)) as f64 * 10.
);

// ndarrays already support simple arithematics out of the box
let rads = degs  * std::f64::consts::PI / 180.;

// F64Array further allows `f64` native methods to be used on the array.
let sin_values = rads.sin();
```