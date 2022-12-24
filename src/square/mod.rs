/// Features for Square arrays.
/// 
/// To illustrate the purpose of this module, we can consider a function of::
/// 
///     | (x, y) | x * y
/// 
/// Assume we are to map each and every element of an Array `A` to every other
/// element of itself and apply this function. Since `a * b` is the same as `b * a`,
/// half of the calculations will be duplicated and wasted.
/// 
/// We can apply the calculations to the lower-left half of the resultant array,
/// and mirror the array along the diagonal. This prevents duplicated calculations
/// without any caching.

pub mod func;
pub mod traits;

pub use func::{
    trapizoid_slices_of_lower_half,
};