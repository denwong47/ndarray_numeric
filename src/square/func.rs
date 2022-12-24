use std::ops::RangeInclusive;

pub fn trapizoid_slices_of_lower_half(n:usize, count:usize) -> Vec<RangeInclusive<usize>> {
    let mut result:Vec<RangeInclusive<usize>> = Vec::new();

    let mut n0 = 0_usize;
    let area = (n*n) as f64 / 2. / count as f64;
    
    while n0 < n {
        let n0d = {
            (n0.pow(2) as f64 + 2. * area).sqrt()
            - n0 as f64
        }.round() as usize;

        let n1 = (n0 + n0d as usize).max(n0).min(n);
    
        result.push(n0..=n1);
        
        n0 = n1+1;
    }

    return result;
}