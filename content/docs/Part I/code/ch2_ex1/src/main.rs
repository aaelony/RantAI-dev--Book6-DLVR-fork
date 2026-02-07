use nalgebra as na;
use na::DMatrix;

fn main() {
    let a = DMatrix::<f64>::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let svd = a.svd(true, true);  // Perform SVD
    println!("Singular values: {:?}", svd.singular_values);
}
