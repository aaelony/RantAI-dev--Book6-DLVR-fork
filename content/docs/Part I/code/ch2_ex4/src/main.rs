use ndarray;
use ndarray::{Array1, arr1};

fn main() {
    let data1: Array1<f64> = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let data2: Array1<f64> = arr1(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mean1 = data1.mean().unwrap();
    let mean2 = data2.mean().unwrap();

    let covariance = (data1.clone() - mean1) * (data2.clone() - mean2);
    let covariance_mean = covariance.mean().unwrap();
    println!("Covariance: {}", covariance_mean);
}
