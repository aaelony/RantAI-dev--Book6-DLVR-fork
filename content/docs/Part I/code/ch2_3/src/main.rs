use ndarray;
use ndarray::{Array1, arr1};

fn main() {
    let data: Array1<f64> = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    let mean = data.mean().unwrap();
    println!("Mean: {}", mean);

    let variance = data.mapv(|x| (x - mean).powi(2)).mean().unwrap();
    println!("Variance: {}", variance);
}
