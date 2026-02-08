use ndarray::{Array1, Array2, arr1, arr2};
use rayon::prelude::*;

fn sequential_gradient_descent(weights: &mut Array1<f64>, inputs: &Array2<f64>, targets: &Array1<f64>, learning_rate: f64, num_epochs: usize) {
    for _ in 0..num_epochs {
        let predictions = inputs.dot(weights);
        let errors = &predictions - targets;
        let gradient = inputs.t().dot(&errors) / targets.len() as f64;
        *weights = weights.clone() - learning_rate * gradient;
    }
}

fn parallel_gradient_descent(weights: &mut Array1<f64>, inputs: &Array2<f64>, targets: &Array1<f64>, learning_rate: f64, num_epochs: usize) {
    for _ in 0..num_epochs {
        let predictions = inputs.dot(weights);
        let errors = &predictions - targets;

        // Collect each column of `inputs` as `Array1` for parallel iteration
        let columns: Vec<Array1<f64>> = (0..inputs.ncols())
            .map(|i| inputs.column(i).to_owned())
            .collect();

        // Compute the gradient in parallel
        let gradient: Vec<f64> = columns
            .par_iter()
            .map(|col| col.dot(&errors) / targets.len() as f64)
            .collect();

        // Convert Vec<f64> to Array1<f64>
        let gradient = Array1::from(gradient);

        *weights = weights.clone() - learning_rate * gradient;
    }
}

fn main() {
    let inputs: Array2<f64> = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let targets: Array1<f64> = arr1(&[3.0, 7.0, 11.0]);
    let mut weights: Array1<f64> = arr1(&[0.1, 0.1]);

    println!("Running Sequential Gradient Descent...");
    sequential_gradient_descent(&mut weights, &inputs, &targets, 0.01, 1000);
    println!("Optimized weights (Sequential): {:?}", weights);

    weights.fill(0.1); // Reset weights for parallel run

    println!("Running Parallel Gradient Descent using Rayon...");
    parallel_gradient_descent(&mut weights, &inputs, &targets, 0.01, 1000);
    println!("Optimized weights (Parallel): {:?}", weights);
}
