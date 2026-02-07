use ndarray::{Array1, Array2};

fn mini_batch_gradient_descent(weights: &mut Array1<f64>, inputs: &Array2<f64>, targets: &Array1<f64>, batch_size: usize, learning_rate: f64, num_epochs: usize) {
    let num_samples = inputs.nrows();
    for _ in 0..num_epochs {
        let mut total_gradient = Array1::zeros(weights.len());
        for (batch_inputs, batch_targets) in inputs.axis_chunks_iter(ndarray::Axis(0), batch_size).zip(targets.axis_chunks_iter(ndarray::Axis(0), batch_size)) {
            let predictions = batch_inputs.dot(weights);
            let errors = predictions - batch_targets;
            let gradient = batch_inputs.t().dot(&errors) / batch_targets.len() as f64;
            total_gradient += &gradient;
        }
        *weights = weights.clone() - learning_rate * total_gradient / (num_samples as f64 / batch_size as f64);
    }
}

fn main() {
    let inputs: Array2<f64> = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
    let targets: Array1<f64> = ndarray::arr1(&[3.0, 7.0, 11.0, 15.0]);
    let mut weights: Array1<f64> = ndarray::arr1(&[0.1, 0.1]);

    mini_batch_gradient_descent(&mut weights, &inputs, &targets, 2, 0.01, 1000);
    println!("Optimized weights: {:?}", weights);
}
