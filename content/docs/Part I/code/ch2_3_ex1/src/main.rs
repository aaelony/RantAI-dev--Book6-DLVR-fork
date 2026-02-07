use ndarray::{Array1, Array2, arr1, arr2};

fn gradient_descent(weights: &mut Array1<f64>, learning_rate: f64, gradient: &Array1<f64>) {
    for i in 0..weights.len() {
        weights[i] -= learning_rate * gradient[i];
    }
}

fn compute_loss_gradient(weights: &Array1<f64>, inputs: &Array2<f64>, targets: &Array1<f64>) -> Array1<f64> {
    let predictions = inputs.dot(weights);
    let errors = &predictions - targets;
    inputs.t().dot(&errors) / targets.len() as f64
}

fn main() {
    let inputs: Array2<f64> = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let targets: Array1<f64> = arr1(&[5.0, 6.0]);
    let mut weights: Array1<f64> = arr1(&[0.1, 0.1]);

    let learning_rate = 0.01;
    let num_iterations = 1000;

    for _ in 0..num_iterations {
        let gradient = compute_loss_gradient(&weights, &inputs, &targets);
        gradient_descent(&mut weights, learning_rate, &gradient);
    }

    println!("Optimized weights: {:?}", weights);
}
