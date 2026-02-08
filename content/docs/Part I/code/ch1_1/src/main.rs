use rand::Rng;
use plotters::prelude::*;
use std::f64;

// Function to generate synthetic 2D dataset
fn generate_data(n_samples: usize) -> (Vec<(f64, f64)>, Vec<i32>) {
    let mut rng = rand::thread_rng();
    let mut data = Vec::new();
    let mut labels = Vec::new();

    for _ in 0..n_samples {
        let x1: f64 = rng.gen_range(-1.0..1.0);
        let x2: f64 = rng.gen_range(-1.0..1.0);
        let label = if x1 + x2 > 0.0 { 1 } else { -1 }; // Linearly separable condition
        data.push((x1, x2));
        labels.push(label);
    }

    (data, labels)
}

// Perceptron Learning Algorithm
fn perceptron_learning_algorithm(
    data: &[(f64, f64)],
    labels: &[i32],
    learning_rate: f64,
    max_iter: usize,
) -> (f64, f64, f64) {
    let mut w1 = 0.0;
    let mut w2 = 0.0;
    let mut b = 0.0;

    for _ in 0..max_iter {
        let mut converged = true;

        for (i, &(x1, x2)) in data.iter().enumerate() {
            let y = labels[i];
            let prediction = if w1 * x1 + w2 * x2 + b > 0.0 { 1 } else { -1 };

            if prediction != y {
                // Update weights and bias
                w1 += learning_rate * (y - prediction) as f64 * x1;
                w2 += learning_rate * (y - prediction) as f64 * x2;
                b += learning_rate * (y - prediction) as f64;
                converged = false;
            }
        }

        if converged {
            break;
        }
    }

    (w1, w2, b)
}

// Visualization of data and decision boundary using plotters
fn visualize(
    data: &[(f64, f64)],
    labels: &[i32],
    weights: (f64, f64, f64),
    filename: &str,
) {
    let (w1, w2, b) = weights;

    let root = BitMapBackend::new(filename, (800, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Perceptron Learning Algorithm", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-1.5..1.5, -1.5..1.5)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    // Plot data points
    for (i, &(x1, x2)) in data.iter().enumerate() {
        let color = if labels[i] == 1 { &RED } else { &BLUE };
        chart.draw_series(PointSeries::of_element(
            [(x1, x2)],
            5,
            color,
            &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
        )).unwrap();
    }

    // Plot decision boundary
    let x_min = -1.5;
    let x_max = 1.5;
    let y_min = (-b - w1 * x_min) / w2;
    let y_max = (-b - w1 * x_max) / w2;

    chart
        .draw_series(LineSeries::new([(x_min, y_min), (x_max, y_max)], &BLACK))
        .unwrap();

    root.present().unwrap();
}

fn main() {
    let n_samples = 200;
    let (data, labels) = generate_data(n_samples);

    let learning_rate = 0.1;
    let max_iter = 1000;
    let weights = perceptron_learning_algorithm(&data, &labels, learning_rate, max_iter);

    println!("Learned weights: {:?}", weights);

    visualize(&data, &labels, weights, "perceptron_visualization.png");
}
