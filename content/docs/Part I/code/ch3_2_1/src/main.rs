use ndarray::{Array4, Array, s};
use rand::distr::Uniform;
use rand::Rng;

// Convolution operation
fn conv2d(input: &Array4<f64>, filter: &Array4<f64>, stride: usize, padding: usize) -> Array4<f64> {
    let (input_height, input_width, input_depth, _) = (
        input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
    let num_filters = filter.shape()[3];

    let output_height = (input_height - filter.shape()[0] + 2 * padding) / stride + 1;
    let output_width = (input_width - filter.shape()[1] + 2 * padding) / stride + 1;
    let mut output = Array::zeros((output_height, output_width, num_filters, 1));

    for n in 0..num_filters {
        for i in 0..output_height {
            for j in 0..output_width {
                for d in 0..input_depth {
                    let input_region = input.slice(s![
                        i * stride..i * stride + filter.shape()[0],
                        j * stride..j * stride + filter.shape()[1],
                        d,
                        0
                    ]);

                    let filter_region = filter.slice(s![.., .., d, n]);
                    output[[i, j, n, 0]] += (&input_region * &filter_region).sum();
                }
            }
        }
    }
    output
}

fn main() {
    let mut rng = rand::rng();
    let input = Array4::from_shape_fn((28, 28, 3, 1), |_| rng.sample(Uniform::new(0.0, 1.0).unwrap())); // Example input
    let filter = Array4::from_shape_fn((3, 3, 3, 32), |_| rng.sample(Uniform::new(-1.0, 1.0).unwrap())); // 32 filters
    let output = conv2d(&input, &filter, 1, 0); // No padding, stride 1
    println!("Output shape: {:?}", output.shape());
}
