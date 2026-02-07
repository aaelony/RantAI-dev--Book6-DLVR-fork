use ndarray::Array2;
use rand::Rng;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

struct NeuralNetwork {
    weights_input_hidden: Array2<f64>,
    weights_hidden_output: Array2<f64>,
    learning_rate: f64,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let mut rng = rand::rng();
        let weights_input_hidden = Array2::from_shape_fn((input_size, hidden_size), |_| rng.random::<f64>());
        let weights_hidden_output = Array2::from_shape_fn((hidden_size, output_size), |_| rng.random::<f64>());

        NeuralNetwork {
            weights_input_hidden,
            weights_hidden_output,
            learning_rate,
        }
    }

    fn forward(&self, input: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let hidden_input = input.dot(&self.weights_input_hidden);
        let hidden_output = hidden_input.mapv(sigmoid);
        let final_input = hidden_output.dot(&self.weights_hidden_output);
        let final_output = final_input.mapv(sigmoid);
        (hidden_output, final_output)
    }

    fn backward(&mut self, input: &Array2<f64>, hidden_output: &Array2<f64>, output: &Array2<f64>, target: &Array2<f64>) {
        let output_error = target - output;
        let output_delta = &output_error * &output.mapv(sigmoid_derivative);

        let hidden_error = output_delta.dot(&self.weights_hidden_output.t());
        let hidden_delta = &hidden_error * &hidden_output.mapv(sigmoid_derivative);

        // Print values during the backward pass
        println!("Output error:\n{}", output_error);
        println!("Output delta:\n{}", output_delta);
        println!("Hidden error:\n{}", hidden_error);
        println!("Hidden delta:\n{}", hidden_delta);

        self.weights_hidden_output.scaled_add(self.learning_rate, &hidden_output.t().dot(&output_delta));
        self.weights_input_hidden.scaled_add(self.learning_rate, &input.t().dot(&hidden_delta));
    }

    fn train(&mut self, input: &Array2<f64>, target: &Array2<f64>) {
        let (hidden_output, final_output) = self.forward(input);

        // Print values during the forward pass
        println!("Input:\n{}", input);
        println!("Hidden layer output:\n{}", hidden_output);
        println!("Final output (predictions):\n{}", final_output);
        println!("Target:\n{}", target);

        self.backward(input, &hidden_output, &final_output, target);
    }
}

fn main() {
    let input = Array2::from_elem((1, 3), 1.0);
    let target = Array2::from_elem((1, 1), 0.5);
    let mut nn = NeuralNetwork::new(3, 5, 1, 0.1);

    nn.train(&input, &target);
}
