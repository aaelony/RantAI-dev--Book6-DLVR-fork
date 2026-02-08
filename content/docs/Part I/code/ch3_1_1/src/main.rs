use ndarray::Array2;
use rand::Rng;

// Sigmoid activation function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Derivative of sigmoid function for backpropagation
fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

struct NeuralNetwork {
    #[allow(dead_code)]
    input_size: usize,
    #[allow(dead_code)]
    hidden_size: usize,
    #[allow(dead_code)]
    output_size: usize,
    weights_input_hidden: Array2<f64>,
    weights_hidden_output: Array2<f64>,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights_input_hidden = Array2::from_shape_fn((input_size, hidden_size), |_| rng.random::<f64>());
        let weights_hidden_output = Array2::from_shape_fn((hidden_size, output_size), |_| rng.random::<f64>());

        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            weights_hidden_output,
        }
    }

    #[allow(dead_code)]
    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let hidden_input = input.dot(&self.weights_input_hidden);
        let hidden_output = hidden_input.mapv(sigmoid);
        let final_input = hidden_output.dot(&self.weights_hidden_output);
        final_input.mapv(sigmoid)
    }

    // Backward pass and weight updates
    fn train(&mut self, input: &Array2<f64>, target: &Array2<f64>, learning_rate: f64) {
        // Forward pass
        let hidden_input = input.dot(&self.weights_input_hidden);
        let hidden_output = hidden_input.mapv(sigmoid);
        let final_input = hidden_output.dot(&self.weights_hidden_output);
        let final_output = final_input.mapv(sigmoid);

        println!("Forward pass:");
        println!("Hidden layer input:\n{}", hidden_input);
        println!("Hidden layer output (after sigmoid):\n{}", hidden_output);
        println!("Output layer input:\n{}", final_input);
        println!("Output layer output (predictions, after sigmoid):\n{}", final_output);

        // Error in output layer
        let output_error = target - &final_output;
        let output_delta = &output_error * &final_output.mapv(sigmoid_derivative);

        println!("\nBackward pass:");
        println!("Output error:\n{}", output_error);
        println!("Output delta (error * sigmoid derivative):\n{}", output_delta);

        // Error in hidden layer
        let hidden_error = output_delta.dot(&self.weights_hidden_output.t());
        let hidden_delta = &hidden_error * &hidden_output.mapv(sigmoid_derivative);

        println!("Hidden layer error:\n{}", hidden_error);
        println!("Hidden delta (error * sigmoid derivative):\n{}", hidden_delta);

        // Update weights with scaled addition
        self.weights_hidden_output.scaled_add(learning_rate, &hidden_output.t().dot(&output_delta));
        self.weights_input_hidden.scaled_add(learning_rate, &input.t().dot(&hidden_delta));

        println!("\nWeight updates:");
        println!("Updated weights (input to hidden):\n{}", self.weights_input_hidden);
        println!("Updated weights (hidden to output):\n{}", self.weights_hidden_output);
    }
}

fn main() {
    let mut nn = NeuralNetwork::new(3, 5, 1);
    let input = Array2::from_elem((1, 3), 1.0);
    let target = Array2::from_elem((1, 1), 0.5);
    nn.train(&input, &target, 0.1);
}
