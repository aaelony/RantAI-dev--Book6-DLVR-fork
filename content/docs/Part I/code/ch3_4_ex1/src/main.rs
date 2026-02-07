use ndarray::Array2;
use rand::Rng;
use rand::distr::Uniform;

// Adam optimizer implementation
struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: Array2<f64>,
    v: Array2<f64>,
    t: usize,
}

impl Adam {
    fn new(input_size: usize, output_size: usize, learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        let m = Array2::zeros((input_size, output_size));
        let v = Array2::zeros((input_size, output_size));
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m,
            v,
            t: 0,
        }
    }

    fn update(&mut self, weights: &mut Array2<f64>, gradients: &Array2<f64>) {
        self.t += 1;

        // Update biased first moment estimate
        self.m = self.beta1 * &self.m + (1.0 - self.beta1) * gradients;

        // Update biased second moment estimate
        self.v = self.beta2 * &self.v + (1.0 - self.beta2) * gradients.mapv(|g| g * g);

        // Compute bias-corrected first and second moment estimates
        let m_hat = &self.m / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = &self.v / (1.0 - self.beta2.powi(self.t as i32));

        // Update weights
        *weights -= &(self.learning_rate * m_hat / (v_hat.mapv(f64::sqrt) + self.epsilon));
    }
}

fn main() {
    let input_size = 3;
    let output_size = 2;

    // Initialize weights and gradients with random values
    let mut rng = rand::thread_rng();
    let mut weights = Array2::from_shape_fn((input_size, output_size), |_| rng.sample(Uniform::new(-1.0, 1.0).unwrap()));
    let gradients = Array2::from_shape_fn((input_size, output_size),   |_| rng.sample(Uniform::new(-0.1, 0.1).unwrap()));

    // Adam optimizer with specified parameters
    let mut adam = Adam::new(input_size, output_size, 0.001, 0.9, 0.999, 1e-8);

    // Run optimizer updates
    for _ in 0..1000 {
        adam.update(&mut weights, &gradients);
    }

    println!("Updated weights: {:?}", weights);
}
