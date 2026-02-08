use ndarray::{Array2, Array1};
use rand::Rng;
use rand::distr::Uniform;

// Recurrent step
fn rnn_step(x_t: &Array1<f64>, h_t_prev: &Array1<f64>, w_xh: &Array2<f64>, w_hh: &Array2<f64>) -> Array1<f64> {
    let h_t = w_xh.dot(x_t) + w_hh.dot(h_t_prev);
    h_t.mapv(|x| x.tanh())
}

fn main() {
    let mut rng = rand::rng();

    // Generate random input array for 10 timesteps and 5 features
    let input = Array2::from_shape_fn((10, 5), |_| rng.sample(Uniform::new(0.0, 1.0).unwrap()));

    // Initialize hidden state with zeros (5 hidden units)
    let mut hidden_state = Array1::zeros(5);

    // Generate random weights for w_xh and w_hh
    let w_xh = Array2::from_shape_fn((5, 5), |_| rng.sample(Uniform::new(-1.0, 1.0).unwrap()));
    let w_hh = Array2::from_shape_fn((5, 5), |_| rng.sample(Uniform::new(-1.0, 1.0).unwrap()));

    // Iterate over each timestep in the input sequence
    for t in 0..input.shape()[0] {
        hidden_state = rnn_step(&input.row(t).to_owned(), &hidden_state, &w_xh, &w_hh);
        println!("Hidden state at timestep {}: {:?}", t, hidden_state);
    }
}
