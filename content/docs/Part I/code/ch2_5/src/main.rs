use rand;
use rand_distr::{Normal, Distribution};

fn main() {
    let normal = Normal::new(0.0, 1.0).unwrap();  // mean = 0, standard deviation = 1
    let mut rng = rand::thread_rng();

    let sample: f64 = normal.sample(&mut rng);
    println!("Random sample from Gaussian distribution: {}", sample);
}
