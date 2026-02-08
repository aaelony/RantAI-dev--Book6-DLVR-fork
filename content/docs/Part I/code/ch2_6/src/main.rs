use rand;
use rand_distr::{Distribution as RandDistribution, Normal};
use statrs::statistics::{Data, Distribution as StatrsDistribution};

fn main() {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();

    // Generate a sample of random data
    let samples: Vec<f64> = (0..1000).map(|_| normal.sample(&mut rng)).collect();

    // Use statrs crate for statistical analysis
    let data = Data::new(samples);

    println!("Mean: {}", data.mean().unwrap_or_default());
    println!("Variance: {}", data.variance().unwrap_or_default());
    println!("Skewness: {}", data.skewness().unwrap_or_default());
}
