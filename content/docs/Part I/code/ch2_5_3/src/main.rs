use ndarray::Array1;

fn clip_gradients(gradient: &mut Array1<f64>, clip_value: f64) {
    for grad in gradient.iter_mut() {
        if *grad > clip_value {
            *grad = clip_value;
        } else if *grad < -clip_value {
            *grad = -clip_value;
        }
    }
}

fn main() {
    let mut gradient = ndarray::arr1(&[0.5, 1.2, -3.5, 0.8]);
    clip_gradients(&mut gradient, 1.0);
    println!("Clipped gradient: {:?}", gradient);
}
