use ndarray::{Array2, arr2};

fn parallel_matrix_multiplication(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let mut result = Array2::zeros((a.nrows(), b.ncols()));
    result.axis_iter_mut(ndarray::Axis(0))
        .into_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            row.assign(&a.row(i).dot(b));
        });
    result
}

fn main() {
    let a: Array2<f64> = arr2(&[[1., 2.], [3., 4.]]);
    let b: Array2<f64> = arr2(&[[5., 6.], [7., 8.]]);
    let result = parallel_matrix_multiplication(&a, &b);
    println!("Parallel Matrix product:\n{}", result);
}
