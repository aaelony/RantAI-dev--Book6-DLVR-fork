use plotters::prelude::*;
use rand::Rng;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};
use tch::nn::ModuleT;

fn main() -> anyhow::Result<()> {
    // 1. Generate 2D synthetic datasets with circular pattern
    let n_samples = 1000;
    let mut rng = rand::thread_rng();
    let mut data = Vec::new();
    let mut labels = Vec::new();

    for _ in 0..n_samples {
        let r = rng.gen_range(0.0..2.0);
        let theta = rng.gen_range(0.0..(2.0 * std::f64::consts::PI));
        let x = r * theta.cos();
        let y = r * theta.sin();
        data.push([x, y]);
        labels.push(if r < 1.0 { 0 } else { 1 });
    }

    let data: Tensor = Tensor::of_slice2(&data)
        .to_kind(Kind::Float)
        .to_device(Device::Cpu);
    let labels: Tensor = Tensor::of_slice(&labels)
        .to_kind(Kind::Int64)
        .to_device(Device::Cpu);

    // 2. Define Multi-Layer Perceptron with 6 hidden layers (8 neurons each)
    let vs = nn::VarStore::new(Device::Cpu);
    let net = nn::seq()
        .add(nn::linear(&vs.root(), 2, 8, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(&vs.root(), 8, 8, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(&vs.root(), 8, 8, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(&vs.root(), 8, 8, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(&vs.root(), 8, 8, Default::default()))
        .add_fn(|x| x.relu())
        .add(nn::linear(&vs.root(), 8, 2, Default::default()));

    // 3. Train the model using Adam optimizer
    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

    for epoch in 1..=500 {
        let preds = net.forward_t(&data, true);
        let loss = preds.cross_entropy_for_logits(&labels);
        opt.backward_step(&loss);

        if epoch % 50 == 0 {
            println!("Epoch: {}, Loss: {:.4}", epoch, loss.double_value(&[]));
        }
    }

    // 4. Evaluate and visualize the results
    let preds = net.forward_t(&data, false).argmax(1, false);
    let accuracy = preds.eq_tensor(&labels).to_kind(Kind::Float).mean(Kind::Float);
    println!("Accuracy: {:.2}%", accuracy.double_value(&[]) * 100.0);

    // Visualization setup
    let root = BitMapBackend::new("classification_visualization.png", (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .caption("MLP Classification and Predictions", ("sans-serif", 30))
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-2.5..2.5, -2.5..2.5)?;

    chart.configure_mesh().draw()?;

    // Plot decision boundary
    let resolution = 200;
    let mut grid_data = vec![];
    for i in 0..resolution {
        for j in 0..resolution {
            let x = -2.5 + 5.0 * (i as f64) / (resolution as f64);
            let y = -2.5 + 5.0 * (j as f64) / (resolution as f64);
            grid_data.push([x, y]);
        }
    }

    let grid_tensor: Tensor = Tensor::of_slice2(&grid_data)
        .to_kind(Kind::Float)
        .to_device(Device::Cpu);
    let grid_preds = net.forward_t(&grid_tensor, false).argmax(1, false);

    let grid_points: Vec<(f64, f64, u8)> = grid_data
        .iter()
        .zip(grid_preds.iter::<i64>().unwrap())
        .map(|(coords, label)| (coords[0], coords[1], label as u8))
        .collect();

    chart.draw_series(
        grid_points.iter().map(|(x, y, label)| {
            let color = if *label == 0 { &BLUE.mix(0.2) } else { &RED.mix(0.2) };
            Circle::new((*x, *y), 1, color.filled())
        }),
    )?;

    // Plot original data points
    let data_points: Vec<((f64, f64), i64)> = data
        .to_kind(Kind::Double)
        .chunk(2, 1)
        .iter()
        .zip(labels.iter::<i64>().unwrap())
        .map(|(coords, label)| {
            let x = coords.double_value(&[0]);
            let y = coords.double_value(&[1]);
            ((x, y), label)
        })
        .collect();

    chart.draw_series(
        data_points
            .iter()
            .filter(|(_, label)| *label == 0)
            .map(|((x, y), _)| Circle::new((*x, *y), 3, BLUE.filled())),
    )?;

    chart.draw_series(
        data_points
            .iter()
            .filter(|(_, label)| *label == 1)
            .map(|((x, y), _)| Circle::new((*x, *y), 3, RED.filled())),
    )?;

    root.present()?;
    println!("Visualization saved to classification_visualization.png");

    Ok(())
}
