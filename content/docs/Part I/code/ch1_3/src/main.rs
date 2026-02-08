use plotters::prelude::*;
use rand::Rng;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};
use tch::nn::ModuleT;

fn main() -> anyhow::Result<()> {
    // Generate 2D synthetic datasets with circular pattern
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

    // Define Multi-Layer Perceptron with 6 hidden layers (8 neurons each)
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

    // Train the model using Adam optimizer
    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
    let mut loss_history = vec![];

    for epoch in 1..=2000 {
        let preds = net.forward_t(&data, true);
        let loss = preds.cross_entropy_for_logits(&labels);
        opt.backward_step(&loss);

        // Store loss for visualization
        loss_history.push(loss.double_value(&[]));

        // Print progress every 50 epochs
        if epoch % 50 == 0 {
            println!("Epoch: {}, Loss: {:.4}", epoch, loss.double_value(&[]));
        }
    }

    // Visualize loss curve
    plot_loss_curve("loss_curve.png", &loss_history)?;

    // Visualize heatmap of gradients
    let grads = visualize_gradients(&vs)?;
    plot_gradients("gradient_heatmap.png", grads)?;

    Ok(())
}

fn plot_loss_curve(filename: &str, loss_history: &[f64]) -> anyhow::Result<()> {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_loss = *loss_history.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss Curve", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..loss_history.len(), 0.0..max_loss)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        loss_history.iter().enumerate().map(|(x, &y)| (x, y)),
        &BLUE,
    ))?;

    root.present()?;
    println!("Loss curve saved to {}", filename);
    Ok(())
}

fn visualize_gradients(vs: &nn::VarStore) -> Result<Tensor, tch::TchError> {
    // Extract gradients from all parameters
    let grads: Vec<Tensor> = vs
        .variables()
        .values()
        .map(|var| {
            let grad = var.grad();
            if grad.numel() == 0 {
                Tensor::zeros_like(var)
            } else {
                grad
            }
        })
        .collect();

    // Concatenate gradients into a single tensor for visualization
    let flattened_grads = Tensor::cat(&grads.iter().map(|g| g.view(-1)).collect::<Vec<_>>(), 0);
    Ok(flattened_grads)
}

fn plot_gradients(filename: &str, gradients: Tensor) -> anyhow::Result<()> {
    // Extract gradient data as a Vec<f32>
    let tensor = gradients.abs().to_kind(Kind::Float).view([-1]);
    let numel = tensor.size()[0] as usize;
    let gradient_data: Vec<f32> = (0..numel)
        .map(|i| tensor.double_value(&[i as i64]) as f32)
        .collect();

    // Find the maximum gradient value
    let max_grad = gradient_data.iter().cloned().fold(0.0_f32, f32::max);

    // Set up the plot
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Gradient Heatmap", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..gradient_data.len(), 0.0..max_grad as f64)?;

    chart.configure_mesh().draw()?;

    // Plot the gradient data as a line series
    chart.draw_series(LineSeries::new(
        gradient_data.iter().enumerate().map(|(x, &y)| (x, y as f64)),
        &RED,
    ))?;

    root.present()?;
    println!("Gradient heatmap saved to {}", filename);
    Ok(())
}
