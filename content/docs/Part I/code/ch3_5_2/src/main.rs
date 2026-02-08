use tch::{nn, nn::Module, Device, Tensor};
use tch::Kind;

fn l2_regularization(weights: &Tensor, lambda: f64) -> Tensor {
    weights.pow_tensor_scalar(2).sum(Kind::Float) * lambda
}

fn dropout(input: &Tensor, dropout_rate: f64) -> Tensor {
    input.dropout(dropout_rate, true)
}

fn precision_recall_f1(y_true: &Tensor, y_pred: &Tensor, threshold: f64) -> (f64, f64, f64) {
    // Convert Bool prediction to Float so all comparisons are same-type
    let pred_class = y_pred.ge(threshold).to_kind(Kind::Float);

    // Create comparison scalars on the same device and dtype as y_true
    let one = y_true.ones_like();
    let zero = y_true.zeros_like();

    println!("[DEBUG] pred_class shape: {:?}, kind: {:?}", pred_class.size(), pred_class.kind());
    println!("[DEBUG] y_true    shape: {:?}, kind: {:?}", y_true.size(), y_true.kind());
    println!("[DEBUG] pred_class sum (num predicted positive): {:.1}", pred_class.sum(Kind::Float).double_value(&[]));
    println!("[DEBUG] y_true    sum (num actual positive):     {:.1}", y_true.sum(Kind::Float).double_value(&[]));

    // TP: predicted positive AND actually positive
    let true_positive = pred_class
        .eq_tensor(&one)
        .logical_and(&y_true.eq_tensor(&one))
        .to_kind(Kind::Float)
        .sum(Kind::Float);

    // FP: predicted positive AND actually negative
    let false_positive = pred_class
        .eq_tensor(&one)
        .logical_and(&y_true.eq_tensor(&zero))
        .to_kind(Kind::Float)
        .sum(Kind::Float);

    // FN: predicted negative AND actually positive
    let false_negative = pred_class
        .eq_tensor(&zero)
        .logical_and(&y_true.eq_tensor(&one))
        .to_kind(Kind::Float)
        .sum(Kind::Float);

    // TN: predicted negative AND actually negative
    let true_negative = pred_class
        .eq_tensor(&zero)
        .logical_and(&y_true.eq_tensor(&zero))
        .to_kind(Kind::Float)
        .sum(Kind::Float);

    let tp_val = true_positive.double_value(&[]);
    let fp_val = false_positive.double_value(&[]);
    let fn_val = false_negative.double_value(&[]);
    let tn_val = true_negative.double_value(&[]);

    println!("[DEBUG] TP: {tp_val}, FP: {fp_val}, FN: {fn_val}, TN: {tn_val}");
    println!("[DEBUG] Confusion matrix total: {}", tp_val + fp_val + fn_val + tn_val);

    let precision = if tp_val + fp_val > 0.0 {
        tp_val / (tp_val + fp_val)
    } else {
        0.0
    };

    let recall = if tp_val + fn_val > 0.0 {
        tp_val / (tp_val + fn_val)
    } else {
        0.0
    };

    let f1 = if precision + recall > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else {
        0.0
    };

    println!("[DEBUG] Precision: {precision:.4}, Recall: {recall:.4}, F1: {f1:.4}");

    (precision, recall, f1)
}

fn main() {
    let device = Device::cuda_if_available();
    println!("[DEBUG] Using device: {:?}", device);

    let vs = nn::VarStore::new(device);

    let net = nn::seq()
        .add(nn::linear(vs.root() / "layer1", 3, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs.root() / "layer2", 128, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs.root() / "layer3", 64, 1, Default::default()));

    let xs = Tensor::randn(&[100, 3], (Kind::Float, device));
    let ys = Tensor::randint(2, &[100, 1], (Kind::Float, device));

    println!("[DEBUG] ys:");
    for i in 0..ys.size()[0] {
        //println!("\t{}: {:.1}", i, ys.double_value(&[i, 0]));
        print!("{:.1}, ", ys.double_value(&[i,0]));
    }
    println!(" ");

    println!("[DEBUG] xs shape: {:?}, kind: {:?}", xs.size(), xs.kind());
    println!("[DEBUG] ys shape: {:?}, kind: {:?}", ys.size(), ys.kind());
    println!("[DEBUG] ys distribution: {:.1} ones out of {} samples",
        ys.sum(Kind::Float).double_value(&[]), ys.size()[0]);

    // Dropout applied during forward pass
    let _dropped_out_input = dropout(&xs, 0.5);

    // L2 regularization
    let logits = net.forward(&xs);
    let _loss = l2_regularization(&logits, 0.01);

    println!("[DEBUG] logits shape: {:?}, kind: {:?}", logits.size(), logits.kind());
    println!("[DEBUG] logits min: {:.4}, max: {:.4}, mean: {:.4}",
        logits.min().double_value(&[]),
        logits.max().double_value(&[]),
        logits.mean(Kind::Float).double_value(&[]));

    // Apply sigmoid to convert logits to probabilities
    let predictions = logits.sigmoid();

    println!("[DEBUG] predictions (sigmoid) min: {:.4}, max: {:.4}, mean: {:.4}",
        predictions.min().double_value(&[]),
        predictions.max().double_value(&[]),
        predictions.mean(Kind::Float).double_value(&[]));
    println!("[DEBUG] predictions >= 0.5 count: {:.0}",
        predictions.ge(0.5).to_kind(Kind::Float).sum(Kind::Float).double_value(&[]));

    // Precision, Recall, F1 Score
    let (precision, recall, f1) = precision_recall_f1(&ys, &predictions, 0.5);
    println!("Precision: {:.4}, Recall: {:.4}, F1 Score: {:.4}", precision, recall, f1);
}
