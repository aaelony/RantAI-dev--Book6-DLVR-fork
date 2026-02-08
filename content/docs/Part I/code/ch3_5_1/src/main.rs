use ndarray::{Array2, Array1};
use rand::Rng;
use rand::distr::Uniform;
use std::f64;

// L2 regularization
fn l2_regularization(weights: &Array2<f64>, lambda: f64) -> f64 {
    let squared = weights.mapv(|w| w.powi(2));
    let sum_sq = squared.sum();
    let loss = lambda * sum_sq;
    eprintln!("[DEBUG l2_regularization] weights = {:?}", weights);
    eprintln!("[DEBUG l2_regularization] squared = {:?}", squared);
    eprintln!("[DEBUG l2_regularization] sum_sq = {}, lambda = {}, loss = {}", sum_sq, lambda, loss);
    loss
}

// Dropout implementation
fn dropout(input: &Array2<f64>, dropout_rate: f64) -> Array2<f64> {
    let mut rng = rand::rng();
    let dropout_mask = Array2::from_shape_fn(input.dim(), |idx| {
        let r: f64 = rng.random();
        let keep = if r < dropout_rate { 0.0 } else { 1.0 };
        eprintln!("[DEBUG dropout] idx={:?}, rand={:.4}, rate={}, mask={}", idx, r, dropout_rate, keep);
        keep
    });
    eprintln!("[DEBUG dropout] input      = {:?}", input);
    eprintln!("[DEBUG dropout] mask       = {:?}", dropout_mask);
    let result = input * &dropout_mask;
    eprintln!("[DEBUG dropout] result     = {:?}", result);
    let all_zero = result.iter().all(|&v| v == 0.0);
    if all_zero {
        eprintln!("[WARN  dropout] All values were zeroed out! With {} elements and rate={}, this has probability {:.4}",
            input.len(), dropout_rate, dropout_rate.powi(input.len() as i32));
    }
    result
}

// Precision, Recall, F1 score
fn precision_recall_f1(y_true: &Array1<f64>, y_pred: &Array1<f64>, threshold: f64) -> (f64, f64, f64) {
    let mut true_positive = 0.0;
    let mut false_positive = 0.0;
    let mut false_negative = 0.0;

    eprintln!("[DEBUG precision_recall_f1] threshold = {}", threshold);
    for (i, (&true_val, &pred_val)) in y_true.iter().zip(y_pred.iter()).enumerate() {
        let pred_class = if pred_val >= threshold { 1.0 } else { 0.0 };
        let label = if pred_class == 1.0 && true_val == 1.0 {
            true_positive += 1.0;
            "TP"
        } else if pred_class == 1.0 && true_val == 0.0 {
            false_positive += 1.0;
            "FP"
        } else if pred_class == 0.0 && true_val == 1.0 {
            false_negative += 1.0;
            "FN"
        } else {
            "TN"
        };
        eprintln!("[DEBUG precision_recall_f1] [{}] true={}, pred={}, pred_class={} -> {}", i, true_val, pred_val, pred_class, label);
    }

    eprintln!("[DEBUG precision_recall_f1] TP={}, FP={}, FN={}", true_positive, false_positive, false_negative);

    let precision = if true_positive + false_positive == 0.0 {
        eprintln!("[WARN  precision_recall_f1] precision=0 because TP+FP=0 (no positive predictions)");
        0.0
    } else {
        true_positive / (true_positive + false_positive)
    };

    let recall = if true_positive + false_negative == 0.0 {
        eprintln!("[WARN  precision_recall_f1] recall=0 because TP+FN=0 (no actual positives)");
        0.0
    } else {
        true_positive / (true_positive + false_negative)
    };

    let f1 = if precision + recall == 0.0 {
        eprintln!("[WARN  precision_recall_f1] f1=0 because precision+recall=0");
        0.0
    } else {
        2.0 * (precision * recall) / (precision + recall)
    };

    eprintln!("[DEBUG precision_recall_f1] precision={:.4}, recall={:.4}, f1={:.4}", precision, recall, f1);
    (precision, recall, f1)
}

// AUC-ROC calculation
fn auc_roc(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let mut sorted_pred: Vec<_> = y_pred.iter().zip(y_true.iter()).collect();
    sorted_pred.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap());

    eprintln!("[DEBUG auc_roc] sorted (pred, true): {:?}", sorted_pred.iter().map(|(p, t)| (**p, **t)).collect::<Vec<_>>());

    let mut tpr = vec![];
    let mut fpr = vec![];

    let total_positives = y_true.iter().filter(|&&y| y == 1.0).count() as f64;
    let total_negatives = y_true.iter().filter(|&&y| y == 0.0).count() as f64;

    eprintln!("[DEBUG auc_roc] total_positives={}, total_negatives={}", total_positives, total_negatives);

    if total_positives == 0.0 {
        eprintln!("[WARN  auc_roc] No positive samples! AUC is undefined.");
    }
    if total_negatives == 0.0 {
        eprintln!("[WARN  auc_roc] No negative samples! AUC is undefined.");
    }

    let mut true_positives = 0.0;
    let mut false_positives = 0.0;

    for (i, &(_, &true_val)) in sorted_pred.iter().enumerate() {
        if true_val == 1.0 {
            true_positives += 1.0;
        } else {
            false_positives += 1.0;
        }

        let cur_tpr = true_positives / total_positives;
        let cur_fpr = false_positives / total_negatives;
        eprintln!("[DEBUG auc_roc] step {}: true_val={}, TP={}, FP={}, TPR={:.4}, FPR={:.4}", i, true_val, true_positives, false_positives, cur_tpr, cur_fpr);
        tpr.push(cur_tpr);
        fpr.push(cur_fpr);
    }

    let mut auc = 0.0;
    for i in 1..tpr.len() {
        let delta_x = fpr[i] - fpr[i - 1];
        let avg_y = (tpr[i] + tpr[i - 1]) / 2.0;
        let contrib = delta_x * avg_y;
        eprintln!("[DEBUG auc_roc] trapezoid {}: delta_x={:.4}, avg_y={:.4}, contrib={:.4}, running_auc={:.4}", i, delta_x, avg_y, contrib, auc + contrib);
        auc += contrib;
    }

    eprintln!("[DEBUG auc_roc] final AUC = {:.4}", auc);
    auc
}

fn main() {
    eprintln!("=== Debug Tracing Enabled ===\n");

    let input_size = 3;
    let hidden_size = 5;
    let output_size = 1;
    let dropout_rate = 0.5;
    let lambda = 0.01;

    eprintln!("[DEBUG main] input_size={}, hidden_size={}, output_size={}", input_size, hidden_size, output_size);
    eprintln!("[DEBUG main] dropout_rate={}, lambda={}\n", dropout_rate, lambda);

    // Simulate Neural Network's input and output for demonstration purposes
    let mut rng = rand::rng();
    let input = Array2::from_shape_fn((1, input_size), |_| rng.sample(Uniform::new(0.0, 1.0).unwrap()));
    let weights = Array2::from_shape_fn((hidden_size, output_size), |_| rng.sample(Uniform::new(0.0, 1.0).unwrap()));

    eprintln!("[DEBUG main] input = {:?}", input);
    eprintln!("[DEBUG main] weights = {:?}\n", weights);

    // Apply L2 regularization
    let l2_loss = l2_regularization(&weights, lambda);
    println!("L2 Regularization Loss: {}", l2_loss);

    eprintln!();

    // Apply Dropout to the input
    let dropped_input = dropout(&input, dropout_rate);
    println!("Input after Dropout: {:?}", dropped_input);

    eprintln!();

    // Evaluation metrics
    let y_true = Array1::from(vec![1.0, 0.0, 1.0, 0.0, 1.0]);
    let y_pred = Array1::from(vec![0.9, 0.1, 0.8, 0.4, 0.6]);

    eprintln!("[DEBUG main] y_true = {:?}", y_true);
    eprintln!("[DEBUG main] y_pred = {:?}\n", y_pred);

    let (precision, recall, f1) = precision_recall_f1(&y_true, &y_pred, 0.5);
    println!("Precision: {:.4}, Recall: {:.4}, F1 Score: {:.4}", precision, recall, f1);

    eprintln!();

    // AUC-ROC Calculation
    let auc = auc_roc(&y_true, &y_pred);
    println!("AUC-ROC Score: {:.4}", auc);
}
