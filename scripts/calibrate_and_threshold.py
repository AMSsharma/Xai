import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_auc_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.models.dataset import MultimodalDataset
from backend.models.multimodal_model import MultimodalClassifier
from scripts.train_models import calculate_metrics, compute_ece

PROCESSED_DIR = "data/processed"
IMAGES_DIR = "data/raw/images"
MODELS_DIR = "backend/models"
SCALING_PARAMS_PATH = os.path.join(PROCESSED_DIR, "scaling_params.json")
CALIBRATION_PARAMS_PATH = os.path.join(PROCESSED_DIR, "calibration_params.json")
CALIBRATION_RESULTS_PATH = os.path.join(PROCESSED_DIR, "calibration_results.json")
PLOT_PATH = os.path.join(PROCESSED_DIR, "eda", "calibration_curves.png")

def get_logits_and_labels(model, dataloader, device):
    """Run model on dataloader and collect raw logits and true labels."""
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for img, tab, labels in dataloader:
            img, tab = img.to(device), tab.to(device)
            logits = model(img, tab)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
            
    return np.concatenate(all_logits), np.concatenate(all_labels)

def nll_loss_func(temperature, logits, labels):
    """Compute Negative Log-Likelihood (NLL) loss for a given temperature."""
    # logits shape: [N, 2]
    scaled_logits = logits / temperature
    
    # Softmax stable computation
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # NLL for class labels
    # labels can be 0 or 1
    eps = 1e-15
    nll = -np.mean(np.log(probs[np.arange(len(labels)), labels] + eps))
    return nll

def main():
    print("=== Phase 19, 17, 18: Calibration & Cost-Sensitive Optimization ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    val_dataset = MultimodalDataset(
        csv_path=os.path.join(PROCESSED_DIR, "val_split.csv"),
        images_dir=IMAGES_DIR,
        scaling_params_path=SCALING_PARAMS_PATH,
        is_training=False,
        fit_scaler=False
    )
    test_dataset = MultimodalDataset(
        csv_path=os.path.join(PROCESSED_DIR, "test_split.csv"),
        images_dir=IMAGES_DIR,
        scaling_params_path=SCALING_PARAMS_PATH,
        is_training=False,
        fit_scaler=False
    )
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. Load Model
    model = MultimodalClassifier(tabular_dim=6, pretrained=False)
    model_path = os.path.join(MODELS_DIR, "multimodal_fusion.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Run train_models first.")
        return
        
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # 3. Extract Logits
    print("Extracting validation and test logits...")
    val_logits, val_labels = get_logits_and_labels(model, val_loader, device)
    test_logits, test_labels = get_logits_and_labels(model, test_loader, device)
    
    # 4. Fit Temperature Scaling on Validation Logits
    print("Optimizing temperature scaling...")
    # Initial temperature guess is 1.0
    res = minimize(nll_loss_func, x0=[1.0], args=(val_logits, val_labels), method='L-BFGS-B', bounds=[(0.01, 10.0)])
    optimal_temp = float(res.x[0])
    print(f"Optimal Temperature: {optimal_temp:.4f} (Validation NLL: {res.fun:.4f})")
    
    # Compute calibrated validation probabilities
    # val_logits / optimal_temp -> softmax
    val_scaled_logits = val_logits / optimal_temp
    val_exp = np.exp(val_scaled_logits - np.max(val_scaled_logits, axis=1, keepdims=True))
    val_calibrated_probs = val_exp / np.sum(val_exp, axis=1, keepdims=True)
    val_pos_probs_calibrated = val_calibrated_probs[:, 1]
    
    val_raw_exp = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))
    val_raw_probs = val_raw_exp / np.sum(val_raw_exp, axis=1, keepdims=True)
    val_pos_probs_raw = val_raw_probs[:, 1]
    
    # 5. Cost-Sensitive Threshold Optimization on Validation Set
    # Costs: FN = 10.0, FP = 1.0
    cost_fn = 10.0
    cost_fp = 1.0
    
    thresholds = np.linspace(0.01, 0.99, 99)
    best_cost = float('inf')
    best_threshold = 0.5
    
    for t in thresholds:
        preds = (val_pos_probs_calibrated >= t).astype(int)
        fn_count = np.sum((val_labels == 1) & (preds == 0))
        fp_count = np.sum((val_labels == 0) & (preds == 1))
        total_cost = (fn_count * cost_fn) + (fp_count * cost_fp)
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = float(t)
            
    print(f"Optimal Threshold (Val): {best_threshold:.4f} (Total Cost: {best_cost:.1f})")
    
    # Save Calibration Parameters
    cal_params = {
        "temperature": optimal_temp,
        "optimal_threshold": best_threshold,
        "cost_matrix": {"fn_cost": cost_fn, "fp_cost": cost_fp}
    }
    with open(CALIBRATION_PARAMS_PATH, 'w') as f:
        json.dump(cal_params, f, indent=2)
    print(f"Saved calibration parameters to {CALIBRATION_PARAMS_PATH}")
    
    # 6. Apply to Test Set & Evaluate
    print("\nEvaluating on Test Set...")
    # Raw probabilities
    test_raw_exp = np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True))
    test_raw_probs = test_raw_exp / np.sum(test_raw_probs := test_raw_exp, axis=1, keepdims=True)
    test_pos_probs_raw = test_raw_probs[:, 1]
    
    # Calibrated probabilities
    test_scaled_logits = test_logits / optimal_temp
    test_scaled_exp = np.exp(test_scaled_logits - np.max(test_scaled_logits, axis=1, keepdims=True))
    test_calibrated_probs = test_scaled_exp / np.sum(test_scaled_exp, axis=1, keepdims=True)
    test_pos_probs_calibrated = test_calibrated_probs[:, 1]
    
    # Evaluate raw metrics at default threshold 0.5
    raw_metrics = calculate_metrics(test_labels, test_raw_probs)
    raw_fn = raw_metrics['confusion_matrix']['fn']
    raw_fp = raw_metrics['confusion_matrix']['fp']
    raw_cost = (raw_fn * cost_fn) + (raw_fp * cost_fp)
    
    # Evaluate calibrated metrics at default threshold 0.5
    calibrated_metrics_default_t = calculate_metrics(test_labels, test_calibrated_probs)
    
    # Evaluate calibrated metrics at optimal threshold
    # Note: calculate_metrics uses threshold 0.5 internally.
    # We will compute the custom threshold metrics manually or wrap them.
    test_preds_opt = (test_pos_probs_calibrated >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds_opt).ravel()
    
    opt_sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    opt_specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    opt_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    opt_f1 = float(2 * opt_precision * opt_sensitivity / (opt_precision + opt_sensitivity)) if (opt_precision + opt_sensitivity) > 0 else 0.0
    opt_cost = (fn * cost_fn) + (fp * cost_fp)
    
    final_metrics_opt_t = {
        "auroc": float(roc_auc_score(test_labels, test_pos_probs_calibrated)),
        "ece": compute_ece(test_pos_probs_calibrated, test_labels),
        "brier_score": float(np.mean((test_pos_probs_calibrated - test_labels) ** 2)),
        "sensitivity": opt_sensitivity,
        "specificity": opt_specificity,
        "precision": opt_precision,
        "f1": opt_f1,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "total_cost": float(opt_cost)
    }
    
    # Save results comparisons
    results = {
        "raw_model_t_0_5": {
            "auroc": raw_metrics["auroc"],
            "ece": raw_metrics["ece"],
            "brier_score": raw_metrics["brier_score"],
            "sensitivity": raw_metrics["sensitivity"],
            "specificity": raw_metrics["specificity"],
            "f1": raw_metrics["f1"],
            "total_cost": float(raw_cost),
            "confusion_matrix": raw_metrics["confusion_matrix"]
        },
        "calibrated_model_t_0_5": {
            "auroc": calibrated_metrics_default_t["auroc"],
            "ece": calibrated_metrics_default_t["ece"],
            "brier_score": calibrated_metrics_default_t["brier_score"],
            "sensitivity": calibrated_metrics_default_t["sensitivity"],
            "specificity": calibrated_metrics_default_t["specificity"],
            "f1": calibrated_metrics_default_t["f1"],
            "total_cost": float((calibrated_metrics_default_t['confusion_matrix']['fn'] * cost_fn) + 
                                (calibrated_metrics_default_t['confusion_matrix']['fp'] * cost_fp)),
            "confusion_matrix": calibrated_metrics_default_t["confusion_matrix"]
        },
        "calibrated_model_opt_t": final_metrics_opt_t
    }
    
    with open(CALIBRATION_RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n--- TEST COMPARISON RESULTS ---")
    print(f"Calibration metrics:")
    print(f"  Raw Brier Score:        {results['raw_model_t_0_5']['brier_score']:.4f} | Calibrated: {results['calibrated_model_opt_t']['brier_score']:.4f}")
    print(f"  Raw ECE:                {results['raw_model_t_0_5']['ece']:.4f} | Calibrated: {results['calibrated_model_opt_t']['ece']:.4f}")
    print(f"Decision metrics (Threshold 0.5 vs Optimal {best_threshold:.4f}):")
    print(f"  Raw Model (t=0.5)     | Sensitivity: {results['raw_model_t_0_5']['sensitivity']:.4f} | Specificity: {results['raw_model_t_0_5']['specificity']:.4f} | Cost: {results['raw_model_t_0_5']['total_cost']:.1f}")
    print(f"  Calibrated (t=0.5)    | Sensitivity: {results['calibrated_model_t_0_5']['sensitivity']:.4f} | Specificity: {results['calibrated_model_t_0_5']['specificity']:.4f} | Cost: {results['calibrated_model_t_0_5']['total_cost']:.1f}")
    print(f"  Calibrated (t={best_threshold:.4f}) | Sensitivity: {results['calibrated_model_opt_t']['sensitivity']:.4f} | Specificity: {results['calibrated_model_opt_t']['specificity']:.4f} | Cost: {results['calibrated_model_opt_t']['total_cost']:.1f}")
    
    # 7. Generate Calibration Curves Plot
    print("\nPlotting reliability diagrams...")
    fraction_of_positives_raw, mean_predicted_value_raw = calibration_curve(test_labels, test_pos_probs_raw, n_bins=10)
    fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(test_labels, test_pos_probs_calibrated, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    plt.plot(mean_predicted_value_raw, fraction_of_positives_raw, "s-", color='#e74c3c', label=f"Raw Model (ECE={results['raw_model_t_0_5']['ece']:.3f})")
    plt.plot(mean_predicted_value_cal, fraction_of_positives_cal, "o-", color='#2ecc71', label=f"Calibrated Model (ECE={results['calibrated_model_opt_t']['ece']:.3f})")
    plt.title("Reliability Diagram (Probability Calibration Comparison)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"Calibration plot saved to {PLOT_PATH}")
    print("Post-processing calibration and decision threshold tuning complete!")

if __name__ == "__main__":
    main()
