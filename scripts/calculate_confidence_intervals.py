import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.models.dataset import MultimodalDataset
from backend.models.multimodal_model import MultimodalClassifier
from backend.models.tabular_model import TabularClassifier
from backend.models.image_model import ImageClassifier
from scripts.train_models import calculate_metrics, compute_ece

PROCESSED_DIR = "data/processed"
IMAGES_DIR = "data/raw/images"
MODELS_DIR = "backend/models"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

SCALING_PARAMS_PATH = os.path.join(PROCESSED_DIR, "scaling_params.json")
CALIBRATION_PARAMS_PATH = os.path.join(PROCESSED_DIR, "calibration_params.json")
OUTPUT_PATH = os.path.join(REPORTS_DIR, "test_confidence_intervals.csv")

def bootstrap_metrics(df, pred_probs, labels, unique_patients, B=2000, threshold=0.5, cost_fn=10.0, cost_fp=1.0):
    """
    Perform patient-level bootstrapping.
    """
    # unique_patients is unique list of patient IDs in the test set
    n_patients = len(unique_patients)
    rng = np.random.default_rng(42)
    
    # Store bootstrap results
    metrics_list = []
    
    # We map patient_id to indices in the dataset
    patient_to_idx = df.groupby('patient_id').indices
    
    for b in range(B):
        # Sample patients with replacement
        sampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
        
        # Collect all sample indices corresponding to sampled patients
        bootstrap_indices = []
        for p in sampled_patients:
            bootstrap_indices.extend(patient_to_idx[p])
            
        bootstrap_indices = np.array(bootstrap_indices)
        
        # Extract predictions and labels for this bootstrap sample
        b_probs = pred_probs[bootstrap_indices]
        b_labels = labels[bootstrap_indices]
        
        # Calculate standard metrics at target threshold
        # Positive class probability is b_probs[:, 1]
        pos_probs = b_probs[:, 1]
        b_preds = (pos_probs >= threshold).astype(int)
        
        # AUROC
        try:
            auroc = roc_auc_score(b_labels, pos_probs)
        except Exception:
            auroc = 0.5
            
        # AUPRC
        precision_curve, recall_curve, _ = precision_recall_curve(b_labels, pos_probs)
        auprc = auc(recall_curve, precision_curve)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(b_labels, b_preds).ravel()
        
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
        ece = compute_ece(pos_probs, b_labels)
        brier = float(np.mean((pos_probs - b_labels) ** 2))
        cost = float(fn * cost_fn + fp * cost_fp)
        
        metrics_list.append({
            "auroc": auroc,
            "auprc": auprc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
            "ece": ece,
            "brier_score": brier,
            "cost": cost
        })
        
    # Compute CI percentiles (2.5th and 97.5th)
    metrics_df = pd.DataFrame(metrics_list)
    ci_results = {}
    for col in metrics_df.columns:
        mean_val = float(metrics_df[col].mean())
        lower_ci = float(np.percentile(metrics_df[col], 2.5))
        upper_ci = float(np.percentile(metrics_df[col], 97.5))
        ci_results[col] = (mean_val, lower_ci, upper_ci)
        
    return ci_results

def main():
    print("=== Upgraded Statistical Evaluation: Bootstrap CIs ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    test_dataset = MultimodalDataset(
        csv_path=os.path.join(PROCESSED_DIR, "test_split.csv"),
        images_dir=IMAGES_DIR,
        scaling_params_path=SCALING_PARAMS_PATH,
        is_training=False,
        fit_scaler=False
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_df = test_dataset.df
    unique_patients = test_df['patient_id'].unique()
    test_labels = test_df['label'].values
    
    # Cost Parameters
    cost_fn = 10.0
    cost_fp = 1.0
    
    # Load calibration parameters
    temp = 1.0
    opt_t = 0.5
    if os.path.exists(CALIBRATION_PARAMS_PATH):
        with open(CALIBRATION_PARAMS_PATH, 'r') as f:
            cal_params = json.load(f)
            temp = cal_params.get("temperature", 1.0)
            opt_t = cal_params.get("optimal_threshold", 0.5)
            
    print(f"Loaded calibration params. Temp={temp:.4f}, Optimal Threshold={opt_t:.4f}")
    
    # Models to evaluate
    models_to_run = [
        ("Tabular-only", TabularClassifier(input_dim=6), "tabular_only.pth"),
        ("Image-only", ImageClassifier(pretrained=False), "image_only.pth"),
        ("Multimodal Fusion", MultimodalClassifier(tabular_dim=6, pretrained=False), "multimodal_fusion.pth")
    ]
    
    records = []
    
    for model_name, model, weight_file in models_to_run:
        weight_path = os.path.join(MODELS_DIR, weight_file)
        if not os.path.exists(weight_path):
            print(f"Skipping {model_name} as weights {weight_path} do not exist.")
            continue
            
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Get predictions
        all_logits = []
        with torch.no_grad():
            for img, tab, _ in test_loader:
                img, tab = img.to(device), tab.to(device)
                if model_name == "Image-only":
                    outputs = model(img)
                elif model_name == "Tabular-only":
                    outputs = model(tab)
                else:
                    outputs = model(img, tab)
                all_logits.append(outputs.cpu().numpy())
                
        logits = np.concatenate(all_logits, axis=0)
        
        # 1. Evaluate at default threshold (t=0.5) raw (temp=1.0)
        raw_probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        raw_probs /= np.sum(raw_probs, axis=1, keepdims=True)
        
        print(f"Running Bootstrap (B=2000) for {model_name} (t=0.5, Raw)...")
        raw_ci = bootstrap_metrics(test_df, raw_probs, test_labels, unique_patients, B=2000, threshold=0.5)
        
        # Add to records
        for metric, (est, low, upp) in raw_ci.items():
            if metric == "cost":
                continue # don't report default cost for all
            records.append({
                "Model": model_name,
                "Metric": metric,
                "Threshold": "0.5 (Raw)",
                "Estimate": est,
                "Lower 95% CI": low,
                "Upper 95% CI": upp
            })
            
        # 2. Evaluate Calibrated model (temp=temp) if multimodal
        if model_name == "Multimodal Fusion":
            scaled_logits = logits / temp
            cal_probs = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            cal_probs /= np.sum(cal_probs, axis=1, keepdims=True)
            
            print(f"Running Bootstrap (B=2000) for {model_name} (t=0.5, Calibrated)...")
            cal_ci = bootstrap_metrics(test_df, cal_probs, test_labels, unique_patients, B=2000, threshold=0.5)
            
            for metric, (est, low, upp) in cal_ci.items():
                if metric == "cost":
                    continue
                records.append({
                    "Model": f"{model_name} (Calibrated)",
                    "Metric": metric,
                    "Threshold": "0.5 (Calibrated)",
                    "Estimate": est,
                    "Lower 95% CI": low,
                    "Upper 95% CI": upp
                })
                
            # 3. Evaluate Calibrated model at optimal cost-optimized threshold (t=opt_t)
            print(f"Running Bootstrap (B=2000) for {model_name} (t={opt_t:.4f}, Cost-Optimized Calibrated)...")
            opt_ci = bootstrap_metrics(test_df, cal_probs, test_labels, unique_patients, B=2000, threshold=opt_t, cost_fn=cost_fn, cost_fp=cost_fp)
            
            for metric, (est, low, upp) in opt_ci.items():
                records.append({
                    "Model": f"{model_name} (Calibrated)",
                    "Metric": metric,
                    "Threshold": f"{opt_t:.2f} (Cost-Optimized)",
                    "Estimate": est,
                    "Lower 95% CI": low,
                    "Upper 95% CI": upp
                })
                
    ci_df = pd.DataFrame(records)
    ci_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Bootstrap statistical analysis complete! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
