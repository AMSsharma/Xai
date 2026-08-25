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
from backend.models.image_model import ImageClassifier
from backend.models.tabular_model import TabularClassifier
from scripts.train_models import compute_ece

PROCESSED_DIR = "data/processed"
IMAGES_DIR = "data/raw/images"
MODELS_DIR = "backend/models"
REPORTS_DIR = "reports"
SCALING_PARAMS_PATH = os.path.join(PROCESSED_DIR, "scaling_params.json")
OUTPUT_PATH = os.path.join(REPORTS_DIR, "statistical_model_comparison.csv")

def main():
    print("=== Upgraded Statistical Evaluation: Paired Model Comparison ===")
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
    
    # Load Models
    img_model = ImageClassifier(pretrained=False)
    img_weight_path = os.path.join(MODELS_DIR, "image_only.pth")
    if not os.path.exists(img_weight_path):
        print(f"Missing weights: {img_weight_path}")
        return
    img_model.load_state_dict(torch.load(img_weight_path, map_location=device))
    img_model = img_model.to(device).eval()
    
    multi_model = MultimodalClassifier(tabular_dim=6, pretrained=False)
    multi_weight_path = os.path.join(MODELS_DIR, "multimodal_fusion.pth")
    if not os.path.exists(multi_weight_path):
        print(f"Missing weights: {multi_weight_path}")
        return
    multi_model.load_state_dict(torch.load(multi_weight_path, map_location=device))
    multi_model = multi_model.to(device).eval()
    
    # Run predictions
    img_logits_list = []
    multi_logits_list = []
    
    with torch.no_grad():
        for img, tab, _ in test_loader:
            img, tab = img.to(device), tab.to(device)
            img_logits_list.append(img_model(img).cpu().numpy())
            multi_logits_list.append(multi_model(img, tab).cpu().numpy())
            
    img_logits = np.concatenate(img_logits_list, axis=0)
    multi_logits = np.concatenate(multi_logits_list, axis=0)
    
    # Softmax probabilities
    def softmax(l):
        e = np.exp(l - np.max(l, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)
        
    img_probs = softmax(img_logits)
    multi_probs = softmax(multi_logits)
    
    # Group patient indices
    patient_to_idx = test_df.groupby('patient_id').indices
    n_patients = len(unique_patients)
    rng = np.random.default_rng(42)
    B = 2000
    
    differences = {
        "auroc": [],
        "ece": [],
        "brier_score": []
    }
    
    for b in range(B):
        # Sample patients with replacement
        sampled = rng.choice(unique_patients, size=n_patients, replace=True)
        bootstrap_indices = []
        for p in sampled:
            bootstrap_indices.extend(patient_to_idx[p])
        bootstrap_indices = np.array(bootstrap_indices)
        
        b_labels = test_labels[bootstrap_indices]
        
        # Image-only metrics
        b_img_probs = img_probs[bootstrap_indices, 1]
        try:
            auroc_img = roc_auc_score(b_labels, b_img_probs)
        except:
            auroc_img = 0.5
        ece_img = compute_ece(b_img_probs, b_labels)
        brier_img = np.mean((b_img_probs - b_labels) ** 2)
        
        # Multimodal metrics
        b_multi_probs = multi_probs[bootstrap_indices, 1]
        try:
            auroc_multi = roc_auc_score(b_labels, b_multi_probs)
        except:
            auroc_multi = 0.5
        ece_multi = compute_ece(b_multi_probs, b_labels)
        brier_multi = np.mean((b_multi_probs - b_labels) ** 2)
        
        # Calculate paired differences (Multimodal - Image-only)
        differences["auroc"].append(auroc_multi - auroc_img)
        differences["ece"].append(ece_multi - ece_img)
        differences["brier_score"].append(brier_multi - brier_img)
        
    # Analyze differences
    records = []
    for metric, diffs in differences.items():
        diffs = np.array(diffs)
        mean_diff = float(np.mean(diffs))
        lower_ci = float(np.percentile(diffs, 2.5))
        upper_ci = float(np.percentile(diffs, 97.5))
        
        # Calculate two-tailed bootstrap p-value
        # p = 2 * min(prop >= 0, prop <= 0)
        p_val = float(2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0)))
        
        records.append({
            "Comparison": "Multimodal vs Image-only",
            "Metric": metric,
            "Point Estimate Diff": mean_diff,
            "Lower 95% CI": lower_ci,
            "Upper 95% CI": upper_ci,
            "p-value": p_val,
            "Statistical Significance": "Yes" if p_val < 0.05 else "No"
        })
        
    comparison_df = pd.DataFrame(records)
    comparison_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Statistical comparison complete! Saved to {OUTPUT_PATH}")
    print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    main()
