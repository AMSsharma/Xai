import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter, ImageEnhance
import io

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.models.dataset import MultimodalDataset
from backend.models.multimodal_model import MultimodalClassifier
from scripts.train_models import calculate_metrics

PROCESSED_DIR = "data/processed"
IMAGES_DIR = "data/raw/images"
MODELS_DIR = "backend/models"
SCALING_PARAMS_PATH = os.path.join(PROCESSED_DIR, "scaling_params.json")
CALIBRATION_PARAMS_PATH = os.path.join(PROCESSED_DIR, "calibration_params.json")
ROBUSTNESS_RESULTS_PATH = os.path.join(PROCESSED_DIR, "robustness_results.json")

# Custom Dataset that applies on-the-fly image perturbations for robustness testing
class PerturbedDataset(MultimodalDataset):
    def __init__(self, csv_path, images_dir, scaling_params_path=None, perturbation_type=None, tab_shift=False):
        super(PerturbedDataset, self).__init__(csv_path, images_dir, scaling_params_path, is_training=False, fit_scaler=False)
        self.perturbation_type = perturbation_type
        self.tab_shift = tab_shift

    def __getitem__(self, idx):
        row = self.df.iloc[idx].copy()
        
        # Apply tabular shift for external dataset simulation
        if self.tab_shift:
            # Shift SpO2 down by 2% (representing sicker patients or calibration offset)
            row['spo2'] = max(50, row['spo2'] - 2)
            # Shift temperature up by 0.2C
            row['temperature'] = row['temperature'] + 0.2
            
        img_path = os.path.join(self.images_dir, row['filename'])
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Apply image perturbations
            if self.perturbation_type == "blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=3.0))
            elif self.perturbation_type == "noise":
                # Add Gaussian noise
                img_arr = np.array(image).astype(np.float32)
                noise = np.random.normal(0, 25, img_arr.shape)
                img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(img_arr)
            elif self.perturbation_type == "low_contrast":
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(0.4) # reduce contrast by 60%
            elif self.perturbation_type == "low_res":
                # Downsample to 64x64 and upscale to 224x224 (lossy resolution decay)
                image = image.resize((64, 64), Image.Resampling.BILINEAR)
                image = image.resize((224, 224), Image.Resampling.BILINEAR)
                
            img_tensor = self.image_transforms(image)
        except Exception as e:
            print(f"Error: {e}")
            img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            
        tab_tensor = self.preprocess_tabular(row)
        label = torch.tensor(int(row['label']), dtype=torch.long)
        
        return img_tensor, tab_tensor, label

def evaluate_loader(model, loader, device, temperature=1.0):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for img, tab, labels in loader:
            img, tab = img.to(device), tab.to(device)
            logits = model(img, tab)
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return calculate_metrics(np.array(all_labels), np.array(all_probs))

def main():
    print("=== Phase 27 & 28: Robustness Testing & External Validation ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load calibration params for temperature scaling
    temp = 1.0
    if os.path.exists(CALIBRATION_PARAMS_PATH):
        with open(CALIBRATION_PARAMS_PATH, 'r') as f:
            cal_params = json.load(f)
            temp = cal_params.get("temperature", 1.0)
            
    # Load Model
    model = MultimodalClassifier(tabular_dim=6, pretrained=False)
    model_path = os.path.join(MODELS_DIR, "multimodal_fusion.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Run train_models first.")
        return
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    csv_path = os.path.join(PROCESSED_DIR, "test_split.csv")
    
    # 1. Evaluate clean test set (Internal baseline)
    clean_ds = PerturbedDataset(csv_path, IMAGES_DIR, SCALING_PARAMS_PATH, perturbation_type=None)
    clean_loader = DataLoader(clean_ds, batch_size=32, shuffle=False)
    clean_metrics = evaluate_loader(model, clean_loader, device, temp)
    
    # 2. Evaluate Image Perturbations (Robustness)
    perturbations = ["blur", "noise", "low_contrast", "low_res"]
    perturbation_results = {}
    
    for pert in perturbations:
        print(f"Evaluating model robustness against: {pert}...")
        ds = PerturbedDataset(csv_path, IMAGES_DIR, SCALING_PARAMS_PATH, perturbation_type=pert)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        metrics = evaluate_loader(model, loader, device, temp)
        perturbation_results[pert] = metrics

    # 3. Evaluate Shifted Dataset B (Simulated External Validation)
    print("\nEvaluating model on Dataset B (Simulated External Validation Shift)...")
    # Dataset B combines: scanner noise (noise) + clinical vital offsets (tab_shift)
    ds_b = PerturbedDataset(csv_path, IMAGES_DIR, SCALING_PARAMS_PATH, perturbation_type="noise", tab_shift=True)
    loader_b = DataLoader(ds_b, batch_size=32, shuffle=False)
    external_metrics = evaluate_loader(model, loader_b, device, temp)
    
    # 4. Save results comparison
    summary = {
        "internal_validation": clean_metrics,
        "robustness_perturbations": perturbation_results,
        "external_validation": external_metrics
    }
    
    with open(ROBUSTNESS_RESULTS_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("\n--- ROBUSTNESS RESULTS ---")
    print(f"{'Condition':20s} | {'AUROC':6s} | {'AUPRC':6s} | {'Sensitivity':11s} | {'ECE':6s}")
    print("-" * 59)
    print(f"{'Original Test':20s} | {clean_metrics['auroc']:.4f} | {clean_metrics['auprc']:.4f} | {clean_metrics['sensitivity']:.4f}       | {clean_metrics['ece']:.4f}")
    for pert, m in perturbation_results.items():
        print(f"{pert:20s} | {m['auroc']:.4f} | {m['auprc']:.4f} | {m['sensitivity']:.4f}       | {m['ece']:.4f}")
        
    print("\n--- EXTERNAL VALIDATION COMPARISON ---")
    print(f"Internal AUROC:  {clean_metrics['auroc']:.4f} | External Validation B AUROC:  {external_metrics['auroc']:.4f}")
    print(f"Internal AUPRC:  {clean_metrics['auprc']:.4f} | External Validation B AUPRC:  {external_metrics['auprc']:.4f}")
    print(f"Internal Recall: {clean_metrics['sensitivity']:.4f} | External Validation B Recall: {external_metrics['sensitivity']:.4f}")
    print(f"Internal ECE:    {clean_metrics['ece']:.4f} | External Validation B ECE:    {external_metrics['ece']:.4f}")
    
    print(f"\nAll results saved to {ROBUSTNESS_RESULTS_PATH}")
    print("Robustness testing and external validation complete!")

if __name__ == "__main__":
    main()
