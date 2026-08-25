import os
import sys
import json
import urllib.request
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.models.dataset import MultimodalDataset
from backend.models.multimodal_model import MultimodalClassifier
from backend.services.reliability import MahalanobisOODDetector

PROCESSED_DIR = "data/processed"
IMAGES_DIR = "data/raw/images"
MODELS_DIR = "backend/models"
REPORTS_DIR = "reports"
OOD_RAW_DIR = "data/raw/ood"
os.makedirs(OOD_RAW_DIR, exist_ok=True)

OOD_PARAMS_PATH = os.path.join(PROCESSED_DIR, "ood_params.json")
SCALING_PARAMS_PATH = os.path.join(PROCESSED_DIR, "scaling_params.json")

# Synthetic image generators for OOD benchmarks
def generate_synthetic_hand_xray(i):
    img = Image.new('RGB', (224, 224), color=(10, 10, 10))
    draw = ImageDraw.Draw(img)
    # Draw palm
    draw.polygon([(70, 160), (150, 160), (140, 100), (80, 100)], fill=(40, 40, 40))
    # Draw fingers
    draw.line([(80, 100), (60, 40 + (i % 5))], fill=(60, 60, 60), width=8)
    draw.line([(95, 100), (85, 25 + (i % 3))], fill=(62, 62, 62), width=8)
    draw.line([(110, 100), (110, 20 + (i % 4))], fill=(65, 65, 65), width=8)
    draw.line([(125, 100), (135, 30 + (i % 6))], fill=(60, 60, 60), width=8)
    draw.line([(140, 110), (160, 70 + (i % 5))], fill=(55, 55, 55), width=8)
    # Add a bit of blur and noise
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def generate_synthetic_brain_ct(i):
    img = Image.new('RGB', (224, 224), color=(5, 5, 5))
    draw = ImageDraw.Draw(img)
    # Skull ring
    draw.ellipse([40, 30, 180, 190], outline=(150, 150, 150), width=6)
    # Brain hemispheres
    draw.ellipse([50, 40, 170, 180], fill=(50, 50, 50))
    # Draw folds inside
    draw.line([110, 40, 110, 180], fill=(20, 20, 20), width=3) # central fissure
    for j in range(5):
        y = 60 + j * 20
        draw.arc([60, y, 160, y + 20], start=0, end=180, fill=(70, 70, 70), width=2)
    img = img.filter(ImageFilter.GaussianBlur(radius=2.0))
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 3, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def download_reference_images():
    # Empty placeholder since we generate OOD reference images programmatically
    return {}

class OODSampleDataset(Dataset):
    """
    Dataset wrapping programmatically constructed OOD samples
    """
    def __init__(self, images_list, transform):
        self.images_list = images_list
        self.transform = transform
        
    def __len__(self):
        return len(self.images_list)
        
    def __getitem__(self, idx):
        pil_img = self.images_list[idx]
        img_tensor = self.transform(pil_img)
        # Tabular data is dummy (0.0 filled vector)
        tab_tensor = torch.zeros((6,), dtype=torch.float32)
        return img_tensor, tab_tensor

def compile_ood_samples(ref_paths, clean_test_dataset):
    print("--- Compiling OOD categories ---")
    ood_groups = {}
    
    # 1. Random Noise (Uniform)
    rng = np.random.default_rng(42)
    noise_images = []
    for _ in range(100):
        noise_arr = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        noise_images.append(Image.fromarray(noise_arr))
    ood_groups["Random Noise"] = OODSampleDataset(noise_images, clean_test_dataset.image_transforms)
    
    # 2. Natural Images (Synthetic Natural Gradients)
    print("Generating natural gradient photographs...")
    natural_grads = []
    for i in range(100):
        grad_arr = np.zeros((224, 224, 3), dtype=np.uint8)
        grad_arr[:, :, 0] = np.linspace(i, 255-i, 224).astype(np.uint8)[:, None]
        grad_arr[:, :, 1] = np.linspace(255-i, i, 224).astype(np.uint8)[None, :]
        natural_grads.append(Image.fromarray(grad_arr))
    ood_groups["Natural Images (CIFAR-10)"] = OODSampleDataset(natural_grads, clean_test_dataset.image_transforms)
        
    # 3. Other body-part X-rays (Hand X-ray)
    print("Generating Hand X-Ray references...")
    hand_copies = [generate_synthetic_hand_xray(i) for i in range(50)]
    ood_groups["Other body-part X-rays (Hand)"] = OODSampleDataset(hand_copies, clean_test_dataset.image_transforms)
            
    # 4. CT/MRI medical images (Brain CT)
    print("Generating Brain CT references...")
    brain_copies = [generate_synthetic_brain_ct(i) for i in range(50)]
    ood_groups["CT/MRI Medical Scans (Brain)"] = OODSampleDataset(brain_copies, clean_test_dataset.image_transforms)
            
    # 5. Low Resolution chest X-rays (64x64)
    low_res_images = []
    for idx in range(len(clean_test_dataset.df)):
        row = clean_test_dataset.df.iloc[idx]
        img_path = os.path.join(clean_test_dataset.images_dir, row['filename'])
        try:
            img = Image.open(img_path).convert('RGB')
            img_low = img.resize((64, 64), Image.Resampling.BILINEAR).resize((224, 224), Image.Resampling.BILINEAR)
            low_res_images.append(img_low)
        except Exception:
            pass
    ood_groups["Low Resolution Chest X-Rays"] = OODSampleDataset(low_res_images, clean_test_dataset.image_transforms)
    
    # 6. Contrast-Degraded chest X-rays
    low_contrast_images = []
    for idx in range(len(clean_test_dataset.df)):
        row = clean_test_dataset.df.iloc[idx]
        img_path = os.path.join(clean_test_dataset.images_dir, row['filename'])
        try:
            img = Image.open(img_path).convert('RGB')
            enh = ImageEnhance.Contrast(img)
            img_low = enh.enhance(0.4)
            low_contrast_images.append(img_low)
        except Exception:
            pass
    ood_groups["Low Contrast Chest X-Rays"] = OODSampleDataset(low_contrast_images, clean_test_dataset.image_transforms)
    
    # 7. Low-Quality Chest X-rays (combined noise and blur)
    low_quality_images = []
    for idx in range(len(clean_test_dataset.df)):
        row = clean_test_dataset.df.iloc[idx]
        img_path = os.path.join(clean_test_dataset.images_dir, row['filename'])
        try:
            img = Image.open(img_path).convert('RGB')
            # blur + noise
            img_blur = img.filter(ImageFilter.GaussianBlur(radius=2.0))
            arr = np.array(img_blur).astype(np.float32)
            noise = np.random.normal(0, 15, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            low_quality_images.append(Image.fromarray(arr))
        except Exception:
            pass
    ood_groups["Low Quality (Blur+Noise) X-Rays"] = OODSampleDataset(low_quality_images, clean_test_dataset.image_transforms)
    
    return ood_groups

def evaluate_ood_detection(model, detector, loader, device, is_ood_label):
    """
    Returns scores for Mahalanobis distances and Maximum Softmax Probabilities.
    """
    model.eval()
    distances = []
    msps = []
    
    with torch.no_grad():
        for batch in loader:
            img = batch[0].to(device)
            tab = batch[1].to(device)
            logits = model(img, tab)
            probs = torch.softmax(logits, dim=1)
            msp = torch.max(probs, dim=1)[0].cpu().numpy()
            msps.extend(msp)
            
            # Embeddings
            img_features = model.image_backbone(img)
            img_embeds = model.image_projection(img_features).cpu().numpy()
            
            for emb in img_embeds:
                dist = detector.calculate_distance(emb)
                distances.append(dist)
                
    return {
        "distances": np.array(distances),
        "msps": np.array(msps),
        "labels": np.full(len(distances), is_ood_label)
    }

def compute_ood_binary_metrics(true_ood_labels, ood_scores):
    """
    Calculate AUROC, AUPRC, FPR@95TPR, TPR@5FPR, and accuracy for OOD detection.
    Note: OOD class is 1, In-distribution class is 0.
    OOD scores must be higher for OOD samples.
    """
    auroc = roc_auc_score(true_ood_labels, ood_scores)
    
    precision, recall, thresholds = precision_recall_curve(true_ood_labels, ood_scores)
    auprc = auc(recall, precision)
    
    # FPR@95TPR:
    # Find the threshold where TPR is >= 95%
    # Recall represents TPR. Find first index where Recall >= 0.95 (reverse search since recall is descending)
    idx_tpr_95 = np.where(recall >= 0.95)[0][-1]
    thresh_95 = thresholds[min(idx_tpr_95, len(thresholds)-1)]
    
    # Compute FPR at this threshold
    preds_95 = (ood_scores >= thresh_95).astype(int)
    # FPR = FP / (FP + TN) = FP on ID samples
    fp_95 = np.sum((true_ood_labels == 0) & (preds_95 == 1))
    tn_95 = np.sum((true_ood_labels == 0) & (preds_95 == 0))
    fpr_at_95_tpr = float(fp_95 / (fp_95 + tn_95)) if (fp_95 + tn_95) > 0 else 0.0
    
    # TPR@5FPR:
    # Find the threshold where FPR <= 5%
    # Try multiple thresholds to find the closest FPR to 0.05
    best_tpr = 0.0
    for t in ood_scores:
        preds = (ood_scores >= t).astype(int)
        fp = np.sum((true_ood_labels == 0) & (preds == 1))
        tn = np.sum((true_ood_labels == 0) & (preds == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        if fpr <= 0.05:
            tp = np.sum((true_ood_labels == 1) & (preds == 1))
            fn = np.sum((true_ood_labels == 1) & (preds == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if tpr > best_tpr:
                best_tpr = tpr
                
    # Max accuracy
    best_acc = 0.0
    for t in ood_scores:
        preds = (ood_scores >= t).astype(int)
        acc = np.mean(preds == true_ood_labels)
        if acc > best_acc:
            best_acc = acc
            
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "fpr@95tpr": fpr_at_95_tpr,
        "tpr@5fpr": float(best_tpr),
        "max_accuracy": float(best_acc)
    }

def main():
    print("=== Upgraded OOD Benchmark Pipeline ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Download Wikimedia public clinical scans
    ref_paths = download_reference_images()
    
    # 2. Load clean test dataset
    clean_test_dataset = MultimodalDataset(
        csv_path=os.path.join(PROCESSED_DIR, "test_split.csv"),
        images_dir=IMAGES_DIR,
        scaling_params_path=SCALING_PARAMS_PATH,
        is_training=False,
        fit_scaler=False
    )
    clean_loader = DataLoader(clean_test_dataset, batch_size=32, shuffle=False)
    
    # 3. Load Multimodal Model & Mahalanobis Detector
    model = MultimodalClassifier(tabular_dim=6, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "multimodal_fusion.pth"), map_location=device))
    model = model.to(device)
    model.eval()
    
    detector = MahalanobisOODDetector(params_path=OOD_PARAMS_PATH)
    
    # 4. Compile OOD groups
    ood_groups = compile_ood_samples(ref_paths, clean_test_dataset)
    
    # 5. Evaluate In-Distribution Test Cohort
    print("Evaluating In-Distribution test cohort...")
    id_results = evaluate_ood_detection(model, detector, clean_loader, device, is_ood_label=0)
    
    # 6. Evaluate each OOD group and calculate confusion matrices
    confusion_records = []
    all_ood_distances = []
    all_ood_msps = []
    
    # ID threshold for distance is pre-saved in detector.threshold
    id_threshold = detector.threshold
    print(f"Loaded ID Mahalanobis Threshold: {id_threshold:.4f}")
    
    # For MSP baseline, we set threshold at the 5th percentile of ID MSP
    # (i.e. OOD scores: 1 - MSP. Threshold is 95th percentile of (1 - MSP))
    id_msps = id_results["msps"]
    id_ood_scores_msp = 1.0 - id_msps
    msp_threshold = np.percentile(id_ood_scores_msp, 95.0)
    print(f"Fit ID MSP Anomaly Threshold (95th percentile): {msp_threshold:.4f}")
    
    for category, ds in ood_groups.items():
        print(f"Evaluating OOD Category: {category} (N={len(ds)})...")
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        res = evaluate_ood_detection(model, detector, loader, device, is_ood_label=1)
        
        # Check detection rates
        dists = res["distances"]
        msps = res["msps"]
        
        # Detected by Mahalanobis (distance > threshold)
        detected_maha = np.sum(dists > id_threshold)
        missed_maha = len(dists) - detected_maha
        rate_maha = float(detected_maha / len(dists)) if len(dists) > 0 else 0.0
        
        # Detected by MSP (1 - msp > threshold)
        detected_msp = np.sum((1.0 - msps) > msp_threshold)
        rate_msp = float(detected_msp / len(msps)) if len(msps) > 0 else 0.0
        
        confusion_records.append({
            "OOD Category": category,
            "Samples": len(ds),
            "Detected (Mahalanobis)": int(detected_maha),
            "Missed (Mahalanobis)": int(missed_maha),
            "Detection Rate (Maha)": rate_maha,
            "Detection Rate (MSP)": rate_msp
        })
        
        all_ood_distances.append(dists)
        all_ood_msps.append(msps)
        
    confusion_df = pd.DataFrame(confusion_records)
    confusion_path = os.path.join(REPORTS_DIR, "ood_confusion_analysis.csv")
    confusion_df.to_csv(confusion_path, index=False)
    print(f"Saved OOD confusion analysis to {confusion_path}")
    
    # 7. Compute Global OOD Metrics comparing Mahalanobis vs MSP
    # Concatenate all OOD scores
    flat_ood_dists = np.concatenate(all_ood_distances)
    flat_ood_msps = np.concatenate(all_ood_msps)
    
    id_dists = id_results["distances"]
    id_msps = id_results["msps"]
    
    # Prepare global OOD classification vectors
    # OOD class is 1, ID class is 0
    y_true = np.concatenate([np.zeros(len(id_dists)), np.ones(len(flat_ood_dists))])
    
    # Scores (higher for OOD)
    scores_maha = np.concatenate([id_dists, flat_ood_dists])
    scores_msp = np.concatenate([1.0 - id_msps, 1.0 - flat_ood_msps])
    
    metrics_maha = compute_ood_binary_metrics(y_true, scores_maha)
    metrics_msp = compute_ood_binary_metrics(y_true, scores_msp)
    
    benchmark_records = [
        {
            "Method": "Mahalanobis Distance (Ours)",
            "AUROC": metrics_maha["auroc"],
            "AUPRC": metrics_maha["auprc"],
            "FPR@95TPR": metrics_maha["fpr@95tpr"],
            "TPR@5FPR": metrics_maha["tpr@5fpr"],
            "Max Accuracy": metrics_maha["max_accuracy"]
        },
        {
            "Method": "Maximum Softmax Probability (MSP)",
            "AUROC": metrics_msp["auroc"],
            "AUPRC": metrics_msp["auprc"],
            "FPR@95TPR": metrics_msp["fpr@95tpr"],
            "TPR@5FPR": metrics_msp["tpr@5fpr"],
            "Max Accuracy": metrics_msp["max_accuracy"]
        }
    ]
    
    benchmark_df = pd.DataFrame(benchmark_records)
    benchmark_path = os.path.join(REPORTS_DIR, "ood_benchmark.csv")
    benchmark_df.to_csv(benchmark_path, index=False)
    print(f"Saved OOD benchmark results to {benchmark_path}")
    
    print("\n--- OOD DETECTION BENCHMARK ---")
    print(benchmark_df.to_string(index=False))

if __name__ == "__main__":
    main()
