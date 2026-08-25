import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, classification_report

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom architectures
from backend.models.dataset import MultimodalDataset
from backend.models.image_model import ImageClassifier
from backend.models.tabular_model import TabularClassifier
from backend.models.multimodal_model import MultimodalClassifier

PROCESSED_DIR = "data/processed"
IMAGES_DIR = "data/raw/images"
MODELS_DIR = "backend/models"
SCALING_PARAMS_PATH = os.path.join(PROCESSED_DIR, "scaling_params.json")
EXPERIMENTS_PATH = os.path.join(PROCESSED_DIR, "experiments.json")

def compute_ece(probs, labels, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(probs)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find samples in this bin
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)

def calculate_metrics(y_true, y_pred_probs):
    """
    Compute Accuracy, AUROC, AUPRC, Sensitivity (Recall), Specificity, Precision, F1, ECE, and Confusion Matrix.
    """
    # y_pred_probs is shape [n_samples, 2]
    # We take probability of positive class (class 1, Pneumonia)
    pos_probs = y_pred_probs[:, 1]
    y_pred = (pos_probs >= 0.5).astype(int)
    
    auroc = float(roc_auc_score(y_true, pos_probs))
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, pos_probs)
    auprc = float(auc(recall_curve, precision_curve))
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = float(tp + tn) / float(tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    ece = compute_ece(pos_probs, y_true)
    
    # Brier Score
    brier = float(np.mean((pos_probs - y_true) ** 2))
    
    return {
        "accuracy": accuracy,
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "ece": ece,
        "brier_score": brier,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

def train_one_model(model_name, model, train_loader, val_loader, test_loader, device, class_weights, epochs=5, lr=3e-4):
    print(f"\n--- Training {model_name} ---")
    model = model.to(device)
    
    # Define Loss with class weights to handle imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    best_epoch = 0
    model_path = os.path.join(MODELS_DIR, f"{model_name.lower().replace('-', '_')}.pth")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for img, tab, labels in train_loader:
            img, tab, labels = img.to(device), tab.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass depending on model type
            if model_name == "Image-only":
                outputs = model(img)
            elif model_name == "Tabular-only":
                outputs = model(tab)
            else: # Multimodal
                outputs = model(img, tab)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * img.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss /= len(train_loader.dataset)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for img, tab, labels in val_loader:
                img, tab, labels = img.to(device), tab.to(device), labels.to(device)
                
                if model_name == "Image-only":
                    outputs = model(img)
                elif model_name == "Tabular-only":
                    outputs = model(tab)
                else:
                    outputs = model(img, tab)
                    
                loss = criterion(outputs, labels)
                val_loss += loss.item() * img.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")
        
        # Checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)
            
    print(f"Saving best weights from Epoch {best_epoch} (Val Loss: {best_val_loss:.4f}) to {model_path}")
    
    # Load best weights for test evaluation
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for img, tab, labels in test_loader:
            img, tab, labels = img.to(device), tab.to(device), labels.to(device)
            
            if model_name == "Image-only":
                outputs = model(img)
            elif model_name == "Tabular-only":
                outputs = model(tab)
            else:
                outputs = model(img, tab)
                
            probs = torch.softmax(outputs, dim=1)
            
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            
    metrics = calculate_metrics(np.array(test_labels), np.array(test_probs))
    print(f"Test Results for {model_name}:")
    print(f"  AUROC:       {metrics['auroc']:.4f}")
    print(f"  AUPRC:       {metrics['auprc']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1:          {metrics['f1']:.4f}")
    print(f"  ECE:         {metrics['ece']:.4f}")
    
    return metrics

def main():
    print("=== Phase 7, 8, 9, 10, 11: Baseline and Multimodal Experiments ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize Datasets & DataLoaders
    # Note: scaling params are computed ONLY on the training split, saving params
    train_dataset = MultimodalDataset(
        csv_path=os.path.join(PROCESSED_DIR, "train_split.csv"),
        images_dir=IMAGES_DIR,
        scaling_params_path=SCALING_PARAMS_PATH,
        is_training=True,
        fit_scaler=True
    )
    
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
    
    # Compute Class Weights from train dataset to address class imbalance
    train_labels = train_dataset.df['label'].values
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    # Inverse frequency weighting
    w_0 = total_samples / (2.0 * class_counts[0])
    w_1 = total_samples / (2.0 * class_counts[1])
    class_weights = torch.tensor([w_0, w_1], dtype=torch.float32)
    print(f"Class counts in training: Normal={class_counts[0]}, Pneumonia={class_counts[1]}")
    print(f"Calculated class weights: Normal={w_0:.3f}, Pneumonia={w_1:.3f}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize experiments log
    experiments_log = {}
    if os.path.exists(EXPERIMENTS_PATH):
        try:
            with open(EXPERIMENTS_PATH, 'r') as f:
                experiments_log = json.load(f)
        except Exception:
            pass

    # 2. Train Tabular-only MLP Baseline
    tab_model = TabularClassifier(input_dim=6)
    tab_metrics = train_one_model("Tabular-only", tab_model, train_loader, val_loader, test_loader, device, class_weights, epochs=5, lr=1e-3)
    experiments_log["Tabular-only Baseline"] = {
        "metrics": tab_metrics,
        "hyperparams": {"lr": 1e-3, "epochs": 5, "batch_size": 32}
    }
    
    # 3. Train Image-only ResNet18 Baseline
    img_model = ImageClassifier(pretrained=True)
    img_metrics = train_one_model("Image-only", img_model, train_loader, val_loader, test_loader, device, class_weights, epochs=5, lr=3e-4)
    experiments_log["Image-only Baseline"] = {
        "metrics": img_metrics,
        "hyperparams": {"lr": 3e-4, "epochs": 5, "batch_size": 32}
    }
    
    # 4. Train Multimodal Fusion Network
    fusion_model = MultimodalClassifier(tabular_dim=6, pretrained=True)
    fusion_metrics = train_one_model("Multimodal-Fusion", fusion_model, train_loader, val_loader, test_loader, device, class_weights, epochs=5, lr=3e-4)
    experiments_log["Multimodal Fusion Model"] = {
        "metrics": fusion_metrics,
        "hyperparams": {"lr": 3e-4, "epochs": 5, "batch_size": 32}
    }
    
    # Save experiments log
    with open(EXPERIMENTS_PATH, 'w') as f:
        json.dump(experiments_log, f, indent=2)
    print(f"\nAll experiments logged and saved to {EXPERIMENTS_PATH}")
    
    # 5. Output comparative table
    print("\n=== EXPERIMENT COMPARISON ===")
    print(f"{'Model Name':25s} | {'Accuracy':8s} | {'Precision':9s} | {'Recall':6s} | {'AUROC':6s} | {'ECE':6s}")
    print("-" * 78)
    for model_name, data in experiments_log.items():
        m = data["metrics"]
        print(f"{model_name:25s} | {m.get('accuracy', 0.0):.4f}   | {m.get('precision', 0.0):.4f}    | {m['sensitivity']:.4f} | {m['auroc']:.4f} | {m['ece']:.4f}")
        
    print("\nModel training phase complete!")

if __name__ == "__main__":
    main()
