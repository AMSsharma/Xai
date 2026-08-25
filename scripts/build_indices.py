import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.models.dataset import MultimodalDataset
from backend.models.multimodal_model import MultimodalClassifier
from backend.services.reliability import MahalanobisOODDetector
from backend.services.retrieval import SimilarCaseRetrieval

PROCESSED_DIR = "data/processed"
IMAGES_DIR = "data/raw/images"
MODELS_DIR = "backend/models"
SCALING_PARAMS_PATH = os.path.join(PROCESSED_DIR, "scaling_params.json")
OOD_PARAMS_PATH = os.path.join(PROCESSED_DIR, "ood_params.json")
RETRIEVAL_DB_PATH = os.path.join(PROCESSED_DIR, "retrieval_db.json")

def main():
    print("=== MLOps Step: Building OOD and Case Retrieval Indices ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    train_dataset = MultimodalDataset(
        csv_path=os.path.join(PROCESSED_DIR, "train_split.csv"),
        images_dir=IMAGES_DIR,
        scaling_params_path=SCALING_PARAMS_PATH,
        is_training=False,
        fit_scaler=False
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    # 2. Load Model
    model = MultimodalClassifier(tabular_dim=6, pretrained=False)
    model_path = os.path.join(MODELS_DIR, "multimodal_fusion.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Run train_models first.")
        return
        
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # 3. Extract Projection Embeddings
    print("Extracting 128-dimensional image projection embeddings for training set...")
    embeddings = []
    metadata = []
    
    with torch.no_grad():
        for idx, (img, tab, labels) in enumerate(train_loader):
            img = img.to(device)
            # Run image branch projection only
            img_features = model.image_backbone(img)
            img_embeds = model.image_projection(img_features)
            
            embeddings.append(img_embeds.cpu().numpy())
            
            # Map batch indices back to original dataframe rows
            batch_size = img.size(0)
            start_row = idx * 32
            for i in range(batch_size):
                row = train_dataset.df.iloc[start_row + i]
                metadata.append({
                    "patient_id": row["patient_id"],
                    "filename": row["filename"],
                    "label": int(row["label"]),
                    "age": float(row["age"]),
                    "gender": row["gender"],
                    "temperature": float(row["temperature"]),
                    "spo2": int(row["spo2"])
                })
                
    embeddings = np.concatenate(embeddings, axis=0) # Shape: [N_train, 128]
    print(f"Extracted {len(embeddings)} training embeddings. Shape: {embeddings.shape}")
    
    # 4. Fit & Save OOD Detector (Mahalanobis Distance)
    print("\nFitting Mahalanobis OOD Detector...")
    detector = MahalanobisOODDetector()
    detector.fit(embeddings, threshold_percentile=95.0)
    detector.save_params(OOD_PARAMS_PATH)
    print(f"OOD parameters saved to {OOD_PARAMS_PATH}")
    
    # 5. Build & Save Similar-Case Retrieval Database
    print("\nBuilding Case Retrieval Index...")
    retrieval = SimilarCaseRetrieval()
    retrieval.save_database(RETRIEVAL_DB_PATH, embeddings, metadata)
    print(f"Retrieval database saved to {RETRIEVAL_DB_PATH}")
    print("OOD and Retrieval index build complete!")

if __name__ == "__main__":
    main()
