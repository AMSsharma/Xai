import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.models.dataset import MultimodalDataset
from backend.models.multimodal_model import MultimodalClassifier
from backend.services.reliability import predict_with_mc_dropout, MahalanobisOODDetector
from backend.services.retrieval import SimilarCaseRetrieval
from scripts.robustness_and_shift import PerturbedDataset

PROCESSED_DIR = "data/processed"
IMAGES_DIR = "data/raw/images"
MODELS_DIR = "backend/models"
SCALING_PARAMS_PATH = os.path.join(PROCESSED_DIR, "scaling_params.json")
CALIBRATION_PARAMS_PATH = os.path.join(PROCESSED_DIR, "calibration_params.json")
OOD_PARAMS_PATH = os.path.join(PROCESSED_DIR, "ood_params.json")
RETRIEVAL_DB_PATH = os.path.join(PROCESSED_DIR, "retrieval_db.json")

def main():
    print("=== Phase 47: Final End-to-End Demo Cases ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load configuration and model
    with open(CALIBRATION_PARAMS_PATH, 'r') as f:
        cal_params = json.load(f)
        temp = cal_params["temperature"]
        optimal_threshold = cal_params["optimal_threshold"]
        
    model = MultimodalClassifier(tabular_dim=6, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "multimodal_fusion.pth"), map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load OOD detector and Retrieval Index
    ood_detector = MahalanobisOODDetector(params_path=OOD_PARAMS_PATH)
    retrieval_index = SimilarCaseRetrieval(db_path=RETRIEVAL_DB_PATH)
    
    # Load clean test set to select cases
    test_ds = MultimodalDataset(
        csv_path=os.path.join(PROCESSED_DIR, "test_split.csv"),
        images_dir=IMAGES_DIR,
        scaling_params_path=SCALING_PARAMS_PATH,
        is_training=False,
        fit_scaler=False
    )
    
    print("\n--- CASE A: HIGH-CONFIDENCE IN-DISTRIBUTION PNEUMONIA ---")
    # Select a clear pneumonia case (label 1)
    pneumonia_rows = test_ds.df[test_ds.df['label'] == 1]
    # We will pick the first one
    idx_a = pneumonia_rows.index[0]
    img_a, tab_a, lbl_a = test_ds[idx_a]
    
    # Vitals of Case A
    row_a = test_ds.df.iloc[idx_a]
    print(f"Patient ID:  {row_a['patient_id']}")
    print(f"Clinical Vitals: Age={row_a['age']}, Temp={row_a['temperature']}C, SpO2={row_a['spo2']}%, HR={row_a['heart_rate']} bpm, Cough={row_a['cough_severity']}")
    
    # Predict
    img_tensor_a = img_a.unsqueeze(0).to(device)
    tab_tensor_a = tab_a.unsqueeze(0).to(device)
    
    mc_res_a = predict_with_mc_dropout(model, img_tensor_a, tab_tensor_a, temperature=temp, n_iter=15)
    with torch.no_grad():
        img_features_a = model.image_backbone(img_tensor_a)
        img_embed_a = model.image_projection(img_features_a).cpu().numpy()[0]
    ood_res_a = ood_detector.is_ood(img_embed_a)
    
    print("AI Assessment:")
    print(f"  Calibrated Prob of Pneumonia: {(mc_res_a['mean_probability']*100):.1f}%")
    print(f"  Uncertainty Category:         {mc_res_a['confidence_category']} (std={mc_res_a['uncertainty']:.4f})")
    print(f"  OOD Risk Level:               {ood_res_a['ood_risk']} (dist={ood_res_a['distance']:.4f})")
    
    
    print("\n--- CASE B: AMBIGUOUS / BORDERLINE CLINICAL CASE ---")
    # Manually construct a case with borderline vitals (temperature 37.6C, SpO2 94%, heart rate 108 bpm) 
    # to test model behavior under ambiguous tabular features and standard image
    # We will use an image from test set but replace tabular vitals
    idx_b = 5 # select a normal image
    img_b, _, _ = test_ds[idx_b]
    row_b = test_ds.df.iloc[idx_b].copy()
    
    # Override vitals to be ambiguous
    row_b['temperature'] = 37.6
    row_b['spo2'] = 94
    row_b['heart_rate'] = 108
    row_b['cough_severity'] = "Moderate"
    
    # Reprocess tabular
    tab_b = test_ds.preprocess_tabular(row_b)
    
    print(f"Patient ID:  {row_b['patient_id']} (Borderline vitals manually overlaid)")
    print(f"Clinical Vitals: Age={row_b['age']}, Temp={row_b['temperature']}C, SpO2={row_b['spo2']}%, HR={row_b['heart_rate']} bpm, Cough={row_b['cough_severity']}")
    
    img_tensor_b = img_b.unsqueeze(0).to(device)
    tab_tensor_b = tab_b.unsqueeze(0).to(device)
    
    mc_res_b = predict_with_mc_dropout(model, img_tensor_b, tab_tensor_b, temperature=temp, n_iter=15)
    with torch.no_grad():
        img_features_b = model.image_backbone(img_tensor_b)
        img_embed_b = model.image_projection(img_features_b).cpu().numpy()[0]
    ood_res_b = ood_detector.is_ood(img_embed_b)
    
    print("AI Assessment:")
    print(f"  Calibrated Prob of Pneumonia: {(mc_res_b['mean_probability']*100):.1f}%")
    print(f"  Uncertainty Category:         {mc_res_b['confidence_category']} (std={mc_res_b['uncertainty']:.4f})")
    print(f"  OOD Risk Level:               {ood_res_b['ood_risk']} (dist={ood_res_b['distance']:.4f})")
    
    
    print("\n--- CASE C: OUT-OF-DISTRIBUTION (OOD) ANOMALY CASE ---")
    # Case C: A completely non-medical image (Random pixel noise) representing a severe scanner malfunction or wrong file upload
    # We generate a random noise array of shape (224, 224, 3)
    rng_c = np.random.default_rng(99)
    noise_arr = rng_c.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    image_c = Image.fromarray(noise_arr)
    
    # We use Case A's vitals for Case C to prove the anomaly is triggered by the image
    row_c = test_ds.df.iloc[idx_a]
    tab_c = test_ds.preprocess_tabular(row_c)
    
    print(f"Patient ID:  {row_c['patient_id']} (Non-Medical Image: Random Pixel Noise)")
    print(f"Clinical Vitals: Age={row_c['age']}, Temp={row_c['temperature']}C, SpO2={row_c['spo2']}%, HR={row_c['heart_rate']} bpm, Cough={row_c['cough_severity']}")
    
    # Image transformation
    transform_c = test_ds.image_transforms
    img_c_tensor = transform_c(image_c)
    
    img_tensor_c = img_c_tensor.unsqueeze(0).to(device)
    tab_tensor_c = tab_c.unsqueeze(0).to(device)
    
    mc_res_c = predict_with_mc_dropout(model, img_tensor_c, tab_tensor_c, temperature=temp, n_iter=15)
    with torch.no_grad():
        img_features_c = model.image_backbone(img_tensor_c)
        img_embed_c = model.image_projection(img_features_c).cpu().numpy()[0]
    ood_res_c = ood_detector.is_ood(img_embed_c)
    
    print("AI Assessment:")
    print(f"  Calibrated Prob of Pneumonia: {(mc_res_c['mean_probability']*100):.1f}%")
    print(f"  Uncertainty Category:         {mc_res_c['confidence_category']} (std={mc_res_c['uncertainty']:.4f})")
    print(f"  OOD Risk Level:               {ood_res_c['ood_risk']} (dist={ood_res_c['distance']:.4f})")
    print(f"  OOD Alert Status:             {'OOD WARNING ACTIVATED' if ood_res_c['is_ood'] else 'Normal'}")
    
    print("\nEnd-to-End Demo evaluation complete!")

if __name__ == "__main__":
    main()
