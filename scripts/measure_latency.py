import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.models.dataset import MultimodalDataset
from backend.models.multimodal_model import MultimodalClassifier
from backend.services.reliability import predict_with_mc_dropout, MahalanobisOODDetector
from backend.services.explainability import GradCAM, explain_tabular_counterfactual

PROCESSED_DIR = "data/processed"
IMAGES_DIR = "data/raw/images"
MODELS_DIR = "backend/models"
SCALING_PARAMS_PATH = os.path.join(PROCESSED_DIR, "scaling_params.json")
CALIBRATION_PARAMS_PATH = os.path.join(PROCESSED_DIR, "calibration_params.json")
OOD_PARAMS_PATH = os.path.join(PROCESSED_DIR, "ood_params.json")

def main():
    print("=== Upgraded Latency Profiling Suite ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load scaling and calibration parameters
    with open(SCALING_PARAMS_PATH, 'r') as f:
        scaling_params = json.load(f)
    with open(CALIBRATION_PARAMS_PATH, 'r') as f:
        cal_params = json.load(f)
        temp = cal_params["temperature"]
        opt_t = cal_params["optimal_threshold"]
        
    # Load Model
    model = MultimodalClassifier(tabular_dim=6, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "multimodal_fusion.pth"), map_location=device))
    model = model.to(device).eval()
    
    # Load OOD Detector
    detector = MahalanobisOODDetector()
    if os.path.exists(OOD_PARAMS_PATH):
        detector.load_params(OOD_PARAMS_PATH)
        
    # Get a sample data point from test split
    test_csv = os.path.join(PROCESSED_DIR, "test_split.csv")
    test_df = pd.read_csv(test_csv)
    sample_row = test_df.iloc[0]
    sample_img_path = os.path.join(IMAGES_DIR, sample_row["filename"])
    
    # Let's define the profiling rounds
    n_rounds = 100
    
    # 1. Profile Preprocessing
    preproc_times = []
    dataset_helper = MultimodalDataset(test_csv, IMAGES_DIR, SCALING_PARAMS_PATH, is_training=False, fit_scaler=False)
    
    for _ in range(n_rounds):
        t0 = time.perf_counter()
        pil_img = Image.open(sample_img_path).convert('RGB')
        img_tensor = dataset_helper.image_transforms(pil_img).unsqueeze(0).to(device)
        
        # Preprocess tabular
        gender_code = 0 if sample_row['gender'] == 'M' else 1
        cough_mapping = {"Absent": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
        cough_code = cough_mapping.get(sample_row['cough_severity'], 0)
        
        tab_arr = np.zeros(6, dtype=np.float32)
        tab_arr[0] = (sample_row['age'] - scaling_params['age']['mean']) / scaling_params['age']['std']
        tab_arr[1] = (sample_row['temperature'] - scaling_params['temperature']['mean']) / scaling_params['temperature']['std']
        tab_arr[2] = (sample_row['spo2'] - scaling_params['spo2']['mean']) / scaling_params['spo2']['std']
        tab_arr[3] = (sample_row['heart_rate'] - scaling_params['heart_rate']['mean']) / scaling_params['heart_rate']['std']
        tab_arr[4] = float(gender_code)
        tab_arr[5] = float(cough_code)
        tab_tensor = torch.tensor(tab_arr, dtype=torch.float32).unsqueeze(0).to(device)
        
        preproc_times.append((time.perf_counter() - t0) * 1000) # in ms
        
    # 2. Profile Model Single Inference (no dropout)
    inference_times = []
    with torch.no_grad():
        for _ in range(n_rounds):
            t0 = time.perf_counter()
            logits = model(img_tensor, tab_tensor)
            _ = torch.softmax(logits, dim=1)
            inference_times.append((time.perf_counter() - t0) * 1000)
            
    # 3. Profile MC Dropout (N=15)
    mc_times = []
    for _ in range(n_rounds):
        t0 = time.perf_counter()
        _ = predict_with_mc_dropout(model, img_tensor, tab_tensor, temperature=temp, n_iter=15)
        mc_times.append((time.perf_counter() - t0) * 1000)
        
    # 4. Profile OOD Anomaly Calculation
    ood_times = []
    with torch.no_grad():
        img_features = model.image_backbone(img_tensor)
        img_embed = model.image_projection(img_features).cpu().numpy()[0]
    
    for _ in range(n_rounds):
        t0 = time.perf_counter()
        _ = detector.calculate_distance(img_embed)
        ood_times.append((time.perf_counter() - t0) * 1000)
        
    # 5. Profile Grad-CAM Explanation
    gradcam_times = []
    gcam = GradCAM(model)
    for _ in range(20): # Grad-CAM is heavier, 20 runs
        t0 = time.perf_counter()
        _ = gcam.generate_heatmap(img_tensor, tab_tensor, target_class=1)
        gradcam_times.append((time.perf_counter() - t0) * 1000)
    gcam.remove_hooks()
    
    # 6. Profile Counterfactual Boundary search
    cf_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = explain_tabular_counterfactual(model, img_tensor, tab_tensor, sample_row, scaling_params, opt_t)
        cf_times.append((time.perf_counter() - t0) * 1000)
        
    # Report Latency Metrics
    profiling_data = [
        ("Feature Preprocessing", preproc_times),
        ("Standard Inference (No Dropout)", inference_times),
        ("MC Dropout Inference (N=15)", mc_times),
        ("OOD Anomaly Detection", ood_times),
        ("Grad-CAM Generation", gradcam_times),
        ("Counterfactual Search", cf_times)
    ]
    
    results = []
    print("\n--- Latency Benchmark Results ---")
    for name, times in profiling_data:
        times = np.array(times)
        mean_val = np.mean(times)
        median_val = np.median(times)
        p95_val = np.percentile(times, 95)
        
        print(f"{name}:")
        print(f"  Mean:   {mean_val:.2f} ms")
        print(f"  Median: {median_val:.2f} ms")
        print(f"  P95:    {p95_val:.2f} ms")
        
        results.append({
            "Component": name,
            "Mean Latency (ms)": mean_val,
            "Median Latency (ms)": median_val,
            "P95 Latency (ms)": p95_val
        })
        
    df_out = pd.DataFrame(results)
    df_out.to_csv("reports/latency_profile.csv", index=False)
    print("\nSaved latency profile to reports/latency_profile.csv")

if __name__ == "__main__":
    main()
