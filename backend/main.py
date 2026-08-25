import os
import sys
import json
import base64
from io import BytesIO
from datetime import datetime
from PIL import Image
import numpy as np
import cv2

import torch
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.database import init_db, get_db, CaseAudit
from backend.models.dataset import MultimodalDataset
from backend.models.multimodal_model import MultimodalClassifier
from backend.services.explainability import GradCAM, explain_tabular_perturbation, explain_tabular_counterfactual
from backend.services.reliability import predict_with_mc_dropout, MahalanobisOODDetector
from backend.services.retrieval import SimilarCaseRetrieval

app = FastAPI(title="Clinical AI Decision Intelligence API", version="1.0.0")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins in dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
PROCESSED_DIR = "data/processed"
STATIC_DIR = "backend/static"
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Mount static files (uploads and eda plots)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/plots", StaticFiles(directory=os.path.join(PROCESSED_DIR, "eda")), name="plots")

# Global variables for models and services loaded at startup
MODEL = None
TEMPERATURE = 1.0
OPTIMAL_THRESHOLD = 0.5
SCALING_PARAMS = None
OOD_DETECTOR = None
RETRIEVAL_INDEX = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def startup_event():
    global MODEL, TEMPERATURE, OPTIMAL_THRESHOLD, SCALING_PARAMS, OOD_DETECTOR, RETRIEVAL_INDEX
    print("=== FastAPI Starting Up: Preloading models and services ===")
    
    # 1. Initialize SQLite Database
    init_db()
    
    # 2. Load Tabular Scaling Params
    scaling_params_path = os.path.join(PROCESSED_DIR, "scaling_params.json")
    if os.path.exists(scaling_params_path):
        with open(scaling_params_path, 'r') as f:
            SCALING_PARAMS = json.load(f)
        print("Preloaded tabular scaling parameters.")
    else:
        print("Warning: Scaling parameters not found!")
        
    # 3. Load Calibration Parameters (Temperature & Optimal Threshold)
    cal_params_path = os.path.join(PROCESSED_DIR, "calibration_params.json")
    if os.path.exists(cal_params_path):
        with open(cal_params_path, 'r') as f:
            cal_params = json.load(f)
            TEMPERATURE = cal_params.get("temperature", 1.0)
            OPTIMAL_THRESHOLD = cal_params.get("optimal_threshold", 0.5)
        print(f"Preloaded calibration parameters. Temp: {TEMPERATURE:.4f}, Threshold: {OPTIMAL_THRESHOLD:.4f}")
        
    # 4. Load Multimodal PyTorch Model
    MODEL = MultimodalClassifier(tabular_dim=6, pretrained=False)
    model_path = "backend/models/multimodal_fusion.pth"
    if os.path.exists(model_path):
        MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
        MODEL.to(DEVICE)
        MODEL.eval()
        print("Preloaded Multimodal Fusion Model.")
    else:
        print("Warning: Multimodal weights not found!")
        
    # 5. Load OOD Detector
    ood_params_path = os.path.join(PROCESSED_DIR, "ood_params.json")
    OOD_DETECTOR = MahalanobisOODDetector(params_path=ood_params_path)
    print("Preloaded Mahalanobis OOD Detector.")
    
    # 6. Load Case Retrieval Index
    db_path = os.path.join(PROCESSED_DIR, "retrieval_db.json")
    RETRIEVAL_INDEX = SimilarCaseRetrieval(db_path=db_path)
    print("Preloaded Similar Case Retrieval Index.")
    print("Startup loading complete!")

def preprocess_tabular_input(age, gender, temperature, spo2, heart_rate, cough_severity):
    """
    Scale continuous vitals and encode categoricals for model ingestion.
    """
    # Continuous variables
    vitals = []
    continuous_cols = ['age', 'temperature', 'spo2', 'heart_rate']
    inputs = {'age': age, 'temperature': temperature, 'spo2': spo2, 'heart_rate': heart_rate}
    
    for col in continuous_cols:
        val = inputs[col]
        # Use preloaded scaling params
        mean = SCALING_PARAMS[col]["mean"]
        std = SCALING_PARAMS[col]["std"]
        vitals.append((val - mean) / std)
        
    # Categoricals
    gender_mapping = {"M": 0, "F": 1}
    cough_mapping = {"Absent": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
    
    gender_code = gender_mapping.get(gender, 0)
    cough_code = cough_mapping.get(cough_severity, 0)
    
    tabular_vector = np.array(vitals + [gender_code, cough_code], dtype=np.float32)
    return torch.tensor(tabular_vector, dtype=torch.float32).unsqueeze(0) # add batch dim

@app.post("/api/predict")
async def predict_case(
    image: UploadFile = File(...),
    patient_id: str = Form(...),
    age: float = Form(...),
    gender: str = Form(...),
    temperature: float = Form(...),
    spo2: int = Form(...),
    heart_rate: int = Form(...),
    cough_severity: str = Form(...),
    db: Session = Depends(get_db)
):
    if MODEL is None or SCALING_PARAMS is None:
        raise HTTPException(status_code=500, detail="Models/configurations not loaded.")
        
    # 1. Save uploaded image to disk for auditing and Grad-CAM visualization
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp_str}_{image.filename}"
    saved_image_path = os.path.join(UPLOADS_DIR, safe_filename)
    
    try:
        content = await image.read()
        pil_img = Image.open(BytesIO(content)).convert('RGB')
        pil_img.save(saved_image_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")
        
    # 2. Image transformation for PyTorch
    transform = transforms = MultimodalDataset("", "", scaling_params_path=None, is_training=False).image_transforms
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE) # Shape: [1, 3, 224, 224]
    
    # 3. Preprocess tabular inputs
    try:
        tab_tensor = preprocess_tabular_input(age, gender, temperature, spo2, heart_rate, cough_severity).to(DEVICE)
    except Exception as e:
         raise HTTPException(status_code=400, detail=f"Metadata preprocessing failed: {e}")
         
    # 4. Perform MC Dropout Predictions (estimates Calibrated Probability and Uncertainty)
    mc_results = predict_with_mc_dropout(MODEL, img_tensor, tab_tensor, temperature=TEMPERATURE, n_iter=15)
    calibrated_prob = mc_results["mean_probability"]
    uncertainty = mc_results["uncertainty"]
    confidence_cat = mc_results["confidence_category"]
    
    # Raw logit prediction (no dropout, default forward pass)
    with torch.no_grad():
        raw_logits = MODEL(img_tensor, tab_tensor)
        raw_probs = torch.softmax(raw_logits, dim=1).cpu().numpy()[0]
        raw_prob = float(raw_probs[1])
        
    # Classification decision based on clinical optimal threshold
    predicted_class = 1 if calibrated_prob >= OPTIMAL_THRESHOLD else 0
    
    # 5. Out-of-Distribution (OOD) Detection
    # Extract image embedding
    with torch.no_grad():
        img_features = MODEL.image_backbone(img_tensor)
        img_embed = MODEL.image_projection(img_features).cpu().numpy()[0]
        
    ood_result = OOD_DETECTOR.is_ood(img_embed)
    
    # OOD Safety Gate: Suppress prediction metrics if input is OOD
    diagnosis_available = True
    safety_message = "Normal Operation"
    if ood_result["is_ood"] or ood_result["ood_risk"] in ["HIGH", "CRITICAL"]:
        diagnosis_available = False
        safety_message = "OUT-OF-DISTRIBUTION INPUT — CLINICIAN REVIEW REQUIRED"
        predicted_class = -1
        calibrated_prob = 0.0
        raw_prob = 0.0
    
    # 6. Retrieve Similar Historical Cases
    similar_cases = RETRIEVAL_INDEX.retrieve(img_embed, k=3)
    
    # 7. Write Audit log in SQLite
    new_audit = CaseAudit(
        patient_id=patient_id,
        filename=safe_filename,
        age=age,
        gender=gender,
        temperature=temperature,
        spo2=spo2,
        heart_rate=heart_rate,
        cough_severity=cough_severity,
        prediction_class=predicted_class,
        raw_probability=raw_prob,
        calibrated_probability=calibrated_prob,
        uncertainty=uncertainty,
        confidence_category=confidence_cat,
        is_ood=ood_result["is_ood"],
        ood_distance=ood_result["distance"],
        ood_risk=ood_result["ood_risk"]
    )
    db.add(new_audit)
    db.commit()
    db.refresh(new_audit)
    
    return {
        "case_id": new_audit.id,
        "patient_id": patient_id,
        "filename": safe_filename,
        "prediction_class": predicted_class,
        "raw_probability": raw_prob,
        "calibrated_probability": calibrated_prob,
        "uncertainty": uncertainty,
        "confidence_category": confidence_cat,
        "is_ood": ood_result["is_ood"],
        "ood_risk": ood_result["ood_risk"],
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "similar_cases": similar_cases,
        "timestamp": new_audit.timestamp.isoformat(),
        "diagnosis_available": diagnosis_available,
        "safety_message": safety_message
    }

@app.post("/api/explain")
async def generate_explanation(
    case_id: int = Form(...),
    target_class: int = Form(1),
    db: Session = Depends(get_db)
):
    if MODEL is None or SCALING_PARAMS is None:
         raise HTTPException(status_code=500, detail="Models/configurations not loaded.")
         
    case = db.query(CaseAudit).filter(CaseAudit.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case audit record not found.")
        
    img_path = os.path.join(UPLOADS_DIR, case.filename)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Original uploaded file not found on server.")
        
    try:
        pil_img = Image.open(img_path).convert('RGB')
        transform = MultimodalDataset("", "", scaling_params_path=None, is_training=False).image_transforms
        img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        tab_tensor = preprocess_tabular_input(
            case.age, case.gender, case.temperature, case.spo2, case.heart_rate, case.cough_severity
        ).to(DEVICE)
        
        # 1. Image Explanations: Grad-CAM
        gcam = GradCAM(MODEL)
        heatmap = gcam.generate_heatmap(img_tensor, tab_tensor, target_class=target_class)
        gcam.remove_hooks()
        
        # Convert original image to OpenCV format
        orig_cv = cv2.imread(img_path)
        orig_cv = cv2.resize(orig_cv, (224, 224))
        
        # Resize heatmap and apply colormap
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(orig_cv, 0.6, heatmap_color, 0.4, 0)
        
        # Convert images back to base64 to return directly in JSON
        _, orig_buf = cv2.imencode('.png', orig_cv)
        _, heat_buf = cv2.imencode('.png', heatmap_color)
        _, over_buf = cv2.imencode('.png', overlay)
        
        orig_b64 = base64.b64encode(orig_buf).decode('utf-8')
        heat_b64 = base64.b64encode(heat_buf).decode('utf-8')
        over_b64 = base64.b64encode(over_buf).decode('utf-8')
        
        # 2. Tabular Explanations: Perturbation Attributions
        tab_attributions = explain_tabular_perturbation(MODEL, img_tensor, tab_tensor, case, SCALING_PARAMS, OPTIMAL_THRESHOLD)
        
        # 3. Tabular Counterfactual Explanations
        tab_counterfactual = explain_tabular_counterfactual(MODEL, img_tensor, tab_tensor, case, SCALING_PARAMS, OPTIMAL_THRESHOLD)
        
        return {
            "case_id": case_id,
            "target_class": target_class,
            "original_image_base64": f"data:image/png;base64,{orig_b64}",
            "heatmap_base64": f"data:image/png;base64,{heat_b64}",
            "overlay_base64": f"data:image/png;base64,{over_b64}",
            "tabular_attributions": tab_attributions,
            "tabular_counterfactual": tab_counterfactual
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {e}")

@app.post("/api/override")
async def clinician_override(
    case_id: int = Form(...),
    clinician_label: int = Form(...),
    override_reason: str = Form(...),
    db: Session = Depends(get_db)
):
    case = db.query(CaseAudit).filter(CaseAudit.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case record not found.")
        
    case.clinician_label = clinician_label
    case.override_reason = override_reason
    case.override_timestamp = datetime.utcnow()
    db.commit()
    
    return {"message": "Clinician audit override saved successfully."}

@app.get("/api/metrics")
async def get_system_metrics():
    """
    Serve training, calibration, and robustness statistics.
    """
    experiments = {}
    calibration = {}
    robustness = {}
    
    # Try reading the precomputed JSON files
    try:
        with open(os.path.join(PROCESSED_DIR, "experiments.json"), 'r') as f:
            experiments = json.load(f)
        with open(os.path.join(PROCESSED_DIR, "calibration_results.json"), 'r') as f:
            calibration = json.load(f)
        with open(os.path.join(PROCESSED_DIR, "robustness_results.json"), 'r') as f:
            robustness = json.load(f)
    except Exception as e:
        print(f"Error reading precomputed results: {e}")
        
    tabular_feature_audit = []
    univariate_results = []
    test_confidence_intervals = []
    ablation_study = []
    ood_benchmark = []
    ood_confusion_analysis = []
    human_model_agreement = []
    
    try:
        if os.path.exists("reports/tabular_feature_audit.csv"):
            tabular_feature_audit = pd.read_csv("reports/tabular_feature_audit.csv").to_dict(orient="records")
        if os.path.exists("reports/univariate_results.csv"):
            univariate_results = pd.read_csv("reports/univariate_results.csv").to_dict(orient="records")
        if os.path.exists("reports/test_confidence_intervals.csv"):
            test_confidence_intervals = pd.read_csv("reports/test_confidence_intervals.csv").to_dict(orient="records")
        if os.path.exists("reports/ablation_study.csv"):
            ablation_study = pd.read_csv("reports/ablation_study.csv").to_dict(orient="records")
        if os.path.exists("reports/ood_benchmark.csv"):
            ood_benchmark = pd.read_csv("reports/ood_benchmark.csv").to_dict(orient="records")
        if os.path.exists("reports/ood_confusion_analysis.csv"):
            ood_confusion_analysis = pd.read_csv("reports/ood_confusion_analysis.csv").to_dict(orient="records")
        if os.path.exists("reports/human_model_agreement.csv"):
            human_model_agreement = pd.read_csv("reports/human_model_agreement.csv").to_dict(orient="records")
    except Exception as e:
        print(f"Error loading CSV reports: {e}")
        
    return {
        "experiments": experiments,
        "calibration": calibration,
        "robustness": robustness,
        "tabular_feature_audit": tabular_feature_audit,
        "univariate_results": univariate_results,
        "test_confidence_intervals": test_confidence_intervals,
        "ablation_study": ablation_study,
        "ood_benchmark": ood_benchmark,
        "ood_confusion_analysis": ood_confusion_analysis,
        "human_model_agreement": human_model_agreement
    }

@app.get("/api/cases")
async def get_all_cases(db: Session = Depends(get_db)):
    cases = db.query(CaseAudit).order_by(CaseAudit.timestamp.desc()).all()
    return cases

@app.get("/api/failures")
async def get_model_failures(db: Session = Depends(get_db)):
    """
    Find case records that represent failures or concerns:
    1. False Positives / False Negatives (where clinician_label is set and differs from predicted_class)
    2. High Uncertainty (uncertainty standard deviation > 0.15)
    3. Highly unfamiliar / Out-of-Distribution cases (is_ood = True)
    """
    # Select cases with is_ood = True OR high uncertainty OR overrides
    failures = db.query(CaseAudit).filter(
        (CaseAudit.is_ood == True) | 
        (CaseAudit.uncertainty > 0.15) | 
        (CaseAudit.clinician_label != None)
    ).order_by(CaseAudit.timestamp.desc()).all()
    return failures

@app.get("/api/health")
def health_check():
    return {
        "status": "HEALTHY",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None
    }

# Serve built React frontend if it exists
FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))

if os.path.exists(FRONTEND_DIST):
    @app.get("/{rest_of_path:path}")
    async def serve_spa(rest_of_path: str):
        # Serve actual files if they exist in dist
        file_path = os.path.join(FRONTEND_DIST, rest_of_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        # Otherwise fallback to index.html for SPA routing
        if rest_of_path.startswith("api/") or rest_of_path.startswith("static/") or rest_of_path.startswith("plots/"):
            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
else:
    @app.get("/")
    def root_fallback():
        return {"message": "FastAPI is running. Build the frontend via 'npm run build' inside the frontend directory to serve the UI."}

if __name__ == "__main__":
    import uvicorn
    # Read environment variables for production binding
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    reload = False if "PORT" in os.environ else True
    print(f"Starting server on {host}:{port} (reload={reload})...")
    uvicorn.run("main:app", host=host, port=port, reload=reload)
