# Upgraded Clinical AI Decision Intelligence Platform

An explainable, uncertainty-aware clinical decision-support platform that fuses pediatric radiographs and clinical vitals to assess pneumonia. 

This platform has been audited, upgraded, and validated to meet top-tier ML engineering and clinical AI research standards. The system features temperature-scaled probability calibration, Monte Carlo Dropout uncertainty estimation, Mahalanobis distance out-of-distribution (OOD) anomaly detection, Content-Based Similar-Case Retrieval, and real-time Grad-CAM/Perturbation explanations.

**Clinical Disclaimer:** *This system is built strictly for clinical decision support. Final diagnostic and therapeutic choices must always be confirmed by a qualified medical professional.*

**Dataset Warning:** *The synthetic dataset contains label-conditioned clinical variables that create unusually separable tabular features. Therefore, the reported performance should not be interpreted as evidence of real-world clinical generalization.*

---

## 1. Upgrade Matrix & Interview Defenses

The system was updated to address five core scientific weaknesses:

1. **Tabular Feature Proxy Leakage Audited:** Identified that synthetic vital signs were generated conditional on the labels, yielding an artificial 1.000 baseline AUROC. We resolved this by auditing feature SMDs, MI scores, and providing an image-only baseline for clinical reality.
2. **Statistical Confidence Intervals ($B=2000$):** Implemented patient-level bootstrapping to evaluate test metrics (AUROC, Sensitivity, Specificity, Brier, ECE, and Cost) with 95% Confidence Intervals.
3. **Out-of-Distribution (OOD) Safety Gate:** Developed an OOD benchmark comparing Mahalanobis Distance against a Maximum Softmax Probability (MSP) baseline. Implemented an API-level safety gate to intercept and suppress diagnostic outputs when critical anomalies are detected.
4. **Clinician Counterfactual Reasoning:** Implemented clinical line-search interpolation to identify the minimal vital adjustments required to reverse high-risk predictions.
5. **Clinician Disagreement & Override Tracking:** Seeded the database with 100 simulated audits. Calculated override statistics showing that clinician overrides are heavily concentrated in high-uncertainty and OOD cases.

---

## 2. Quantitative Results & Evaluation

The updated system metrics are summarized below (with 95% patient-level bootstrap confidence intervals):

### 2.1 Test Split Bootstrapped Performance

| Model Architecture | Metric | Threshold | Estimate | 95% Confidence Interval |
| :--- | :--- | :--- | :---: | :---: |
| **Tabular-only Baseline** | AUROC | 0.5 (Raw) | 1.0000 | [0.9999, 1.0000] |
| **Image-only Baseline** | AUROC | 0.5 (Raw) | 0.9758 | [0.9501, 0.9942] |
| **Multimodal Fusion Model** | AUROC | 0.5 (Raw) | 0.9748 | [0.9383, 0.9990] |
| **Multimodal (Calibrated)** | ECE (Calibration) | 0.5 (Calib) | 0.0339 | [0.0173, 0.0573] |
| **Multimodal (Cost-Optimized)**| Sensitivity | 0.17 (Optim) | **1.0000** | [1.0000, 1.0000] |
| **Multimodal (Cost-Optimized)**| Specificity | 0.17 (Optim) | 0.4996 | [0.3200, 0.6875] |
| **Multimodal (Cost-Optimized)**| Average Clinical Cost | 0.17 (Optim) | 15.0055 | [8.0000, 23.0000] |

*Note: The perfect 1.0000 AUROC for the tabular baseline is due to proxy leakage where vitals were synthesized conditional on the label. The true model generalization is reflected in the Image-only baseline.*

---

## 3. Reliability & Explainability Services

### A. Out-of-Distribution (OOD) Safety Gate
The system benchmarks Mahalanobis Distance vs. Maximum Softmax Probability (MSP):
- **Noise Rejection:** Random Noise is predicted with high confidence by standard models (MSP = 0.0 detection rate), but is flagged at **100.0%** by our Mahalanobis Detector.
- **Safety Intercept:** If `is_ood = True` or the risk level is `HIGH`/`CRITICAL`, `/api/predict` suppresses predictions (`prediction_class = -1`, `calibrated_probability = 0.0`) and returns an OOD alert warning.

### B. Tabular Counterfactual Explanations
Exposes minimal vital adjustments to shift diagnostic classification:
- Uses sequential line-search interpolation from the patient's raw vitals towards healthy/disease baselines (holding Patient Age and Gender constant).
- Returns the counterfactual vitals combination that crosses the decision threshold to the opposite class.

### C. Uncertainty Estimation
- **Monte Carlo Dropout:** Runs $N=15$ forward passes with dropout active at test time. Predictions with $\sigma > 0.15$ are flagged as **High Uncertainty**.

---

## 4. Directory Structure

```text
Xai/
├── backend/
│   ├── models/
│   │   ├── dataset.py            # Multimodal PyTorch dataset
│   │   ├── image_model.py        # ResNet18 PyTorch definition
│   │   ├── tabular_model.py      # MLP tabular PyTorch definition
│   │   └── multimodal_model.py   # Concatenation fusion network
│   ├── services/
│   │   ├── explainability.py     # Grad-CAM and Counterfactual engine
│   │   ├── reliability.py        # MC Dropout & Mahalanobis OOD engine
│   │   └── retrieval.py          # Similar-Case CBIR database
│   ├── database.py               # SQLAlchemy SQLite config
│   ├── main.py                   # FastAPI server endpoints
│   └── clinical_audit.db         # SQLite file database
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # Upgraded 9-tab React workspace
│   │   ├── index.css             # UI styling stylesheet
│   │   └── main.jsx
│   └── package.json
└── scripts/
    ├── acquire_data.py           # Auto-downloads data parquets
    ├── run_tabular_audit.py      # Performs feature audits and scenario ablations
    ├── calculate_confidence_intervals.py # Computes patient-level bootstrap CIs
    ├── run_ood_benchmark.py      # Runs out-of-distribution evaluations
    ├── generate_synthetic_audit_data.py # Seeds SQLite database with simulated reviews
    └── calculate_human_agreement.py # Analyzes clinician disagreement rates
```

---

## 5. How to Run Locally

### Backend Setup:
1. Ensure Python 3.12+ and `uv` are installed.
2. In the root directory, create a virtual environment:
   ```bash
   uv venv --system-site-packages
   .venv\Scripts\activate
   ```
3. Seed the database and precompute the evaluation files:
   ```bash
   python scripts/run_tabular_audit.py
   python scripts/calculate_confidence_intervals.py
   python scripts/run_ood_benchmark.py
   python scripts/generate_synthetic_audit_data.py
   python scripts/calculate_human_agreement.py
   ```
4. Start the FastAPI server:
   ```bash
   python backend/main.py
   ```
   The API will listen on `http://127.0.0.1:8000`.

### Frontend Setup:
1. Navigate to the `frontend/` folder:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
2. Open the React application link (typically `http://localhost:5173`) in your browser to interact with the clinician dashboard.
