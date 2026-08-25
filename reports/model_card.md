# MODEL CARD: CLINICAL PNEUMONIA SCREENING SUPPORT v1.4.2

This Model Card details the architecture, evaluation parameters, and deployment boundaries of the multimodal decision-support classifier.

---

## 1. Model Details
- **Architecture:** Late-fusion neural network combining:
  - **Visual Backbone:** Pretrained ResNet18 (frozen feature extractor projected to 128 dimensions via a linear projection layer).
  - **Tabular Backbone:** Multi-Layer Perceptron (MLP) mapping 6 clinical variables to 32 dimensions.
  - **Fusion Method:** Concatenation of embeddings (160 dimensions total) passed to a classification head (fully connected layers with dropout, outputting 2 logits).
- **Post-Processing Calibration:** Validation-fitted Temperature Scaling ($T=1.5951$) applied to output logits before thresholding.
- **Decision Threshold:** Cost-optimized classification threshold ($t^*=0.1700$) chosen to minimize missed cases.

---

## 2. Intended Use
- **Primary Audience:** Pediatric clinicians and radiological reviewers.
- **Intended Use Case:** Clinical decision support to assist in screening pediatric chest radiographs and patient vitals for pneumonia markers.
- **Operational Mode:** Interactive screening assistant. The model provides visual overlays (Grad-CAM), uncertainty alerts, and counterfactual sensitivity analyses to augment human diagnostic reviews.

---

## 3. Non-Intended Use
- **No Autonomous Diagnostics:** This model is not intended to operate as a standalone diagnostic tool or to replace qualified medical reviews.
- **No Clinical Intervention Actions:** Decisions regarding patient discharge, hospitalization, or therapeutic prescriptions must not be based solely on model predictions.
- **Adult Cohorts:** Not trained or validated on adult radiological configurations.

---

## 4. Dataset Details
- **Visual Data Source:** Chest X-ray images resampled from the Hugging Face Pediatric Chest X-Ray cohort (Normal vs. Pneumonia).
- **Clinical Tabular Source:** Age, Biological Sex, and four vital signs (Body Temperature, SpO₂, Heart Rate, Cough Severity) synthetically generated conditional on the classification target.

---

## 5. Training Procedure
- **Optimizer:** Adam with cross-entropy loss.
- **Modality Splitting:** Frozen visual weights extracted from pre-trained multimodal model. Ablation classifiers trained on training split embeddings.
- **Calibration Scaling:** Temperature scaling optimized using Validation ECE.

---

## 6. Evaluation Cohort
- **Validation Split:** 16 patient-isolated records.
- **Sacred Test Split:** 16 patient-isolated records (190 total images). Patient disjointness programmatically verified ($\text{overlap} = 0$).

---

## 7. Point Estimates & Confidence Intervals (Patient-Level Bootstrap, B = 2000)

Evaluations represent a patient-level bootstrap simulation ($B=2000$) to guarantee confidence estimates are not artificially inflated by correlated radiographs:

- **Multimodal Fusion (Calibrated, cost-optimized, t=0.17):**
  - **AUROC:** 0.9748 (95% CI: [0.9383, 0.9990])
  - **AUPRC:** 0.9942 (95% CI: [0.9840, 0.9998])
  - **Sensitivity:** 1.0000 (95% CI: [1.0000, 1.0000])
  - **Specificity:** 0.4996 (95% CI: [0.3200, 0.6875])
  - **Expected Calibration Error (ECE):** 0.0339 (95% CI: [0.0173, 0.0573])
  - **Average Diagnostic Cost:** 15.0055 (95% CI: [8.0000, 23.0000])

---

## 8. Known Limitations & Dataset Artifacts
- **Proxy Target Leakage:** The synthetic dataset contains label-conditioned clinical variables that create unusually separable tabular features. Therefore, the reported performance should not be interpreted as evidence of real-world clinical generalization.
- **Wide Confidence Intervals:** Due to the small size of the test split cohort (16 patients, 190 images), specificity demonstrates wide intervals ([32.0%, 68.75%]). This represents an honest estimate of cohort statistical uncertainty.

---

## 9. OOD Anomaly Behavior & Robustness
- **OOD Detection:** Distance-based Mahalanobis OOD detector detects **Random Noise** anomalies at **100%**, preventing NN overconfidence failures.
- **Robustness Attenuation:** System AUROC collapses from **0.9748** to **0.9634** under severe image downsampling (64x64 resolution), indicating the visual backbone's sensitivity to resolution loss.

---

## 10. Explainability Services
- **Visual Saliency:** Layer-4 activation mapping via Grad-CAM highlights radiographic regions of interest.
- **Sensitivity Boundary Analysis:** Non-causal counterfactual line-search calculates the minimum clinical vital adjustments required to cross the model decision boundary.

---

## 11. Safety & Ethical Considerations
- **OOD Safety Gate:** Predict gateway suppresses predictions when critical anomalies are flagged, forcing clinician reviews.
- **Uncertainty Flags:** Cases with MC dropout standard deviation $>0.15$ are flagged in red to alert users of low statistical confidence.

---

## 12. Clinical Validation Status

> [!IMPORTANT]
> **Clinical Validation Status Warning:**
> **This is a research/portfolio prototype and has not undergone prospective clinical validation.** All human-in-the-loop consensus records used in evaluations are simulated demonstration data, and no real clinical trials or clinician studies have been performed.
