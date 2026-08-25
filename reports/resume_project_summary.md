# RESUME PROJECT SUMMARY: CLINICAL AI DECISION PLATFORM

This artifact provides resume-ready bullet points tailored to Data Science, Machine Learning Engineering, and AI Engineering roles, utilizing actual measured performance metrics from the validation audits.

---

## 1. Data Scientist Resume Version

Emphasizes: proxy target leakage audits, multimodal validation, patient-level bootstrap statistical rigor, probability calibration, and cost-sensitive decision theory.

- **Audited Multimodal Data Quality & Target Leakage:** Audited clinical data parquets to uncover target proxy leakage in synthetic vitals (univariate SpO₂ and Temperature AUROCs $\ge 0.99$). Disclosed this dataset artifact via structural card warnings and established the visual model (test AUROC: 0.9758) as the true clinical baseline.
- **Enforced Statistical Rigor with Patient-Level Bootstrap:** Built a patient-level bootstrap resampling framework ($B=2000$) to evaluate test metrics. Preserved patient isolation (0% patient ID overlap across splits) and constructed robust 95% confidence intervals to evaluate specificity ([0.3200, 0.6875]) and average cost.
- **Optimized Clinical Utility via Calibration & Cost-Sensitive Decisions:** Applied Temperature Scaling ($T=1.5951$) to reduce Expected Calibration Error (ECE) to 3.39% (95% CI: [1.73%, 5.73%]). Implemented clinical utility optimization (FN cost = 10, FP cost = 1) to determine the optimal validation decision boundary ($t=0.17$), yielding 100.0% Sensitivity.

---

## 2. Machine Learning Engineer Resume Version

Emphasizes: low-latency inference, out-of-distribution (OOD) safety gates, API design, production databases, and system audit logs.

- **Developed Multi-modal Late Fusion Inference Pipeline:** Deployed a late-fusion PyTorch classifier combining a frozen ResNet18 visual backbone and tabular MLP vital encoder. Optimized feature extraction and preprocessing routines to run inference under low latencies.
- **Implemented Distance-Based OOD Safety Gate:** Programmed a Mahalanobis distance out-of-distribution (OOD) detector on visual projection embeddings. Benchmarked performance against a Maximum Softmax Probability (MSP) baseline, achieving 100% anomaly detection on random noise and automatically suppressing prediction outputs.
- **Engineered Clinician Audit & SQL Logging Systems:** Designed a relational database schema in SQLite/SQLAlchemy to capture patient metadata, model confidence categories, MC Dropout uncertainty standard deviations, OOD indicators, and clinician overrides, logging case telemetry for audit reviews.

---

## 3. AI Engineer Resume Version

Emphasizes: explainability (XAI), uncertainty estimation, clinician-in-the-loop interfaces, and interactive dashboards.

- **Integrated Multimodal Explainability Engines:** Deployed a dual-explainability engine combining layer-4 Grad-CAM visual saliency maps and tabular perturbation attributions. Built a non-causal counterfactual line-search solver to compute minimal vital adjustments required to cross model decision boundaries.
- **Implemented Monte Carlo Dropout Uncertainty Flags:** Formulated real-time uncertainty estimation using $N=15$ stochastic forward passes with dropout active at test-time. Flagged predictions with standard deviations $>0.15$ in the user interface to alert clinicians of low model confidence.
- **Designed 9-Tab Clinician Dashboard Interface:** Designed and developed a clinical decision support React SPA. Unified prediction forms, Grad-CAM/overlay visualizers, reliability scatterplots, bootstrap performance tables, override portals, and database audit logs.
