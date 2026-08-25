# Dataset Card - Clinical AI Decision Support Cohort

This dataset card describes the training, validation, and test cohorts used to build and evaluate the Clinical AI Decision Intelligence Platform.

## 1. Cohort Composition & Statistics

The dataset is divided patient-wise to prevent data leakage (no images from the same patient can overlap across splits).

| Cohort Split | Patient Count | Image Count | Normal Images (0) | Pneumonia Images (1) | Prevalence (% Pneumonia) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Cohort** | 536 | 954 | 246 | 708 | 74.2% |
| **Validation Cohort** | 115 | 227 | 56 | 171 | 75.3% |
| **Test Cohort** | 115 | 190 | 30 | 160 | 84.2% |
| **Total Cohort** | **766** | **1371** | **332** | **1039** | **75.8%** |

---

## 2. Feature Definitions & Ranges

The dataset contains one radiographic modality and six clinical tabular features:

| Column Name | Feature Type | Units / Categories | Description | Value Range |
| :--- | :--- | :--- | :--- | :--- |
| `filename` | Image | Radiograph JPEG | Chest X-ray image (2D grayscale, resized to 224x224) | N/A |
| `age` | Tabular | Years | Pediatric patient age | $1.0$ to $5.0$ years |
| `gender` | Tabular | `M`, `F` | Biological sex of the child | Male (0) or Female (1) |
| `temperature` | Tabular | Degrees Celsius | Body temperature measurement | $35.8^\circ\text{C}$ to $41.2^\circ\text{C}$ |
| `spo2` | Tabular | Percent (%) | Oxygen saturation level measured by pulse oximetry | $50\%$ to $100\%$ |
| `heart_rate` | Tabular | Beats per minute | Pulse heart rate frequency | $65$ to $185$ bpm |
| `cough_severity`| Tabular | `Absent`, `Mild`, `Moderate`, `Severe` | Clinician-assessed cough severity rating | Ordered categories (0 to 3) |

---

## 3. Synthetic Tabular Data Warnings

> [!WARNING]
> **Synthetic Proxy Artifact Alert:**
> The clinical vitals (`temperature`, `spo2`, `heart_rate`, `cough_severity`) in this dataset were programmatically generated from Gaussian and multinomial distributions conditioned directly on the pneumonia label (from the underlying chest X-ray image dataset) in `scripts/acquire_data.py`.
>
> The synthetic dataset contains label-conditioned clinical variables that create unusually separable tabular features. Therefore, the reported performance should not be interpreted as evidence of real-world clinical generalization.
> 
> Because of this generation logic:
> 1. Vitals represent a perfect proxy of the label, making them perfectly separable.
> 2. Model performance on these tabular variables is artificially high ($\text{AUROC} = 1.0$) and does not represent real-world clinical performance where vitals are noisy, overlapping, and highly non-specific.
> 3. Multimodal fusion does not show discriminative advantages over the tabular-only model on this dataset.
> 
> **Evaluation Guideline:** Evaluators must interpret the perfect tabular performance as a structural dataset limitation. Robustness tests, out-of-distribution (OOD) checks, and explainability mechanisms must be used to test the model's reliance on these features.
