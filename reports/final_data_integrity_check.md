# CLINICAL DATA INTEGRITY & VALIDATION ISOLATION AUDIT

**Date:** August 2026  
**Auditor:** Senior Clinical AI Scientist & ML Engineer  
**Status:** COMPLETE & FROZEN  

---

## 1. Split Isolation & Patient-Level Disjointness

To prevent validation bias, patient-level isolation was enforced at split generation. Because multiple chest radiographs can belong to a single patient, splitting at the image level causes patient-wise leakage.

We verified the disjointness of patient IDs across the three splits programmatically:

$$\text{Train Patients} \cap \text{Validation Patients} = \emptyset$$
$$\text{Train Patients} \cap \text{Test Patients} = \emptyset$$
$$\text{Validation Patients} \cap \text{Test Patients} = \emptyset$$

### 1.1 Patient ID Counts
- **Total Patients:** 80
- **Train Patients:** 48 patients (60%)
- **Validation Patients:** 16 patients (20%)
- **Test Patients:** 16 patients (20%)

### 1.2 Programmatic Verification Check
Running the pipeline leak verification yields the following overlap matrix:
- **Train $\cap$ Validation Overlap:** 0 patients
- **Train $\cap$ Test Overlap:** 0 patients
- **Validation $\cap$ Test Overlap:** 0 patients

*Status:* **PASSED (0% Patient ID Overlap Across Splits)**

---

## 2. Duplicate Records & Data Consistency

We checked for image duplicates and patient record duplicates:
- **Duplicate Images:** Verified that each filename is unique in the metadata. No duplicate image files are shared across split folders.
- **Duplicate Patients:** Verified that each patient ID has a unique biological sex and age record. Vitals are unique per patient cohort instance.

*Status:* **PASSED**

---

## 3. Preprocessing & Downstream Parameter Fitting Workflow

To prevent data leakage, training parameters must be computed strictly on the training set and applied downstream. The pipeline isolates parameter estimation as follows:

| Process / Parameter | Fitted on Cohort | Evaluated on Cohort | Reason for Isolation |
| :--- | :--- | :--- | :--- |
| **Tabular Scaler Parameters** | **TRAIN** (`train_split.csv`) | Val / Test | Normalizes features using training mean and standard deviation. Fitting on validation or test would leak population statistics. |
| **Model Weights (PyTorch)** | **TRAIN** (`train_split.csv`) | Val / Test | Learns weights using backpropagation on training samples only. |
| **Temperature Calibration ($T$)** | **VALIDATION** (`val_split.csv`) | Test | Temperature scaling scaling optimizes ECE on out-of-training logits. Fitting on Train underestimates error; fitting on Test causes leakage. |
| **Decision Threshold ($t^*$)** | **VALIDATION** (`val_split.csv`) | Test | Grid-search optimizes clinical cost functions (FN weight=10, FP weight=1) on the validation split. |
| **OOD Distance Threshold** | **TRAIN** (`train_split.csv`) | Test | Mahalanobis Covariance matrix ($\Sigma^{-1}$) and mean vector ($\mu$) are computed on normal train representations. The 95th percentile OOD threshold is determined on Train. |
| **Final Evaluation** | **TEST** (`test_split.csv`) | *Sacred Test Set* | Serves as the final unbiased benchmark. No parameters, thresholds, or hyperparameters are tuned using test data. |

---

## 4. Preprocessing Leakage Audit Checklist
- [x] **No Test-Set Fit:** Are scaling parameters ($\mu, \sigma$) for vitals fit on the test set? *No. They are loaded from `data/processed/scaling_params.json` which is written strictly during training.*
- [x] **No Test-Set Calibration:** Is temperature calibration scaled using test logits? *No. Fitted on validation split.*
- [x] **No Test-Set Decision Threshold:** Is the clinical decision threshold ($t = 0.17$) tuned to optimize test performance? *No. It was selected on validation cost logs.*
- [x] **No Test-Set OOD Tuning:** Is the OOD threshold selected to maximize test detection rates? *No. It is the 95th percentile of distances computed on training images.*

---

## 5. Conclusions

The pipeline enforces data isolation rules. All statistical metrics (Confidence Intervals) computed on the test set represent an unbiased estimate of generalization.
