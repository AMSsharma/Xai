# Clinical AI decision support - Dataset Leakage Audit

This audit evaluates dataset construction and preprocessing stages to establish the validity and clinical reproducibility of model performance claims.

## 1. Split Isolation Verification

We verified patient-wise isolation on the split files in [`data/processed/`](file:///c:/Users/thele/Xai/data/processed):
*   `train_split.csv`
*   `val_split.csv`
*   `test_split.csv`

### Patient ID Intersections:
*   $\text{Train} \cap \text{Validation} = \emptyset$ (0 overlapping patients)
*   $\text{Train} \cap \text{Test} = \emptyset$ (0 overlapping patients)
*   $\text{Validation} \cap \text{Test} = \emptyset$ (0 overlapping patients)

Patient-level split isolation is successfully maintained, preventing direct patient leakage.

---

## 2. Feature-Level Audit & Proxy Leakage Analysis

Although patient split isolation is correct, the continuous and categorical patient vitals have severe clinical proxy leakage. They were synthesized directly using the target label as a distribution condition in [`scripts/acquire_data.py`](file:///c:/Users/thele/Xai/scripts/acquire_data.py).

Below is the analysis of each input variable:

| Feature Name | Potential Leakage? | Available at Inference? | Why? | Decision |
| :--- | :---: | :---: | :--- | :--- |
| **`age`** | No | Yes | Sampled uniformly from $1.0$ to $5.0$ regardless of class label. No statistical difference across classes. | **Keep** as a control demographic feature. |
| **`gender`** | No | Yes | Sampled uniformly (50% Male, 50% Female) regardless of label. No class association. | **Keep** as a control demographic feature. |
| **`temperature`** | **CRITICAL LEAKAGE** | Yes | Synthesized via label-derived normal distributions: Normal class ($\mu = 36.8^\circ\text{C}, \sigma = 0.25^\circ\text{C}$); Pneumonia class ($\mu = 38.6^\circ\text{C}, \sigma = 0.6^\circ\text{C}$). The distributions are almost non-overlapping. | **Retain but Flag:** This feature is a direct proxy for the label. Document as a dataset limitation. |
| **`spo2`** | **CRITICAL LEAKAGE** | Yes | Synthesized via label-derived normal distributions: Normal class ($\mu = 98.5\%, \sigma = 0.8\%$); Pneumonia class ($\mu = 90.5\%, \sigma = 3.5\%$). Almost completely separates the positive and negative class. | **Retain but Flag:** This is a direct proxy for the label. Document as a dataset limitation. |
| **`heart_rate`** | **CRITICAL LEAKAGE** | Yes | Synthesized via label-derived normal distributions: Normal class ($\mu = 95\text{ bpm}, \sigma = 8\text{ bpm}$); Pneumonia class ($\mu = 132\text{ bpm}, \sigma = 12\text{ bpm}$). | **Retain but Flag:** Significant distribution shift. Document as a dataset limitation. |
| **`cough_severity`** | **CRITICAL LEAKAGE** | Yes | Synthesized via label-derived multinomial distributions. Normal class has 70% Absent, 0% Severe; Pneumonia class has 5% Absent, 40% Severe. | **Retain but Flag:** Direct label proxy. Document as a dataset limitation. |

---

## 3. Duplicate and Contamination Audit
- **Image Duplicates:** The raw dataset contained 14 duplicate files (same image registered under slightly different IDs/names). These were identified in [`data_quality_report.json`](file:///c:/Users/thele/Xai/data/raw/data_quality_report.json) and cleaned in [`split_data.py`](file:///c:/Users/thele/Xai/scripts/split_data.py) prior to model training, avoiding validation bias.
- **Normalization Leakage:** Tabular scaling statistics ($\mu, \sigma$ for Z-score normalization) were fitted strictly on `train_split.csv` and stored in `scaling_params.json` to normalize the validation and test datasets. This prevents parameter leakage from validation/test datasets back into training.

---

## 4. Scientific Conclusion & Implications
The perfect performance ($\text{AUROC} = 1.0$) of the Tabular MLP baseline is not a result of superior modeling. It is an artifact of the dataset generation process where clinical variables represent **synthetic label proxies**. 

**Implication for Multimodal Fusion:** Because the tabular variables contain complete information about the label, the multimodal fusion network cannot improve upon the tabular baseline. The image features, which have natural noise and clinical variability, are mathematically redundant when fused with these leakage proxies. 
This must be presented as a core research insight rather than an engineering error.
