# CLINICAL AI PLATFORM: INTERVIEW DEFENSE GUIDE

This guide provides technically rigorous, evidence-backed answers to 30 critical questions about the platform's dataset, modeling, statistics, calibration, decision theory, OOD detection, explainability, human oversight, and robustness.

---

## 1. Dataset & Leakage

### Q1: Why did your tabular model achieve AUROC 1.0?
**Answer:** The tabular model achieved an AUROC of 1.0000 on the test split because the continuous vitals (temperature, SpO₂, heart rate) and categorical variables (cough severity) were synthetically generated conditional on the labels (`label = 0` vs. `label = 1`) in the initial dataset script. This created complete class separability in the feature space.

### Q2: How did you detect the problem?
**Answer:** I wrote and ran a feature diagnostics suite (`scripts/run_tabular_audit.py`) that performed univariate statistical evaluations. The script calculated the Standarized Mean Difference (SMD), Mutual Information (MI), and univariate AUROCs. SpO₂ and Temperature yielded individual univariate AUROCs of $1.0000$ and $0.9999$ on the test split, which is a clear indicator of target-proxy leakage.

### Q3: What caused the target proxy?
**Answer:** The root cause was inside [`scripts/acquire_data.py`](file:///c:/Users/thele/Xai/scripts/acquire_data.py), where Gaussian vital distributions for `temperature`, `spo2`, and `heart_rate` were parameterized conditional on the label (e.g., normal patients had a mean temperature of $37.0^\circ\text{C}$ and standard deviation of $0.4$, while pneumonia cases had a mean of $38.8^\circ\text{C}$ and standard deviation of $0.6$). This conditional generation linked the features deterministically to the label.

### Q4: Why didn't you hide/fix it?
**Answer:** Hiding the leakage would be scientifically dishonest and represents a major red flag in professional ML engineering. Instead of artificially modifying the synthetic data to look "imperfect", I audited, documented, and exposed it. I established the **Image-only model** (AUROC: 0.9758, 95% CI: [0.9501, 0.9942]) as the true clinical baseline, ensuring our findings are transparent and scientifically defensible.

---

## 2. Modeling & Ablations

### Q5: Why ResNet18?
**Answer:** ResNet18 is a lightweight, standard convolutional architecture. For a cohort size of 1,371 chest radiographs, larger architectures (like ResNet50 or DenseNet) are highly prone to overfitting on image margins and noise. ResNet18 serves as an ideal baseline feature extractor that can be trained efficiently on CPU/GPU without massive computational overhead.

### Q6: Why multimodal fusion?
**Answer:** In acute clinical diagnostics, chest radiographs are rarely interpreted in isolation. Clinicians integrate visual signs with patient vitals (fever, oxygen levels, heart rate) to make diagnostic support decisions. Late-fusion multimodal networks (concatenating image projection features with MLP vital embeddings) mimic this clinical workflow.

### Q7: Why doesn't multimodal fusion outperform tabular?
**Answer:** Because the tabular variables already contain complete information about the label (AUROC: 1.0000), the multimodal model cannot mathematically improve upon them. The image features introduce natural noise, which is redundant when concatenated with these perfect target-proxy variables.

### Q8: What did the ablation study teach you?
**Answer:** The ablation study (documented in `reports/ablation_study.csv`) proved that when we remove tabular variables, performance drops to the Image-only baseline (AUROC: 0.9758). It also proved that corrupting visual features (blur, noise) degrades the multimodal model's performance slightly, showing that the classifier does utilize both modalities but is heavily anchored by the separable tabular features.

---

## 3. Statistics & Confidence Intervals

### Q9: Why patient-level bootstrap?
**Answer:** Standard image-level bootstrapping treats individual radiographs as independent. In clinical datasets, a patient can have multiple follow-up radiographs. Treating these as independent samples causes split leakage and underestimates metric variance. Patient-level bootstrapping isolates patient IDs first, samples patients with replacement, and retains all corresponding images, resulting in statistically honest interval bounds.

### Q10: Why 2000 iterations?
**Answer:** $B=2000$ is the standard empirical threshold in bootstrap literature to construct stable and accurate tail percentiles (2.5th and 97.5th percentiles for a 95% CI). Running fewer iterations (e.g., $B=200$ or $500$) leads to high variance in the interval boundaries.

### Q11: What do your confidence intervals tell you?
**Answer:** Our confidence intervals represent the true statistical uncertainty of the model on this cohort. For example, the Multimodal model's specificity is estimated at **49.96%** under a 0.17 threshold, with a wide 95% CI of **[32.0%, 68.75%]**. This wide interval is an honest reflection of our small test cohort size (16 patients, 190 images).

### Q12: Why isn't 100% sensitivity sufficient?
**Answer:** While achieving 100.0% Sensitivity (95% CI: [1.0000, 1.0000]) means we catch all pneumonia cases, the cost-sensitive threshold shift drops our Specificity to 49.96%. In a real clinical setting, this low specificity leads to high false alarm rates (alarm fatigue) and unnecessary clinical diagnostics.

---

## 4. Probability Calibration

### Q13: Why temperature scaling?
**Answer:** Neural networks optimized via cross-entropy tend to produce overconfident probability estimates that do not reflect true empirical probabilities. Temperature scaling divides logits by a learned scalar $T > 1$ before the softmax function, softening probabilities and improving calibration without changing the argmax decision boundary.

### Q14: What is ECE?
**Answer:** Expected Calibration Error (ECE) measures the difference between model confidence and actual accuracy. It partitions predictions into equal-width bins (e.g., $M=10$), calculates the absolute difference between average confidence and accuracy in each bin, and takes the weighted average:
$$ECE = \sum_{m=1}^M \frac{|B_m|}{N} |acc(B_m) - conf(B_m)|$$

### Q15: Why must calibration be fitted on validation?
**Answer:** Fitting temperature calibration on the training set leads to overfitting, as training logits are highly over-separable. Fitting on the test set is a severe data leakage violation. Fitting on the independent validation split ($T=1.5951$) ensures that calibration generalises to unseen test data.

---

## 5. Decision Theory

### Q16: Why threshold 0.17?
**Answer:** Standard classifiers default to $t=0.5$. In clinical medicine, a False Negative (missed pneumonia, leading to sepsis) is far more dangerous than a False Positive (unnecessary antibiotic/readmit). We defined a cost matrix: $C(FN) = 10$ and $C(FP) = 1$. The optimal threshold that minimizes the expected cost on the validation split is **0.17**.

### Q17: Why not threshold 0.5?
**Answer:** At $t=0.5$, the model prioritizes accuracy, which tolerates False Negatives. In clinical diagnostics, false negatives carry a high cost, so shifting the decision boundary to $t=0.17$ ensures we catch all potential cases (100% Sensitivity).

### Q18: How would a hospital choose a threshold?
**Answer:** A hospital administrator or clinical steering committee would define the threshold by adjusting the cost weights ($C(FN)$ and $C(FP)$) based on resource constraints (available beds, cost of diagnostic audits, and safety risks).

---

## 6. Out-of-Distribution (OOD) Detection

### Q19: Why Mahalanobis distance?
**Answer:** Mahalanobis Distance measures the distance of a test image projection embedding from the multivariate Gaussian cluster of normal training images, accounting for covariance:
$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$
Unlike Euclidean distance, it scales features by their variance and covariance, making it highly robust to multidimensional correlations.

### Q20: Why compare against MSP?
**Answer:** Maximum Softmax Probability (MSP) is the standard baseline for neural network confidence. Deep learning classifiers tend to output high confidence (large MSP) even on random noise. Benchmarking against MSP demonstrates that distance-based OOD detectors are necessary to catch extreme anomalies.

### Q21: What types of OOD does your model detect poorly?
**Answer:** The model struggles to detect low-level image corruptions, such as low contrast or blur (detection rate $\le 7\%$). Because these remain visually similar to chest radiographs, they lie close to the in-distribution image cluster, though they degrade classification accuracy.

### Q22: What happens when OOD is detected?
**Answer:** The predict API suppresses the normal prediction outputs (`prediction_class = -1`, calibrated probability set to `0.0`), triggers an active safety warning flag, and returns the message: `OUT-OF-DISTRIBUTION INPUT — CLINICIAN REVIEW REQUIRED`.

---

## 7. Explainability & Counterfactuals

### Q23: Why Grad-CAM?
**Answer:** Grad-CAM uses the gradients of the class score flowing into the final convolutional layer of the ResNet18 visual backbone to generate a coarse localization map highlighting the regions of the image that contributed most to the prediction.

### Q24: What are Grad-CAM's limitations?
**Answer:** Grad-CAM only visualizes the final conv layer feature map, which is low resolution (7x7 in ResNet18). It is not fine-grained, can be visually noisy, and is susceptible to adversarial input perturbations.

### Q25: Why isn't a counterfactual causal?
**Answer:** A counterfactual in this platform is a **non-causal sensitivity boundary analysis**. It shows what minimal input variations would change the model's decision. It does *not* imply that changing these clinical parameters (e.g., lowering a patient's temperature or raising SpO₂) will medically improve the patient's condition.

---

## 8. Human Oversight & Robustness

### Q26: Why clinician override?
**Answer:** In safety-critical ML, human-in-the-loop oversight is essential. Logging overrides to `clinical_audit.db` tracks clinician-model disagreement, allowing for continuous auditing and offline model refinement.

### Q27: Are your clinician results real?
**Answer:** **No. The clinician override dataset is simulated demonstration data.** No real clinician trials or user studies were performed. The 100 cases in the database were programmatically seeded to demonstrate the audit trail and oversight telemetry.

### Q28: How would real reviewer feedback be incorporated?
**Answer:** In production, reviewer overrides would be stored in the SQL database. Disagreement cases would be flagged for consensus double-reading. The confirmed cases would then be added to the training set for model retraining (active learning).

### Q29: What did the low-resolution experiment reveal?
**Answer:** Downsampling radiographs to 64x64 caused the AUROC to collapse to **0.9634**, showing that model generalization is highly sensitive to input resolution. This underscores the need for OOD safety gates to reject low-quality inputs.

### Q30: What would you do before deploying this to a real hospital?
**Answer:** Before clinical deployment, I would:
1. Validate the model on a completely external, prospective clinical cohort with real, unsynthesized EHR vitals.
2. Undergo rigorous multi-reader clinician consensus evaluations.
3. Perform cybersecurity audits on the image upload endpoints and SQL database.
4. Obtain regulatory clearance (e.g., FDA 510(k)).
