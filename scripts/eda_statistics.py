import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
import cv2

PROCESSED_DIR = "data/processed"
TRAIN_SPLIT_PATH = os.path.join(PROCESSED_DIR, "train_split.csv")
IMAGES_DIR = "data/raw/images"
EDA_OUT_DIR = os.path.join(PROCESSED_DIR, "eda")

def main():
    print("=== Phase 5 & 6: Exploratory Data Analysis & Hypotheses ===")
    
    if not os.path.exists(TRAIN_SPLIT_PATH):
        print(f"Error: Training split {TRAIN_SPLIT_PATH} not found. Run split_data first.")
        return
        
    os.makedirs(EDA_OUT_DIR, exist_ok=True)
    
    # Load training data for analysis (we do EDA ONLY on the training split to prevent visual leakage)
    df = pd.read_csv(TRAIN_SPLIT_PATH)
    print(f"Loaded training split with {len(df)} records for EDA.")
    
    # 1. Image Statistics (Brightness and Contrast)
    print("\nComputing image statistics (brightness and contrast)...")
    brightness_list = []
    contrast_list = []
    
    for idx, row in df.iterrows():
        img_path = os.path.join(IMAGES_DIR, row['filename'])
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                brightness_list.append(np.mean(img))
                contrast_list.append(np.std(img))
            else:
                brightness_list.append(np.nan)
                contrast_list.append(np.nan)
        except Exception:
            brightness_list.append(np.nan)
            contrast_list.append(np.nan)
            
    df['brightness'] = brightness_list
    df['contrast'] = contrast_list
    
    # Clean any NaN image values
    df = df.dropna(subset=['brightness', 'contrast'])
    
    # 2. Descriptive Stats by Target
    print("\nGenerating descriptive statistics...")
    normal_df = df[df['label'] == 0]
    pneumonia_df = df[df['label'] == 1]
    
    continuous_vars = ['age', 'temperature', 'spo2', 'heart_rate', 'brightness', 'contrast']
    desc_stats = {}
    
    for var in continuous_vars:
        desc_stats[var] = {
            "normal": {
                "mean": float(normal_df[var].mean()),
                "std": float(normal_df[var].std()),
                "median": float(normal_df[var].median())
            },
            "pneumonia": {
                "mean": float(pneumonia_df[var].mean()),
                "std": float(pneumonia_df[var].std()),
                "median": float(pneumonia_df[var].median())
            }
        }
        
    # 3. Hypothesis Testing
    print("\nPerforming statistical hypothesis tests...")
    test_results = {}
    
    # H1: Vitals and demographics differ between Normal and Pneumonia
    # Using Mann-Whitney U test (non-parametric t-test alternative, robust to non-normality)
    for var in continuous_vars:
        stat, pval = stats.mannwhitneyu(normal_df[var], pneumonia_df[var], alternative='two-sided')
        test_results[var] = {
            "test": "Mann-Whitney U",
            "statistic": float(stat),
            "p_value": float(pval),
            "significant": bool(pval < 0.05)
        }
        
    # Categorical: Gender vs Target (Chi-Square)
    contingency_gender = pd.crosstab(df['gender'], df['label'])
    chi2_g, p_g, dof_g, ex_g = stats.chi2_contingency(contingency_gender)
    test_results["gender"] = {
        "test": "Chi-Square Independence",
        "statistic": float(chi2_g),
        "p_value": float(p_g),
        "significant": bool(p_g < 0.05),
        "contingency_table": contingency_gender.to_dict()
    }
    
    # Categorical: Cough Severity vs Target (Chi-Square)
    contingency_cough = pd.crosstab(df['cough_severity'], df['label'])
    chi2_c, p_c, dof_c, ex_c = stats.chi2_contingency(contingency_cough)
    test_results["cough_severity"] = {
        "test": "Chi-Square Independence",
        "statistic": float(chi2_c),
        "p_value": float(p_c),
        "significant": bool(p_c < 0.05),
        "contingency_table": contingency_cough.to_dict()
    }

    # Save Stats JSON
    stats_output = {
        "descriptive_statistics": desc_stats,
        "hypothesis_tests": test_results
    }
    
    with open(os.path.join(EDA_OUT_DIR, "eda_statistics.json"), 'w') as f:
        json.dump(stats_output, f, indent=2)
        
    # 4. Generate Visualizations (saved for UI/report)
    print("\nGenerating and saving EDA plots...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Age vs SpO2 jointplot or scatter split by diagnosis
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='temperature', y='spo2', hue='label', palette={0: '#2ecc71', 1: '#e74c3c'}, alpha=0.7)
    plt.title("Clinical Scatter: Temperature vs SpO2 by Case Type")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("SpO2 (%)")
    plt.legend(labels=["Normal", "Pneumonia"])
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUT_DIR, "temp_vs_spo2.png"), dpi=150)
    plt.close()
    
    # Plot 2: Boxplots of SpO2 and Temp
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(ax=axes[0], data=df, x='label', y='spo2', hue='label', palette={0: '#2ecc71', 1: '#e74c3c'}, legend=False)
    axes[0].set_title("SpO2 Distribution by Diagnosis")
    axes[0].set_xticklabels(["Normal", "Pneumonia"])
    axes[0].set_ylabel("SpO2 (%)")
    axes[0].set_xlabel("")
    
    sns.boxplot(ax=axes[1], data=df, x='label', y='temperature', hue='label', palette={0: '#2ecc71', 1: '#e74c3c'}, legend=False)
    axes[1].set_title("Temperature Distribution by Diagnosis")
    axes[1].set_xticklabels(["Normal", "Pneumonia"])
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].set_xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUT_DIR, "vitals_boxplots.png"), dpi=150)
    plt.close()

    # Plot 3: Image Brightness vs Contrast
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=df, x='brightness', y='contrast', hue='label', fill=True, alpha=0.5, palette={0: '#2ecc71', 1: '#e74c3c'})
    plt.title("Image Feature Densities: Brightness vs Contrast")
    plt.xlabel("Mean Pixel Value (Brightness)")
    plt.ylabel("Standard Deviation (Contrast)")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUT_DIR, "image_stats_density.png"), dpi=150)
    plt.close()

    print("\n--- STATISTICAL ANALYSIS RESULTS ---")
    for var, res in test_results.items():
        sig_str = "SIGNIFICANT" if res['significant'] else "NOT SIGNIFICANT"
        print(f"Variable '{var:15s}' | p-val: {res['p_value']:.4e} | {sig_str}")
        
    print("\nScientific reminder:")
    print("  - These tests show statistical associations within our cohort.")
    print("  - They do NOT prove causality. Standard clinical reasoning must be applied.")
    print(f"\nAll plots and statistics written to {EDA_OUT_DIR}/")

if __name__ == "__main__":
    main()
