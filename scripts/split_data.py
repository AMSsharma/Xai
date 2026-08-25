import os
import pandas as pd
import numpy as np
import json

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
METADATA_PATH = os.path.join(RAW_DIR, "metadata.csv")
QUALITY_REPORT_PATH = os.path.join(RAW_DIR, "data_quality_report.json")

def main():
    print("=== Phase 4: Data Leakage Investigation & Splitting ===")
    
    if not os.path.exists(METADATA_PATH) or not os.path.exists(QUALITY_REPORT_PATH):
        print("Error: Missing raw metadata or quality report. Run acquire_data and data_quality first.")
        return

    # Load data
    df = pd.read_csv(METADATA_PATH)
    with open(QUALITY_REPORT_PATH, 'r') as f:
        quality_report = json.load(f)
        
    print(f"Initial raw records: {len(df)}")
    
    # 1. Clean duplicates identified in quality report
    duplicate_files = [pair[0] for pair in quality_report['details']['duplicate_images_pairs']]
    df_cleaned = df[~df['filename'].isin(duplicate_files)].copy()
    print(f"Dropped {len(df) - len(df_cleaned)} duplicate files. Cleaned records: {len(df_cleaned)}")
    
    # 2. Get unique patients and their labels
    patient_df = df_cleaned.groupby('patient_id').agg({
        'label': 'first',
        'image_id': 'count'
    }).rename(columns={'image_id': 'image_count'}).reset_index()
    
    total_patients = len(patient_df)
    print(f"Unique patients: {total_patients}")
    
    # 3. Deterministic split by patient ID
    # Use seed 42 to ensure reproducibility
    rng = np.random.default_rng(42)
    
    # Shuffle patients
    shuffled_indices = rng.permutation(total_patients)
    patient_df_shuffled = patient_df.iloc[shuffled_indices].copy()
    
    # Define split bounds (70% / 15% / 15%)
    train_bound = int(0.70 * total_patients)
    val_bound = int(0.85 * total_patients)
    
    train_patients = patient_df_shuffled.iloc[:train_bound]['patient_id'].tolist()
    val_patients = patient_df_shuffled.iloc[train_bound:val_bound]['patient_id'].tolist()
    test_patients = patient_df_shuffled.iloc[val_bound:]['patient_id'].tolist()
    
    # Map back to image level
    train_split = df_cleaned[df_cleaned['patient_id'].isin(train_patients)].copy()
    val_split = df_cleaned[df_cleaned['patient_id'].isin(val_patients)].copy()
    test_split = df_cleaned[df_cleaned['patient_id'].isin(test_patients)].copy()
    
    # 4. Leakage Verification (MANDATORY)
    train_p_set = set(train_patients)
    val_p_set = set(val_patients)
    test_p_set = set(test_patients)
    
    overlap_train_val = train_p_set.intersection(val_p_set)
    overlap_train_test = train_p_set.intersection(test_p_set)
    overlap_val_test = val_p_set.intersection(test_p_set)
    
    print("\n--- LEAKAGE VERIFICATION ---")
    print(f"Overlap Train & Val:  {len(overlap_train_val)}")
    print(f"Overlap Train & Test: {len(overlap_train_test)}")
    print(f"Overlap Val & Test:   {len(overlap_val_test)}")
    
    # Assertions to prevent compilation of leaked splits
    assert len(overlap_train_val) == 0, "DATA LEAKAGE DETECTED between Train and Validation!"
    assert len(overlap_train_test) == 0, "DATA LEAKAGE DETECTED between Train and Test!"
    assert len(overlap_val_test) == 0, "DATA LEAKAGE DETECTED between Validation and Test!"
    
    print("VERIFICATION SUCCESS: 0% patient ID overlap across splits.")
    
    # 5. Print statistics for each split
    print("\n--- SPLIT STATISTICS ---")
    for name, split in [("TRAIN", train_split), ("VALIDATION", val_split), ("TEST", test_split)]:
        num_images = len(split)
        num_patients = split['patient_id'].nunique()
        normal_count = len(split[split['label'] == 0])
        pneumonia_count = len(split[split['label'] == 1])
        pct_pneumonia = (pneumonia_count / num_images) * 100 if num_images > 0 else 0
        
        print(f"{name:10s}: Patients={num_patients:4d}, Images={num_images:4d} | "
              f"Normal={normal_count:3d}, Pneumonia={pneumonia_count:4d} ({pct_pneumonia:.1f}% Pneumonia)")
              
    # 6. Save split sheets
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_split.to_csv(os.path.join(PROCESSED_DIR, "train_split.csv"), index=False)
    val_split.to_csv(os.path.join(PROCESSED_DIR, "val_split.csv"), index=False)
    test_split.to_csv(os.path.join(PROCESSED_DIR, "test_split.csv"), index=False)
    
    # Save a JSON file detailing the splits
    split_meta = {
        "train": {"patients": len(train_p_set), "images": len(train_split)},
        "val": {"patients": len(val_p_set), "images": len(val_split)},
        "test": {"patients": len(test_p_set), "images": len(test_split)}
    }
    with open(os.path.join(PROCESSED_DIR, "split_summary.json"), 'w') as f:
        json.dump(split_meta, f, indent=2)
        
    print(f"\nFinal split files written to {PROCESSED_DIR}/")
    print("Leakage investigation and splitting complete!")

if __name__ == "__main__":
    main()
