import os
import pandas as pd
import numpy as np
import hashlib
import json
from PIL import Image
import cv2

RAW_DIR = "data/raw"
IMAGES_DIR = os.path.join(RAW_DIR, "images")
METADATA_PATH = os.path.join(RAW_DIR, "metadata.csv")
REPORT_PATH = os.path.join(RAW_DIR, "data_quality_report.json")

def get_file_hash(path):
    """Compute MD5 hash of file."""
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def main():
    print("=== Phase 3: Data Quality Engineering ===")
    
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file {METADATA_PATH} not found. Run acquisition first.")
        return

    # Load metadata
    df = pd.read_csv(METADATA_PATH)
    total_records = len(df)
    print(f"Loaded metadata with {total_records} records.")

    corrupted_images = []
    hash_to_files = {}
    duplicate_images = []
    dim_anomalies = []
    stats_anomalies = []
    
    print("\nRunning image checks (corruption, duplicate files, dimensions, and statistics)...")
    for idx, row in df.iterrows():
        img_name = row['filename']
        img_path = os.path.join(IMAGES_DIR, img_name)
        
        # 1. Corruption check
        if not os.path.exists(img_path):
            corrupted_images.append((img_name, "File does not exist"))
            continue
            
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verify image integrity
            
            # Reopen to read dimensions and compute stats
            img_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_cv is None:
                corrupted_images.append((img_name, "OpenCV failed to read"))
                continue
                
            h, w = img_cv.shape
            
            # 2. Dimensions check
            if h < 100 or w < 100:
                dim_anomalies.append((img_name, f"Too small: {w}x{h}"))
                
            # 3. Image stats check (blank or extreme contrast)
            mean_val = np.mean(img_cv)
            std_val = np.std(img_cv)
            if std_val < 5:  # flat/blank image
                stats_anomalies.append((img_name, f"Blank image (std={std_val:.2f})"))
            elif mean_val < 5 or mean_val > 250:
                stats_anomalies.append((img_name, f"Extreme brightness (mean={mean_val:.2f})"))
                
            # 4. Hash duplicate check
            file_hash = get_file_hash(img_path)
            if file_hash in hash_to_files:
                duplicate_images.append((img_name, hash_to_files[file_hash]))
                hash_to_files[file_hash].append(img_name)
            else:
                hash_to_files[file_hash] = [img_name]
                
        except Exception as e:
            corrupted_images.append((img_name, str(e)))

    # 5. Missing values check in metadata
    missing_labels = df['label'].isna().sum()
    missing_metadata = df[['age', 'gender', 'temperature', 'spo2', 'heart_rate', 'cough_severity']].isna().sum().to_dict()

    # 6. Patient duplicate checks
    # Check if any patient ID appears with conflicting labels
    patient_label_counts = df.groupby('patient_id')['label'].nunique()
    conflicting_patients = patient_label_counts[patient_label_counts > 1].index.tolist()
    
    # 7. Metadata range checks (sanity checks)
    anomalous_vitals = []
    for idx, row in df.iterrows():
        pid = row['patient_id']
        age = row['age']
        temp = row['temperature']
        spo2 = row['spo2']
        hr = row['heart_rate']
        
        # Clinical bounds check
        if age < 0 or age > 120:
            anomalous_vitals.append((pid, "age", age))
        if temp < 30 or temp > 45:
            anomalous_vitals.append((pid, "temperature", temp))
        if spo2 < 50 or spo2 > 100:
            anomalous_vitals.append((pid, "spo2", spo2))
        if hr < 30 or hr > 250:
            anomalous_vitals.append((pid, "heart_rate", hr))

    # Compile report dictionary
    report = {
        "summary": {
            "total_records": total_records,
            "valid_records": total_records - len(corrupted_images),
            "corrupted_images_count": len(corrupted_images),
            "duplicate_files_count": len(duplicate_images),
            "dimension_anomalies_count": len(dim_anomalies),
            "statistical_anomalies_count": len(stats_anomalies),
            "missing_labels_count": int(missing_labels),
            "conflicting_patients_count": len(conflicting_patients),
            "anomalous_vitals_count": len(anomalous_vitals)
        },
        "details": {
            "corrupted_images": corrupted_images,
            "duplicate_images_pairs": duplicate_images[:20],  # cap details
            "dimension_anomalies": dim_anomalies,
            "statistical_anomalies": stats_anomalies,
            "missing_metadata_columns": {k: int(v) for k, v in missing_metadata.items()},
            "conflicting_patients": conflicting_patients,
            "anomalous_vitals": anomalous_vitals[:20]
        }
    }

    # Save report
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n=== DATA QUALITY REPORT ===")
    print(f"Total Images Analyzed: {report['summary']['total_records']}")
    print(f"Corrupted Images:     {report['summary']['corrupted_images_count']}")
    print(f"Duplicate Files:      {report['summary']['duplicate_files_count']}")
    print(f"Dimension Anomalies:  {report['summary']['dimension_anomalies_count']}")
    print(f"Stats Anomalies:      {report['summary']['statistical_anomalies_count']}")
    print(f"Missing Labels:       {report['summary']['missing_labels_count']}")
    print(f"Conflicting Patients: {report['summary']['conflicting_patients_count']}")
    print(f"Anomalous Vitals:     {report['summary']['anomalous_vitals_count']}")
    
    print(f"\nReport written to {REPORT_PATH}")
    print("Data quality checks complete!")

if __name__ == "__main__":
    main()
