import os
import urllib.request
import pandas as pd
import numpy as np
from PIL import Image
import io
import sys

# Constants
REPO_ID = "hf-vision/chest-xray-pneumonia"
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"
FILES_TO_DOWNLOAD = {
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
    "train_2": "data/train-00002-of-00007.parquet"
}

RAW_DIR = "data/raw"
IMAGES_DIR = os.path.join(RAW_DIR, "images")

def download_file(url, dest_path):
    """Download file with progress printing."""
    print(f"Downloading {url} -> {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = min(100, readsofar * 100 / totalsize)
            if blocknum % 100 == 0:  # limit output frequency
                sys.stdout.write(f"\rProgress: {percent:.1f}% ({readsofar/(1024*1024):.1f}MB / {totalsize/(1024*1024):.1f}MB)")
                sys.stdout.flush()
        else:
            sys.stdout.write(f"\rRead {readsofar} bytes")
            sys.stdout.flush()
            
    urllib.request.urlretrieve(url, dest_path, reporthook)
    print("\nDownload complete.")

def extract_patient_id(path, label):
    """
    Extract patient ID from filename.
    Pneumonia: person100_bacteria_475.jpeg -> person100
    Normal: NORMAL2-IM-1427-0001.jpeg -> NORMAL2-IM-1427
            IM-0001-0001.jpeg -> IM-0001
    """
    basename = os.path.basename(path)
    if label == 1: # Pneumonia
        if basename.startswith("person"):
            parts = basename.split("_")
            return parts[0]
        else:
            return basename.split(".")[0][:10]
    else: # Normal
        if basename.startswith("NORMAL"):
            parts = basename.split("-")
            if len(parts) >= 3:
                return "-".join(parts[:3])
            return "-".join(parts[:2])
        elif basename.startswith("IM"):
            parts = basename.split("-")
            if len(parts) >= 2:
                return "-".join(parts[:2])
            return basename.split(".")[0]
        else:
            return basename.split(".")[0][:10]

def main():
    print("=== Phase 2: Data Acquisition Pipeline ===")
    
    # 1. Download Parquet files
    downloaded_paths = {}
    for key, relative_path in FILES_TO_DOWNLOAD.items():
        url = BASE_URL + relative_path
        dest = os.path.join(RAW_DIR, os.path.basename(relative_path))
        downloaded_paths[key] = dest
        if not os.path.exists(dest):
            download_file(url, dest)
        else:
            print(f"File {dest} already exists, skipping download.")
            
    # 2. Load Parquet files and combine
    print("\nReading parquet files...")
    dfs = []
    for key, path in downloaded_paths.items():
        df = pd.read_parquet(path)
        df['source_split'] = key
        dfs.append(df)
    
    raw_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(raw_df)} raw records.")
    
    # 3. Extract and save images, collect file metadata
    print("\nExtracting and saving images to raw/images/...")
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    metadata_records = []
    
    for idx, row in raw_df.iterrows():
        img_dict = row['image']
        label = int(row['label'])
        source_split = row['source_split']
        
        img_bytes = img_dict['bytes']
        orig_path = img_dict['path']
        
        # Save image file
        filename = f"{idx:05d}_{os.path.basename(orig_path)}"
        dest_img_path = os.path.join(IMAGES_DIR, filename)
        
        try:
            image = Image.open(io.BytesIO(img_bytes))
            image.save(dest_img_path)
            
            # Extract patient ID
            patient_id = extract_patient_id(orig_path, label)
            
            metadata_records.append({
                "image_id": idx,
                "filename": filename,
                "original_path": orig_path,
                "patient_id": patient_id,
                "label": label,
                "source_split": source_split,
                "width": image.width,
                "height": image.height
            })
        except Exception as e:
            print(f"Error processing image {idx} ({orig_path}): {e}")
            
    metadata_df = pd.DataFrame(metadata_records)
    print(f"Successfully processed {len(metadata_df)} images.")
    
    # 4. Generate patient-consistent clinical metadata
    print("\nGenerating patient-consistent clinical variables...")
    # Find all unique patient IDs
    unique_patients = metadata_df['patient_id'].unique()
    print(f"Found {len(unique_patients)} unique patient IDs.")
    
    # We must assign a consensus label to each patient to avoid patient label conflict
    # (Though in this dataset patient labels should be 100% consistent)
    patient_labels = metadata_df.groupby('patient_id')['label'].agg(lambda x: x.value_counts().index[0]).to_dict()
    
    # Deterministic generation using patient_id as seed for reproducibility
    patient_metadata = []
    for pid in unique_patients:
        plabel = patient_labels[pid]
        
        # Use hash of patient_id as random seed
        seed = abs(hash(pid)) % (2**32)
        rng = np.random.default_rng(seed)
        
        # Pediatric patient age: 1.0 to 5.0 years (12 to 60 months)
        age = round(rng.uniform(1.0, 5.0), 1)
        gender = "M" if rng.uniform() > 0.5 else "F"
        
        if plabel == 0:  # Normal
            # Temp: Mean 36.8, SD 0.25 (Celsius)
            temperature = round(rng.normal(36.8, 0.25), 1)
            # SpO2: Mean 98.5%, SD 0.8% (cap at 100)
            spo2 = int(min(100, round(rng.normal(98.5, 0.8))))
            # Heart rate: Mean 95, SD 8
            heart_rate = int(rng.normal(95, 8))
            # Cough Severity: Absent (70%), Mild (25%), Moderate (5%), Severe (0%)
            cough_choice = rng.choice(["Absent", "Mild", "Moderate", "Severe"], p=[0.70, 0.25, 0.05, 0.00])
        else:  # Pneumonia
            # Temp: Mean 38.6, SD 0.6 (fever)
            temperature = round(rng.normal(38.6, 0.6), 1)
            # SpO2: Mean 90.5%, SD 3.5% (mild to moderate hypoxia)
            spo2 = int(min(99, round(rng.normal(90.5, 3.5))))
            # Heart rate: Mean 132, SD 12 (tachycardia)
            heart_rate = int(rng.normal(132, 12))
            # Cough Severity: Absent (5%), Mild (15%), Moderate (40%), Severe (40%)
            cough_choice = rng.choice(["Absent", "Mild", "Moderate", "Severe"], p=[0.05, 0.15, 0.40, 0.40])
            
        patient_metadata.append({
            "patient_id": pid,
            "age": age,
            "gender": gender,
            "temperature": temperature,
            "spo2": spo2,
            "heart_rate": heart_rate,
            "cough_severity": cough_choice
        })
        
    patient_df = pd.DataFrame(patient_metadata)
    
    # Merge metadata together
    final_df = pd.merge(metadata_df, patient_df, on="patient_id", how="left")
    
    # Save the raw metadata
    metadata_path = os.path.join(RAW_DIR, "metadata.csv")
    final_df.to_csv(metadata_path, index=False)
    print(f"Saved merged patient metadata sheet to {metadata_path}")
    print("\nDataset Card Preview:")
    print(f"Total Records: {len(final_df)}")
    print(f"Normal Cases (0): {len(final_df[final_df['label'] == 0])}")
    print(f"Pneumonia Cases (1): {len(final_df[final_df['label'] == 1])}")
    print(f"Unique Patients: {len(unique_patients)}")
    print(f"Acquisition Pipeline Complete!")

if __name__ == "__main__":
    main()
