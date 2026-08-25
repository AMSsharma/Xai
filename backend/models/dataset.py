import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import torchvision.transforms as transforms

class MultimodalDataset(Dataset):
    """
    Custom PyTorch Dataset for Multimodal Chest X-Ray & Clinical Tabular Data.
    """
    def __init__(self, csv_path, images_dir, scaling_params_path=None, is_training=False, fit_scaler=False):
        if csv_path:
            self.df = pd.read_csv(csv_path)
        else:
            self.df = None
        self.images_dir = images_dir
        self.is_training = is_training
        
        # Categorical mappings
        self.cough_mapping = {"Absent": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
        self.gender_mapping = {"M": 0, "F": 1}
        
        # Fit or load tabular scaling parameters
        self.continuous_cols = ['age', 'temperature', 'spo2', 'heart_rate']
        
        if fit_scaler and self.df is not None:
            # Fit scaling parameters ONLY on the training split
            self.scaling_params = {}
            for col in self.continuous_cols:
                mean = float(self.df[col].mean())
                std = float(self.df[col].std())
                self.scaling_params[col] = {"mean": mean, "std": std if std > 0 else 1.0}
            
            if scaling_params_path:
                os.makedirs(os.path.dirname(scaling_params_path), exist_ok=True)
                with open(scaling_params_path, 'w') as f:
                    json.dump(self.scaling_params, f, indent=2)
                print(f"Tabular scaling parameters fitted and saved to {scaling_params_path}")
        else:
            if scaling_params_path and os.path.exists(scaling_params_path):
                with open(scaling_params_path, 'r') as f:
                    self.scaling_params = json.load(f)
            else:
                # Fallback to standard defaults if no scaling file exists
                print("Warning: Scaling params path not found, using generic defaults.")
                self.scaling_params = {
                    "age": {"mean": 3.0, "std": 1.2},
                    "temperature": {"mean": 37.8, "std": 1.0},
                    "spo2": {"mean": 94.0, "std": 5.0},
                    "heart_rate": {"mean": 110.0, "std": 20.0}
                }
                
        # Define image transforms
        # 1. Medically reasonable training transforms: mild rotations and contrast adjustments.
        #    We avoid vertical flips because heart/lung orientation is clinically fixed.
        if self.is_training:
            self.image_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def preprocess_tabular(self, row):
        """Scale continuous vitals and encode categorical variables."""
        # Scale continuous
        vitals = []
        for col in self.continuous_cols:
            val = float(row[col])
            mean = self.scaling_params[col]["mean"]
            std = self.scaling_params[col]["std"]
            scaled_val = (val - mean) / std
            vitals.append(scaled_val)
            
        # Encode categorical
        gender_code = self.gender_mapping.get(row['gender'], 0)
        cough_code = self.cough_mapping.get(row['cough_severity'], 0)
        
        # Merge into a single tabular tensor
        # Size: 4 scaled continuous + 1 gender code + 1 cough severity code = 6 features
        tabular_vector = np.array(vitals + [gender_code, cough_code], dtype=np.float32)
        return torch.tensor(tabular_vector, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, row['filename'])
        try:
            # Read as RGB
            image = Image.open(img_path).convert('RGB')
            img_tensor = self.image_transforms(image)
        except Exception as e:
            # Fallback for error/corrupt image
            print(f"Error loading image {img_path}, using dummy tensor: {e}")
            img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            
        # Get preprocessed tabular data
        tab_tensor = self.preprocess_tabular(row)
        
        # Label
        label = torch.tensor(int(row['label']), dtype=torch.long)
        
        return img_tensor, tab_tensor, label
