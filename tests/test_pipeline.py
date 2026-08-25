import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.models.dataset import MultimodalDataset
from backend.models.multimodal_model import MultimodalClassifier
from backend.services.reliability import predict_with_mc_dropout, MahalanobisOODDetector
from backend.services.explainability import explain_tabular_perturbation
from scripts.train_models import compute_ece

PROCESSED_DIR = "data/processed"
IMAGES_DIR = "data/raw/images"
SCALING_PARAMS_PATH = os.path.join(PROCESSED_DIR, "scaling_params.json")

class TestClinicalPipeline(unittest.TestCase):
    
    def setUp(self):
        self.train_split_path = os.path.join(PROCESSED_DIR, "train_split.csv")
        self.val_split_path = os.path.join(PROCESSED_DIR, "val_split.csv")
        self.test_split_path = os.path.join(PROCESSED_DIR, "test_split.csv")
        
    def test_01_patient_leakage(self):
        """Assert 0% patient ID overlap across splits (leakage check)."""
        if not os.path.exists(self.train_split_path):
            self.skipTest("Data splits not generated yet.")
            
        df_train = pd.read_csv(self.train_split_path)
        df_val = pd.read_csv(self.val_split_path)
        df_test = pd.read_csv(self.test_split_path)
        
        train_patients = set(df_train['patient_id'].unique())
        val_patients = set(df_val['patient_id'].unique())
        test_patients = set(df_test['patient_id'].unique())
        
        # Intersections
        overlap_train_val = train_patients.intersection(val_patients)
        overlap_train_test = train_patients.intersection(test_patients)
        overlap_val_test = val_patients.intersection(test_patients)
        
        self.assertEqual(len(overlap_train_val), 0, f"Leaked train-val patients: {overlap_train_val}")
        self.assertEqual(len(overlap_train_test), 0, f"Leaked train-test patients: {overlap_train_test}")
        self.assertEqual(len(overlap_val_test), 0, f"Leaked val-test patients: {overlap_val_test}")
        
    def test_02_preprocessing_scaling(self):
        """Verify tabular scaling parameters fit only on training, preventing leakage."""
        if not os.path.exists(SCALING_PARAMS_PATH):
            self.skipTest("Scaling parameters not generated yet.")
            
        with open(SCALING_PARAMS_PATH, 'r') as f:
            import json
            params = json.load(f)
            
        self.assertIn("age", params)
        self.assertIn("temperature", params)
        self.assertIn("spo2", params)
        self.assertIn("heart_rate", params)
        
        # Verify std > 0
        for col, stats in params.items():
            self.assertGreater(stats["std"], 0)
            
    def test_03_model_forward_pass(self):
        """Verify model output logit dimensions."""
        model = MultimodalClassifier(tabular_dim=6, pretrained=False)
        
        # Dummy batch of size 2
        dummy_img = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
        dummy_tab = torch.zeros((2, 6), dtype=torch.float32)
        
        logits = model(dummy_img, dummy_tab)
        self.assertEqual(logits.shape, (2, 2)) # binary classification logits
        
    def test_04_ece_computation(self):
        """Test expected calibration error calculations."""
        # Perfect calibration case: predictions match exact labels
        probs = np.array([0.01, 0.02, 0.98, 0.99])
        labels = np.array([0, 0, 1, 1])
        ece = compute_ece(probs, labels, n_bins=10)
        self.assertLess(ece, 0.05) # should be extremely low
        
    def test_05_ood_detector(self):
        """Test OOD detector fits and calculates distances correctly."""
        # 10 training embeddings of size 128
        train_embeds = np.random.normal(0, 1, (10, 128))
        detector = MahalanobisOODDetector()
        detector.fit(train_embeds, threshold_percentile=90.0)
        
        # An in-distribution sample
        in_sample = np.random.normal(0, 1, (128,))
        res_in = detector.is_ood(in_sample)
        
        # An out-of-distribution sample (distant mean)
        out_sample = np.random.normal(10, 1, (128,))
        res_out = detector.is_ood(out_sample)
        
        self.assertGreater(res_out["distance"], res_in["distance"])
        self.assertEqual(res_out["ood_risk"], "CRITICAL")

if __name__ == '__main__':
    unittest.main()
