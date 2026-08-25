import torch
import numpy as np
import json

def enable_dropout(model):
    """
    Force dropout layers to remain in training mode during evaluation.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def predict_with_mc_dropout(model, img_tensor, tab_tensor, temperature=1.0, n_iter=15):
    """
    Perform Monte Carlo Dropout predictions to estimate calibrated probability and uncertainty.
    """
    model.eval()
    enable_dropout(model) # force dropout layers active
    
    probs_list = []
    
    with torch.no_grad():
        for _ in range(n_iter):
            logits = model(img_tensor, tab_tensor)
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=1)
            probs_list.append(probs.cpu().numpy()[0])
            
    # Convert to array: shape [n_iter, 2]
    probs_arr = np.array(probs_list)
    
    # Compute mean and standard deviation for the positive class (class 1, Pneumonia)
    pos_probs = probs_arr[:, 1]
    mean_prob = float(np.mean(pos_probs))
    std_prob = float(np.std(pos_probs))
    
    # Categorize confidence
    # If standard deviation (uncertainty) is high, confidence category is LOW
    if std_prob > 0.15:
        confidence_category = "HIGH UNCERTAINTY"
    elif std_prob > 0.08:
        confidence_category = "MODERATE UNCERTAINTY"
    else:
        confidence_category = "LOW UNCERTAINTY"
        
    return {
        "mean_probability": mean_prob,
        "uncertainty": std_prob,
        "confidence_category": confidence_category,
        "all_probs": pos_probs.tolist()
    }

class MahalanobisOODDetector:
    """
    OOD Anomaly Detector based on Mahalanobis Distance of projection embeddings.
    """
    def __init__(self, params_path=None):
        self.mean = None
        self.cov_inv = None
        self.threshold = None
        
        if params_path and torch.os.path.exists(params_path):
            self.load_params(params_path)

    def fit(self, embeddings, threshold_percentile=95.0):
        """
        Fit detector on training embeddings.
        embeddings shape: [N, D]
        """
        # Compute mean
        self.mean = np.mean(embeddings, axis=0) # Shape: [D]
        
        # Compute Diagonal Covariance (Normalized Euclidean) for high numerical stability
        var = np.var(embeddings, axis=0)
        # Avoid division by zero
        var = np.where(var == 0, 1e-5, var)
        self.cov_inv = np.diag(1.0 / var)
        
        # Calculate distances for all training samples to set the OOD threshold
        dists = [self.calculate_distance(x) for x in embeddings]
        self.threshold = float(np.percentile(dists, threshold_percentile))
        print(f"OOD Detector fitted with diagonal covariance. Threshold ({threshold_percentile}th percentile): {self.threshold:.4f}")

    def calculate_distance(self, embedding):
        """
        Compute Mahalanobis distance for a single embedding vector.
        """
        diff = embedding - self.mean
        dist = np.sqrt(np.dot(np.dot(diff, self.cov_inv), diff.T))
        return float(dist)

    def is_ood(self, embedding):
        """
        Return whether the embedding is Out-of-Distribution, its distance, and OOD risk level.
        """
        dist = self.calculate_distance(embedding)
        is_anomaly = dist > self.threshold
        
        if dist > self.threshold * 1.5:
            risk = "CRITICAL"
        elif dist > self.threshold:
            risk = "HIGH"
        else:
            risk = "LOW"
            
        return {
            "is_ood": bool(is_anomaly),
            "distance": dist,
            "threshold": self.threshold,
            "ood_risk": risk
        }

    def save_params(self, path):
        """Save OOD parameters to a file."""
        state = {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "cov_inv": self.cov_inv.tolist() if self.cov_inv is not None else None,
            "threshold": self.threshold
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_params(self, path):
        """Load OOD parameters from a file."""
        with open(path, 'r') as f:
            state = json.load(f)
        self.mean = np.array(state["mean"]) if state["mean"] is not None else None
        self.cov_inv = np.array(state["cov_inv"]) if state["cov_inv"] is not None else None
        self.threshold = state["threshold"]
