import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Grad-CAM implementation for the ResNet18 backbone within MultimodalClassifier.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # Register hooks on ResNet layer 4 (last conv block)
        self.target_layer = self.model.image_backbone.layer4
        
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, img_tensor, tab_tensor, target_class=1):
        """
        Generate Grad-CAM heatmap for a target class.
        """
        self.model.zero_grad()
        
        # Forward pass
        # img_tensor shape: [1, 3, 224, 224]
        # tab_tensor shape: [1, 6]
        logits = self.model(img_tensor, tab_tensor)
        score = logits[0, target_class]
        
        # Backward pass to calculate gradients
        score.backward()
        
        if self.gradients is None or self.activations is None:
            # Fallback if hooks failed
            return np.zeros((224, 224), dtype=np.float32)
            
        # Global average pooling of gradients
        # activations shape: [1, 512, 7, 7]
        # gradients shape: [1, 512, 7, 7]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # Shape: [1, 512, 1, 1]
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1) # Shape: [1, 7, 7]
        cam = F.relu(cam) # Apply ReLU as we only care about features that positively correlate with class score
        
        # Normalize
        cam = cam.cpu().numpy()[0]
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        # Resize to input image size (224, 224)
        cam_resized = cv2.resize(cam, (224, 224))
        return cam_resized

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def explain_tabular_perturbation(model, img_tensor, scaled_tab, original_row, scaling_params, optimal_threshold=0.5):
    """
    Compute perturbation-based attribution scores for tabular vitals.
    Attribution measures the delta in prediction probability when replacing a feature with its baseline (0).
    """
    model.eval()
    with torch.no_grad():
        # Get baseline prediction probability (class 1, Pneumonia) using actual patient image
        base_logits = model(img_tensor, scaled_tab)
        base_prob = float(torch.softmax(base_logits, dim=1)[0, 1].cpu().numpy())
        
        attributions = {}
        continuous_cols = ['age', 'temperature', 'spo2', 'heart_rate']
        
        # 1. Continuous features perturbation
        for idx, col in enumerate(continuous_cols):
            perturbed_tab = scaled_tab.clone()
            perturbed_tab[0, idx] = 0.0 # set to training mean (0.0 scaled)
            
            p_logits = model(img_tensor, perturbed_tab)
            p_prob = float(torch.softmax(p_logits, dim=1)[0, 1].cpu().numpy())
            
            # If removing this feature decreases pneumonia probability, the feature contributes POSITIVELY to prediction
            delta = base_prob - p_prob
            attributions[col] = delta
            
        # 2. Categorical features
        for idx, col in [(4, 'gender'), (5, 'cough_severity')]:
            perturbed_tab = scaled_tab.clone()
            perturbed_tab[0, idx] = 0.0 # set to baseline
            
            p_logits = model(img_tensor, perturbed_tab)
            p_prob = float(torch.softmax(p_logits, dim=1)[0, 1].cpu().numpy())
            delta = base_prob - p_prob
            attributions[col] = delta
            
    return attributions

def explain_tabular_counterfactual(model, img_tensor, scaled_tab, original_row, scaling_params, optimal_threshold=0.17):
    """
    Non-causal model sensitivity boundary analysis.
    Finds the threshold-crossing decision boundary along a clinical path
    to identify what minimal input variations would change the model's prediction.
    Age and Gender are kept constant.
    """
    model.eval()
    with torch.no_grad():
        # Get baseline prediction probability (class 1, Pneumonia)
        logits = model(img_tensor, scaled_tab)
        base_prob = float(torch.softmax(logits, dim=1)[0, 1].cpu().numpy())
        current_class = 1 if base_prob >= optimal_threshold else 0
        
        # Define reference boundary search directions
        if current_class == 1:
            boundary_reference = {
                "temperature": 36.8,
                "spo2": 98,
                "heart_rate": 95,
                "cough_severity": "Absent"
            }
        else:
            boundary_reference = {
                "temperature": 38.6,
                "spo2": 90,
                "heart_rate": 132,
                "cough_severity": "Severe"
            }
            
        continuous_cols = ['age', 'temperature', 'spo2', 'heart_rate']
        gender_mapping = {"M": 0, "F": 1}
        cough_mapping = {"Absent": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
        cough_reverse = {0: "Absent", 1: "Mild", 2: "Moderate", 3: "Severe"}
        
        # Extract patient's actual vitals
        def get_val(obj, key):
            if hasattr(obj, key):
                return getattr(obj, key)
            if isinstance(obj, dict):
                return obj.get(key)
            return obj[key]
            
        actual_vitals = {
            "temperature": float(get_val(original_row, "temperature")),
            "spo2": int(get_val(original_row, "spo2")),
            "heart_rate": int(get_val(original_row, "heart_rate")),
            "cough_severity": get_val(original_row, "cough_severity")
        }
        
        # Search path from actual to reference boundary to find the transition point
        n_steps = 20
        found_boundary = False
        boundary_vitals = None
        boundary_prob = None
        
        for step in range(n_steps + 1):
            alpha = step / n_steps
            
            # Interpolate raw values along the search path
            temp_interp = (1 - alpha) * actual_vitals["temperature"] + alpha * boundary_reference["temperature"]
            spo2_interp = int(round((1 - alpha) * actual_vitals["spo2"] + alpha * boundary_reference["spo2"]))
            hr_interp = int(round((1 - alpha) * actual_vitals["heart_rate"] + alpha * boundary_reference["heart_rate"]))
            
            cough_actual_code = cough_mapping.get(actual_vitals["cough_severity"], 0)
            cough_target_code = cough_mapping.get(boundary_reference["cough_severity"], 0)
            cough_interp_code = int(round((1 - alpha) * cough_actual_code + alpha * cough_target_code))
            cough_interp = cough_reverse.get(cough_interp_code, "Absent")
            
            perturbed_tab = scaled_tab.clone()
            
            # Z-Score scale interpolated values
            temp_mean, temp_std = scaling_params["temperature"]["mean"], scaling_params["temperature"]["std"]
            spo2_mean, spo2_std = scaling_params["spo2"]["mean"], scaling_params["spo2"]["std"]
            hr_mean, hr_std = scaling_params["heart_rate"]["mean"], scaling_params["heart_rate"]["std"]
            
            perturbed_tab[0, 1] = (temp_interp - temp_mean) / temp_std
            perturbed_tab[0, 2] = (spo2_interp - spo2_mean) / spo2_std
            perturbed_tab[0, 3] = (hr_interp - hr_mean) / hr_std
            perturbed_tab[0, 5] = float(cough_interp_code)
            
            # Forward pass
            p_logits = model(img_tensor, perturbed_tab)
            p_prob = float(torch.softmax(p_logits, dim=1)[0, 1].cpu().numpy())
            p_class = 1 if p_prob >= optimal_threshold else 0
            
            if p_class != current_class:
                found_boundary = True
                boundary_vitals = {
                    "temperature": round(temp_interp, 1),
                    "spo2": int(spo2_interp),
                    "heart_rate": int(hr_interp),
                    "cough_severity": cough_interp
                }
                boundary_prob = p_prob
                break
                
        if not found_boundary:
            boundary_vitals = boundary_reference
            boundary_prob = 0.0 if current_class == 1 else 1.0
            
        return {
            "current_probability": base_prob,
            "current_decision": "HIGH RISK (Pneumonia)" if current_class == 1 else "LOW RISK (Normal)",
            "counterfactual_vitals": boundary_vitals,
            "counterfactual_probability": boundary_prob,
            "counterfactual_decision": "LOW RISK (Normal)" if current_class == 1 else "HIGH RISK (Pneumonia)",
            "disclaimer": "Model sensitivity boundary counterfactual — NOT a clinical recommendation, causal explanation, or target health directive."
        }
