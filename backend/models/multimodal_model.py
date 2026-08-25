import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalClassifier(nn.Module):
    """
    Multimodal Fusion Network combining Radiograph Images and Patient Clinical Metadata.
    """
    def __init__(self, tabular_dim=6, num_classes=2, pretrained=True):
        super(MultimodalClassifier, self).__init__()
        
        # 1. Image Branch (ResNet18 Feature Extractor)
        if pretrained:
            self.image_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.image_backbone = models.resnet18()
            
        # Freeze early layers for stability
        child_counter = 0
        for child in self.image_backbone.children():
            child_counter += 1
            if child_counter < 7:
                for param in child.parameters():
                    param.requires_grad = False
                    
        # Project ResNet outputs (512) to a joint space of 128
        in_ftrs = self.image_backbone.fc.in_features
        self.image_backbone.fc = nn.Identity()  # remove final ResNet linear classifier
        
        self.image_projection = nn.Sequential(
            nn.Linear(in_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 2. Tabular Branch (MLP Feature Extractor)
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 3. Fusion Classifier
        # Input size: 128 (image representation) + 32 (tabular representation) = 160
        self.fusion_fc = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, img, tab):
        # Extract image embeddings
        img_features = self.image_backbone(img) # Size: [batch_size, 512]
        img_embed = self.image_projection(img_features) # Size: [batch_size, 128]
        
        # Extract tabular embeddings
        tab_embed = self.tabular_encoder(tab) # Size: [batch_size, 32]
        
        # Fuse (Concatenate)
        fused = torch.cat((img_embed, tab_embed), dim=1) # Size: [batch_size, 160]
        
        # Classify
        logits = self.fusion_fc(fused) # Size: [batch_size, num_classes]
        return logits
