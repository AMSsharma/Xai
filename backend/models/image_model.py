import torch
import torch.nn as nn
import torchvision.models as models

class ImageClassifier(nn.Module):
    """
    Image-only Classifier using ResNet18.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(ImageClassifier, self).__init__()
        # Use torchvision.models.resnet18
        if pretrained:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.resnet = models.resnet18()
            
        # Freeze early layers if doing transfer learning (first 6 layers)
        # to prevent overriding general features and save VRAM/compute
        child_counter = 0
        for child in self.resnet.children():
            child_counter += 1
            if child_counter < 7:
                for param in child.parameters():
                    param.requires_grad = False
                    
        # Replace fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)
