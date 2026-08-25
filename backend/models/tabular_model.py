import torch
import torch.nn as nn

class TabularClassifier(nn.Module):
    """
    Tabular-only Classifier using a simple Multi-Layer Perceptron (MLP).
    Input: 6 features (4 scaled continuous, 1 binary gender, 1 ordinal cough severity)
    """
    def __init__(self, input_dim=6, num_classes=2):
        super(TabularClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.net(x)
