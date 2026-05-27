import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.image_model import DeepFakeImageModel

import torch
import torch.nn as nn
import timm

class DeepFakeImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0
        )
        self.classifier = nn.Linear(1280, 2)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
