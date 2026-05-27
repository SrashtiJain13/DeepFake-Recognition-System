import torch
import torch.nn as nn
import timm

class DeepFakeVideoModel(nn.Module):
    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        self.cnn = timm.create_model(
            "resnet18",
            pretrained=pretrained_backbone,
            num_classes=0
        )
        self.lstm = nn.LSTM(512, 256, batch_first=True)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        batch, frames, c, h, w = x.shape
        x = x.view(batch * frames, c, h, w)
        features = self.cnn(x)
        features = features.view(batch, frames, -1)
        _, (h_n, _) = self.lstm(features)
        return self.fc(h_n[-1])
