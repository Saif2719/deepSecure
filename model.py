import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):
    # Load EfficientNet-B4 pretrained model
    model = models.efficientnet_b4(pretrained=True)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    model = get_model()
    x = torch.randn(1, 3, 380, 380)  # EfficientNet-B4 expects 380x380
    y = model(x)
    print("Output shape:", y.shape)
