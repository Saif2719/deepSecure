import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):
    model = models.resnet18(pretrained=True)

    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    model = get_model()
    x = torch.randn(1, 3, 224, 224)  # ResNet prefers 224x224
    y = model(x)
    print("Output shape:", y.shape)
