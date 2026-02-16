import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2, architecture='resnet50', pretrained=True, dropout=0.5):
    """
    Get a pretrained model for deepfake detection with enhanced classifier.
    
    Args:
        num_classes: Number of output classes (default: 2 for real/fake)
        architecture: Model architecture ('resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b3')
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate for regularization (default: 0.5)
    
    Returns:
        PyTorch model with enhanced classifier head
    """
    
    if architecture == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        # Enhanced classifier with dropout
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        
    elif architecture == 'resnet50':
        # Better accuracy than ResNet18, recommended for large datasets
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        # Enhanced classifier with dropout and batch norm
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
    elif architecture == 'efficientnet_b0':
        # Efficient and accurate, good for limited resources
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        
    elif architecture == 'efficientnet_b3':
        # More powerful EfficientNet, better for large datasets
        model = models.efficientnet_b3(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
    elif architecture == 'efficientnet_b4':
        # High-performance EfficientNet, excellent for production
        model = models.efficientnet_b4(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 768),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Dropout(dropout / 2),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 3),
            nn.Linear(256, num_classes)
        )
        
    else:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Choose from: resnet18, resnet50, efficientnet_b0, efficientnet_b3, efficientnet_b4")
    
    return model


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for improved accuracy.
    Use this for final predictions after training individual models.
    """
    def __init__(self, models_list):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models_list)
    
    def forward(self, x):
        # Average predictions from all models
        outputs = []
        for model in self.models:
            outputs.append(torch.softmax(model(x), dim=1))
        return torch.mean(torch.stack(outputs), dim=0)


if __name__ == "__main__":
    print("Testing different architectures:")
    print("-" * 50)
    
    for arch in ['resnet18', 'resnet50', 'efficientnet_b0']:
        model = get_model(architecture=arch)
        x = torch.randn(1, 3, 224, 224)  # Standard input size
        y = model(x)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"{arch}:")
        print(f"  Output shape: {y.shape}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print()
