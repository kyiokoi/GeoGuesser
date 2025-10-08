"""
Simple CNN model using transfer learning with ResNet18
"""
import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=6, pretrained=True):
    """
    Create a ResNet18 model for country classification.
    
    Args:
        num_classes: Number of countries to classify
        pretrained: Use ImageNet pretrained weights
    
    Returns:
        PyTorch model
    """
    # Load pretrained ResNet18
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    
    # Replace final layer for our 6 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = create_model(num_classes=6)
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be [1, 6]
