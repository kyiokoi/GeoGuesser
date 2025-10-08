"""
Evaluate the trained model on test set
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import create_model


# Config
DATA_DIR = Path("Data/Processed")
MODEL_DIR = Path("models")
IMG_SIZE = 224


def evaluate():
    """Evaluate model on test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load class mapping
    with open(MODEL_DIR / "class_to_idx.json", "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Test transforms (no augmentation)
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset = datasets.ImageFolder(DATA_DIR / "Testing", transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model (using original model.py - ResNet18)
    model = create_model(num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pt", map_location=device))
    model = model.to(device)
    model.eval()
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    
    # Classification report
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    output_path = MODEL_DIR / "confusion_matrix.png"
    plt.savefig(output_path)
    print(f"\n✓ Confusion matrix saved to {output_path}")
    
    # Save results
    results = {
        "test_accuracy": float(accuracy),
        "classification_report": classification_report(all_labels, all_preds, 
                                                      target_names=class_names, 
                                                      output_dict=True)
    }
    with open(MODEL_DIR / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {MODEL_DIR / 'test_results.json'}")


if __name__ == "__main__":
    evaluate()
