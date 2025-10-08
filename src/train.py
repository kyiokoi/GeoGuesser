"""
Simple training script for GeoGuesser country classifier
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm
import json

# Import model files (try improved models first)
try:
    from model_improved import create_model, count_parameters  # Best models
except ImportError:
    try:
        from model_custom import create_model, count_parameters  # Custom models
    except ImportError:
        from model import create_model, count_parameters  # Fallback


# Config
DATA_DIR = Path("Data/Processed")
MODEL_DIR = Path("models")
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
IMG_SIZE = 224  # Standard input size

# Model selection: "resnet18"
MODEL_TYPE = "resnet18"

CLASSES = ["Indonesia", "Laos", "Malaysia", "Philippines", "Singapore", "Thailand"]


def get_data_loaders():
    """Create train and validation data loaders with augmentation."""
    
    # Data augmentation for training
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(DATA_DIR / "Training", transform=train_transforms)
    val_dataset = datasets.ImageFolder(DATA_DIR / "Validation", transform=val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, train_dataset.class_to_idx


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train():
    """Main training function."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model directory
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, class_to_idx = get_data_loaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Classes: {list(class_to_idx.keys())}")
    
    # Save class mapping
    with open(MODEL_DIR / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)
    
    # Create model
    model = create_model(num_classes=len(CLASSES), model_type=MODEL_TYPE, pretrained=(MODEL_TYPE=="resnet18"))
    model = model.to(device)
    print(f"Model: {MODEL_TYPE} ({count_parameters(model):,} parameters)")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    # Save final model and history
    torch.save(model.state_dict(), MODEL_DIR / "final_model.pt")
    with open(MODEL_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to {MODEL_DIR.resolve()}")


if __name__ == "__main__":
    train()
