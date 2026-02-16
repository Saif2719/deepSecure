import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from load_dataset import DeepfakeDataset
from model import get_model

# ---------- Config ----------
BATCH_SIZE = 32       # Increased for GPU
EPOCHS = 20           # More epochs with GPU
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-detect GPU
print(f"Using device: {DEVICE}")

# ---------- Dataset Paths ----------
# For local training (default)
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/test"  # Using test as validation since no val folder exists

# For Kaggle training (uncomment and update these paths)
# TRAIN_DIR = "/kaggle/input/your-dataset-name/train"
# VAL_DIR = "/kaggle/input/your-dataset-name/val"

# For Kaggle output (uncomment for Kaggle)
# OUTPUT_DIR = "/kaggle/working"
OUTPUT_DIR = "output"  # Local output directory

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Datasets ----------
train_dataset = DeepfakeDataset(TRAIN_DIR, augment=True)  # Use augmentation for training
val_dataset = DeepfakeDataset(VAL_DIR, augment=False)  # No augmentation for validation

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                         shuffle=True, num_workers=2 if DEVICE == "cuda" else 0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                       shuffle=False, num_workers=2 if DEVICE == "cuda" else 0)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# ---------- Model ----------
model = get_model(num_classes=2)
model.to(DEVICE)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                 factor=0.5, patience=3, verbose=True)

# ---------- Training Loop ----------
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    avg_loss = total_loss / len(train_loader)

    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100. * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] - "
          f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Update learning rate
    scheduler.step(val_acc)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"✓ Saved best model with val_acc: {val_acc:.2f}%")

print("\n" + "="*50)
print("Training finished!")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print("="*50)

# Save final model
final_model_path = os.path.join(OUTPUT_DIR, "deepfake_model.pth")
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")
