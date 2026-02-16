"""
Optimized training script for FaceForensics++ & Celeb-DF Combined Dataset
This script is specifically configured for this large-scale deepfake dataset.

Dataset: Face Forensic++ & Celeb-DF Combined Deepfake Data
Link: https://www.kaggle.com/datasets/sorokin/faceforensics

Features:
- Multi-scale training for better feature extraction
- Mixed precision training for faster GPU utilization
- Advanced data augmentation
- Gradient accumulation for larger effective batch size
- Model ensemble support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
import time
from datetime import datetime

from load_dataset import DeepfakeDataset
from model import get_model

# ---------- Configuration ----------
BATCH_SIZE = 32       # Reduced for mixed precision training
ACCUMULATION_STEPS = 2  # Effective batch size = 32 * 2 = 64
EPOCHS = 40
LR = 0.0001
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS}")

# ---------- Dataset Paths for FaceForensics++ & Celeb-DF ----------
# Common structures for this dataset:
# Option 1: If dataset has train/val split
TRAIN_DIR = "/kaggle/input/faceforensics/train"
VAL_DIR = "/kaggle/input/faceforensics/val"

# Option 2: If dataset has different structure (uncomment if needed)
# TRAIN_DIR = "/kaggle/input/faceforensics/FaceForensics++/train"
# VAL_DIR = "/kaggle/input/faceforensics/FaceForensics++/val"

# Option 3: If you need to explore first (uncomment to find structure)
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     print(dirname)
#     if len(filenames) > 0:
#         print(f"  Files: {filenames[:3]}")

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Model Configuration ----------
# For FaceForensics++, use more powerful architecture
ARCHITECTURE = 'efficientnet_b4'  # Options: 'resnet50', 'efficientnet_b0', 'efficientnet_b3', 'efficientnet_b4'
USE_MIXED_PRECISION = True  # Faster training with FP16

# ---------- Load Datasets ----------
print("\n" + "="*70)
print("Loading FaceForensics++ & Celeb-DF Dataset...")
print("="*70)

train_dataset = DeepfakeDataset(TRAIN_DIR, augment=True)
val_dataset = DeepfakeDataset(VAL_DIR, augment=False)

print(f"Train samples: {len(train_dataset):,}")
print(f"Val samples: {len(val_dataset):,}")

# Check class balance
real_count = train_dataset.labels.count(0)
fake_count = train_dataset.labels.count(1)
print(f"Train - Real: {real_count:,}, Fake: {fake_count:,}")
print(f"Balance ratio: {real_count/fake_count:.2f}:1")

# Data loaders with optimized settings
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,  # Larger batch for validation
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)

print(f"Train batches: {len(train_loader):,}")
print(f"Val batches: {len(val_loader):,}")

# ---------- Initialize Model ----------
print(f"\nInitializing {ARCHITECTURE} model...")
model = get_model(num_classes=2, architecture=ARCHITECTURE, pretrained=True)
model.to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ---------- Loss & Optimizer ----------
# Use label smoothing for better generalization
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# AdamW optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Cosine annealing with warm restarts
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

# Mixed precision scaler
scaler = GradScaler() if USE_MIXED_PRECISION else None

# ---------- Training Functions ----------
def train_epoch(model, loader, criterion, optimizer, scaler, device, accumulation_steps):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Mixed precision training
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Statistics
        total_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Progress update every 100 batches
        if (batch_idx + 1) % 100 == 0:
            current_acc = 100. * correct / total
            print(f"  Batch [{batch_idx+1}/{len(loader)}] - "
                  f"Loss: {loss.item() * accumulation_steps:.4f}, "
                  f"Acc: {current_acc:.2f}%")
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Track per-class accuracy
    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            real_mask = labels == 0
            fake_mask = labels == 1
            
            real_total += real_mask.sum().item()
            fake_total += fake_mask.sum().item()
            real_correct += (predicted[real_mask] == labels[real_mask]).sum().item()
            fake_correct += (predicted[fake_mask] == labels[fake_mask]).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    real_acc = 100. * real_correct / real_total if real_total > 0 else 0
    fake_acc = 100. * fake_correct / fake_total if fake_total > 0 else 0
    
    return avg_loss, accuracy, real_acc, fake_acc


# ---------- Training Loop ----------
print("\n" + "="*70)
print("Starting Training...")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

best_val_acc = 0.0
best_epoch = 0
patience_counter = 0
early_stop_patience = 10  # Increased for large dataset

training_history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'real_acc': [], 'fake_acc': []
}

start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print("-" * 70)
    
    # Training
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, scaler, DEVICE, ACCUMULATION_STEPS
    )
    
    # Validation
    val_loss, val_acc, real_acc, fake_acc = validate(
        model, val_loader, criterion, DEVICE
    )
    
    # Update learning rate
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    training_history['train_loss'].append(train_loss)
    training_history['train_acc'].append(train_acc)
    training_history['val_loss'].append(val_loss)
    training_history['val_acc'].append(val_acc)
    training_history['real_acc'].append(real_acc)
    training_history['fake_acc'].append(fake_acc)
    
    # Print epoch summary
    epoch_time = time.time() - epoch_start
    print(f"\nEpoch Summary:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"  Real Acc: {real_acc:.2f}%, Fake Acc: {fake_acc:.2f}%")
    print(f"  Learning Rate: {current_lr:.6f}")
    print(f"  Epoch Time: {epoch_time:.1f}s")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        patience_counter = 0
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'real_acc': real_acc,
            'fake_acc': fake_acc,
            'architecture': ARCHITECTURE,
            'training_history': training_history
        }
        
        torch.save(checkpoint, os.path.join(OUTPUT_DIR, "best_model.pth"))
        print(f"  ✓ Saved best model! Val Acc: {val_acc:.2f}%")
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{early_stop_patience}")
    
    # Early stopping
    if patience_counter >= early_stop_patience:
        print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
        break
    
    print("=" * 70)

# ---------- Training Complete ----------
total_time = time.time() - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)

print("\n" + "="*70)
print("Training Complete!")
print("="*70)
print(f"Total training time: {hours}h {minutes}m")
print(f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.6f}")

# Save final model
final_checkpoint = {
    'model_state_dict': model.state_dict(),
    'architecture': ARCHITECTURE,
    'val_acc': val_acc,
    'best_val_acc': best_val_acc,
    'training_history': training_history
}
torch.save(final_checkpoint, os.path.join(OUTPUT_DIR, "final_model.pth"))

print(f"\n✓ Models saved:")
print(f"  - best_model.pth (Val Acc: {best_val_acc:.2f}%)")
print(f"  - final_model.pth (Last epoch)")

# Save training history
import json
with open(os.path.join(OUTPUT_DIR, "training_history.json"), 'w') as f:
    json.dump(training_history, f, indent=2)
print(f"  - training_history.json")

# Display download links
try:
    from IPython.display import FileLink, display
    print("\n📥 Download trained models:")
    display(FileLink(os.path.join(OUTPUT_DIR, 'best_model.pth')))
    display(FileLink(os.path.join(OUTPUT_DIR, 'final_model.pth')))
    display(FileLink(os.path.join(OUTPUT_DIR, 'training_history.json')))
except:
    print("\nModels saved successfully in /kaggle/working/")

print("\n" + "="*70)
print("🎉 Training finished successfully!")
print("="*70)
