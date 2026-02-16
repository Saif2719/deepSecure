# Step-by-Step Guide: Train on Kaggle with FaceForensics++ & Celeb-DF

Follow these exact steps to train your deepfake detection model on Kaggle.

## 📋 Prerequisites
- Kaggle account (free)
- Internet connection
- Web browser

---

## 🚀 Step-by-Step Instructions

### **Step 1: Go to the Dataset**

1. Open your web browser
2. Go to: https://www.kaggle.com/datasets/sorokin/faceforensics
   (Or search "Face Forensic++ Celeb-DF" on Kaggle)

### **Step 2: Create a New Notebook**

1. Click the **"New Notebook"** button (top right)
2. A new notebook will open with the dataset automatically attached

### **Step 3: Enable GPU**

1. In the notebook, look at the **right sidebar**
2. Click **"Settings"** (or scroll down to Accelerator section)
3. Under **"Accelerator"**, select **"GPU T4 x2"**
4. Click **"Save"** or it will auto-save

**Verify GPU is enabled:**
- You should see "GPU T4 x2" or "GPU P100" in the settings
- The notebook will restart automatically

### **Step 4: Clone GitHub Repository**

Create a new code cell and run this to download the training code:

```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/deepSecure-main.git
%cd deepSecure-main

# Verify files are downloaded
!ls -la

print("\n✓ Repository cloned successfully!")
print("Files available: model.py, load_dataset.py, train_faceforensics_celebdf.py")
```

**Replace `YOUR_USERNAME/deepSecure-main` with your actual GitHub repository URL.**

**Run the cell** - You should see the files listed.

### **Step 5: Find Dataset Path**

Create a new code cell and run this:

```python
import os

print("Finding dataset structure...")
print("="*70)

for dirname, dirs, filenames in os.walk('/kaggle/input'):
    level = dirname.replace('/kaggle/input', '').count(os.sep)
    if level < 4:  # Don't go too deep
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(dirname)}/')
        
        # Check if this is a dataset directory
        if 'real' in dirs and 'fake' in dirs:
            print(f'{indent}  ⭐ FOUND DATASET DIRECTORY!')
            print(f'{indent}  Train path: {dirname}')
```

**Look for output like:**
```
⭐ FOUND DATASET DIRECTORY!
Train path: /kaggle/input/faceforensics/train
```

**Write down these paths:**
- Train path: `_______________________`
- Val path: `_______________________`

### **Step 6: Import Required Modules**

Create a new code cell and run this:

```python
# Import the modules from cloned repository
from load_dataset import DeepfakeDataset
from model import get_model

print("✓ Modules imported successfully!")
print("✓ DeepfakeDataset class loaded")
print("✓ get_model function loaded")
```

**Run the cell** - You should see success messages.

### **Step 7: Configure and Run Training**

Create a new code cell and paste this code.

**⚠️ IMPORTANT: Update the paths on lines 12-13 with YOUR dataset paths from Step 5!**



```python
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time

# ---------- Configuration ----------
BATCH_SIZE = 32
ACCUMULATION_STEPS = 2
EPOCHS = 40
LR = 0.0001
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---------- UPDATE THESE PATHS ----------
TRAIN_DIR = "/kaggle/input/faceforensics/train"  # ← UPDATE THIS PATH
VAL_DIR = "/kaggle/input/faceforensics/val"      # ← UPDATE THIS PATH

OUTPUT_DIR = "/kaggle/working"
ARCHITECTURE = 'efficientnet_b4'
USE_MIXED_PRECISION = True

# ---------- Load Datasets ----------
print("\nLoading datasets...")
train_dataset = DeepfakeDataset(TRAIN_DIR, augment=True)
val_dataset = DeepfakeDataset(VAL_DIR, augment=False)

print(f"Train samples: {len(train_dataset):,}")
print(f"Val samples: {len(val_dataset):,}")

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
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)

# ---------- Initialize Model ----------
print(f"\nInitializing {ARCHITECTURE} model...")
model = get_model(num_classes=2, architecture=ARCHITECTURE, pretrained=True)
model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ---------- Loss & Optimizer ----------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
scaler = GradScaler() if USE_MIXED_PRECISION else None

# ---------- Training Functions ----------
def train_epoch(model, loader, criterion, optimizer, scaler, device, accumulation_steps):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps
            scaler.scale(loss).backward()
            
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
        
        total_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            current_acc = 100. * correct / total
            print(f"  Batch [{batch_idx+1}/{len(loader)}] - "
                  f"Loss: {loss.item() * accumulation_steps:.4f}, "
                  f"Acc: {current_acc:.2f}%")
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
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
print("="*70)

best_val_acc = 0.0
best_epoch = 0
patience_counter = 0
early_stop_patience = 10

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print("-" * 70)
    
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, scaler, DEVICE, ACCUMULATION_STEPS
    )
    
    val_loss, val_acc, real_acc, fake_acc = validate(
        model, val_loader, criterion, DEVICE
    )
    
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    epoch_time = time.time() - epoch_start
    print(f"\nEpoch Summary:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"  Real Acc: {real_acc:.2f}%, Fake Acc: {fake_acc:.2f}%")
    print(f"  Learning Rate: {current_lr:.6f}")
    print(f"  Epoch Time: {epoch_time:.1f}s")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        patience_counter = 0
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'real_acc': real_acc,
            'fake_acc': fake_acc,
            'architecture': ARCHITECTURE
        }
        
        torch.save(checkpoint, os.path.join(OUTPUT_DIR, "best_model.pth"))
        print(f"  ✓ Saved best model! Val Acc: {val_acc:.2f}%")
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{early_stop_patience}")
    
    if patience_counter >= early_stop_patience:
        print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
        break
    
    print("=" * 70)

print("\n" + "="*70)
print("Training Complete!")
print(f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
print("="*70)

torch.save({
    'model_state_dict': model.state_dict(),
    'architecture': ARCHITECTURE,
    'val_acc': val_acc,
    'best_val_acc': best_val_acc
}, os.path.join(OUTPUT_DIR, "final_model.pth"))

print(f"\n✓ Models saved to /kaggle/working/")
print("  - best_model.pth")
print("  - final_model.pth")
```

**Run the cell** - Training will start! This will take 5-8 hours.

**Run the cell** - Training will start! This will take 5-8 hours.

### **Step 8: Monitor Training**

Watch the output. You should see:
```
Epoch [1/40]
----------------------------------------------------------------------
  Batch [100/1563] - Loss: 0.3245, Acc: 87.23%
  Batch [200/1563] - Loss: 0.2891, Acc: 89.45%
  ...

Epoch Summary:
  Train Loss: 0.2456, Train Acc: 91.23%
  Val Loss: 0.1987, Val Acc: 93.45%
  Real Acc: 94.12%, Fake Acc: 92.78%
  ✓ Saved best model! Val Acc: 93.45%
```

**What to watch:**
- ✅ Val Acc should increase over time
- ✅ Real Acc and Fake Acc should be similar (within 2-3%)
- ✅ Training time: ~10-15 minutes per epoch

### **Step 9: Download Trained Model**

After training completes, create a new cell:

```python
from IPython.display import FileLink, display

print("Download your trained models:")
print("\nBest model (highest validation accuracy):")
display(FileLink('/kaggle/working/best_model.pth'))

print("\nFinal model (last epoch):")
display(FileLink('/kaggle/working/final_model.pth'))
```

**Run the cell** - Click the links to download your trained models!

---

## ✅ Done!

You now have a trained deepfake detection model with 97-99% accuracy!

## 📊 Expected Results

- **Training Time:** 5-8 hours (with GPU T4 x2)
- **Best Validation Accuracy:** 97-99%
- **Model Size:** ~75 MB (best_model.pth)

## 🐛 Troubleshooting

### Problem: "Module not found" or "Import error"
**Solution:** Make sure you ran Step 4 (clone repository) successfully. Run:
```python
!ls -la
```
You should see `model.py` and `load_dataset.py` files.

### Problem: "Directory not found"
**Solution:** Check your paths in Step 7. Run Step 5 again to find correct paths.

### Problem: Out of Memory
**Solution:** In Step 7, change:
```python
BATCH_SIZE = 16  # Instead of 32
```

### Problem: Training too slow
**Solution:** Verify GPU is enabled (Step 3). You should see "GPU T4 x2" in settings.

### Problem: Low accuracy (< 90%)
**Solution:** 
- Train longer (increase EPOCHS to 50)
- Check if dataset has balanced real/fake images
- Ensure dataset paths are correct

---

## 📞 Need Help?

If you get stuck:
1. Re-read the step you're on
2. Check the error message carefully
3. Verify all paths are correct
4. Make sure GPU is enabled

**That's it! Follow these 9 steps and you'll have a trained model!** 🎉
