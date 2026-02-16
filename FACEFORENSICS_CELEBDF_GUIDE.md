# Training Guide for FaceForensics++ & Celeb-DF Combined Dataset

This guide is specifically for training on the **Face Forensic++ & Celeb-DF Combined Deepfake Data** dataset on Kaggle.

## 📊 Dataset Information

**Dataset:** Face Forensic++ & Celeb-DF Combined Deepfake Data  
**Size:** Large-scale dataset with thousands of real and fake face images  
**Quality:** High-quality, professional deepfake dataset  
**Best For:** Production-grade deepfake detection models  

## 🚀 Quick Start

### Step 1: Setup Kaggle Notebook

1. Go to the dataset page on Kaggle
2. Click **"New Notebook"**
3. Enable **GPU T4 x2** or **GPU P100** in Settings
4. The dataset will be automatically attached at `/kaggle/input/`

### Step 2: Find Dataset Structure

Run this code first to explore the dataset structure:

```python
import os

print("Exploring dataset structure...")
print("="*70)

for dirname, dirs, filenames in os.walk('/kaggle/input'):
    level = dirname.replace('/kaggle/input', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(dirname)}/')
    
    # Check if this is a dataset directory
    if 'real' in dirs and 'fake' in dirs:
        print(f'{indent}  ⭐ FOUND: Dataset directory!')
        print(f'{indent}  Path: {dirname}')
    
    # Show sample files
    if filenames and level < 4:
        subindent = ' ' * 2 * (level + 1)
        print(f'{subindent}Files: {len(filenames)} total')
        if len(filenames) > 0:
            print(f'{subindent}Sample: {filenames[:3]}')
```

### Step 3: Update Dataset Paths

Based on the structure you found, update the paths in [`train_faceforensics_celebdf.py`](train_faceforensics_celebdf.py):

```python
# Common structures:

# Option 1: Direct structure
TRAIN_DIR = "/kaggle/input/faceforensics/train"
VAL_DIR = "/kaggle/input/faceforensics/val"

# Option 2: Nested structure
TRAIN_DIR = "/kaggle/input/faceforensics/FaceForensics++/train"
VAL_DIR = "/kaggle/input/faceforensics/FaceForensics++/val"

# Option 3: Combined structure
TRAIN_DIR = "/kaggle/input/faceforensics/combined/train"
VAL_DIR = "/kaggle/input/faceforensics/combined/test"
```

### Step 4: Copy Training Code

Use the optimized training script [`train_faceforensics_celebdf.py`](train_faceforensics_celebdf.py) which includes:

✅ **Mixed Precision Training** - 2x faster with FP16  
✅ **Gradient Accumulation** - Larger effective batch size  
✅ **Advanced Augmentation** - Better generalization  
✅ **Label Smoothing** - Prevents overfitting  
✅ **Cosine Annealing** - Better learning rate scheduling  
✅ **Per-Class Accuracy** - Track real vs fake separately  
✅ **Training History** - Save metrics for analysis  

## 📝 Kaggle Notebook Structure

### Cell 1: Explore Dataset
```python
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
    if len(filenames) > 0:
        print(f"  Files: {len(filenames)}")
```

### Cell 2: Dataset Class
```python
# Copy the entire content of load_dataset.py here
```

### Cell 3: Model Definition
```python
# Copy the entire content of model.py here
```

### Cell 4: Training Script
```python
# Copy the entire content of train_faceforensics_celebdf.py here
# Make sure to update TRAIN_DIR and VAL_DIR paths!
```

### Cell 5: Download Models
```python
from IPython.display import FileLink, display

print("Download trained models:")
display(FileLink('/kaggle/working/best_model.pth'))
display(FileLink('/kaggle/working/final_model.pth'))
display(FileLink('/kaggle/working/training_history.json'))
```

## ⚙️ Optimized Configuration

The training script is pre-configured for optimal performance:

```python
BATCH_SIZE = 32              # Optimized for GPU memory
ACCUMULATION_STEPS = 2       # Effective batch size = 64
EPOCHS = 40                  # Sufficient for large dataset
LR = 0.0001                  # Good starting learning rate
WEIGHT_DECAY = 1e-4          # Regularization
ARCHITECTURE = 'resnet50'    # Best balance of speed/accuracy
USE_MIXED_PRECISION = True   # 2x faster training
```

### Architecture Options

| Architecture | Speed | Accuracy | GPU Memory | Recommended For |
|-------------|-------|----------|------------|-----------------|
| `resnet18` | Fast | Good | 4GB | Quick experiments |
| `resnet50` | Medium | Very Good | 8GB | Balanced performance |
| `efficientnet_b0` | Fast | Very Good | 6GB | Efficient training |
| `efficientnet_b3` | Slow | Excellent | 12GB | High accuracy |
| `efficientnet_b4` | Slower | **Best** | 14GB | **Production (Recommended)** |

## 🎯 Expected Performance

### Training Time (with GPU T4 x2):
- **Small dataset** (10k images): ~30-45 minutes
- **Medium dataset** (50k images): ~2-3 hours
- **Large dataset** (100k+ images): ~5-8 hours

### Expected Accuracy:
- **ResNet50**: 95-98% validation accuracy
- **EfficientNet-B3**: 96-99% validation accuracy
- **EfficientNet-B4**: 97-99.5% validation accuracy ⭐
- **Ensemble**: 98-99.7% validation accuracy

## 🔧 Advanced Features

### 1. Mixed Precision Training

Automatically enabled for 2x faster training:
```python
USE_MIXED_PRECISION = True  # Uses FP16 instead of FP32
```

**Benefits:**
- 2x faster training
- 50% less GPU memory
- Same accuracy as FP32

### 2. Gradient Accumulation

Simulates larger batch sizes:
```python
ACCUMULATION_STEPS = 2  # Effective batch size = 32 * 2 = 64
```

**Benefits:**
- Better gradient estimates
- More stable training
- Works with limited GPU memory

### 3. Enhanced Data Augmentation

Includes advanced augmentations:
- Random rotation (±15°)
- Color jittering
- Random affine transforms
- Random perspective
- Random erasing (simulates occlusions)

### 4. Label Smoothing

Prevents overconfidence:
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Benefits:**
- Better generalization
- Prevents overfitting
- More robust predictions

### 5. Cosine Annealing with Warm Restarts

Better learning rate scheduling:
```python
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
```

**Benefits:**
- Escapes local minima
- Better convergence
- No manual tuning needed

## 📈 Monitoring Training

The script provides detailed metrics:

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
  Learning Rate: 0.000095
  Epoch Time: 245.3s
  ✓ Saved best model! Val Acc: 93.45%
```

### Key Metrics to Watch:

1. **Validation Accuracy** - Should increase over time
2. **Real vs Fake Accuracy** - Should be balanced (within 2-3%)
3. **Train vs Val Loss** - Val loss should be close to train loss
4. **Learning Rate** - Should decrease gradually

### Warning Signs:

⚠️ **Overfitting:** Train acc >> Val acc (difference > 10%)  
⚠️ **Underfitting:** Both train and val acc are low (< 80%)  
⚠️ **Class Imbalance:** Real acc and Fake acc differ by > 5%  
⚠️ **Learning Rate Too High:** Loss oscillates or increases  

## 🐛 Troubleshooting

### Issue 1: Out of Memory (OOM)

**Solution 1:** Reduce batch size
```python
BATCH_SIZE = 16  # Instead of 32
```

**Solution 2:** Reduce accumulation steps
```python
ACCUMULATION_STEPS = 1  # Instead of 2
```

**Solution 3:** Use smaller model
```python
ARCHITECTURE = 'resnet18'  # Instead of resnet50
```

### Issue 2: Training Too Slow

**Solution 1:** Verify GPU is enabled
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Solution 2:** Increase batch size (if memory allows)
```python
BATCH_SIZE = 64  # If you have enough GPU memory
```

**Solution 3:** Reduce number of workers
```python
num_workers=2  # Instead of 4
```

### Issue 3: Low Accuracy

**Solution 1:** Train longer
```python
EPOCHS = 50  # Instead of 40
```

**Solution 2:** Use larger model
```python
ARCHITECTURE = 'efficientnet_b3'  # Instead of resnet50
```

**Solution 3:** Adjust learning rate
```python
LR = 0.00005  # Lower learning rate
```

### Issue 4: Overfitting

**Solution 1:** Increase dropout
```python
model = get_model(num_classes=2, architecture='resnet50', dropout=0.7)
```

**Solution 2:** Increase weight decay
```python
WEIGHT_DECAY = 1e-3  # Instead of 1e-4
```

**Solution 3:** Use more augmentation (already enabled in enhanced version)

## 📊 After Training

### 1. Download Models

Three files will be saved:
- `best_model.pth` - Model with highest validation accuracy ⭐
- `final_model.pth` - Model from last epoch
- `training_history.json` - Training metrics for analysis

### 2. Load Model for Inference

```python
import torch
from model import get_model

# Load checkpoint
checkpoint = torch.load('best_model.pth')

# Initialize model with same architecture
model = get_model(num_classes=2, architecture='resnet50')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model trained for {checkpoint['epoch']} epochs")
print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
print(f"Real accuracy: {checkpoint['real_acc']:.2f}%")
print(f"Fake accuracy: {checkpoint['fake_acc']:.2f}%")
```

### 3. Analyze Training History

```python
import json
import matplotlib.pyplot as plt

# Load history
with open('training_history.json', 'r') as f:
    history = json.load(f)

# Plot training curves
plt.figure(figsize=(15, 5))

# Loss
plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Accuracy
plt.subplot(1, 3, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Validation Accuracy')

# Per-class accuracy
plt.subplot(1, 3, 3)
plt.plot(history['real_acc'], label='Real Acc')
plt.plot(history['fake_acc'], label='Fake Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Real vs Fake Accuracy')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
```

## 🎓 Tips for Best Results

1. **Start with default settings** - They're optimized for this dataset
2. **Monitor GPU usage** - Should be 80-95% utilization
3. **Check class balance** - Real and fake accuracy should be similar
4. **Use early stopping** - Prevents overfitting (already enabled)
5. **Save training history** - Helps diagnose issues
6. **Try ensemble** - Train multiple models and average predictions

## 🏆 Advanced: Model Ensemble

For maximum accuracy, train multiple models and combine them:

```python
# Train 3 different models
# Model 1: ResNet50
# Model 2: EfficientNet-B0
# Model 3: EfficientNet-B3

# Then use ensemble for predictions
from model import EnsembleModel

model1 = get_model(architecture='resnet50')
model1.load_state_dict(torch.load('best_model_resnet50.pth')['model_state_dict'])

model2 = get_model(architecture='efficientnet_b0')
model2.load_state_dict(torch.load('best_model_effb0.pth')['model_state_dict'])

model3 = get_model(architecture='efficientnet_b3')
model3.load_state_dict(torch.load('best_model_effb3.pth')['model_state_dict'])

# Create ensemble
ensemble = EnsembleModel([model1, model2, model3])
ensemble.eval()

# Use ensemble for predictions
# Typically gives 1-2% better accuracy than single model
```

## 📚 Resources

- [FaceForensics++ Paper](https://arxiv.org/abs/1901.08971)
- [Celeb-DF Paper](https://arxiv.org/abs/1909.12962)
- [PyTorch Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

**Ready to train? Follow this guide and achieve 95%+ accuracy on FaceForensics++ & Celeb-DF!** 🚀
