import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from load_dataset import DeepfakeDataset
from model import get_model

# ---------------- CONFIG ----------------
BATCH_SIZE = 2          # EfficientNet-B4 is heavy (T4 safe)
EPOCHS = 10
LR = 0.0001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ---------------- DATASETS ----------------
TRAIN_PATH = "/kaggle/input/1000-videos-split/1000_videos/train"
VAL_PATH   = "/kaggle/input/1000-videos-split/1000_videos/validation"

train_dataset = DeepfakeDataset(TRAIN_PATH)
val_dataset   = DeepfakeDataset(VAL_PATH)

print("Train images:", len(train_dataset))
print("Val images:", len(val_dataset))

# ---------------- DATALOADERS (IMPORTANT FIX) ----------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,          # REQUIRED on Kaggle
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

# ---------------- MODEL ----------------
model = get_model(num_classes=2)
model.to(DEVICE)

# ---------------- LOSS & OPTIMIZER ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAINING LOOP ----------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_loss:.4f}")

print("Training finished")

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "output/deepfake_model.pth")
print("Model saved to output/deepfake_model.pth")
