import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from load_dataset import DeepfakeDataset
from model import get_model

BATCH_SIZE = 2               # EfficientNet-B4 is heavy
EPOCHS = 10
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # auto GPU for Kaggle

# Datasets
train_dataset = DeepfakeDataset("/kaggle/input/1000-videos-split/1000_videos/train")
val_dataset = DeepfakeDataset("/kaggle/input/1000-videos-split/1000_videos/validation")   # FIXED: correct folder name

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = get_model(num_classes=2)
model.to(DEVICE)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------- Training Loop ----------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_loss:.4f}")

print("Training finished")

# Save model
torch.save(model.state_dict(), "output/deepfake_model.pth")
print("Model saved to output/deepfake_model.pth")
