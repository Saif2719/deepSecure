import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from load_dataset import DeepfakeDataset
from model import get_model

device = torch.device("cpu")

# Load dataset
test_dataset = DeepfakeDataset(
    root_dir="dataset/test"   # <-- CORRECT
)

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Load model
model = get_model(num_classes=2)
model.load_state_dict(torch.load("output/deepfake_model.pth", map_location=device))
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Metrics
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("Confusion Matrix:\n", cm)
