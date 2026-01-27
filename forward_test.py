import torch
from torch.utils.data import DataLoader
from load_dataset import DeepfakeDataset
from model import get_model

BATCH_SIZE = 4

dataset = DeepfakeDataset("dataset/processed/train")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = get_model()
model.eval()  # important

images, labels = next(iter(loader))

with torch.no_grad():
    outputs = model(images)

print("Input batch shape:", images.shape)
print("Output shape:", outputs.shape)
print("Labels:", labels)
