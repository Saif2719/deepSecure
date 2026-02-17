import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

        for label_name, label_value in [("real", 0), ("fake", 1)]:
            class_dir = os.path.join(root_dir, label_name)

            if not os.path.exists(class_dir):
                continue

            for img in os.listdir(class_dir):
                if img.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.image_paths.append(os.path.join(class_dir, img))
                    self.labels.append(label_value)

        # EfficientNet-B4 compatible transforms
        self.transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


# 🔒 IMPORTANT: test code MUST be inside this block
if __name__ == "__main__":
    dataset = DeepfakeDataset("/kaggle/input/1000-videos-split/1000_videos/train")  # Kaggle or local path

    real_count = dataset.labels.count(0)
    fake_count = dataset.labels.count(1)

    print("Real images:", real_count)
    print("Fake images:", fake_count)

    img, label = dataset[0]
    print("Sample image shape:", img.shape)  # [3, 380, 380]
    print("Sample label:", label)
