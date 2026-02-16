import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        """
        Dataset for deepfake detection.
        
        Args:
            root_dir: Root directory containing 'real' and 'fake' subdirectories
            augment: Whether to apply data augmentation (use True for training)
        """
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.augment = augment

        # Load images from real and fake directories
        for label_name, label_value in [("real", 0), ("fake", 1)]:
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue
                
            for img in os.listdir(class_dir):
                if img.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.image_paths.append(os.path.join(class_dir, img))
                    self.labels.append(label_value)

        # Enhanced data augmentation for training (improves model generalization)
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                # Random erasing to simulate occlusions
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225]),  # ImageNet normalization
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3))
            ])
        else:
            # Simple transform for validation/testing with center crop
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
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
    dataset = DeepfakeDataset("dataset/processed/train")

    real_count = dataset.labels.count(0)
    fake_count = dataset.labels.count(1)

    print("Real images:", real_count)
    print("Fake images:", fake_count)

    img, label = dataset[0]
    print("Sample image shape:", img.shape)
    print("Sample label:", label)
