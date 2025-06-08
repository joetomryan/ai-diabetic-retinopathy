from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import torch
from torchvision import transforms

class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1])
        image_path = os.path.join(self.image_dir, image_id + ".png")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Cannot load image: {image_path} - {e}")
            # Return a dummy tensor and label to keep DataLoader from crashing
            dummy = torch.zeros(3, 300, 300)  # Matching EfficientNet-B3 input
            return dummy, -1  # Use -1 to indicate invalid label

        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloader(csv_file, image_dir, batch_size=16, shuffle=True, transform=None):
    dataset = RetinopathyDataset(csv_file, image_dir, transform)
    print(f"[INFO] Total entries in dataset: {len(dataset)}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
