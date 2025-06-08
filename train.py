import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataloader import get_dataloader  # Changed import to relative
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training

# 1. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# 2. Load EfficientNet-B3 pre-trained model
model = models.efficientnet_b3(pretrained=True)
num_classes = 5  # Number of DR severity classes
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# 3. Transforms
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

# 4. Data Loaders with pin_memory and num_workers
train_loader = get_dataloader(csv_file='data/train_split.csv', image_dir='data/train_images', batch_size=32, shuffle=True, transform=transform)
val_loader = get_dataloader(csv_file='data/val_split.csv', image_dir='data/train_images', batch_size=32, shuffle=False, transform=transform)


# 5. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()  # For AMP

# 6. Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Mixed precision forward-backward
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_preds / total_preds

        # Validation phase
        model.eval()
        val_correct_preds = 0
        val_total_preds = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast():
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct_preds += torch.sum(preds == labels).item()
                val_total_preds += labels.size(0)

        val_accuracy = val_correct_preds / val_total_preds

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Acc: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"[INFO] Saved new best model with accuracy: {best_accuracy:.4f}")

# 7. Run training
train_model(model, train_loader, val_loader, criterion, optimizer)
