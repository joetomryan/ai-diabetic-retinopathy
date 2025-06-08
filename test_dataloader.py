from src.dataloader import get_dataloader
from torchvision import transforms

def test_dataloader():
    transform = transforms.Compose([
        transforms.Resize((300, 300)),  # EfficientNet-B3 default size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    dataloader = get_dataloader(
        csv_file='data/train.csv',
        image_dir='data/train_images',
        batch_size=4,
        transform=transform
    )

    print("[INFO] Trying to fetch one batch...")
    for images, labels in dataloader:
        print("[SUCCESS] Batch of images shape:", images.shape)
        print("[SUCCESS] Batch of labels:", labels)
        break

if __name__ == "__main__":
    test_dataloader()
