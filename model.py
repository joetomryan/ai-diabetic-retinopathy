import torch
import torch.nn as nn
from torchvision import models

class EfficientNetB3Model(nn.Module):
    def __init__(self, num_classes=5):  # assuming 5 classes for diabetic retinopathy
        super(EfficientNetB3Model, self).__init__()
        
        # Load pre-trained EfficientNet-B3 model
        self.model = models.efficientnet_b3(pretrained=True)
        
        # Modify the classifier to match the number of classes
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# To test if the model works
if __name__ == "__main__":
    model = EfficientNetB3Model()
    # Try to pass a random tensor to check the model's forward pass
    x = torch.randn(4, 3, 300, 300)  # Batch of 4 images, 300x300, 3 channels
    output = model(x)
    print("Output shape:", output.shape)
