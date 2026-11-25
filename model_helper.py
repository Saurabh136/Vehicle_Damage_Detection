
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "saved_model.pth"

DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "Front Breakage",
    "Front Crushed",
    "Front Normal",
    "Rear Breakage",
    "Rear Crushed",
    "Rear Normal"
]


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# ResNet50 Model Architecture
# --------------------------------------------------

class CarClassifierResNet(nn.Module):

    def __init__(self, num_classes=6):

        super().__init__()

        # Reconstruct ResNet50 without downloading ImageNet weights
        self.model = models.resnet50(weights=None)

        # Freeze all backbone parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace original classification head
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

def load_model():

    model = CarClassifierResNet(
        num_classes=len(CLASS_NAMES)
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# Prediction
# --------------------------------------------------


def predict(image: Image.Image, model, confidence_threshold=0.70):

    image = image.convert("RGB")

    image_tensor = IMAGE_TRANSFORM(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        outputs = model(image_tensor)

        # Convert logits into probabilities
        probabilities = torch.softmax(outputs, dim=1)

        # Get the highest probability and predicted class
        confidence, predicted_class = probabilities.max(dim=1)

        confidence = confidence.item()
        predicted_class = predicted_class.item()

    # Reject low-confidence predictions
    if confidence < confidence_threshold:
        return "Unable to classify this image. Please upload a clear vehicle image."

    return CLASS_NAMES[predicted_class]

