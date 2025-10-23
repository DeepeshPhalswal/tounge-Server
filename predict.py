import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import numpy as np
import io
import os

# ======================
# 1. Configuration
# ======================
model_path = "tounge_analysis_ai.pth"   # Path to your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# 2. Fixed Class Names (from your dataset)
# ======================
class_names = [
    "Anemia",
    "Bleeding disorder",
    "Healthy",
    "Hiv",
    "Hyperacidity tounge",
    "Hyperthyroidism",
    "Jaundice",
    "Leprosy",
    "Tuberculosis",
    "Ulcer",
    "Vitamin B12 deficiency"
]
num_classes = len(class_names)

print("📂 Loaded Class Names:")
for i, cls in enumerate(class_names):
    print(f"{i}: {cls}")

# ======================
# 3. Define Transform
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ======================
# 4. Load Model
# ======================
def load_model(model_path, num_classes):
    """Load trained ResNet18 model."""
    model = models.resnet18(weights=None)  # No pretrained weights (fixes warning)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    return model


# Load the model once globally for performance
model = load_model(model_path, num_classes)


# ======================
# 5. Predict Function
# ======================
def predict_image(image_input):
    """
    Predict disease class from an image.
    Accepts either a file path or an OpenCV image (numpy array).
    """
    # Convert image if needed
    if isinstance(image_input, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    elif isinstance(image_input, str) and os.path.exists(image_input):
        image = Image.open(image_input).convert("RGB")
    else:
        raise ValueError("Invalid image input. Must be path or numpy array.")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        prediction = class_names[predicted.item()]

    return prediction


# ======================
# 6. CLI Testing
# ======================
if __name__ == "__main__":
    image_path = input("Enter image path: ").strip()
    if os.path.exists(image_path):
        result = predict_image(image_path)
        print(f"\n✅ Predicted Class: {result}")
    else:
        print("❌ Image not found! Please check the path.")
