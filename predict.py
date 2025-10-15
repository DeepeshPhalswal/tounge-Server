import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import requests

MODEL_PATH = "tounge_analysis_ai.pth"
MODEL_URL = "https://drive.google.com/uc?id=1AbCdEfGhIjKlMnOpQr"  # Replace with your link

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("🔽 Model file not found — downloading...")
        r = requests.get(MODEL_URL, allow_redirects=True)
        open(MODEL_PATH, "wb").write(r.content)
        print("✅ Model downloaded successfully.")
    else:
        print("✅ Model already exists.")

# ======================
# 1. Configuration
# ======================
model_path = "tounge_analysis_ai.pth"   # Path to trained model
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
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    return model

# ======================
# 5. Predict Function
# ======================
def predict_image(image_path):
    model = load_model(model_path, num_classes)
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        prediction = class_names[predicted.item()]
    return prediction

# ======================
# 6. Run Prediction
# ======================
if __name__ == "__main__":
    image_path = input("Enter image path: ").strip()

    if os.path.exists(image_path):
        result = predict_image(image_path)
        print(f"\n✅ Predicted Class: {result}")
    else:
        print("❌ Image not found! Please check the path.")

