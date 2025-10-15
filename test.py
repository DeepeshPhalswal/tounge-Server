import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# ======================
# 1. Configuration
# ======================
model_path = "tounge_analysis_ai.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

# ======================
# 2. Fixed Class Names
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
# 4. Load Datasets
# ======================
data_dir = "dataset"
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ======================
# 5. Load Model
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
# 6. Evaluation Function
# ======================
def evaluate(loader, model, dataset_name="Dataset"):
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Compute accuracy
    correct = sum([p == t for p, t in zip(all_preds, all_labels)])
    accuracy = 100 * correct / len(all_labels)
    print(f"🎯 {dataset_name} Accuracy: {accuracy:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ======================
# 7. Run Evaluation
# ======================
if __name__ == "__main__":
    model = load_model(model_path, num_classes)

    print("\nEvaluating on TRAIN dataset...")
    evaluate(train_loader, model, dataset_name="Train")

    print("\nEvaluating on TEST dataset...")
    evaluate(test_loader, model, dataset_name="Test")
