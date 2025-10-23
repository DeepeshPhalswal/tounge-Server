import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import os

# ======================
# 1. Configuration
# ======================
data_dir = "dataset"          # Path to dataset
batch_size = 32
num_epochs = 50
learning_rate = 0.001
model_path = "tounge_analysis_ai.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🧠 Using device: {device}")

# ======================
# 2. Data Preprocessing
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"📂 Detected Classes: {class_names}")

# ======================
# 3. Model Definition
# ======================
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ======================
# 4. Auto Resume Training
# ======================
start_epoch = 0
train_losses, test_accuracies = [], []

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    train_losses = checkpoint.get("train_losses", [])
    test_accuracies = checkpoint.get("test_accuracies", [])
    start_epoch = checkpoint["epoch"] + 1
    print(f"🔄 Resuming training from epoch {start_epoch+1}/{num_epochs}")
else:
    print("🚀 Starting new training session")

# ======================
# 5. Training Loop with tqdm
# ======================
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"✅ Epoch {epoch+1} complete | Loss: {avg_loss:.4f}")

    # ======================
    # 6. Testing after each epoch
    # ======================
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    test_accuracies.append(acc)
    print(f"🎯 Test Accuracy after Epoch {epoch+1}: {acc:.2f}%")

    # ======================
    # 7. Save Checkpoint
    # ======================
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_losses": train_losses,
        "test_accuracies": test_accuracies
    }, model_path)
    print(f"💾 Model checkpoint saved at '{model_path}'")

print("✅ Training fully complete!")

# ======================
# 8. Plot Accuracy & Loss Graphs
# ======================
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Training Loss", marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(test_accuracies, label="Test Accuracy", color='green', marker='o')
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ======================
# 9. Model Loading & Prediction
# ======================
def load_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    return model

def predict_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# ======================
# 10. Example Usage
# ======================
model = load_model(model_path, num_classes)
print("Predicted:", predict_image(r"dataset\test\Leprosy\Leprosy24.jpg", model))
