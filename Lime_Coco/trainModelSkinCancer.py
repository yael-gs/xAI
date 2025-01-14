import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datasets import load_dataset

# Charger le dataset
ds = load_dataset("marmal88/skin_cancer")
train_dataset = load_dataset("marmal88/skin_cancer", split="train")
valid_dataset = load_dataset("marmal88/skin_cancer", split="validation")
test_dataset = load_dataset("marmal88/skin_cancer", split="test")

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset personnalisé
class SkinCancerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

        self.label_mapping = {label: idx for idx, label in enumerate(set(data['dx'] for data in dataset))}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        image = data['image']
        label = data['dx']
        if self.transform:
            image = self.transform(image)
        label = self.label_mapping[label]
        return image, label

# Charger les datasets
train_data = SkinCancerDataset(train_dataset, transform=transform)
valid_data = SkinCancerDataset(valid_dataset, transform=transform)
test_data = SkinCancerDataset(test_dataset, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Définir le modèle
num_classes = len(set(data['dx'] for data in train_dataset))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Fonction d'entraînement
def train(model, train_loader, valid_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
        train_acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
        model.eval()
        valid_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        valid_acc = correct / len(valid_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {valid_loss:.4f}, Val Acc: {valid_acc:.4f}")

train(model, train_loader, valid_loader, criterion, optimizer, epochs=5)

model_save_path = "skin_cancer_model.pth"
torch.save({
    "model_state_dict": model.state_dict(),
    "label_mapping": train_data.label_mapping
}, model_save_path)

def test(model, test_loader, label_mapping_path):
    checkpoint = torch.load(label_mapping_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    label_mapping = checkpoint["label_mapping"]

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {correct / len(test_loader.dataset):.4f}")
    print("Classes utilisées :")
    print(label_mapping)

test(model, test_loader, model_save_path)
