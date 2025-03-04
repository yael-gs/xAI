import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights  # <--- Import pour VGG16
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

class MyMultiLabelDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        csv_file : Chemin vers un fichier CSV contenant :
                   - une colonne 'filepath' pour l'image
                   - des colonnes pour chaque label (0/1)
        img_dir  : dossier contenant les images
        transform: transformations (Resize, ToTensor, etc.)
        """
        
        # on clippe les valeurs 0/1 pour en faire un problème de classification
        self.label_cols = ["retinopathy_grade","risk_of_macular_edema"]
        self.df = pd.read_csv(csv_file)
        self.df["retinopathy_grade"] = self.df["retinopathy_grade"].clip(0,1).astype("int8")
        self.df["risk_of_macular_edema"] = self.df["risk_of_macular_edema"].clip(0,1).astype("int8")
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, (row['id'] + ".jpg"))
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # Créer le vecteur de labels (0/1)
        labels = torch.from_numpy(row[self.label_cols].values.astype('float32'))
        return image, labels


# =====================
# Hypothèses
# =====================
num_labels = 2
batch_size = 16
num_epochs = 20
lr = 1e-4

# Transforms
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    train_dataset = MyMultiLabelDataset("dataset/train/labels.csv", "dataset/train/images", transform=transform)
    test_dataset  = MyMultiLabelDataset("dataset/test/labels.csv",  "dataset/test/images", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ====== Modification : Utilisation de VGG16 =======
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    
    # Adaptation de la dernière couche
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_labels)
    model = model.to(device)

    # Utilisation de BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    from tqdm import tqdm
    loss_list = []
    for epoch in tqdm(range(num_epochs), total=num_epochs):
        # Entraînement
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)           # shape: (batch_size, num_labels)
            loss = criterion(outputs, labels) # BCEWithLogitsLoss
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Training loss: {loss.item():.4f}")
        loss_list.append(loss.item())
        
        fig = plt.figure()
        ax = fig.subplots()
        ax.plot(loss_list)
        ax.set_yscale("log")
        fig.savefig("training_loss.png")
        plt.close()
        
        # Validation rapide
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(test_dataset)
        print(f"Validation Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "multi_label_vgg16.pth")
    print("Fin de l'entraînement (multi-label) avec VGG16.")
