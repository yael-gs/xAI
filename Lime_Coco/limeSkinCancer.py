import torch
import torch.nn.functional as F
from torchvision import models, transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from skimage.transform import resize

# Charger le modèle et le dictionnaire des classes
checkpoint = torch.load("skin_cancer_model.pth")
label_mapping = checkpoint["label_mapping"]
idx_to_label = {v: k for k, v in label_mapping.items()}  # Dictionnaire inversé

model = models.resnet50(pretrained=False)
num_classes = len(label_mapping)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        image = data['image']
        label = data['dx']
        if self.transform:
            image = self.transform(image)
        return image, label

# Charger le dataset de test
test_dataset = load_dataset("marmal88/skin_cancer", split="test")
test_data = SkinCancerDataset(test_dataset, transform=transform)

# Fonction pour prédire en batch
def batch_predict(images):
    model.eval()
    tensor_images = [
        torch.from_numpy(image).permute(2, 0, 1) if isinstance(image, np.ndarray) else image
        for image in images
    ]
    batch = torch.stack(tensor_images, dim=0).to(device)
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

def explain_image(img, original_img, label_idx, label_mapping, idx_to_label, show_image=True, save_image=False):
    explainer = lime_image.LimeImageExplainer()

    # Convertir Tensor en NumPy pour LIME
    img_np = np.array(img.permute(1, 2, 0).detach().cpu())

    # Convertir l'image originale en NumPy
    if isinstance(original_img, torch.Tensor):
        original_img = original_img.permute(1, 2, 0).detach().cpu().numpy()
    elif isinstance(original_img, np.ndarray):
        pass  # Rien à faire, déjà un tableau NumPy
    else:
        original_img = np.array(original_img)  # PIL -> NumPy

    original_img = original_img / 255.0 if original_img.max() > 1 else original_img

    # Si label_idx est une chaîne, convertir en entier à l'aide de label_mapping
    if isinstance(label_idx, str):
        label_idx = label_mapping[label_idx]

    # Classe réelle et prédiction
    real_label = idx_to_label[label_idx]
    probs = batch_predict([img])
    predicted_label_idx = np.argmax(probs[0])
    predicted_label = idx_to_label[predicted_label_idx]

    # Afficher les résultats
    print(f"Classe réelle : {real_label}")
    print(f"Classe prédite : {predicted_label}")
    print("Probabilités des classes :")
    for idx, prob in enumerate(probs[0]):
        class_name = idx_to_label[idx]
        print(f"  {class_name}: {prob:.5f}")

    # Obtenir une explication LIME
    explanation = explainer.explain_instance(
        img_np,
        batch_predict,
        top_labels=len(idx_to_label),
        hide_color=0,
        num_samples=1000
    )

    if show_image:
        for idx in explanation.top_labels:
            temp, mask = explanation.get_image_and_mask(
                idx, positive_only=False, num_features=10, hide_rest=False
            )

            resized_original_img = resize(original_img, temp.shape[:2], preserve_range=True)

            img_boundary = mark_boundaries(resized_original_img, mask)
            plt.figure()
            plt.title(f"LIME Explanation for Class: {idx_to_label[idx]}")
            plt.imshow(img_boundary)
            plt.axis("off")
        plt.show()

    if save_image:
        for idx in explanation.top_labels:
            temp, mask = explanation.get_image_and_mask(
                idx, positive_only=False, num_features=10, hide_rest=False
            )

            resized_original_img = resize(original_img, temp.shape[:2], preserve_range=True)

            img_boundary = mark_boundaries(resized_original_img, mask)
            plt.imsave(f"LIME_explanation_class_{idx}.png", img_boundary)
        #on sauvegarde l'image de base
        plt.imsave("original_image.png", original_img)
if __name__ == "__main__":
    # Charger une image de test
    image, label = test_data[24]
    original_image = test_dataset[24]['image']

    # Expliquer l'image
    explain_image(
        image,
        original_image,
        label_idx=label,
        label_mapping=label_mapping,
        idx_to_label=idx_to_label,
        show_image=True,
        save_image=True
    )
