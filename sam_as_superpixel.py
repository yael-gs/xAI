import torch
from torchvision import models, transforms
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torch import nn
from lime import lime_image
from skimage.segmentation import mark_boundaries
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import pandas as pd
num_labels = 2
batch_size = 8
num_epochs = 3
lr = 3e-5
# -----
# 1. Config
# -----
device = "cuda" if torch.cuda.is_available() else "cpu"

# Images
image_path = "dataset/test/images/IDRiD_012.jpg"
ground_truth_mask_path = "dataset/test/segmentation/IDRiD_020.tif"
ground_truth_label = pd.read_csv("dataset/test/labels.csv")
ground_truth_label = ground_truth_label.loc[ground_truth_label['id'] == "IDRiD_020"]
ground_truth_label = ground_truth_label[["retinopathy_grade", "risk_of_macular_edema"]]

# SAM
# big one
#sam_checkpoint = "sam_vit_h_4b8939.pth"
#model_type = "vit_h"  # SAM model type

#small one
model_type = "vit_b"
sam_checkpoint = "sam_vit_b_01ec64.pth"  # chemin vers le poids vit_b


# Exemple de transform pour votre modèle
# (Assurez-vous que 'small_model.py' ou la définition de `transform` soit cohérente)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----
# 2. Charger l'image
# -----
# On lit en BGR avec OpenCV, puis on convertit en RGB
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
orig_h, orig_w = image.shape[:2]  # dimensions originales

# -----
# 3. Charger le modèle (2 sorties indépendantes)
# -----
print("Loading model (2 outputs) ...")
classification_model = models.vgg16(weights=None)

in_features = classification_model.classifier[6].in_features
classification_model.classifier[6] = nn.Linear(in_features, num_labels)
# Update the load_state_dict call
classification_model.load_state_dict(
    torch.load("multi_label_vgg16.pth", map_location=device, weights_only=True)
) 

classification_model.to(device)
classification_model.eval()

# -----
# 4. Fonction de prédiction (2 sorties -> sigmoïde)
# -----
def get_classification_model_predictions(img: np.ndarray):
    """
    Convertir l'image NumPy en PIL, passer dans le modèle,
    et renvoyer deux probabilités indépendantes.
    Ex : p1 = p(rétinopathie), p2 = p(œdème).
    """
    pil_img = Image.fromarray(img)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = classification_model(input_tensor)     # shape (1,2)
        probs = torch.sigmoid(logits)                   # Sigmoïde pour multi-label
    
    # On repasse en NumPy
    probs_np = probs.cpu().numpy()  # shape (1,2)
    
    # Exemple : on crée un dictionnaire avec 2 probabilités
    #   retinopathy_grade et risk_of_macular_edema
    #   probs_np[0, 0] => première probabilité
    #   probs_np[0, 1] => deuxième probabilité
    pred = {
        "retinopathy_grade"      : float(probs_np[0,0]),
        "risk_of_macular_edema"  : float(probs_np[0,1])
    }
    return pred

# Exemple : obtenir la prédiction pour l'image chargée
print("Running multi-label prediction on the image...")
preds_dict = get_classification_model_predictions(image)
print("Predictions:", preds_dict)
print("Ground truth:", ground_truth_label.to_dict())

# -----
# 5. Charger SAM
# -----
print("\nLoading SAM ...")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_batch=32,
    min_mask_region_area=1000,  # ou plus, en fonction de votre cas
    # d’autres paramètres si besoin…
)


# -----
# 6. Segmentation SAM (pour LIME)
# -----
def custom_sam_segmentation(rgb_image: np.ndarray):
    """
    Utilise SAM pour générer un ensemble de masques,
    puis fusionne en un label_mask (superpixels).
    """
    masks = mask_generator.generate(rgb_image)
    seg_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.int32)
    for i, m in enumerate(masks):
        seg_mask[m['segmentation']] = i + 1
    return seg_mask

# -----
# 7. classifier_fn pour LIME (multi-label)
# -----
def classifier_fn(images: list):
    """
    LIME appelle cette fonction pour un lot d'images (list de np.array).
    On doit renvoyer un np.array de shape (N, 2) contenant
    [p(label_1), p(label_2)] pour chaque image.
    """
    preds = []
    for img in images:
        pil_image = Image.fromarray(img)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = classification_model(input_tensor)  # shape (1,2)
            # On applique la sigmoïde dans le cas multi-label
            probs = torch.sigmoid(logits)                # shape (1,2)
        preds.append(probs.cpu().numpy())                # (1,2)
    
    # Concatène sur la dimension batch -> shape (N,2)
    return np.concatenate(preds, axis=0)

# -----
# 8. LIME Explanation
# -----
explainer = lime_image.LimeImageExplainer()
print("\nGenerating LIME explanation using SAM segmentation ...")

# top_labels=2, car nous avons 2 "sorties" (en multi-label),
# LIME va s'intéresser aux 2 "classes" (output 0 et output 1).
explanation = explainer.explain_instance(
    image,
    classifier_fn=classifier_fn,
    segmentation_fn=custom_sam_segmentation,
    top_labels=2,
    hide_color=0,
    num_samples=50
)

# Récupérer le label (sortie) que l'on veut visualiser.
# Par exemple, explanation.top_labels[0] est souvent la sortie
# jugée la "plus probable" par LIME, mais en multi-label,
# vous pouvez décider de visualiser la sortie 0 ou 1.
top_label_for_lime = explanation.top_labels[0]

# Extraire l'image avec superpixels d'explication
temp, sam_lime_explanation_mask = explanation.get_image_and_mask(
    label=top_label_for_lime,
    positive_only=True,
    num_features=10,
    hide_rest=False
)

# Surimposer les frontières LIME sur l'image d'origine avec des contours rouges et épais
image_lime_boundaries = mark_boundaries(image, sam_lime_explanation_mask, color=(0, 0, 1), mode='thick')


# Sauvegarder
lime_output_path = "sam_lime_explanation.png"
cv2.imwrite(
    lime_output_path,
    cv2.cvtColor((image_lime_boundaries * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
)
print(f"LIME explanation saved to: {lime_output_path}")

# -----
# 9. Afficher la vérité terrain
# -----
gt_mask_pil = Image.open(ground_truth_mask_path).convert("L")
gt_mask_np = np.array(gt_mask_pil)

# Redimensionne si nécessaire
if gt_mask_np.shape[:2] != (orig_h, orig_w):
    print("Warning: Ground truth mask size is different from the image size. Resizing mask ...")
    gt_mask_np = cv2.resize(gt_mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

image_ground_truth_boundaries = mark_boundaries(image/255, gt_mask_np, color=(0, 0, 1), mode='thick')
gt_output_path = "image_ground_truth_boundaries.png"
cv2.imwrite(
    gt_output_path,
    cv2.cvtColor((image_ground_truth_boundaries * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
)
print(f"Ground-truth boundaries saved to: {gt_output_path}")


print("\nDone.")
