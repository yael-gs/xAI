import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
from lime import lime_image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.segmentation import mark_boundaries
import requests
from io import BytesIO

def download_image(image_url, save_path):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image.save(save_path)
    return image

def import_or_download_image(image_path, image_url, printImage=False):
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Image not found. Downloading from {image_url}...")
        image = download_image(image_url, image_path)

    if printImage:
        image.show()
    return image

def get_input_transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transf

def get_input_tensors(img):
    transf = get_input_transform()
    return transf(img).unsqueeze(0)

def get_pil_transform():
    transf = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.CenterCrop(224)]
    )
    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transf = transforms.Compose([transforms.ToTensor(), normalize])
    return transf

def get_probabilities(img_t, idx2label):
    logits = model(img_t)
    probs = F.softmax(logits, dim=1)
    top_probs, top_classes = probs.topk(5)
    for prob, class_idx in zip(top_probs[0].detach().numpy(), top_classes[0].detach().numpy()):
        print(f"Class: {idx2label[class_idx]} - Probability: {prob:.4f}")

def batch_predict(images):
    model.eval()
    preprocess_transform = get_preprocess_transform()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

def explain(img, idx2label,showImage=False,saveImage=False):
    explainer = lime_image.LimeImageExplainer()
    pill_transf = get_pil_transform()

    explanation = explainer.explain_instance(
        np.array(pill_transf(img)),
        batch_predict,
        top_labels=5,
        hide_color=0,
        num_samples=1000,
    )
    if showImage:
        for label in explanation.top_labels:
            temp, mask = explanation.get_image_and_mask(label, positive_only=False, num_features=10, hide_rest=False)
            img_boundry = mark_boundaries(temp/255.0, mask)
            plt.figure()
            plt.title(f"LIME Explanation for: {idx2label[label]}")
            plt.imshow(img_boundry)
            plt.axis("off")
        plt.show()
    if saveImage:
        for label in explanation.top_labels:
            temp, mask = explanation.get_image_and_mask(label, positive_only=False, num_features=10, hide_rest=False)
            img_boundry = mark_boundaries(temp/255.0, mask)
            plt.imsave(f"LIME_{idx2label[label]}.png", img_boundry)

if __name__ == "__main__":
    model = models.resnet50(pretrained=True)
    model.eval()
    
    IMAGE_PATH = "example_image.jpg"
    IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Golden_retriever_stehfoto.jpg/420px-Golden_retriever_stehfoto.jpg"
    img = import_or_download_image(IMAGE_PATH, IMAGE_URL)

    idx2label, cls2label, cls2idx = [], {}, {}
    with open(os.path.abspath("imagenet_class_index.json"), "r") as read_file:
        class_idx = json.load(read_file)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        cls2label = {
            class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))
        }
        cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}

    img_t = get_input_tensors(img)

    print("5 Predictions:")
    get_probabilities(img_t, idx2label)

    explain(img, idx2label,showImage=True,saveImage=True)
