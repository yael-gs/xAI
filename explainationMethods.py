import torch
import pandas as pd
from lime import lime_image
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import numpy as np
from datasetManager import datasetManager
from modelManager import ModelManager
from torchvision import transforms as T
from skimage.segmentation import mark_boundaries
import cv2


class MainExplainer:
    def __init__(self, explainationMethod):
        assert explainationMethod in ['lime'], "Explaination method not supported"
        self.explainationMethod = explainationMethod
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.explainationMethod == 'lime':
            pass
        

    def explain(self, images, model, segmentationModel):
        if self.explainationMethod == 'lime':
            return self.explain_lime(images, model, segmentationModel)
        
    def explain_lime(self, images, model, transformLime, segmentationModel, num_samples=1000,):
        def classifier_fn(input_batch):
            out = model.inference(input_batch, transform=transformLime)
            return out
        # if not isinstance(images, np.ndarray):
        #     raise TypeError("images must be a numpy array")
            
        explainerObject = lime_image.LimeImageExplainer()
        if segmentationModel.segmentationModelType == 'default':
            explanation = explainerObject.explain_instance(
                images,
                classifier_fn,
                top_labels=2,
                hide_color=0,
                num_samples=num_samples,
            )

        return explanation
    
class segmentationWrapper:
    def __init__(self, segmentationModelType):
        self.segmentationModelType = segmentationModelType
        if segmentationModelType == 'sam':
            self.segmentationModel = SamAutomaticMaskGenerator()
        if segmentationModelType == 'default':
            self.segmentationModel = None

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Initialize dataset manager and get samples
    dm = datasetManager(dataset=1, batch_size=8, num_workers=4, transform=T.Compose([T.Resize((224, 224))]))
    model_input = dm.get_sample_by_class(n_samples=1, rawImage=True)[0]
    # Load model
    model_manager = ModelManager('vgg16', 2, "vgg16_model_2025-03-06_13-28_3.pth")

    # Initialize segmentation wrapper and explainer
    segmenter = segmentationWrapper('default')
    explainer = MainExplainer('lime')
    print('start Explaining')
    # Generate explanation
    explanation = explainer.explain_lime(
        model_input,
        model_manager,
        dm.transform,
        segmenter,
        num_samples=100
    )

    # Visualize results
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        label=top_label,
        positive_only=True, 
        num_features=10,
        hide_rest=False
    )

    # Create visualization with boundaries
    img_with_boundaries = mark_boundaries(model_input, mask, color=(1, 0, 0), mode='thick')

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(model_input)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_with_boundaries)
    plt.title("LIME Explanation")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("lime_explanation.png")
    plt.show()

    print("LIME explanation completed and saved to lime_explanation.png")
