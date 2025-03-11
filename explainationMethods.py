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
        

    def explain(self, model_input, model_manager, datasetManagerObject, segmenter, num_samples=1000):
        if self.explainationMethod == 'lime':
            return self._explain_lime(model_input,
                model_manager,
                datasetManagerObject.transform,
                segmenter,
                num_samples=num_samples
            )
        
    def _explain_lime(self, input_imgs, model, transformLime, segmentationModel, num_samples):
        def classifier_fn(input_batch):
            out = model.inference(input_batch)
            return out
        images = input_imgs
        explainerObject = lime_image.LimeImageExplainer()
        if segmentationModel.segmentationModelType == 'default':
            if isinstance(images, np.ndarray):
                from PIL import Image
                images_pil = Image.fromarray(np.uint8(images))
                images = transformLime(images_pil)
                images = images.numpy()
                images = np.transpose(images, (1, 2, 0))
            else:
                images = transformLime(images)
                images = images.numpy()
                images = np.transpose(images, (1, 2, 0))
            print('---------- Lime Explanation ----------')
            explanation = explainerObject.explain_instance(
                images,
                classifier_fn,
                top_labels=1,
                hide_color=0,
                num_samples=num_samples,
            )
        if segmentationModel.segmentationModelType == 'sam':
            assert False, "Not implemented yet"
        self.explanation = explanation
        self.explanation_images = input_imgs
        return explanation
    

    def show_explanation(self, explanation=None, original_image=None):
        assert explanation is not None or hasattr(self, 'explanation'), "No explanation provided"
        assert original_image is not None or hasattr(self, 'explanation_images'), "No original image provided"
        if self.explainationMethod == 'lime':
            if explanation is None:
                explanation = self.explanation
            if original_image is None:
                original_image = self.explanation_images
            top_label = explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(
                label=top_label,
                positive_only=True, 
                num_features=5,
                hide_rest=False
            )

            mask_resized = cv2.resize(mask.astype(np.uint8), (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            img_with_boundaries = mark_boundaries(original_image, mask_resized, color=(0, 0, 1), mode='thick')

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(img_with_boundaries)
            plt.title("LIME Explanation")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig("lime_explanation.png")
            plt.show()

        

    
class segmentationWrapper:
    def __init__(self, segmentationModelType):
        self.segmentationModelType = segmentationModelType
        if segmentationModelType == 'sam':
            self.segmentationModel = SamAutomaticMaskGenerator()
        if segmentationModelType == 'default':
            self.segmentationModel = None
    
    def train(self):
        pass # Si on veut train la segmentation à un moment 

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dm = datasetManager(dataset=1, batch_size=8, num_workers=4, transform=T.Compose([T.Resize((224, 224))]))
    model_input = dm.get_sample_by_class(n_samples=1, rawImage=True)[0]
    model_manager = ModelManager('vgg16', 2, "vgg16_model_2025-03-06_13-28_3.pth")

    segmenter = segmentationWrapper('default')
    explainer = MainExplainer('lime')
    
    explanation = explainer.explain(
        model_input,
        model_manager,
        dm,
        segmenter,
    )
    explainer.show_explanation()

    

