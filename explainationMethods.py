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
from PIL import Image


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
        if isinstance(images, np.ndarray):
            images_pil = Image.fromarray(np.uint8(images))
            images = transformLime(images_pil)
            images = images.numpy()
            images = np.transpose(images, (1, 2, 0))
        else:
            images = transformLime(images)
            images = images.numpy()
            images = np.transpose(images, (1, 2, 0))
        print('---------- Lime Explanation ----------')

        if segmentationModel.segmentationModelType == 'default':
            explanation = explainerObject.explain_instance(
                images,
                classifier_fn,
                top_labels=2,
                hide_color=0,
                num_samples=num_samples,
            )
        if segmentationModel.segmentationModelType == 'sam':
             explanation = explainerObject.explain_instance(
                images,
                classifier_fn,
                top_labels=2,
                hide_color=0,
                num_samples=num_samples,
                segmentation_fn=segmentationModel.segmentationModel,
            )


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
            # Get positive and negative masks separately
            temp, positive_mask = explanation.get_image_and_mask(
                label=top_label,
                positive_only=True,
                num_features=5,
                hide_rest=False
            )
            
            temp, negative_mask = explanation.get_image_and_mask(
                label=top_label,
                positive_only=False,
                negative_only=True, 
                num_features=5,
                hide_rest=False
            )

            # Resize masks to match original image
            positive_mask_resized = cv2.resize(positive_mask.astype(np.uint8), 
                                             (original_image.shape[1], original_image.shape[0]), 
                                             interpolation=cv2.INTER_NEAREST)
            negative_mask_resized = cv2.resize(negative_mask.astype(np.uint8), 
                                             (original_image.shape[1], original_image.shape[0]), 
                                             interpolation=cv2.INTER_NEAREST)
            
            # Create overlay image
            img_with_boundaries = np.copy(original_image).astype(np.float32) / 255
            
            # Add positive regions (green overlay)
            img_with_boundaries[positive_mask_resized > 0] = img_with_boundaries[positive_mask_resized > 0] * 0.7 + np.array([0, 1, 0]) * 0.3
            
            # Add negative regions (red overlay)
            img_with_boundaries[negative_mask_resized > 0] = img_with_boundaries[negative_mask_resized > 0] * 0.7 + np.array([1, 0, 0]) * 0.3
            
            # Make sure values are in valid range
            img_with_boundaries = np.clip(img_with_boundaries, 0, 1)

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
    def __init__(self, segmentationModelType, file=None,params={}):
        assert segmentationModelType in ['sam', 'default'], "Segmentation model not supported"
        self.segmentationModelType = segmentationModelType

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if segmentationModelType == 'sam':
            model_type = "vit_b"
            assert file is not None, "File path needed for this segmentation model"
            sam = sam_model_registry[model_type](checkpoint=file)
            sam.to(device=self.device)
            mask_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_batch=10,
                min_mask_region_area=params.get('min_mask_area', 5),
                # pred_iou_thresh=params.get('pred_iou_thresh', 0.86),
                crop_n_layers=params.get('crop_n_layers', 2),
                crop_overlap_ratio=params.get('crop_overlap_ratio', 0.25),
                # crop_n_points_downscale_factor=params.get('crop_n_points_downscale_factor', 2),
                # point_grids=params.get('point_grids', None),
                points_per_side=params.get('points_per_side', 32),
                # stability_score_thresh=params.get('stability_score_thresh', 0.92),  # Higher for better segmentation
                # box_nms_thresh=params.get('box_nms_thresh', 0.7)  # For better boundary separation
            )


            def sam_segmentation_fn(rgb_image: np.ndarray):
                masks = mask_generator.generate(rgb_image)
                seg_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.int32)
                for i, m in enumerate(masks):
                    seg_mask[m['segmentation']] = i + 1
                return seg_mask
            
            self.segmentationModel = sam_segmentation_fn
            
        if segmentationModelType == 'default':
            self.segmentationModel = None
    
    def train(self):
        pass # Si on veut train la segmentation à un moment 

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dm = datasetManager(dataset=1, batch_size=8, num_workers=4, transform=T.Compose([T.Resize((224, 224))]))
    model_input = dm.get_sample_by_class(n_samples=1, rawImage=True)[0]
    model_manager = ModelManager('vgg16', 2, "vgg16_model_2025-03-06_13-28_3.pth")
    samParams = {
        'crop_n_layers': 2,
        'crop_overlap_ratio': 0.25,
        'points_per_side': 150

    }
    segmenter = segmentationWrapper('sam', 'sam_vit_b_01ec64.pth')
    # segmenter = segmentationWrapper('default')
    explainer = MainExplainer('lime')
    
    explanation = explainer.explain(
        model_input,
        model_manager,
        dm,
        segmenter,
    )
    explainer.show_explanation()

    

