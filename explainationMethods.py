import torch
from lime import lime_image
#from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import numpy as np
from datasetManager import datasetManager
from modelManager import ModelManager
from torchvision import transforms as T
#from skimage.segmentation import mark_boundaries
import cv2
from PIL import Image
import datetime
import matplotlib.pyplot as plt
import shap


class SAMSegmentationMasker:
    def __init__(self, segmentation_fn, image):
        self.segments = segmentation_fn(image)
        self.nb_segments = np.max(self.segments)

    def mask_image(self, masks, image, background=0):
        masks = masks.astype(bool)
        out = np.zeros((masks.shape[0], *image.shape), dtype=image.dtype)
        for i, mask in enumerate(masks):
            masked_image = image.copy()
            boolean_mask = np.zeros(self.segments.shape, dtype=bool)
            for seg_idx in range(1, self.nb_segments + 1):
                if mask[seg_idx - 1]:
                    boolean_mask |= (self.segments == seg_idx)
            out[i] = masked_image * boolean_mask[..., None] + background * (~boolean_mask[..., None])
        return out


class MainExplainer:
    def __init__(self, explainationMethod):
        assert explainationMethod in ['lime','shap'], "Explaination method not supported"
        self.explainationMethod = explainationMethod
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def explain(self, model_input, model_manager, datasetManagerObject, segmenter, num_samples=1000):
        if self.explainationMethod == 'lime':
            return self._explain_lime(model_input,
                model_manager,
                datasetManagerObject.transform,
                segmenter,
                num_samples=num_samples
            )
        if self.explainationMethod == 'shap':
            return self._explain_shap(model_input,
                model_manager,
                datasetManagerObject.transform,
                segmenter,
                num_samples=num_samples
            )

    def show_explanation(self, explanation=None, original_image=None, save=True):
        if self.explainationMethod == 'lime':
            self._show_explanation_lime(explanation, original_image, save)
        if self.explainationMethod == 'shap':
            self._show_explanation_shap(explanation, original_image, save)

    def _explain_shap(self, input_imgs, model, transformShap, segmentationModel, num_samples):
        def classifier_fn(input_batch):
            output = model.inference(input_batch)

            if output.ndim == 4:
                output = output.mean(axis=(2, 3))
            
            return output
        
        images = input_imgs
        if isinstance(images, np.ndarray):
            images_pil = Image.fromarray(np.uint8(images))
            images = transformShap(images_pil)
            images = images.numpy()
        else:
            images = transformShap(images)
            images = images.numpy()
        images = np.transpose(images, (1, 2, 0))
        
        print('---------- Shap Explanation ----------')
        if segmentationModel.segmentationModelType == 'default':
            images = np.expand_dims(images, axis=0)
            masker = shap.maskers.Image("blur(128,128)", images[0].shape)
            explainer = shap.Explainer(classifier_fn, masker, output_names=['Mild','Severe'])
            shap_values = explainer(images, max_evals=num_samples, batch_size=8, outputs=shap.Explanation.argsort.flip[:4])
        
        if segmentationModel.segmentationModelType == 'sam':
            segmentation_fn = segmentationModel.segmentationModel
            masker = SAMSegmentationMasker(segmentation_fn, (images * 255).astype(np.uint8))
            nb_segments = masker.nb_segments
            print(f"Nombre de segments SAM détectés : {nb_segments}")

            explainer = shap.KernelExplainer(
                lambda z: classifier_fn(masker.mask_image(z, images)),
                np.zeros((1, nb_segments))
            )
            self._last_segments = masker.segments

            class_to_explain = 1
            
            shap_values = explainer.shap_values(
                np.ones((1, nb_segments)),
                nsamples=num_samples
            )

            if isinstance(shap_values, list):
                shap_values_class = shap_values[class_to_explain][0]
            else:
                shap_values_class = shap_values[0, :, class_to_explain] if shap_values.ndim > 2 else shap_values[0]
            
            explain_map = np.zeros(images.shape[:2], dtype=np.float32)
            for seg_idx in range(nb_segments):
                seg_value = shap_values_class[seg_idx]
                if hasattr(seg_value, '__len__') and len(seg_value) > 0:
                    seg_value = seg_value[0]
                explain_map[masker.segments == (seg_idx + 1)] = seg_value



        self.explanation = shap_values if shap_values is not None else explain_map
        self.explanation_images = input_imgs
        
        return shap_values if shap_values is not None else explain_map

    def _show_explanation_shap(self, explanation=None, original_image=None, save=True):
        assert explanation is not None or hasattr(self, 'explanation'), "No explanation provided"
        if explanation is None:
            explanation = self.explanation
        if original_image is None and hasattr(self, 'explanation_images'):
            original_image = self.explanation_images
            
        if isinstance(explanation, shap.Explanation):
            shap.image_plot(explanation, show=False)
        else:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title("Original Image")
            plt.axis('off')
            plt.subplot(1, 2, 2)

            if isinstance(explanation, list) and len(explanation) > 0:
                class_idx = 1
                class_name = "Severe"
                if len(explanation) > class_idx:
                    segment_values = explanation[class_idx][0]
                    self._visualize_shap_segments(original_image, segment_values, class_name)

            elif isinstance(explanation, np.ndarray) and explanation.ndim == 3:
                class_idx = 1
                class_name = "Severe"
                segment_values = explanation[0, :, class_idx]
                self._visualize_shap_segments(original_image, segment_values, class_name)

            elif isinstance(explanation, np.ndarray) and explanation.ndim == 2:
                plt.imshow(original_image)
                max_abs_val = np.max(np.abs(explanation)) if np.size(explanation) > 0 else 1.0
                if max_abs_val == 0:
                    max_abs_val = 1.0
                    
                plt.imshow(explanation, cmap='coolwarm', alpha=0.7, 
                        vmin=-max_abs_val, vmax=max_abs_val)
                plt.title("SHAP Explanation")
            else:
                plt.text(0.5, 0.5, "No valid SHAP explanation data", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes)
                plt.title("Visualization Error")
            
            plt.axis('off')
            
        if save:
            h = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
            plt.savefig(f"shap_explanation_{h}.png")
            
        plt.show()

    def _visualize_shap_segments(self, original_image, segment_values, class_name):
        plt.imshow(original_image)
        if hasattr(self, '_last_segments') and self._last_segments is not None:
            segments = self._last_segments
            small_heatmap = np.zeros(segments.shape, dtype=np.float32)
            for seg_idx in range(len(segment_values)):
                small_heatmap[segments == (seg_idx + 1)] = segment_values[seg_idx]
            if small_heatmap.shape != original_image.shape[:2]:
                heatmap = cv2.resize(
                    small_heatmap, 
                    (original_image.shape[1], original_image.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                heatmap = small_heatmap
            
            max_abs_val = np.max(np.abs(segment_values)) if len(segment_values) > 0 else 1.0
            if max_abs_val == 0:
                max_abs_val = 1.0
            plt.imshow(heatmap, cmap='coolwarm', alpha=0.7, 
                    vmin=-max_abs_val, vmax=max_abs_val)
            plt.title(f"SHAP Explanation for class {class_name}")
            
            plt.colorbar(label="SHAP value")
        else:
            plt.title("SHAP values (segments not available)")
        
        

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
    
    


    def _show_explanation_lime(self, explanation=None, original_image=None, save=True):
        assert explanation is not None or hasattr(self, 'explanation'), "No explanation provided"
        assert original_image is not None or hasattr(self, 'explanation_images'), "No original image provided"
        if explanation is None:
            explanation = self.explanation
        if original_image is None:
            original_image = self.explanation_images
        top_label = explanation.top_labels[0]
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
        positive_mask_resized = cv2.resize(positive_mask.astype(np.uint8), 
                                         (original_image.shape[1], original_image.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
        negative_mask_resized = cv2.resize(negative_mask.astype(np.uint8), 
                                         (original_image.shape[1], original_image.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
        
        img_with_boundaries = np.copy(original_image).astype(np.float32) / 255
        
        img_with_boundaries[positive_mask_resized > 0] = img_with_boundaries[positive_mask_resized > 0] * 0.7 + np.array([0, 1, 0]) * 0.3
        img_with_boundaries[negative_mask_resized > 0] = img_with_boundaries[negative_mask_resized > 0] * 0.7 + np.array([1, 0, 0]) * 0.3
        
        img_with_boundaries = np.clip(img_with_boundaries, 0, 1)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img_with_boundaries)
        plt.title("LIME Explanation for class {}".format(top_label))
        plt.axis('off')
        if save:
            h = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
            plt.savefig(f"lime_explanation_{h}.png")
        plt.tight_layout()           
        plt.show()
        return img_with_boundaries
        

    
class segmentationWrapper:
    def __init__(self, segmentationModelType, file=None,params={}):
        assert segmentationModelType in ['sam', 'default'], "Segmentation model not supported"
        self.segmentationModelType = segmentationModelType


        if segmentationModelType == 'sam':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            #if we want to load from a file
            #sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
            #model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            #sam2 = build_sam2(model_cfg, sam2_checkpoint, device=self.device, apply_postprocessing=False)

            # if we want to load for HGF
            #model_type = "vit_b"
            #self.checkpoint = file
            #assert file is not None, "File path needed for this segmentation model"
            #sam = sam_model_registry[model_type](checkpoint=file)
            #sam.to(device=self.device)
            # sam.eval()
            default_sam_params = {
                'min_mask_region_area': 0,               # ↓ Plus petites régions (défaut 0)
                'pred_iou_thresh': 0.70,                  # ↓ Plus de régions conservées (défaut 0.88)
                'stability_score_thresh': 0.80,           # ↓ Plus permissif (défaut 0.95)
                'crop_n_layers': 2,                       # ↓ Moins de recadrage (défaut 0)
                'crop_overlap_ratio': 0.5,                # ↑ Meilleure couverture (défaut 0.3413)
                'points_per_batch': 8,                   # ↑ Efficacité par batch (défaut 64)
                'crop_n_points_downscale_factor': 1,      # = Garde résolution complète (défaut 1)
                'box_nms_thresh': 0.8,                     # ↑ Garde segments voisins (défaut 0.7)
                'points_per_side':150
            }
            for key, value in params.items():
                if key in default_sam_params:
                    default_sam_params[key] = value
            self.params = default_sam_params
            """
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                min_mask_region_area=default_sam_params['min_mask_region_area'],
                pred_iou_thresh=default_sam_params['pred_iou_thresh'],
                stability_score_thresh=default_sam_params['stability_score_thresh'],
                crop_n_layers=default_sam_params['crop_n_layers'],
                crop_overlap_ratio=default_sam_params['crop_overlap_ratio'],
                points_per_batch=default_sam_params['points_per_batch'],
                crop_n_points_downscale_factor=default_sam_params['crop_n_points_downscale_factor'],
                box_nms_thresh=default_sam_params['box_nms_thresh']
            )
            """
            mask_generator = SAM2AutomaticMaskGenerator.from_pretrained('facebook/sam2-hiera-base-plus',
                min_mask_region_area=default_sam_params['min_mask_region_area'],
                pred_iou_thresh=default_sam_params['pred_iou_thresh'],
                stability_score_thresh=default_sam_params['stability_score_thresh'],
                crop_n_layers=default_sam_params['crop_n_layers'],
                crop_overlap_ratio=default_sam_params['crop_overlap_ratio'],
                points_per_batch=default_sam_params['points_per_batch'],
                crop_n_points_downscale_factor=default_sam_params['crop_n_points_downscale_factor'],
                box_nms_thresh=default_sam_params['box_nms_thresh'],
                points_per_side = default_sam_params['points_per_side']
            )

           
            def sam_segmentation_fn(rgb_image: np.ndarray):
                if rgb_image.ndim == 2:
                    rgb_image = np.stack([rgb_image]*3, axis=-1)
                elif rgb_image.shape[2] == 1:
                    rgb_image = np.repeat(rgb_image, 3, axis=-1)
               
                print("imagSize :",rgb_image.shape)
                print("10*10 pixels au centre de l'image :",rgb_image[107:117,107:117])
                with torch.no_grad():
                    masks = mask_generator.generate(rgb_image)

                seg_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.int32)
                for i, m in enumerate(masks):
                    seg_mask[m['segmentation']] = i + 1
                return seg_mask
            
            self.segmentationModel = sam_segmentation_fn

        if segmentationModelType == 'default':
            self.segmentationModel = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        pass # Si on veut train la segmentation à un moment 

if __name__ == '__main__':
    import time
    dm = datasetManager(dataset=1, batch_size=8, num_workers=4, transform=T.Compose([T.Resize((224, 224))]))
    model_input = dm.get_sample_by_class(n_samples=1, rawImage=True)[0]
    model_manager = ModelManager('swinT', 2, "swinT_model_2025-03-11_17-49_3.pth")
    # samParams = {
    #     'min_mask_area': 5,
    #     'crop_n_layers': 2,
    #     'crop_overlap_ratio': 0.25,
    #     'points_per_side': 150
    # }
    
    samParams = {
        'min_mask_region_area': 4,               # ↓ Plus petites régions (défaut 0)
        'pred_iou_thresh': 0.60,                  # ↓ Plus de régions conservées (défaut 0.88)
        'stability_score_thresh': 0.80,           # ↓ Plus permissif (défaut 0.95)
        'crop_n_layers': 2,                       # ↓ Moins de recadrage (défaut 0)
        'crop_overlap_ratio': 0.45,                # ↑ Meilleure couverture (défaut 0.3413)
        'points_per_batch': 64,                   # ↑ Efficacité par batch (défaut 64)
        'crop_n_points_downscale_factor': 1,      # = Garde résolution complète (défaut 1)
        'box_nms_thresh': 0.8                     # ↑ Garde segments voisins (défaut 0.7)
    }
    
    # segmenter = segmentationWrapper('sam', 'sam_vit_b_01ec64.pth', samParams)
    # segmenter = segmentationWrapper('sam', 'sam_vit_b_01ec64.pth')
    # segmenter = segmentationWrapper('default')
    segmenter = segmentationWrapper('sam', None  , {})
    
    TStart = time.time()

    explainer = MainExplainer('lime')
    explanation = explainer.explain(
        model_input,
        model_manager,
        dm,
        segmenter,
    )
    explainer.show_explanation()


    # explainer = MainExplainer('shap')
    # explanation = explainer.explain(
    #     model_input,
    #     model_manager,
    #     dm,
    #     segmenter,
    #     num_samples=60
    # )
    TFinish = time.time()
    print(f"Time taken: {TFinish - TStart:.2f} seconds")
    print(explanation)
    explainer.show_explanation()


