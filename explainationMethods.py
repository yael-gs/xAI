import torch
from lime import lime_image
#from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_hf
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
from quantus.metrics import Complexity, FaithfulnessEstimate
import quantus
from sklearn.metrics import jaccard_score
from itertools import combinations
import pandas as pd

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
    def __init__(self, explainationMethod, metrics=[]):
        assert explainationMethod in ['lime', 'shap', 'gradcam'], "Explaination method not supported"
        self.explainationMethod = explainationMethod
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = metrics

    def explain(self, model_input, model_manager, datasetManagerObject, segmenter, num_samples=1000, return_metrics=False):
        if self.explainationMethod == 'lime':
            return self._explain_lime(model_input,
                model_manager,
                datasetManagerObject.transform,
                segmenter,
                num_samples=num_samples,
                return_metrics=return_metrics
            )
        if self.explainationMethod == 'shap':
            return self._explain_shap(model_input,
                model_manager,
                datasetManagerObject.transform,
                segmenter,
                num_samples=num_samples,
                return_metrics=return_metrics
            )
        if self.explainationMethod == 'gradcam':
            return self._explain_gradcam(model_input,
                model_manager,
                datasetManagerObject.transform,
                return_metrics=return_metrics
            )
    
    def show_explanation(self, explanation=None, original_image=None, save=True):
        if self.explainationMethod == 'lime':
            self._show_explanation_lime(explanation, original_image, save)
        if self.explainationMethod == 'shap':
            self._show_explanation_shap(explanation, original_image, save)
        if self.explainationMethod == 'gradcam':
            self._show_explanation_gradcam(explanation, original_image, save)
            
    def _compute_jaccard(self, input_imgs, explanation, ground_truth_mask, mapping_clr2cls: dict):
        if self.explainationMethod == 'shap':
            top_label = 1
        else:
            top_label = explanation.top_labels[0]

        if self.explainationMethod == 'gradcam':
            positive_mask_resized = cv2.resize(
                explanation,
                (input_imgs.shape[1], input_imgs.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            _, positive_mask = explanation.get_image_and_mask(
                label=top_label,
                positive_only=True,
                num_features=5,
                hide_rest=False
            )
            
            positive_mask_resized = cv2.resize(
                positive_mask.astype(np.float64), 
                (input_imgs.shape[1], input_imgs.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        
        background_color = (0.0, 0.0, 0.0)
        # Extract unique labels from the ground truth mask (assumes shape is (H, W, 3))
        unique_labels = np.unique(ground_truth_mask.reshape(-1, ground_truth_mask.shape[2]), axis=0)
        unique_labels = np.array([lbl for lbl in unique_labels if not np.all(lbl == background_color)])

        # Precompute binary masks for each unique label only once
        masks = [np.all(ground_truth_mask == lbl, axis=-1) for lbl in unique_labels]
        
        max_jaccard = 0.0
        best_combo = []
        
        # Iterate over all non-empty combinations of precomputed masks
        for r in range(1, len(unique_labels) + 1):
            for combo in combinations(range(len(unique_labels)), r):
                # Combine the masks corresponding to the indices in the current combination
                union_mask = np.any(np.stack([masks[i] for i in combo], axis=0), axis=0)
                
                # Compute the Jaccard score (flattening the arrays for comparison)
                jaccard = jaccard_score(positive_mask_resized.flatten(), union_mask.flatten(), average='binary') 
                #FIXME : Problème avec les méthodes non binaires (autre que Lime)
                
                if jaccard > max_jaccard:
                    max_jaccard = jaccard
                    best_combo = combo

        # Display the labels corresponding to the best combination of sub masks
        best_labels = []
        for idx in best_combo:
            # Extract color array from unique_labels
            color_array = unique_labels[idx]
            rounded_color = [np.floor(val * 100) / 100 for val in color_array]
            rounded_color = list(rounded_color)
            # Proper conversion: convert the NumPy array to a tuple for dictionary lookup
            color_tuple = tuple(rounded_color)
            label = mapping_clr2cls.get(color_tuple, "Unknown")
            best_labels.append(label)
        
        overlap_proportions = {}
        # Ensure the predicted mask is boolean for logical operations.
        predicted_bool = positive_mask_resized.astype(bool)
        for i, mask in enumerate(masks):
            # Calculate the intersection between the ground truth submask and the predicted mask
            intersection = np.logical_and(mask, predicted_bool)
            submask_area = np.sum(mask)
            if submask_area > 0:
                proportion = np.sum(intersection) / submask_area
            else:
                proportion = np.nan  # or 0, depending on how you wish to handle empty masks.
            
            color_array = unique_labels[i]
            rounded_color = [np.floor(val * 100) / 100 for val in color_array]
            rounded_color = list(rounded_color)
            # Proper conversion: convert the NumPy array to a tuple for dictionary lookup
            color_tuple = tuple(rounded_color)
            label = mapping_clr2cls.get(color_tuple, "Unknown")
            overlap_proportions[label] = proportion
        
        return max_jaccard, best_labels, overlap_proportions
        


    def _verify_valid_metric(self, metric):
        metricDict = {
            'COMPLEXITY': Complexity,
            'FAITHFULNESS': FaithfulnessEstimate,
            'ROAD': quantus.ROAD,
            'AUC': quantus.AUC
        }
        metricsParams = {
            "ROAD": {
                "noise": 0.1,
                "percentages":list(range(1, 51, 1)),
                "display_progressbar":True
            },
            'FAITHFULNESS' : {
                #'perturb_func' : quantus.uniform_noise,
                'features_in_step' : 8
            }
        }
        needGT = ['AUC'] #TODO Implémenter pour les segmentation Ground Truth
        assert metric in metricDict, f"Metric {metric} not supported"
        if metric is None:
            return None, None
        if metric in metricDict:
            return metricDict[metric], metricsParams.get(metric, None)
        return None, None
    
    def _compute_metrics(self, input_imgs, explanation, label, model, metrics : str | list[str] = ['ROAD']):
        if isinstance(metrics, str):
            metrics = [metrics]
        final_metric = {}
        if self.explainationMethod == 'gradcam':
            explain_resized = cv2.resize(
                explanation,
                (input_imgs.shape[2], input_imgs.shape[1]),
                interpolation=cv2.INTER_NEAREST
            )
            
        else:
            # print("input_imgs.shape in _compute_metrics:", input_imgs.shape)
            if self.explainationMethod == 'shap':
                top_label = 1
            else:
                top_label = explanation.top_labels[0]

            _, positive_mask = explanation.get_image_and_mask(
                label=top_label,
                positive_only=True,
                num_features=5,
                hide_rest=False
            )
            
            # print("Positive mask shape:", positive_mask.shape)
            assert positive_mask.ndim == 2, f"Expected 2D mask but got shape {positive_mask.shape}"
            
            if input_imgs.ndim == 4:
                img_height = input_imgs.shape[1]
                img_width = input_imgs.shape[2]
            else:
                img_height = input_imgs.shape[0]
                img_width = input_imgs.shape[1]
                input_imgs = np.expand_dims(input_imgs, axis=0)
            
            explain_resized = cv2.resize(
                positive_mask.astype(np.float64), 
                (img_width, img_height), 
                interpolation=cv2.INTER_NEAREST
            )
            
        # print("Positive mask resized shape:", positive_mask_resized.shape)
        # print("Final input_imgs shape:", input_imgs.shape)
        x_batch = np.transpose(input_imgs, (0, 3, 1, 2))
        y_batch = np.array([label])
        a_batch = np.expand_dims(explain_resized, axis=0)
        a_batch = np.expand_dims(a_batch, axis=1)
        print(label)
        # print('y_batch shape:', y_batch.shape)
        # print('y_batch:', y_batch)
        # print("x_batch shape:", x_batch.shape)  # Should be (1, 3, H, W)
        # print("a_batch shape:", a_batch.shape)  # Should be (1, 1, H, W)

        for metric_name in metrics:
            print('Running:', metric_name)
            MetricClass, metric_params = self._verify_valid_metric(metric_name)

            if metric_params is not None:
                metric_instance = MetricClass(**metric_params)
            else:
                metric_instance = MetricClass()
            
            try:
                metric_score = metric_instance(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    device=self.device,
                )
                final_metric[metric_name] = metric_score
                print(f'Score {metric_name}: {metric_score}')
            except Exception as e:
                print(f"Error computing metric {metric_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                final_metric[metric_name] = None

        return final_metric


    def _explain_shap(self, input_imgs, model_manager, transformShap, segmentationModel, num_samples, return_metrics=False):
        def classifier_fn(input_batch):
            output = model_manager.inference(input_batch)
            if output.ndim == 4:
                print('truc chelou')
                output = output.mean(axis=(2, 3))
            
            return output
        
        class ClassifierWrapper(torch.nn.Module):
            def __init__(self, classifier_fn):
                super(ClassifierWrapper, self).__init__()
                self.classifier_fn = classifier_fn
                self.eval()

            def forward(self, x):
                return self.classifier_fn(x, metricsStyle=True)
        
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
        shap_valuesf = None
        explain_map = None
        segments = None
        
        if segmentationModel.segmentationModelType == 'default':
            images = np.expand_dims(images, axis=0)
            masker = shap.maskers.Image("blur(128,128)", images[0].shape)
            explainer = shap.Explainer(classifier_fn, masker, output_names=['Mild','Severe'])
            shap_valuesf = explainer(images, max_evals=num_samples, batch_size=8, outputs=shap.Explanation.argsort.flip[:4])
        
        if segmentationModel.segmentationModelType in ['sam', 'grid']:
            segmentation_fn = segmentationModel.segmentationModel
            masker = SAMSegmentationMasker(segmentation_fn, (images * 255).astype(np.uint8))
            nb_segments = masker.nb_segments
            print(f"Nombre de segments détectés : {nb_segments}")
            torch.cuda.empty_cache()
            explainer = shap.KernelExplainer(
                lambda z: classifier_fn(masker.mask_image(z, images)),
                np.zeros((1, nb_segments))
            )
            self._last_segments = masker.segments
            segments = masker.segments

            class_to_explain = 1
            
            shap_values = explainer.shap_values(
                np.ones((1, nb_segments)),
                nsamples=min(num_samples, 500),
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
        
        wrapped_explanation = ShapExplanationWrapper(
            shap_valuesf if shap_valuesf is not None else explain_map,
            segments=segments,
            original_image=images
        )
        
        if self.metrics is not [] and self.metrics is not None:
            top_label = wrapped_explanation.top_labels[0]
            print(top_label)
            metricsRes = self._compute_metrics(
                input_imgs=images,
                model=ClassifierWrapper(model_manager.inference),
                explanation=wrapped_explanation,
                label=1,
                metrics=self.metrics
            )
            print(metricsRes)
        
        self.explanation = wrapped_explanation
        self.explanation_images = input_imgs
        if return_metrics:
            return wrapped_explanation, metricsRes
        return wrapped_explanation

    def _show_explanation_shap(self, explanation=None, original_image=None, save=True):
        assert explanation is not None or hasattr(self, 'explanation'), "No explanation provided"
        if explanation is None:
            explanation = self.explanation
        if original_image is None and hasattr(self, 'explanation_images'):
            original_image = self.explanation_images
        

        img, mask = explanation.get_image_and_mask(
            label=1,
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        
        print("Mask shape:", mask.shape)
        print("Image shape:", img.shape)
        img = img[0]


        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_image)
        axes[0].set(
            title="Original Image",
            xticks=[],
            yticks=[],
        )
        
        axes[1].imshow(img)
        im = axes[1].imshow(mask, cmap='hot', alpha=0.7)
        fig.colorbar(im, ax=axes[1])
        axes[1].set(
            title="SHAP Explanation",
            xticks=[],
            yticks=[],
        )
            
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
    
    def _explain_lime(self, input_imgs, model_manager : ModelManager, transformLime, segmentationModel, num_samples, return_metrics=False):
        def classifier_fn(input_batch):
            out = model_manager.inference(input_batch)
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
        if segmentationModel.segmentationModelType in ['sam', 'grid']:
            explanation = explainerObject.explain_instance(
                images,
                classifier_fn,
                top_labels=2,
                hide_color=0,
                num_samples=num_samples,
                segmentation_fn=segmentationModel.segmentationModel,
            )
        
        
        class ClassifierWrapper(torch.nn.Module):
            def __init__(self, classifier_fn):
                super(ClassifierWrapper, self).__init__()
                self.classifier_fn = classifier_fn
                self.eval()

            def forward(self, x):
                return self.classifier_fn(x, metricsStyle=True)

        top_label = explanation.top_labels[0]
        if self.metrics is not [] and self.metrics is not None:
            metricsRes = self._compute_metrics(input_imgs=images, model=ClassifierWrapper(model_manager.inference), explanation=explanation, label=top_label, metrics=self.metrics)
            print(metricsRes)
        
        self.explanation = explanation
        self.explanation_images = input_imgs
        if return_metrics:
            return explanation, metricsRes
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
        
    def _activations_hook(self, grad):
        self.gradients = grad

    def _explain_gradcam(self, input_imgs, model_manager, transform, return_metrics=False):
        images = input_imgs
        if isinstance(images, np.ndarray):
            images_pil = Image.fromarray(np.uint8(images))
            images = transform(images_pil)
            images = images.numpy()
        else:
            images = transform(images)
            images = images.numpy()

        images = np.transpose(images, (1, 2, 0))
        images = np.expand_dims(images, axis=0)
        image = images[0]

        if isinstance(image, np.ndarray):
            # Handle numpy arrays - check dimensions
            if image.ndim == 3:  # Single image with channels
                image = torch.from_numpy(image).float()
                # Ensure channels first format (C,H,W)
                if image.shape[2] in [1, 3, 4]:  # If channels are last
                    image = image.permute(2, 0, 1)
                image = image.unsqueeze(0)  # Add batch dimension
            elif image.ndim == 4:  # Batch of images
                image = torch.from_numpy(image).float()
                # Ensure channels first format (B,C,H,W)
                if image.shape[3] in [1, 3, 4]:  # If channels are last
                    image = image.permute(0, 3, 1, 2)
        else:
            # For PyTorch tensor
            if image.dim() == 3:
                image = image.unsqueeze(0)  # Add batch dimension

        print('---------- Grad-CAM Explanation ----------')

        image = image.to(self.device)

        if model_manager.modelType == "vgg16":
            layers_before = [
                model_manager.model.features[:30]
            ]

            layers_after = [
                model_manager.model.features[30:],
                model_manager.model.avgpool,
                lambda x: x.view((1, -1)),
                model_manager.model.classifier
            ]

        if model_manager.modelType == "resnet50":
            layers_before = [
                model_manager.model.conv1,
                model_manager.model.bn1,
                model_manager.model.maxpool,
                model_manager.model.layer1,
                model_manager.model.layer2,
                model_manager.model.layer3,
                model_manager.model.layer4
            ]

            layers_after = [
                model_manager.model.avgpool,
                lambda x: x.view((1, -1)),
                model_manager.model.fc
            ]

        if model_manager.modelType == "swinT":
            layers_before = [
                model_manager.model.features,
                model_manager.model.norm,
                model_manager.model.permute
            ]

            layers_after = [
                model_manager.model.avgpool,
                model_manager.model.flatten,
                model_manager.model.head
            ]

        def forward(x):
            for layer in layers_before:
                x = layer(x)
                x.requires_grad_() # seems to do nothing
                x.register_hook(self._activations_hook)

            for layer in layers_after:
                x = layer(x)

            return x

        model_manager.model.forward = forward

        pred = model_manager.model(image)
        index = pred.argmax(dim=1)

        pred[:, index].backward()
        gradients = self.gradients

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        x = image
        for layer in layers_before:
            x = layer(x)
            activations = x.detach()

        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = torch.tensor(np.maximum(heatmap.cpu().numpy(), 0))

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        class ClassifierWrapper(torch.nn.Module):
            def __init__(self, classifier_fn):
                super(ClassifierWrapper, self).__init__()
                self.classifier_fn = classifier_fn
                self.eval()

            def forward(self, x):
                return self.classifier_fn(x, metricsStyle=True)
        fexplanation = heatmap.squeeze().numpy()
        print('HM:', type(fexplanation), fexplanation.shape)
        
        top_label = index.item()
        if self.metrics is not [] and self.metrics is not None:
            metricsRes = self._compute_metrics(input_imgs=images, model=ClassifierWrapper(model_manager.inference), explanation=fexplanation, label=top_label, metrics=self.metrics)
            print(metricsRes)

        self.explanation = fexplanation
        self.explanation_images = input_imgs
        if return_metrics :
            return fexplanation, metricsRes
        return fexplanation

    def _show_explanation_gradcam(self, explanation=None, original_image=None, save=True):
        assert explanation is not None or hasattr(self, 'explanation'), "No explanation provided"
        if explanation is None:
            explanation = self.explanation
        if original_image is None and hasattr(self, 'explanation_images'):
            original_image = self.explanation_images

        size = (original_image.shape[1], original_image.shape[0])
        heatmap = cv2.resize(explanation, size)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

        image = 0.4 * heatmap.astype(np.float32) + 0.6 * original_image.astype(np.float32)
        image = np.round(image).astype(np.uint8)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.title("Grad-CAM Explanation")
        plt.axis('off')

        if save:
            h = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
            plt.savefig(f"gradcam_explanation_{h}.png")

        plt.tight_layout()
        plt.show()


class segmentationWrapper:
    def __init__(self, segmentationModelType, file=None,params={}):
        assert segmentationModelType in ['sam', 'default', 'grid'], "Segmentation model not supported"
        self.segmentationModelType = segmentationModelType
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        if segmentationModelType == 'sam':
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
            # mask_generator = SamAutomaticMaskGenerator(
            #     model=sam,
            #     min_mask_region_area=default_sam_params['min_mask_region_area'],
            #     pred_iou_thresh=default_sam_params['pred_iou_thresh'],
            #     stability_score_thresh=default_sam_params['stability_score_thresh'],
            #     crop_n_layers=default_sam_params['crop_n_layers'],
            #     crop_overlap_ratio=default_sam_params['crop_overlap_ratio'],
            #     points_per_batch=default_sam_params['points_per_batch'],
            #     crop_n_points_downscale_factor=default_sam_params['crop_n_points_downscale_factor'],
            #     box_nms_thresh=default_sam_params['box_nms_thresh']
            # )
            
            default_sam_params = {
                'min_mask_region_area': 0,                  # ↓ Plus petites régions (défaut 0)
                'pred_iou_thresh': 0.70,                    # ↓ Plus de régions conservées (défaut 0.88)
                'stability_score_thresh': 0.80,             # ↓ Plus permissif (défaut 0.95)
                'crop_n_layers': 2,                         # ↓ Moins de recadrage (défaut 0)
                'crop_overlap_ratio': 0.5,                  # ↑ Meilleure couverture (défaut 0.3413)
                'points_per_batch': 8,                      # ↑ Efficacité par batch (défaut 64)
                'crop_n_points_downscale_factor': 1,        # = Garde résolution complète (défaut 1)
                'box_nms_thresh': 0.8,                      # ↑ Garde segments voisins (défaut 0.7)
                'points_per_side':32
            }
            for key, value in params.items():
                if key in default_sam_params:
                    default_sam_params[key] = value
            self.params = default_sam_params
            
            sam_model = build_sam2_hf('facebook/sam2.1-hiera-tiny', device=self.device, apply_postprocessing=False)
            mask_generator = SAM2AutomaticMaskGenerator(model=sam_model,
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

                if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
                    rgb_image = (rgb_image * 255).astype(np.uint8)

               
                print("imagSize :",rgb_image.shape)
                print("10*10 pixels au centre de l'image :",rgb_image[107:117,107:117])
                with torch.no_grad():
                    masks = mask_generator.generate(rgb_image)

                seg_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.int32)
                for i, m in enumerate(masks):
                    seg_mask[m['segmentation']] = i + 1
                print("Segmentation prête")
                return seg_mask
            
            self.segmentationModel = sam_segmentation_fn

        if segmentationModelType == 'default':
            self.segmentationModel = None
        

        if segmentationModelType == 'grid':
            def grid_segmentation_fn(rgb_image: np.ndarray, grid_size=(8, 8)):
               
                if rgb_image.ndim == 2:
                    rgb_image = np.stack([rgb_image]*3, axis=-1)
                elif rgb_image.shape[2] == 1:
                    rgb_image = np.repeat(rgb_image, 3, axis=-1)
                
                height, width = rgb_image.shape[:2]
                seg_mask = np.zeros((height, width), dtype=np.int32)
                
                cell_height = height // grid_size[0]
                cell_width = width // grid_size[1]
                
                segment_id = 1
                for i in range(grid_size[0]):
                    for j in range(grid_size[1]):
                        top = i * cell_height
                        bottom = min((i + 1) * cell_height, height)
                        left = j * cell_width
                        right = min((j + 1) * cell_width, width)
                        
                        seg_mask[top:bottom, left:right] = segment_id
                        segment_id += 1
                
                print(f"Grid segmentation created with {grid_size[0]}x{grid_size[1]} grid")
                print(f"Total segments: {segment_id - 1}")
                return seg_mask

            grid_size = params.get('grid_size', (8, 8))
            self.segmentationModel = lambda img: grid_segmentation_fn(img, grid_size)

    def train(self):
        pass # Si on veut train la segmentation à un moment 

class ShapExplanationWrapper:
    def __init__(self, shap_values, segments=None, class_names=None, original_image=None):
        self.shap_values = shap_values
        self.segments = segments
        self.class_names = class_names or ["Mild", "Severe"]
        self.original_image = original_image
        
        if isinstance(shap_values, shap.Explanation):
            if len(shap_values.shape) > 2:
                class_importance = np.sum(np.abs(shap_values.values), axis=(1, 2))
                self.top_labels = np.argsort(-class_importance)[0:2]
            else:
                self.top_labels = [0, 1]
        elif isinstance(shap_values, list):
            self.top_labels = list(range(min(2, len(shap_values))))
        else:
            self.top_labels = [1, 0]

    def get_image_and_mask(self, label=1, positive_only=True, negative_only=False, num_features=5, hide_rest=False):
        print(f"SHAP values type: {type(self.shap_values)}")
        if isinstance(self.shap_values, np.ndarray):
            if np.all(self.shap_values == 0):
                print("All SHAP values are zero")
                return self.original_image, np.zeros((224, 224))
                
        if isinstance(self.shap_values, shap.Explanation):
            shap_arr = np.array(self.shap_values.values)  # ex. forme (1,224,224,3,2)
            
            if shap_arr.ndim == 5 and shap_arr.shape[:3] == (1, 224, 224):
                if label < shap_arr.shape[-1]:
                    label_values = shap_arr[..., label]  
                    shap_map = np.sum(label_values, axis=-1)
                    shap_map = shap_map[0]
                else:
                    print(f"Label {label} hors limite. Retour d'un masque nul.")
                    shap_map = np.zeros((224, 224))
            else:
                shap_map = shap_arr
                if shap_map.ndim > 2:
                    print(f"Forme inattendue {shap_map.shape}, on la réduit en 2D par sum.")
                    shap_map = shap_map.sum(axis=tuple(range(shap_map.ndim - 2)))
        else:
            if isinstance(self.shap_values, np.ndarray):
                shap_map = self.shap_values
            else:
                shap_map = np.zeros((224, 224))

        if shap_map.shape != (224, 224):
            print(f"Forme finale inattendue: {shap_map.shape}. On redimensionne en (224,224) si possible...")
            try:
                shap_map = shap_map.reshape((224, 224))
            except:
                shap_map = np.zeros((224, 224))
        # plt.imshow(shap_map) # To Show the SHAP values only
        # plt.show()
        return self.original_image, shap_map

if __name__ == '__main__':
    """
    import time
    dm = datasetManager(dataset=1, batch_size=8, num_workers=4, transform=T.Compose([T.Resize((224, 224))]))
    model_input = dm.get_sample_by_class(n_samples=1, rawImage=True, retrun_id=True, return_labels=True, split='test')
    image_id = model_input[-1]
    image_id = image_id[0]
    print("Image ID : ",image_id)
    model_input = model_input[0][0]
    model_manager = ModelManager('vgg16', 2, "vgg16_model_2025-03-06_13-28_3.pth")
    # samParams = {
    #     'min_mask_area': 5,
    #     'crop_n_layers': 2,
    #     'crop_overlap_ratio': 0.25,
    #     'points_per_side': 150
    # }
    
    samParams = {
        'min_mask_region_area': 4,                  # ↓ Plus petites régions (défaut 0)
        'pred_iou_thresh': 0.60,                    # ↓ Plus de régions conservées (défaut 0.88)
        'stability_score_thresh': 0.80,             # ↓ Plus permissif (défaut 0.95)
        'crop_n_layers': 2,                         # ↓ Moins de recadrage (défaut 0)
        'crop_overlap_ratio': 0.45,                 # ↑ Meilleure couverture (défaut 0.3413)
        'points_per_batch': 8,                      # ↑ Efficacité par batch (défaut 64)
        'crop_n_points_downscale_factor': 1,        # = Garde résolution complète (défaut 1)
        'box_nms_thresh': 0.8,                      # ↑ Garde segments voisins (défaut 0.7)
        'points_per_side': 12
    }
    
    segmenter = segmentationWrapper('default')
    # segmenter = segmentationWrapper('sam', None, {})
    # segmenter = segmentationWrapper('grid')
    
    TStart = time.time()

    # explainer = MainExplainer('gradcam', metrics = ['ROAD', 'FAITHFULNESS', 'COMPLEXITY'])
    # explainer = MainExplainer('shap', metrics = ['ROAD', 'FAITHFULNESS', 'COMPLEXITY'])
    explainer = MainExplainer('shap', metrics = ['ROAD', 'COMPLEXITY'])

    explanation = explainer.explain(
        model_input,
        model_manager,
        dm,
        segmenter,
        num_samples=1000
    )
    # explainer.show_explanation()
    ground_truth_mask = dm.get_ground_segmentation(img_id=image_id, apply_transform=False)
    if explainer.explainationMethod == 'lime' :
        print("Jaccard index for bets sub masks combination : ", explainer._compute_jaccard(model_input, explanation, ground_truth_mask, dm.gt_msk_clr2cls))
    
    TFinish = time.time()
    print(f"Time taken: {TFinish - TStart:.2f} seconds")
    print(explanation)
    # explainer.show_explanation()
    """
    
    