import os
import pickle as pkl
from datasetManager import datasetManager
from modelManager import ModelManager
from explainationMethods import MainExplainer, segmentationWrapper
import torchvision.transforms as T

results = []
model_manager = ModelManager('vgg16', 2, "vgg16_model_2025-03-06_13-28_3.pth")
segmenter = segmentationWrapper('grid')
explainer = MainExplainer('lime', metrics = ['ROAD', 'FAITHFULNESS', 'COMPLEXITY'])

dm = datasetManager(dataset=1, batch_size=8, num_workers=4, transform=T.Compose([T.Resize((224, 224))]))
n_samples = len(os.listdir('dataset/test/images'))
print(n_samples)
model_inputs = dm.get_sample_by_class(n_samples=n_samples, rawImage=True, retrun_id=True, return_labels=True, split='test')
for k in range(len(model_inputs[-1])):
    model_input = model_inputs[0][k]
    image_id = model_inputs[3][k]
    label = model_inputs[2][k]
    print(image_id)
    explanation, metricRes = explainer.explain(
        model_input,
        model_manager,
        dm,
        segmenter,
        num_samples=1000,
        return_metrics=True
    )
    ground_truth_mask = dm.get_ground_segmentation(img_id=image_id, apply_transform=False)
    if explainer.explainationMethod == 'lime' :
        max_jaccard, best_labels, overlap_proportions = explainer._compute_jaccard(model_input, explanation, ground_truth_mask, dm.gt_msk_clr2cls)
    
    results.append([image_id, label, metricRes, (max_jaccard, best_labels, overlap_proportions)])
    print("#############")

print(results)
with open('results_lime.pkl', 'wb') as f:
    pkl.dump(results, f)
    f.close()