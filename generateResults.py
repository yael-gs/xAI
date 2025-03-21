import os
import pickle as pkl
from datasetManager import datasetManager
from modelManager import ModelManager
from explainationMethods import MainExplainer, segmentationWrapper
import torchvision.transforms as T


USER = 'YAEL'

samParams = {
    'min_mask_region_area': 4,                  
    'pred_iou_thresh': 0.60,                  
    'stability_score_thresh': 0.80,             
    'crop_n_layers': 2,                         
    'crop_overlap_ratio': 0.45,                 
    'points_per_batch': 8,                      
    'crop_n_points_downscale_factor': 1,        
    'box_nms_thresh': 0.8,                      
    'points_per_side': 32
}


AttributionDict = {
    "BIGGY" : ('lime','sam'),
    "COCO" : ('shap','sam'),
    "YAEL" : ('lime','grid'),
    "JOJO" : ('shap','grid'),
    "FILIPE" : ('lime','default'),
}
#Il restera ('shap','default'), ('gradcam','default')

Attribution = AttributionDict[USER]
# Attribution = ('lime','grid') #Modifier ici pour custom 

LIndex = ["IDRiD_012" , "IDRiD_020" , "IDRiD_027" , "IDRiD_028" , "IDRiD_029" , "IDRiD_048" , "IDRiD_053" , "IDRiD_057" , "IDRiD_058" , "IDRiD_059"]
results = []
model_manager = ModelManager('vgg16', 2, "vgg16_model_2025-03-06_13-28_3.pth")
segmenter = segmentationWrapper(Attribution[1],params={'grid_size': (15, 15)} if Attribution[1] == 'grid' else (samParams if Attribution[1] == 'sam' else {}))

explainer = MainExplainer(Attribution[0], metrics = ['ROAD', 'FAITHFULNESS', 'COMPLEXITY'])

dm = datasetManager(dataset=1, batch_size=8, num_workers=4, transform=T.Compose([T.Resize((224, 224))]))
# n_samples = len(os.listdir('dataset/test/images'))
# print(n_samples)
model_inputs = dm.get_sample_by_ID(LIndex=LIndex, rawImage=True, retrun_id=True, return_labels=True, split='test')
for k in range(len(model_inputs[-1])):
    print(k/len(model_inputs[-1]))
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
    else:
        max_jaccard, best_labels, overlap_proportions = 0, 0, 0

    results.append([image_id, label, metricRes, (max_jaccard, best_labels, overlap_proportions)])
    print("#############")

print(results)
with open(f'results_{Attribution[0]}_{Attribution[1]}.pkl', 'wb') as f:
    pkl.dump(results, f)
    f.close()