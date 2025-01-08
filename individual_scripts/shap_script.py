import json

import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import matplotlib.pyplot as plt
import shap


def main():
    #load pre-trained model and data
    model = ResNet50(weights="imagenet")
    X, y = shap.datasets.imagenet50()

    # getting ImageNet 1000 class names
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    with open(shap.datasets.cache(url)) as file:
        class_names = [v[1] for v in json.load(file).values()]

    def f(x):
        tmp = x.copy()
        preprocess_input(tmp)
        return model(tmp)

   


    # define a masker that is used to mask out partitions of the input image.
    masker = shap.maskers.Image("inpaint_telea", X[0].shape)

    # create an explainer with model and image masker
    explainer = shap.Explainer(f, masker, output_names=class_names)

    # here we explain two images using 500 evaluations of the underlying model to estimate the SHAP values
    shap_values = explainer(X[1:3], max_evals=100, batch_size=50, outputs=shap.Explanation.argsort.flip[:4])


    

    # output with shap values
    fig = plt.figure()             # <---- initialize figure `fig`
    shap.image_plot(shap_values)
    save_path = 'shap_summary_plot.png'
    fig.savefig(save_path)         # <---- save `fig` (not current figure)
    plt.close(fig) 
    print("fig saved")


if __name__ == "__main__":
    main()
    
