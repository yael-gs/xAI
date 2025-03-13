import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import os
import matplotlib.pyplot as plt
import io
import torch
from datasetManager import datasetManager
from modelManager import ModelManager
from explainationMethods import MainExplainer, segmentationWrapper
from torchvision import transforms as T

st.set_page_config(
    page_title="XAI Visualization Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Dataset Images"
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None
if 'image_info' not in st.session_state:
    st.session_state.image_info = {}

st.title("🔍 Explainable AI for Medical Images")
st.markdown("Visualize and understand model predictions using LIME and SHAP explanations")

with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Model")
    model_type = st.selectbox(
        "Select model architecture",
        ["vgg16", "resnet50", "swinT"]
    )
    
    model_weights = {
        "vgg16": "vgg16_model_2025-03-06_13-28_3.pth",
        "resnet50": "resnet50_model_2025-03-06_13-31_3.pth",
        "swinT": "swinT_model_2025-03-06_13-35_3.pth"
    }
    
    st.subheader("Explanation Method")
    xai_method = st.radio("Choose explanation method", ["lime", "shap"])
    
    st.subheader("Segmentation")
    seg_method = st.radio("Choose segmentation method", ["default", "sam"])
    
    sam_params = {}
    if seg_method == "sam":
        #sam_file = st.text_input("SAM model path", "sam_vit_b_01ec64.pth")
        sam_file = None
        
        with st.expander("Advanced SAM parameters"):
            sam_params['min_mask_region_area'] = st.slider("Min mask region area", 0, 20, 10)
            sam_params['pred_iou_thresh'] = st.slider("IoU threshold", 0.0, 1.0, 0.60, 0.05)
            sam_params['stability_score_thresh'] = st.slider("Stability threshold", 0.0, 1.0, 0.60, 0.05)
            sam_params['crop_n_layers'] = st.slider("Crop layers", 0, 3, 0)
            sam_params['points_per_side'] = st.segmented_control("Points per side",[32, 64, 128], default=32)
            sam_params['crop_overlap_ratio'] = st.slider("Overlap ratio", 0.0, 1.0, 0.45, 0.05)
    
    st.subheader("Explanation Settings")
    num_samples = st.slider("Number of samples", 20, 2000, 140, step=20)

@st.cache_resource
def get_dataset_manager():
    return datasetManager(dataset=1, batch_size=8, num_workers=4, 
                         transform=T.Compose([T.Resize((224, 224))]))

dm = get_dataset_manager()

@st.cache_resource
def get_model_manager(model_type, weight_file):
    return ModelManager(model_type, 2, weight_file)

try:
    model = get_model_manager(model_type, model_weights[model_type])
    st.sidebar.success(f"Model loaded: {model_type}")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {str(e)}")

tab1, tab2 = st.tabs(["Dataset Images", "Upload Image"])

if tab1._active:
    active_tab = "Dataset Images"
else:
    active_tab = "Upload Image"
st.session_state.current_tab = active_tab

if 'previous_tab' not in st.session_state:
    st.session_state.previous_tab = active_tab
elif st.session_state.previous_tab != active_tab:
    if 'explanation_image' in st.session_state:
        del st.session_state.explanation_image
    if 'prediction' in st.session_state:
        del st.session_state.prediction
    st.session_state.previous_tab = active_tab

with tab1:
    st.header("Select Image from Dataset")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        class_filter = st.selectbox("Filter by class", ["Any", "Mild", "Severe"])
    with col2:
        num_images = st.slider("Number of images to display", 1, 10, 4)
    with col3:
        select_split = st.selectbox("Select split", ["train", "test",'all'], index=2)

    retinopathy_class = None
    if class_filter == "Mild":
        retinopathy_class = 0
    elif class_filter == "Severe":
        retinopathy_class = 1
    
    if st.button("Load Images", key="load_dataset_images"):
        with st.spinner("Loading images..."):
            samples = dm.get_sample_by_class(
                n_samples=num_images,
                retinopathy_class=retinopathy_class,
                split=select_split,
                return_labels=True,
                rawImage=True,
                retrun_id=True
            )
            print("samples",samples)            
            if samples:
                st.session_state.dataset_images = samples[0]
                st.session_state.dataset_labels = samples[1]
                st.session_state.image_filenames = [f"image_{i}" for i in range(len(samples[0]))]
                st.session_state.image_ids = samples[3]

    if 'dataset_images' in st.session_state and 'dataset_labels' in st.session_state:
        st.write(f"Found {len(st.session_state.dataset_images)} images to display")
        
        cols = st.columns(len(st.session_state.dataset_images))
        for i, (col, img, label, image_id) in enumerate(zip(cols, st.session_state.dataset_images, st.session_state.dataset_labels,st.session_state.image_ids)):
            with col:
                st.image(img, caption=f"Class: {label}", use_container_width=True)
                if st.button(f"Select", key=f"select_dataset_{i}"):
                    st.session_state.selected_image = img.copy()
                    st.session_state.source = "dataset"
                    st.session_state.selected_label = label
                    st.session_state.image_info = {
                        'id': image_id,
                        'name': st.session_state.image_filenames[i] if hasattr(st.session_state, 'image_filenames') else f"image_{i}",
                        'class': label
                    }
                    st.session_state.explanation_ready = True
                    st.rerun()

with tab2:
    st.header("Upload Your Own Image")
    
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(img)
        st.image(img_array, caption="Uploaded Image", use_container_width=True)
        if st.button("Use This Image", key="use_uploaded_image"):
            st.session_state.selected_image = img_array.copy()
            st.session_state.source = "upload"
            if 'selected_label' in st.session_state:
                del st.session_state.selected_label
            
            image_name = uploaded_file.name if hasattr(uploaded_file, 'name') else "uploaded_image"
            st.session_state.image_info = {
                'id': 'uploaded_img',
                'name': image_name,
                'class': 'Unknown'
            }
            st.session_state.explanation_ready = True
            st.rerun()

if st.session_state.get('explanation_ready') and st.session_state.get('selected_image') is not None:
    st.header("Generate Explanation")
    st.subheader("Selected Image")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image(st.session_state.selected_image, caption="Selected Image", use_container_width=True)
            
        if 'selected_label' in st.session_state:
            st.info(f"Class: {st.session_state.selected_label}")
        
        if st.button("Generate Explanation", key="generate_explanation_button"):
            with st.spinner("Generating explanation... This may take a while"):
                try:
                    if seg_method == "sam":
                        segmenter = segmentationWrapper('sam', sam_file, sam_params)
                    else:
                        segmenter = segmentationWrapper('default')
                    explainer = MainExplainer(xai_method)
                    start_time = time.time()
                    explanation = explainer.explain(
                        st.session_state.selected_image,
                        model,
                        dm,
                        segmenter,
                        num_samples=num_samples
                    )
                    end_time = time.time()
                    fig = plt.figure(figsize=(12, 6))
                    result = explainer.show_explanation(explanation, st.session_state.selected_image, save=False)
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    st.session_state.explanation_image = buf
                    st.session_state.explanation_time = end_time - start_time
                    if st.session_state.source == "dataset":
                        image = T.ToTensor()(st.session_state.selected_image)
                        image = T.Resize((224, 224))(image).unsqueeze(0)
                    else:
                        image = T.Compose([
                            T.ToTensor(),
                            T.Resize((224, 224))
                        ])(Image.fromarray(st.session_state.selected_image)).unsqueeze(0)
                    prediction = model.inference(image)
                    st.session_state.prediction = prediction
                    if select_split == 'test':
                        try:
                            st.session_state.ground_segmentation = dm.get_ground_segmentation(st.session_state.image_info['id'])
                            fig_seg = plt.figure(figsize=(5, 5))
                            plt.imshow(st.session_state.ground_segmentation)
                            plt.title("Ground Truth Segmentation")
                            plt.tight_layout()
                            plt.axis('off')
                            buf_seg = io.BytesIO()
                            fig_seg.savefig(buf_seg, format='png')
                            buf_seg.seek(0)
                            plt.close(fig_seg)
                            st.session_state.segmentation_image = buf_seg
                        except Exception as e:
                            st.warning(f"Could not load ground truth segmentation: {str(e)}")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error generating explanation: {str(e)}")
    
    with col2:
        if select_split == 'test' and 'ground_segmentation' in st.session_state:
            col21, col22 = st.columns([2,1])
            with col21:
                if 'explanation_image' in st.session_state:
                    st.subheader(f"{xai_method.upper()} Explanation")
                    st.image(st.session_state.explanation_image, use_container_width=True)
                    st.success(f"Explanation generated in {st.session_state.explanation_time:.2f} seconds")
                    
                    if 'prediction' in st.session_state:
                        pred = st.session_state.prediction[0]
                        st.subheader("Model Prediction")
                        df = pd.DataFrame({
                            "Class": ["Mild", "Severe"],
                            "Probability": [f"{pred[0]:.4f}", f"{pred[1]:.4f}"]
                        })
                        st.table(df)
            with col22:
                st.subheader("Ground Truth")
                st.image(st.session_state.segmentation_image, use_container_width=True)
        else:
            if 'explanation_image' in st.session_state:
                st.subheader(f"{xai_method.upper()} Explanation")
                st.image(st.session_state.explanation_image, use_container_width=True)
                st.success(f"Explanation generated in {st.session_state.explanation_time:.2f} seconds")
                
                if 'prediction' in st.session_state:
                    pred = st.session_state.prediction[0]
                    st.subheader("Model Prediction")
                    df = pd.DataFrame({
                        "Class": ["Mild", "Severe"],
                        "Probability": [f"{pred[0]:.4f}", f"{pred[1]:.4f}"]
                    })
                    st.table(df)



