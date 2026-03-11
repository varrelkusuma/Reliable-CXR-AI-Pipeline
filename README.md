# Reliable-CXR-AI-Pipeline
This repository contains the official implementation and research code for [A Step Towards Reliable and Trustworthy Artificial Intelligence for Chest X-Ray Imaging]. The project focuses on implementing pre-processing, modelling, prediction, xai and report generation using 2 CNNs and 2 ViT for 12 different pathologies.

## 📌 Overview  
Four deep learning architectures are compared in this study to find the right model to use with pre-processed CXR images: ResNet50, DenseNet121, Swin Trans-formers (Base), and Vision Transformers (ViT/16-B). All models were trained under identical conditions and parameters for fair comparison, with implementation done using PyTorch library. All models were pre-trained on ImageNet, and BCEWithLogitsLoss is selected as the loss function, with pos_weight assigned to it. Other parameters included AdamW as optimizer, a learning rate of 5e6, a weight decay of 0.01, and a maximum of 30 epochs with early stopping.

## 📂 Repository Structure  
```text
├── main/                       # Primary experiment modules
│   ├── data-processing/        # Initial cleaning and formatting
│   ├── image-preprocessing/    # Create pre-processed image
│   ├── image-upscaling/        # Resolution enhancement scripts
│   ├── mimic-relabeling/       # Label harmonization for MIMIC-CXR
│   ├── multiclass-modelling/   # Model training logic
│   ├── ood-test/               # Out-of-Distribution evaluation
│   ├── report-generation/      # Automated metric reporting
│   ├── visual-diagram/         # Figure generation for papers
│   └── xai-implementation/     # Explainability (Grad-CAM, SHAP, etc.)
├── model/                      # Training and Inference logic
│   ├── training/               # Training loops and scripts
│   └── inference/              # Prediction and result analysis
├── npy/                        # Weights and pre-calculated statistics
├── split/                      # Standardized data partitions
│   ├── train.csv               # Training set
│   ├── selection.csv           # Validation/Selection set
│   └── test.csv                # Standardized test set
├── util/                       # Shared helper functions
└── requirements.txt            # Python dependencies
```

## 🚀 Getting Started  
1. Installation
Clone the repository and install the required dependencies:
```
git clone https://github.com/varrelkusuma/Reliable-CXR-AI-Pipeline
cd Reliable-CXR-AI-Pipeline
pip install -r requirements.txt
```
2. Data Preparation
To ensure reproducibility, we provide standardized splits in the /split folder.
You could use the code for training and inference provided in the /model folder.

## 🔍 Research & Analysis  
1. Out-of-Distribution (OOD) Testing
We evaluate the model's generalizability across different datasets. Refer to main/ood-test/ for the evaluation suite.

2. Explainable AI (XAI)
We evaluate a population-level activation map that you could see on main/xai-implementation/ for the final result

3. Report Generation
We create a code to generate radiology report using the 12 Swin-B models trained in this study. Refer to main/report-generation/ for the result.
