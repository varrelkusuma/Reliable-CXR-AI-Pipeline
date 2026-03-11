# Reliable-CXR-AI-Pipeline
This repository contains the official implementation and research code for [A Step Towards Reliable and Trustworthy Artificial Intelligence for Chest X-Ray Imaging]. The project focuses on implementing pre-processing, modelling, prediction, xai and report generation using 2 CNNs and 2 ViT for 12 different pathologies.

📌 Overview
Four deep learning architectures are compared in this study to find the right model to use with pre-processed CXR images: ResNet50, DenseNet121, Swin Trans-formers (Base), and Vision Transformers (ViT/16-B). All models were trained under identical conditions and parameters for fair comparison, with implementation done using PyTorch library. All models were pre-trained on ImageNet, and BCEWithLogitsLoss is selected as the loss function, with pos_weight assigned to it. Other parameters included AdamW as optimizer, a learning rate of 5e6, a weight decay of 0.01, and a maximum of 30 epochs with early stopping.

📂 Repository Structure
├── main/                       # Core experiment pipeline
│   ├── data-processing/        # Initial data cleaning and preparation
│   ├── image-upscaling/        # Resolution enhancement scripts
│   ├── mimic-relabeling/       # Scripts for label correction/harmonization
│   ├── multiclass-modelling/   # Main model architecture and training logic
│   ├── ood-test/               # Out-of-Distribution (OOD) evaluation
│   ├── report-generation/      # Automated generation of findings/metrics
│   ├── visual-diagram/         # Scripts to generate figures for the paper
│   └── xai-implementation/     # Explainable AI (SHAP, Grad-CAM, etc.)
├── model/
│   ├── training/               # Training loops and configuration
│   └── inference/              # Scripts for running predictions and results
├── npy/                        # Saved weights (pos_weight, sample_weight)
├── split/                      # Standardized data splits for reproducibility
│   ├── train.csv
│   ├── selection.csv
│   └── test.csv
├── util/                       # Helper functions and shared utilities
└── requirements.txt            # Software dependencies

🚀 Getting Started
1. Installation
Clone the repository and install the required dependencies:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt

2. Data Preparation
To ensure reproducibility, we provide standardized splits in the /split folder.
You could use the code for training and inference provided in the /model folder..

🔍 Research & Analysis
Out-of-Distribution (OOD) Testing
We evaluate the model's generalizability across different datasets. Refer to main/ood-test/ for the evaluation suite.

Explainable AI (XAI)
We evaluate a population-level activation map that you could see on main/xai-implementation/ for the final result

Report Generation
We create a code to generate radiology report using the 12 Swin-B models trained in this study. Refer to main/report-generation/ for the result.
