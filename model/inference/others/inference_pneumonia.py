import os
import json
import numpy as np
import pandas as pd
import torchxrayvision as xrv
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    precision_score, recall_score, average_precision_score, 
    matthews_corrcoef, brier_score_loss, precision_recall_curve
)
import sys

# Add path to your utility folder
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from util.PNGDatasetNormalized import PNGFolderDataset

# ==========================================
# 1. HELPER CLASSES
# ==========================================

class XRVCorrectionDataset(torch.utils.data.Dataset):
    """
    Wraps standard XRV datasets (range -1024 to 1024) and 
    rescales them to [0, 1] to match PNGFolderDataset.
    Exposes essential attributes for MergeDataset.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        if hasattr(dataset, 'csv'): self.csv = dataset.csv
        if hasattr(dataset, 'pathologies'): self.pathologies = dataset.pathologies
        if hasattr(dataset, 'labels'): self.labels = dataset.labels 
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample['img']
        # Rescale [-1024, 1024] -> [0, 1]
        img = (img + 1024) / 2048.0
        img = np.clip(img, 0, 1)
        if img.ndim == 2: img = np.expand_dims(img, 0)
        sample['img'] = img
        return sample

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    imgs, labs, idxs = [], [], []
    for b in batch:
        img = torch.tensor(b["img"])
        lab = torch.tensor(b["lab"])
        # Ensure 3-channel
        if img.ndim == 3 and img.shape[0] == 1: img = img.repeat(3, 1, 1)
        elif img.ndim == 2: img = img.unsqueeze(0).repeat(3, 1, 1)
        imgs.append(img)
        labs.append(lab)
        idxs.append(b["idx"])
    return {"img": torch.stack(imgs), "lab": torch.stack(labs), "idx": idxs}

# ==========================================
# 2. DATA LOADING (MATCHING TRAINING SETUP)
# ==========================================
print("Loading Validation Datasets...")

# A. Load XRV Raw
d_chex_raw = xrv.datasets.CheX_Dataset(
    imgpath="/home/varrel/imaging/all_images/CheX_dataset/images_all/CheXpert-v1.0/preprocess7",
    csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/train_chex_selection_split_xrv.csv",
    views=["PA","AP"], unique_patients=False)

d_nih_raw = xrv.datasets.NIH_Dataset(
    imgpath="/home/varrel/imaging/all_images/NIH_dataset/images_all/preprocess7",
    csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/nih_selection_split_xrv.csv",
    views=["PA","AP"], unique_patients=False)

d_mimic_raw = xrv.datasets.MIMIC_Dataset(
    imgpath="/home/varrel/imaging/all_images/MIMIC_dataset/images_all/preprocess7",
    csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/mimic_selection_split_xrv.csv",
    metacsvpath="/home/varrel/imaging/all_images/MIMIC_dataset/mimic-cxr-2.0.0-metadata.csv",
    views=["PA","AP"], unique_patients=False)

d_rsna_raw = xrv.datasets.RSNA_Pneumonia_Dataset(
    imgpath="/home/varrel/imaging/all_images/RSNA_dataset/stage_2_train_images_jpg/preprocess7",
    csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/rsna_selection_split_xrv.csv",
    views=["PA","AP"])

# B. Relabel Raw XRV
classes = ['Pneumonia']
xrv.datasets.relabel_dataset(classes, d_chex_raw)
xrv.datasets.relabel_dataset(classes, d_nih_raw)
xrv.datasets.relabel_dataset(classes, d_mimic_raw)
xrv.datasets.relabel_dataset(classes, d_rsna_raw)

# C. Apply Wrapper (Fix Range to [0, 1])
d_chex_selection = XRVCorrectionDataset(d_chex_raw)
d_nih_selection = XRVCorrectionDataset(d_nih_raw)
d_mimic_selection = XRVCorrectionDataset(d_mimic_raw)
d_rsna_selection = XRVCorrectionDataset(d_rsna_raw)

# D. Load PNG Datasets (Already [0, 1])
pc_png_selection = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/PC_dataset/images_all/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/pc_selection_split_filtered.csv", id_col="ImageID")

vindr_png_selection = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/VinDr_dataset/images_all/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/vin_selection_split.csv", id_col="image_id")

openi_png_selection = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/Openi_dataset/images_all/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/openi_selection_split_filtered.csv", id_col="filename")

pp_png_selection = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/PP_dataset/images_all/Pediatric Chest X-ray Pneumonia/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/pp_selection_split.csv", id_col="image_id"
)

# E. Relabel PNG
xrv.datasets.relabel_dataset(classes, pc_png_selection)
xrv.datasets.relabel_dataset(classes, vindr_png_selection)
xrv.datasets.relabel_dataset(classes, openi_png_selection)
xrv.datasets.relabel_dataset(classes, pp_png_selection)

# F. Merge
dmerge_selection = xrv.datasets.MergeDataset([
    d_nih_selection, d_chex_selection, d_mimic_selection, 
    pc_png_selection, openi_png_selection, vindr_png_selection
])

# G. Loaders
dataset_loaders = { "Merged": torch.utils.data.DataLoader(dmerge_selection, batch_size=16, shuffle=False, num_workers=8, collate_fn=safe_collate)}

# ---- Load best binary models ----
model_paths = {
    "DenseNet": "/home/varrel/image_model_dict/final/fixed_densenet121_pneumonia_model_epoch1.pth",
    "ResNet": "/home/varrel/image_model_dict/final/fixed_resnet50_pneumonia_model_epoch1.pth",
    "VitB": "/home/varrel/image_model_dict/final/fixed_vitb_pneumonia_model_epoch1.pth",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name, path):
    """Load the correct architecture (DenseNet121 or Swin-B) with 1 output."""
    
    if model_name == "DenseNet":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, 1)

    elif model_name == "SwinB":
        weights = models.Swin_B_Weights.IMAGENET1K_V1
        model = models.swin_b(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, 1)

    elif model_name == "ResNet":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)

    elif model_name == "VitB":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)
        in_features = model.heads[-1].in_features
        model.heads = nn.Linear(in_features, 1)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Load checkpoint
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model

models_dict = { name: load_model(name, path) for name, path in model_paths.items() }

# ---- Inference Function ----
def run_inference(model, loader):
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            inputs = batch["img"].to(device)
            labels = batch["lab"].cpu().numpy()
            labels = np.nan_to_num(labels, nan=0.0).astype(int)

            outputs = model(inputs).squeeze(1)
            outputs = torch.clamp(outputs, min=-20, max=20)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return all_probs, all_preds, all_labels

# ---- Helper to calculate Mean ± Std ----
def get_stats(values):
    clean_vals = [v for v in values if not np.isnan(v)]
    if len(clean_vals) == 0:
        return 0.0, 0.0
    return np.mean(clean_vals), np.std(clean_vals)

# ---- Helper to find Optimal F1 Threshold ----
def get_optimal_f1_threshold(y_true, y_probs):
    """Finds the threshold that maximizes F1 score using Precision-Recall Curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Calculate F1 for all thresholds (add epsilon to avoid /0)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    
    best_idx = np.argmax(f1_scores)
    
    # Handle edge case where best_idx is the last element
    if best_idx < len(thresholds):
        best_thresh = thresholds[best_idx]
    else:
        best_thresh = thresholds[-1]
        
    return best_thresh, f1_scores[best_idx]

# ---- Bootstrapping Logic ----
def compute_bootstrap_stats(labels, preds, probs, n_bootstraps=1000):
    boot_stats = {
        "accuracy": [], "f1": [], "precision": [], "recall": [], 
        "mcc": [], "roc_auc": [], "auprc": [], "brier_score": []
    }

    for _ in range(n_bootstraps):
        indices = resample(np.arange(len(labels)), replace=True)
        bs_probs = probs[indices]
        bs_preds = preds[indices]
        bs_labels = labels[indices]
        
        boot_stats["accuracy"].append(accuracy_score(bs_labels, bs_preds))
        boot_stats["precision"].append(precision_score(bs_labels, bs_preds, zero_division=0))
        boot_stats["recall"].append(recall_score(bs_labels, bs_preds, zero_division=0))
        boot_stats["f1"].append(f1_score(bs_labels, bs_preds, zero_division=0))
        boot_stats["mcc"].append(matthews_corrcoef(bs_labels, bs_preds))
        boot_stats["brier_score"].append(brier_score_loss(bs_labels, bs_probs))

        try:
            if len(np.unique(bs_labels)) > 1:
                boot_stats["roc_auc"].append(roc_auc_score(bs_labels, bs_probs))
                boot_stats["auprc"].append(average_precision_score(bs_labels, bs_probs))
            else:
                boot_stats["roc_auc"].append(np.nan)
                boot_stats["auprc"].append(np.nan)
        except ValueError:
            boot_stats["roc_auc"].append(np.nan)
            boot_stats["auprc"].append(np.nan)

    final_metrics = {}
    
    for metric_name, values in boot_stats.items():
        clean_vals = [v for v in values if not np.isnan(v)]
        if not clean_vals: 
            mean_val, std_val = 0.0, 0.0
        else:
            mean_val, std_val = np.mean(clean_vals), np.std(clean_vals)
            
        final_metrics[f"{metric_name}_mean"] = mean_val
        final_metrics[f"{metric_name}_std"] = std_val

    return final_metrics

final_results = {}

# ---- MAIN LOOP ----
for model_name, model in models_dict.items():
    print(f"\n===== Evaluating model: {model_name} =====")
    final_results[model_name] = {}

    # 1. OPTIMIZE THRESHOLD ON MERGED DATASET
    # We prioritize the 'Merged' dataset to find the global optimal threshold.
    # We run this first so we can apply this specific threshold to all sub-datasets.
    print("--- Step 1: Finding Optimal Threshold on Merged Selection Set ---")
    
    if "Merged" in dataset_loaders:
        merged_probs, _, merged_labels = run_inference(model, dataset_loaders["Merged"])
        best_thresh, max_f1 = get_optimal_f1_threshold(merged_labels, merged_probs)
        print(f" -> Optimal Threshold: {best_thresh:.4f} (Max F1 on Merged: {max_f1:.4f})")
    else:
        # Fallback if 'Merged' isn't in keys (though your code has it)
        print("Warning: 'Merged' dataset not found. Defaulting to 0.5")
        best_thresh = 0.5
        merged_probs, merged_labels = None, None

    # 2. EVALUATE ALL DATASETS
    for ds_name, loader in dataset_loaders.items():
        print(f"\n--- Dataset: {ds_name} ---")

        # Optimization: If we are evaluating 'Merged', reuse the data from Step 1
        if ds_name == "Merged" and merged_probs is not None:
            probs, labels = merged_probs, merged_labels
            print("(Using cached inference results)")
        else:
            probs, _, labels = run_inference(model, loader)

        # APPLY OPTIMAL THRESHOLD
        # We recalculate preds using the best_thresh found in Step 1
        preds_opt = (probs >= best_thresh).astype(int)

        # Run Bootstrapping
        metrics = compute_bootstrap_stats(labels, preds_opt, probs, n_bootstraps=1000)

        # Add the threshold used to the results dict for reference
        metrics['threshold_used'] = float(best_thresh)
        final_results[model_name][ds_name] = metrics

        # Print Results
        print(f"{ds_name} Results (Mean ± Std) [Thresh: {best_thresh:.4f}]:")
        print(f"  Acc:       {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
        print(f"  F1:        {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
        print(f"  MCC:       {metrics['mcc_mean']:.4f} ± {metrics['mcc_std']:.4f}")
        print(f"  Precision: {metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}")
        print(f"  Recall:    {metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}")
        print(f"  AUC-ROC:   {metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}")
        print(f"  AUPRC:     {metrics['auprc_mean']:.4f} ± {metrics['auprc_std']:.4f}")
        print(f"  Brier:     {metrics['brier_score_mean']:.4f} ± {metrics['brier_score_std']:.4f}")

# Save results
save_path = "/home/varrel/image_model_dict/final/others_pneumonia_inference.json"
with open(save_path, "w") as f:
    json.dump(final_results, f, indent=4)

print(f"\nSaved per-dataset metrics to: {save_path}")