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

# B. Relabel Raw XRV
classes = ['Pleural Effusion']
d_mimic_raw.pathologies = [
    "Pleural Effusion" if p == "Effusion" else p
    for p in d_mimic_raw.pathologies
]
d_chex_raw.pathologies = [
    "Pleural Effusion" if p == "Effusion" else p
    for p in d_chex_raw.pathologies
]
d_nih_raw.pathologies = [
    "Pleural Effusion" if p == "Effusion" else p
    for p in d_nih_raw.pathologies
]
xrv.datasets.relabel_dataset(classes, d_chex_raw)
xrv.datasets.relabel_dataset(classes, d_nih_raw)
xrv.datasets.relabel_dataset(classes, d_mimic_raw)

# C. Apply Wrapper (Fix Range to [0, 1])
d_chex_selection = XRVCorrectionDataset(d_chex_raw)
d_nih_selection = XRVCorrectionDataset(d_nih_raw)
d_mimic_selection = XRVCorrectionDataset(d_mimic_raw)

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

# E. Relabel PNG
xrv.datasets.relabel_dataset(classes, pc_png_selection)
xrv.datasets.relabel_dataset(classes, vindr_png_selection)
xrv.datasets.relabel_dataset(classes, openi_png_selection)

# F. Merge
dmerge_selection = xrv.datasets.MergeDataset([
    d_nih_selection, d_chex_selection, d_mimic_selection, 
    pc_png_selection, openi_png_selection, vindr_png_selection
])

# G. Loaders
dataset_loaders = {
    "NIH": torch.utils.data.DataLoader(d_nih_selection, batch_size=16, shuffle=False, num_workers=8, collate_fn=safe_collate),
    "OpenI": torch.utils.data.DataLoader(openi_png_selection, batch_size=16, shuffle=False, num_workers=8, collate_fn=safe_collate),
    "CheXpert": torch.utils.data.DataLoader(d_chex_selection, batch_size=16, shuffle=False, num_workers=8, collate_fn=safe_collate),
    "MIMIC": torch.utils.data.DataLoader(d_mimic_selection, batch_size=16, shuffle=False, num_workers=8, collate_fn=safe_collate),
    "PC": torch.utils.data.DataLoader(pc_png_selection, batch_size=16, shuffle=False, num_workers=8, collate_fn=safe_collate),
    "VinDR": torch.utils.data.DataLoader(vindr_png_selection, batch_size=16, shuffle=False, num_workers=8, collate_fn=safe_collate),
    "Merged": torch.utils.data.DataLoader(dmerge_selection, batch_size=16, shuffle=False, num_workers=8, collate_fn=safe_collate)
}

# ---- Load best binary models ----
model_paths = {
    "SwinB": "/home/varrel/image_model_dict/final/fixed_swinb_effusion_model_epoch5.pth",
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

# ---- MAIN EXECUTION LOOP (BINARY ADJUSTED) ----
final_results = {}
global_threshold = None # Variable name changed to singular for binary

# A. FIND GLOBAL THRESHOLD
print("\n=== Step 1: Finding Global Threshold (Merged Selection) ===")
if "Merged" in dataset_loaders:
    # Note: run_inference returns (probs, _, labels) or (probs, labels) depending on your version.
    # We use `_` to safely ignore the middle value if it exists, or handle 2 return values.
    # SAFEST WAY:
    merged_results = run_inference(list(models_dict.values())[0], dataset_loaders["Merged"])
    if len(merged_results) == 3:
        merged_probs, _, merged_labels = merged_results
    else:
        merged_probs, merged_labels = merged_results
    
    # Calculate threshold (Binary expects 1D arrays or single column)
    # We flatten inputs just in case shape is (N, 1)
    global_threshold, _ = get_optimal_f1_threshold(merged_labels.flatten(), merged_probs.flatten())
    print(f"Global Threshold Found: {global_threshold:.4f}")
else:
    print("Warning: 'Merged' dataset not found. Defaulting to 0.5")
    global_threshold = 0.5


# B. EVALUATE ALL DATASETS
for model_name, model in models_dict.items():
    print(f"\n===== Evaluating model: {model_name} =====")
    final_results[model_name] = {}

    for ds_name, loader in dataset_loaders.items():
        print(f"\n--- Dataset: {ds_name} ---")
        final_results[model_name][ds_name] = {}

        # 1. Inference
        if ds_name == "Merged" and "Merged" in dataset_loaders:
            probs, labels = merged_probs, merged_labels
            print("(Using cached inference results)")
        else:
            inf_res = run_inference(model, loader)
            if len(inf_res) == 3:
                probs, _, labels = inf_res
            else:
                probs, labels = inf_res
            
        # ----------------------------------------------------
        # SCENARIO 1: GLOBAL THRESHOLD (Robustness)
        # ----------------------------------------------------
        preds_global = (probs >= global_threshold).astype(int)
        
        # Binary compute_bootstrap_stats expects inputs, not multi-label inputs
        metrics_global = compute_bootstrap_stats(labels, preds_global, probs, n_bootstraps=1000)
        
        # Save
        final_results[model_name][ds_name]["global"] = metrics_global
        final_results[model_name][ds_name]["global"]["threshold_used"] = float(global_threshold)
        
        # BINARY PRINTING (Flat keys)
        print(f"\n[GLOBAL Threshold {global_threshold:.4f}] Results:")
        print(f"  Acc:       {metrics_global['accuracy_mean']:.4f} ± {metrics_global['accuracy_std']:.4f}")
        print(f"  F1:        {metrics_global['f1_mean']:.4f} ± {metrics_global['f1_std']:.4f}")
        print(f"  MCC:       {metrics_global['mcc_mean']:.4f} ± {metrics_global['mcc_std']:.4f}")
        print(f"  Precision: {metrics_global['precision_mean']:.4f} ± {metrics_global['precision_std']:.4f}")
        print(f"  Recall:    {metrics_global['recall_mean']:.4f} ± {metrics_global['recall_std']:.4f}")
        print(f"  AUC-ROC:   {metrics_global['roc_auc_mean']:.4f} ± {metrics_global['roc_auc_std']:.4f}")
        print(f"  AUPRC:     {metrics_global['auprc_mean']:.4f} ± {metrics_global['auprc_std']:.4f}")
        print(f"  Brier:     {metrics_global['brier_score_mean']:.4f} ± {metrics_global['brier_score_std']:.4f}")

        # ----------------------------------------------------
        # SCENARIO 2: LOCAL THRESHOLD (Capacity)
        # ----------------------------------------------------
        local_threshold, _ = get_optimal_f1_threshold(labels.flatten(), probs.flatten())
        preds_local = (probs >= local_threshold).astype(int)
        metrics_local = compute_bootstrap_stats(labels, preds_local, probs, n_bootstraps=1000)
        
        # Save
        final_results[model_name][ds_name]["local"] = metrics_local
        final_results[model_name][ds_name]["local"]["threshold_used"] = float(local_threshold)

        print(f"\n[LOCAL Threshold {local_threshold:.4f}] Results:")
        print(f"  Acc:       {metrics_local['accuracy_mean']:.4f} ± {metrics_local['accuracy_std']:.4f}")
        print(f"  F1:        {metrics_local['f1_mean']:.4f} ± {metrics_local['f1_std']:.4f}")
        # Only print summary for local to save space, or repeat full block if desired
        print(f"  AUC-ROC:   {metrics_local['roc_auc_mean']:.4f} ± {metrics_local['roc_auc_std']:.4f}")

# Save results
save_path = "/home/varrel/image_model_dict/final/fixed_effusion_inference.json"
with open(save_path, "w") as f:
    json.dump(final_results, f, indent=4)

print(f"\nSaved per-dataset metrics to: {save_path}")