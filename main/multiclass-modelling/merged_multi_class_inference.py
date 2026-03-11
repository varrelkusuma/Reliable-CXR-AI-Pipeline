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

# ---- 1. SETUP & HELPER CLASSES ----
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from util.PNGDatasetNormalized import PNGFolderDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class XRVCorrectionDataset(torch.utils.data.Dataset):
    """
    Wraps standard XRV datasets (range -1024 to 1024) and 
    rescales them to [0, 1] to match PNGFolderDataset.
    Exposes essential attributes for MergeDataset.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        # Expose attributes required by xrv.datasets.MergeDataset
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
        
        # Ensure correct dimensions
        if img.ndim == 2: 
            img = np.expand_dims(img, 0)
            
        sample['img'] = img
        return sample

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    imgs, labs, idxs = [], [], []
    
    for b in batch:
        img = torch.tensor(b["img"])
        lab = torch.tensor(b["lab"])

        # Ensure 3-channel
        if img.ndim == 3 and img.shape[0] == 1: 
            img = img.repeat(3, 1, 1)
        elif img.ndim == 2: 
            img = img.unsqueeze(0).repeat(3, 1, 1)
            
        imgs.append(img)
        labs.append(lab)
        idxs.append(b["idx"])
        
    return {
        "img": torch.stack(imgs), 
        "lab": torch.stack(labs), 
        "idx": idxs
    }

# ---- 2. DATA LOADING ----
print("Loading Validation Datasets...")

# A. Load Raw XRV Datasets
d_chex_raw = xrv.datasets.CheX_Dataset(
    imgpath="/home/varrel/imaging/all_images/CheX_dataset/images_all/CheXpert-v1.0/preprocess7",
    csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/train_chex_selection_split_xrv.csv",
    views=["PA","AP"], unique_patients=False
)

d_nih_raw = xrv.datasets.NIH_Dataset(
    imgpath="/home/varrel/imaging/all_images/NIH_dataset/images_all/preprocess7",
    csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/nih_selection_split_xrv.csv",
    views=["PA","AP"], unique_patients=False
)

d_mimic_raw = xrv.datasets.MIMIC_Dataset(
    imgpath="/home/varrel/imaging/all_images/MIMIC_dataset/images_all/preprocess7",
    csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/mimic_selection_split_xrv.csv",
    metacsvpath="/home/varrel/imaging/all_images/MIMIC_dataset/mimic-cxr-2.0.0-metadata.csv",
    views=["PA","AP"], unique_patients=False
)

# B. Define Multi-Label Classes & Fix Pathologies
classes = [
    "Edema", "Consolidation", "Pleural Effusion", 
    "Cardiomegaly", "Atelectasis", "Lung Opacity"
]

# Rename specific pathologies to match the target classes
d_mimic_raw.pathologies = ["Pleural Effusion" if p == "Effusion" else p for p in d_mimic_raw.pathologies]
d_chex_raw.pathologies = ["Pleural Effusion" if p == "Effusion" else p for p in d_chex_raw.pathologies]
d_nih_raw.pathologies = ["Pleural Effusion" if p == "Effusion" else p for p in d_nih_raw.pathologies]
d_nih_raw.pathologies = ["Lung Opacity" if p == "Infiltration" else p for p in d_nih_raw.pathologies]

# Relabel Raw Datasets
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
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/pc_selection_split_filtered.csv", 
    id_col="ImageID"
)

vindr_png_selection = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/VinDr_dataset/images_all/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/vin_selection_split.csv", 
    id_col="image_id"
)

openi_png_selection = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/Openi_dataset/images_all/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/openi_selection_split.csv", 
    id_col="filename"
)

# E. Relabel PNG Datasets
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


# ---- 3. MODEL LOADING ----
model_path = "/home/varrel/image_model_dict/swinb_multilabel_model_epoch5.pth"

print(f"Loading multi-label model from {model_path}...")
model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
in_features = model.head.in_features
model.head = nn.Linear(in_features, 6)
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

# ---- 4. INFERENCE HELPER ----
def run_inference(model, loader):
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            inputs = batch["img"].to(device)
            labels = batch["lab"].cpu().numpy()
            labels = np.nan_to_num(labels, nan=0.0)
            all_labels.append(labels)

            outputs = model(inputs) 
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)

    probs_matrix = np.vstack(all_probs)
    labels_matrix = np.vstack(all_labels)
    return probs_matrix, labels_matrix

# ---- 5. METRIC HELPERS ----
def get_optimal_thresholds(y_true, y_probs, class_names):
    """
    Finds the best threshold for F1 score for EACH class independently.
    Includes safety checks for empty classes.
    """
    best_thresholds = []
    print("\n--- Optimizing Thresholds per Class ---") 
    for i, cls in enumerate(class_names):
        if np.sum(y_true[:, i]) == 0:
            best_thresholds.append(0.5)
            continue

        precisions, recalls, thresholds = precision_recall_curve(y_true[:, i], y_probs[:, i])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        
        f1_scores = np.nan_to_num(f1_scores)
        best_idx = np.argmax(f1_scores)
        
        if best_idx < len(thresholds):
            best_thresh = thresholds[best_idx]
        else:
            best_thresh = thresholds[-1]
            
        best_thresholds.append(best_thresh)
        
    return np.array(best_thresholds)

def compute_bootstrap_stats(labels, preds, probs, n_bootstraps=1000):
    boot_stats = {
        "macro_acc": [], "macro_f1": [], "macro_rocauc": [],
        "macro_precision": [], "macro_recall": [], "macro_mcc": [], 
        "macro_auprc": [], "macro_brier": [],
        "per_class": {cls: {"acc": [], "f1": [], "rocauc": [], "precision": [], "recall": [], "mcc": [], "auprc": [], "brier": []} for cls in classes}
    }

    n_samples = len(labels)
    
    for _ in range(n_bootstraps):
        indices = resample(np.arange(n_samples), replace=True)
        bs_probs = probs[indices]
        bs_preds = preds[indices]
        bs_labels = labels[indices]
        
        # Macro
        boot_stats["macro_acc"].append(accuracy_score(bs_labels, bs_preds))
        boot_stats["macro_f1"].append(f1_score(bs_labels, bs_preds, average="macro", zero_division=0))
        boot_stats["macro_precision"].append(precision_score(bs_labels, bs_preds, average="macro", zero_division=0))
        boot_stats["macro_recall"].append(recall_score(bs_labels, bs_preds, average="macro", zero_division=0))
        
        try:
            boot_stats["macro_rocauc"].append(roc_auc_score(bs_labels, bs_probs, average="macro"))
            boot_stats["macro_auprc"].append(average_precision_score(bs_labels, bs_probs, average="macro"))
        except ValueError:
            boot_stats["macro_rocauc"].append(np.nan)
            boot_stats["macro_auprc"].append(np.nan)

        mcc_list, brier_list = [], []
        
        for i, cls in enumerate(classes):
            y_true_cls = bs_labels[:, i]
            y_pred_cls = bs_preds[:, i]
            y_prob_cls = bs_probs[:, i]
            
            # Simple metrics
            boot_stats["per_class"][cls]["acc"].append(accuracy_score(y_true_cls, y_pred_cls))
            boot_stats["per_class"][cls]["f1"].append(f1_score(y_true_cls, y_pred_cls, zero_division=0))
            boot_stats["per_class"][cls]["precision"].append(precision_score(y_true_cls, y_pred_cls, zero_division=0))
            boot_stats["per_class"][cls]["recall"].append(recall_score(y_true_cls, y_pred_cls, zero_division=0))
            
            mcc = matthews_corrcoef(y_true_cls, y_pred_cls)
            brier = brier_score_loss(y_true_cls, y_prob_cls)
            
            boot_stats["per_class"][cls]["mcc"].append(mcc)
            boot_stats["per_class"][cls]["brier"].append(brier)
            mcc_list.append(mcc)
            brier_list.append(brier)

            # Probabilistic metrics
            try:
                if len(np.unique(y_true_cls)) > 1:
                    boot_stats["per_class"][cls]["rocauc"].append(roc_auc_score(y_true_cls, y_prob_cls))
                    boot_stats["per_class"][cls]["auprc"].append(average_precision_score(y_true_cls, y_prob_cls))
                else:
                    boot_stats["per_class"][cls]["rocauc"].append(np.nan)
                    boot_stats["per_class"][cls]["auprc"].append(np.nan)
            except ValueError:
                boot_stats["per_class"][cls]["rocauc"].append(np.nan)
                boot_stats["per_class"][cls]["auprc"].append(np.nan)
                
        boot_stats["macro_mcc"].append(np.mean(mcc_list))
        boot_stats["macro_brier"].append(np.mean(brier_list))

    final_metrics = {"macro": {}, "per_class": {}}
    
    def get_stats(values):
        clean = [v for v in values if not np.isnan(v)]
        if not clean: return np.nan, np.nan 
        return float(np.mean(clean)), float(np.std(clean))
    
    for metric in ["acc", "f1", "precision", "recall", "mcc", "rocauc", "auprc", "brier"]:
        m, s = get_stats(boot_stats[f"macro_{metric}"])
        final_metrics["macro"][f"{metric}_mean"] = m
        final_metrics["macro"][f"{metric}_std"] = s
        
    for cls in classes:
        final_metrics["per_class"][cls] = {}
        for metric in ["acc", "f1", "precision", "recall", "mcc", "rocauc", "auprc", "brier"]:
            m, s = get_stats(boot_stats["per_class"][cls][metric])
            final_metrics["per_class"][cls][f"{metric}_mean"] = m
            final_metrics["per_class"][cls][f"{metric}_std"] = s
            
    return final_metrics

# ---- 6. MAIN EXECUTION LOOP ----
final_results = {}
cached_merged_probs = None
cached_merged_labels = None

if "Merged" in dataset_loaders:
    print("Pre-loading Merged inference results (for cache)...")
    cached_merged_probs, cached_merged_labels = run_inference(model, dataset_loaders["Merged"])

print("\n=== Evaluating Datasets (With Local Thresholds) ===")

for ds_name, loader in dataset_loaders.items():
    print(f"\n--- Dataset: {ds_name} ---")
    
    # 1. Get Inference Results
    if ds_name == "Merged" and cached_merged_probs is not None:
        probs, labels = cached_merged_probs, cached_merged_labels
        print("(Using cached inference results)")
    else:
        probs, labels = run_inference(model, loader)
        
    # 2. LOCAL OPTIMIZATION: Calculate Thresholds for THIS Dataset
    # This ensures "Fair Comparison" with binary models that used local thresholds
    local_thresholds = get_optimal_thresholds(labels, probs, classes)
    
    # 3. Apply Local Thresholds
    # Broadcast: (N, 6) >= (6,)
    preds_opt = (probs >= local_thresholds).astype(int)
    
    # 4. Run Bootstrapping
    metrics = compute_bootstrap_stats(labels, preds_opt, probs, n_bootstraps=1000)
    
    # 5. Store Results & Thresholds
    metrics["thresholds_used"] = {cls: float(t) for cls, t in zip(classes, local_thresholds)}
    metrics["threshold_type"] = "Local (Per-Dataset Optimization)"
    
    final_results[ds_name] = metrics
    
    # Print Summary
    print(f"  Macro F1:  {metrics['macro']['f1_mean']:.4f} ± {metrics['macro']['f1_std']:.4f}")
    print(f"  Macro AUC: {metrics['macro']['rocauc_mean']:.4f} ± {metrics['macro']['rocauc_std']:.4f}")
    print(f"  Macro MCC: {metrics['macro']['mcc_mean']:.4f} ± {metrics['macro']['mcc_std']:.4f}")

# ---- 7. SAVE RESULTS ----
# Save matrices (Merged only)
if cached_merged_probs is not None:
    np.save("/home/varrel/image_model_dict/swinb_multilabel_merged_probs_local.npy", cached_merged_probs)

output_path = "/home/varrel/image_model_dict/swinb_multilabel_inference_metrics_local_opt.json"
with open(output_path, "w") as f:
    json.dump(final_results, f, indent=4)

print(f"\nMetrics saved to {output_path}")