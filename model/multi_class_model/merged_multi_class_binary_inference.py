import os
import numpy as np
import torchxrayvision as xrv
import torch
import torchvision.models as models
import torch.nn as nn
import torch
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from util.PNGDataset import PNGFolderDataset

def safe_collate(batch):
    batch = [b for b in batch if b is not None]

    imgs = []
    labs = []

    for b in batch:
        img = torch.tensor(b["img"])
        lab = torch.tensor(b["lab"])

        # Ensure image is 3-channel
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.repeat(3, 1, 1)  # grayscale → RGB
        elif img.ndim == 2:
            img = img.unsqueeze(0).repeat(3, 1, 1)  # 2D → 3D RGB
        elif img.shape[0] == 3:
            pass  # already RGB
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        imgs.append(img)
        labs.append(lab)

    return {
        "img": torch.stack(imgs),
        "lab": torch.stack(labs),
        "idx": [b["idx"] for b in batch],
    }

"""--- Load Processed Datasets ---"""
d_chex_selection = xrv.datasets.CheX_Dataset(imgpath="/home/varrel/imaging/all_images/CheX_dataset/images_all/CheXpert-v1.0/preprocess2",
                                         csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/train_chex_selection_split_xrv.csv",
                                         views=["PA","AP"], unique_patients=False)
d_nih_selection = xrv.datasets.NIH_Dataset(imgpath="/home/varrel/imaging/all_images/NIH_dataset/images_all/preprocess3",
                                           csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/nih_selection_split_xrv.csv",
                                           views=["PA","AP"], unique_patients=False)
d_mimic_selection = xrv.datasets.MIMIC_Dataset(imgpath="/home/varrel/imaging/all_images/MIMIC_dataset/images_all/preprocess2",
                                     csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/mimic_selection_split_xrv.csv",
                                     metacsvpath="/home/varrel/imaging/all_images/MIMIC_dataset/mimic-cxr-2.0.0-metadata.csv",
                                     views=["PA","AP"], unique_patients=False)

pc_png_selection = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/PC_dataset/images_all/preprocess2",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/pc_selection_split_filtered.csv",
    id_col="ImageID"
)

vindr_png_selection = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/VinDr_dataset/images_all/preprocess2",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/vin_selection_split.csv",
    id_col="image_id"
)

openi_png_selection = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/Openi_dataset/images_all/preprocess2",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/openi_selection_split.csv",
    id_col="filename"
)


classes = ['Edema','Consolidation','Pleural Effusion','Cardiomegaly','Atelectasis','Lung Opacity']

d_mimic_selection.pathologies = ["Pleural Effusion" if p == "Effusion" else p for p in d_mimic_selection.pathologies]
d_chex_selection.pathologies = ["Pleural Effusion" if p == "Effusion" else p for p in d_chex_selection.pathologies]
d_nih_selection.pathologies = ["Pleural Effusion" if p == "Effusion" else p for p in d_nih_selection.pathologies]
d_nih_selection.pathologies = ["Lung Opacity" if p == "Infiltration" else p for p in d_nih_selection.pathologies]

xrv.datasets.relabel_dataset(classes, openi_png_selection)
xrv.datasets.relabel_dataset(classes, d_nih_selection)
xrv.datasets.relabel_dataset(classes, d_chex_selection)
xrv.datasets.relabel_dataset(classes, d_mimic_selection)
xrv.datasets.relabel_dataset(classes, pc_png_selection)
xrv.datasets.relabel_dataset(classes, vindr_png_selection)

dmerge_selection = xrv.datasets.MergeDataset([d_chex_selection, d_mimic_selection, pc_png_selection, openi_png_selection, vindr_png_selection, d_nih_selection])
selection_loader = torch.utils.data.DataLoader(dmerge_selection, batch_size=16, shuffle=False, num_workers=8, collate_fn=safe_collate)

# ---- Load best binary models ----
model_paths = {
    "Edema": "/home/varrel/image_model_dict/edema_best_model.pth",
    "Consolidation": "/home/varrel/image_model_dict/consolidation_best_model.pth",
    "Pleural Effusion": "/home/varrel/image_model_dict/effusion_best_model.pth",
    "Cardiomegaly": "/home/varrel/image_model_dict/cardiomegaly_best_model.pth",
    "Atelectasis": "/home/varrel/image_model_dict/atelectasis_best_model.pth",
    "Lung Opacity": "/home/varrel/image_model_dict/opacity_best_model.pth"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_dict = {}
for cls, path in model_paths.items():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    models_dict[cls] = model

# Run inference
all_probs = {cls: [] for cls in classes}
all_preds = {cls: [] for cls in classes}
all_labels = []

with torch.no_grad():
    for batch in tqdm(selection_loader, desc="Inference"):
        inputs = batch["img"].to(device)
        labels = batch["lab"].cpu().numpy()
        labels = np.nan_to_num(labels, nan=0.0)
        all_labels.append(labels)
        for cls, model in models_dict.items():
            outputs = model(inputs).squeeze(1)
            outputs = torch.clamp(outputs, min=-20, max=20)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs[cls].extend(probs)
            all_preds[cls].extend(preds)

# Convert to matrices [N_samples, 6]
probs_matrix = np.column_stack([all_probs[cls] for cls in classes])
preds_matrix = np.column_stack([all_preds[cls] for cls in classes])
labels_matrix = np.vstack(all_labels)

print("Probabilities shape:", probs_matrix.shape)
print("Predictions shape:", preds_matrix.shape)
print("Labels shape:", labels_matrix.shape)

# ---- Macro metrics ----
macro_acc = accuracy_score(labels_matrix, preds_matrix)
macro_f1  = f1_score(labels_matrix, preds_matrix, average="macro")
try:
    macro_rocauc = roc_auc_score(labels_matrix, probs_matrix, average="macro")
except ValueError:
    macro_rocauc = float("nan")

print(f"Macro Accuracy={macro_acc:.4f}, F1={macro_f1:.4f}, ROC AUC={macro_rocauc:.4f}")

# ---- Per-class metrics ----
per_class_acc = {}
per_class_f1 = {}
per_class_rocauc = {}

for i, cls in enumerate(classes):
    per_class_acc[cls] = accuracy_score(labels_matrix[:, i], preds_matrix[:, i])
    try:
        per_class_f1[cls] = f1_score(labels_matrix[:, i], preds_matrix[:, i])
    except ValueError:
        per_class_f1[cls] = float("nan")
    try:
        per_class_rocauc[cls] = roc_auc_score(labels_matrix[:, i], probs_matrix[:, i])
    except ValueError:
        per_class_rocauc[cls] = float("nan")

print("\nPer-class metrics:")
for cls in classes:
    print(f"{cls}: Acc={per_class_acc[cls]:.4f}, F1={per_class_f1[cls]:.4f}, ROC AUC={per_class_rocauc[cls]:.4f}")

# ---- Save results ----
np.save("/home/varrel/image_model_dict/multimodel_probs.npy", probs_matrix)
np.save("/home/varrel/image_model_dict/multimodel_preds.npy", preds_matrix)

results = {
    "macro_acc": macro_acc,
    "macro_f1": macro_f1,
    "macro_rocauc": macro_rocauc,
    "per_class_acc": per_class_acc,
    "per_class_f1": per_class_f1,
    "per_class_rocauc": per_class_rocauc
}

# Save metrics JSON
with open("/home/varrel/image_model_dict/multimodel_binary_inference_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nMetrics saved to /home/varrel/image_model_dict/multimodel_binary_inference_metrics.json")