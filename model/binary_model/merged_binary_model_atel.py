import os
import json
import numpy as np
import torchxrayvision as xrv
import torch
import gc
import torch.optim as optim
import time
import torchvision.models as models
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import WeightedRandomSampler

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
d_rsna_train = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath="/home/varrel/imaging/all_images/RSNA_dataset/stage_2_train_images_jpg/preprocess2/",
                                                   csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/rsna_train_split_xrv.csv",
                                                   views=["PA","AP"])
d_rsna_selection = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath="/home/varrel/imaging/all_images/RSNA_dataset/stage_2_train_images_jpg/preprocess2/",
                                                       csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/rsna_selection_split_xrv.csv",
                                                       views=["PA","AP"])
d_chex_train = xrv.datasets.CheX_Dataset(imgpath="/home/varrel/imaging/all_images/CheX_dataset/images_all/CheXpert-v1.0/preprocess2",
                                         csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/train_chex_train_split_xrv.csv",
                                         views=["PA","AP"], unique_patients=False)
d_chex_selection = xrv.datasets.CheX_Dataset(imgpath="/home/varrel/imaging/all_images/CheX_dataset/images_all/CheXpert-v1.0/preprocess2",
                                         csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/train_chex_selection_split_xrv.csv",
                                         views=["PA","AP"], unique_patients=False)
d_nih_train = xrv.datasets.NIH_Dataset(imgpath="/home/varrel/imaging/all_images/NIH_dataset/images_all/preprocess3",
                                       csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/nih_train_split_xrv.csv",
                                       views=["PA","AP"], unique_patients=False)
d_nih_selection = xrv.datasets.NIH_Dataset(imgpath="/home/varrel/imaging/all_images/NIH_dataset/images_all/preprocess3",
                                           csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/nih_selection_split_xrv.csv",
                                           views=["PA","AP"], unique_patients=False)
d_mimic_train = xrv.datasets.MIMIC_Dataset(imgpath="/home/varrel/imaging/all_images/MIMIC_dataset/images_all/preprocess2",
                                     csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/mimic_train_split_xrv.csv",
                                     metacsvpath="/home/varrel/imaging/all_images/MIMIC_dataset/mimic-cxr-2.0.0-metadata.csv",
                                     views=["PA","AP"], unique_patients=False)
d_mimic_selection = xrv.datasets.MIMIC_Dataset(imgpath="/home/varrel/imaging/all_images/MIMIC_dataset/images_all/preprocess2",
                                     csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/mimic_selection_split_xrv.csv",
                                     metacsvpath="/home/varrel/imaging/all_images/MIMIC_dataset/mimic-cxr-2.0.0-metadata.csv",
                                     views=["PA","AP"], unique_patients=False)

pc_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/PC_dataset/images_all/preprocess2",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/pc_train_split_filtered.csv",
    id_col="ImageID"
)
pc_png_selection = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/PC_dataset/images_all/preprocess2",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/pc_selection_split_filtered.csv",
    id_col="ImageID"
)

vindr_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/VinDr_dataset/images_all/preprocess2",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/vin_train_split.csv",
    id_col="image_id"
)
vindr_png_selection = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/VinDr_dataset/images_all/preprocess2",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/vin_selection_split.csv",
    id_col="image_id"
)

siim_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/SIIM_dataset/flat_png",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/siim_train_split.csv",
    id_col="ImageId"
)
siim_png_selection = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/SIIM_dataset/flat_png",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/siim_selection_split.csv",
    id_col="ImageId"
)

pp_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/PP_dataset/images_all/flat_png/train",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/pp_train_split.csv",
    id_col="image_id"
)
pp_png_selection = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/PP_dataset/images_all/flat_png/selection",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/pp_selection_split.csv",
    id_col="image_id"
)

openi_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/Openi_dataset/images_all/preprocess2",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/openi_train_split.csv",
    id_col="filename"
)
openi_png_selection = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/Openi_dataset/images_all/preprocess2",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/openi_selection_split.csv",
    id_col="filename"
)


classes = ['Atelectasis']
xrv.datasets.relabel_dataset(classes, openi_png_train)
xrv.datasets.relabel_dataset(classes, openi_png_selection)
xrv.datasets.relabel_dataset(classes, d_nih_train)
xrv.datasets.relabel_dataset(classes, d_nih_selection)
xrv.datasets.relabel_dataset(classes, d_chex_train)
xrv.datasets.relabel_dataset(classes, d_chex_selection)
xrv.datasets.relabel_dataset(classes, d_mimic_train)
xrv.datasets.relabel_dataset(classes, d_mimic_selection)
xrv.datasets.relabel_dataset(classes, pc_png_train)
xrv.datasets.relabel_dataset(classes, pc_png_selection)
xrv.datasets.relabel_dataset(classes, vindr_png_train)
xrv.datasets.relabel_dataset(classes, vindr_png_selection)
#xrv.datasets.relabel_dataset(classes, siim_png_train)
#xrv.datasets.relabel_dataset(classes, siim_png_selection)
#xrv.datasets.relabel_dataset(classes, ox_png_train)
#xrv.datasets.relabel_dataset(classes, ox_png_selection)
#xrv.datasets.relabel_dataset(classes, d_rsna_train)
#xrv.datasets.relabel_dataset(classes, d_rsna_selection)
#xrv.datasets.relabel_dataset(classes, pp_png_train)
#xrv.datasets.relabel_dataset(classes, pp_png_selection)

dmerge_train = xrv.datasets.MergeDataset([d_nih_train, d_chex_train, d_mimic_train, pc_png_train, openi_png_train, vindr_png_train])
dmerge_selection = xrv.datasets.MergeDataset([d_nih_selection, d_chex_selection, d_mimic_selection, pc_png_selection, openi_png_selection, vindr_png_selection])

# ---- Load precomputed weights ----
sample_weights = np.load("/home/varrel/GitHub/CXR-Analysis/model/atelectasis_train_weights.npy")
pos_weight_value = np.load("/home/varrel/GitHub/CXR-Analysis/model/atelectasis_pos_weight.npy")[0]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = torch.utils.data.DataLoader(dmerge_train, batch_size=16, sampler=sampler, num_workers=8, collate_fn=safe_collate)
selection_loader = torch.utils.data.DataLoader(dmerge_selection, batch_size=16, shuffle=False, num_workers=8, collate_fn=safe_collate)

# Baseline ResNet50 (no pretrained weights)
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-6)

pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

scaler = torch.amp.GradScaler("cuda")
num_epochs = 30

# ---- History dict to store metrics ----
history = {
    "train_loss": [],
    "val_loss": [],
    "val_acc": [],
    "val_rocauc": [],
    "val_f1": []
}

# ---- Early stopping parameters (F1-based) ----
patience = 5
best_val_f1 = float("-inf")
epochs_no_improve = 0

print("Starting training...")

for epoch in range(num_epochs):
    start_time = time.time()

    # ---- Training ----
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        inputs = batch["img"].to(device)
        labels = batch["lab"][:,0].to(device).float()

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(inputs).squeeze(1)
            # clamp logits to avoid extreme values
            outputs = torch.clamp(outputs, min=-20, max=20)
            labels = torch.nan_to_num(labels, nan=0.0)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in selection_loader:
            inputs = batch["img"].to(device)
            labels = batch["lab"][:,0].to(device).float()

            with torch.amp.autocast("cuda"):
                outputs = model(inputs).squeeze(1)
                outputs = torch.clamp(outputs, min=-20, max=20)
                labels = torch.nan_to_num(labels, nan=0.0)
                loss = criterion(outputs, labels)

            val_loss += loss.item()

            # sanitize predictions
            probs = torch.sigmoid(outputs).cpu().numpy()
            probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.int().cpu().numpy())

    avg_val_loss = val_loss / len(selection_loader)
    val_acc = accuracy_score(all_labels, all_preds)

    # safe metric calculation
    try:
        val_rocauc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        val_rocauc = float("nan")

    try:
        val_f1 = f1_score(all_labels, all_preds)
    except ValueError:
        val_f1 = float("nan")

    end_time = time.time()
    epoch_time = end_time - start_time

    # ---- Save checkpoint ----
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'val_acc': val_acc,
        'val_rocauc': val_rocauc,
        'val_f1': val_f1,
        'pos_weight': pos_weight_value,
    }, f"/home/varrel/image_model_dict/atelectasis_model_epoch{epoch+1}.pth")

    # ---- Append metrics ----
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(val_acc)
    history["val_rocauc"].append(val_rocauc)
    history["val_f1"].append(val_f1)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {val_acc:.4f}, "
          f"Val ROC AUC: {val_rocauc:.4f}, "
          f"Val F1: {val_f1:.4f}, "
          f"Time: {epoch_time:.2f} sec")

    # ---- Early stopping check (F1-based) ----
    if val_f1 > best_val_f1:
        torch.save(model.state_dict(), "/home/varrel/image_model_dict/atelectasis_best_model.pth")
        best_val_f1 = val_f1
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1} (F1)")
            break

# ---- Save history JSON ----
with open("/home/varrel/image_model_dict/atelectasis_training_history.json", "w") as f:
    json.dump(history, f)

torch.cuda.empty_cache()
gc.collect()