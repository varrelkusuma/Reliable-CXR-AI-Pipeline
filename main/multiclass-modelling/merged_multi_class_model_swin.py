import os
import json
import numpy as np
import torchxrayvision as xrv
import torch
import gc
import torch.optim as optim
import time
import torchvision.models as models
import torchvision.transforms as T
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from torch.utils.data import WeightedRandomSampler
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from util.PNGDatasetNormalized import PNGFolderDataset

class XRVCorrectionDataset(torch.utils.data.Dataset):
    """
    Wraps standard XRV datasets (range -1024 to 1024) and 
    rescales them to [0, 1] to match PNGFolderDataset.
    Exposes essential attributes for MergeDataset.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        
        # CRITICAL: Forward metadata so MergeDataset can see it!
        if hasattr(dataset, 'csv'):
            self.csv = dataset.csv
        if hasattr(dataset, 'pathologies'):
            self.pathologies = dataset.pathologies
        if hasattr(dataset, 'labels'):
            self.labels = dataset.labels  # <--- This was missing before!
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample['img'] # numpy array, usually -1024 to 1024
        
        # Rescale [-1024, 1024] -> [0, 1]
        img = (img + 1024) / 2048.0
        
        # Clip to ensure safety
        img = np.clip(img, 0, 1)
        
        # Ensure 3-channel dimension if missing
        if img.ndim == 2:
            img = np.expand_dims(img, 0)
            
        sample['img'] = img
        return sample

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
# 1. Load XRV Datasets
d_chex_train_raw = xrv.datasets.CheX_Dataset(imgpath="/home/varrel/imaging/all_images/CheX_dataset/images_all/CheXpert-v1.0/preprocess7",
                                         csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/train_chex_train_split_xrv.csv",
                                         views=["PA","AP"], unique_patients=False)
d_chex_sel_raw = xrv.datasets.CheX_Dataset(imgpath="/home/varrel/imaging/all_images/CheX_dataset/images_all/CheXpert-v1.0/preprocess7",
                                       csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/train_chex_selection_split_xrv.csv",
                                       views=["PA","AP"], unique_patients=False)

d_nih_train_raw = xrv.datasets.NIH_Dataset(imgpath="/home/varrel/imaging/all_images/NIH_dataset/images_all/preprocess7",
                                       csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/nih_train_split_xrv.csv",
                                       views=["PA","AP"], unique_patients=False)
d_nih_sel_raw = xrv.datasets.NIH_Dataset(imgpath="/home/varrel/imaging/all_images/NIH_dataset/images_all/preprocess7",
                                     csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/nih_selection_split_xrv.csv",
                                     views=["PA","AP"], unique_patients=False)
d_mimic_train_raw = xrv.datasets.MIMIC_Dataset(imgpath="/home/varrel/imaging/all_images/MIMIC_dataset/images_all/preprocess7",
                                           csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/mimic_train_split_xrv.csv",
                                           metacsvpath="/home/varrel/imaging/all_images/MIMIC_dataset/mimic-cxr-2.0.0-metadata.csv",
                                           views=["PA","AP"], unique_patients=False)
d_mimic_sel_raw = xrv.datasets.MIMIC_Dataset(imgpath="/home/varrel/imaging/all_images/MIMIC_dataset/images_all/preprocess7",
                                         csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/mimic_selection_split_xrv.csv",
                                         metacsvpath="/home/varrel/imaging/all_images/MIMIC_dataset/mimic-cxr-2.0.0-metadata.csv",
                                         views=["PA","AP"], unique_patients=False)

# 2. Relabel BEFORE Wrapping (Relabel works on the .csv attribute)
classes = ['Edema','Consolidation','Pleural Effusion','Cardiomegaly','Atelectasis','Lung Opacity']
d_mimic_train_raw.pathologies = ["Pleural Effusion" if p == "Effusion" else p for p in d_mimic_train_raw.pathologies]
d_mimic_sel_raw.pathologies = ["Pleural Effusion" if p == "Effusion" else p for p in d_mimic_sel_raw.pathologies]
d_chex_train_raw.pathologies = ["Pleural Effusion" if p == "Effusion" else p for p in d_chex_train_raw.pathologies]
d_chex_sel_raw.pathologies = ["Pleural Effusion" if p == "Effusion" else p for p in d_chex_sel_raw.pathologies]
d_nih_train_raw.pathologies = ["Pleural Effusion" if p == "Effusion" else p for p in d_nih_train_raw.pathologies]
d_nih_sel_raw.pathologies = ["Pleural Effusion" if p == "Effusion" else p for p in d_nih_sel_raw.pathologies]
d_nih_train_raw.pathologies = ["Lung Opacity" if p == "Infiltration" else p for p in d_nih_train_raw.pathologies]
d_nih_sel_raw.pathologies = ["Lung Opacity" if p == "Infiltration" else p for p in d_nih_sel_raw.pathologies]
xrv.datasets.relabel_dataset(classes, d_chex_train_raw)
xrv.datasets.relabel_dataset(classes, d_chex_sel_raw)
xrv.datasets.relabel_dataset(classes, d_nih_train_raw)
xrv.datasets.relabel_dataset(classes, d_nih_sel_raw)
xrv.datasets.relabel_dataset(classes, d_mimic_train_raw)
xrv.datasets.relabel_dataset(classes, d_mimic_sel_raw)

# 3. Apply Correction Wrapper (Force XRV to [0, 1] range)
d_chex_train = XRVCorrectionDataset(d_chex_train_raw)
d_chex_selection = XRVCorrectionDataset(d_chex_sel_raw)
d_nih_train = XRVCorrectionDataset(d_nih_train_raw)
d_nih_selection = XRVCorrectionDataset(d_nih_sel_raw)
d_mimic_train = XRVCorrectionDataset(d_mimic_train_raw)
d_mimic_selection = XRVCorrectionDataset(d_mimic_sel_raw)

# 4. Load PNG Datasets (These are already [0, 1] from your new code)
pc_png_train = PNGFolderDataset(img_dir="/home/varrel/imaging/all_images/PC_dataset/images_all/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/pc_train_split_filtered.csv",
    id_col="ImageID"
)
pc_png_selection = PNGFolderDataset(img_dir="/home/varrel/imaging/all_images/PC_dataset/images_all/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/pc_selection_split_filtered.csv",
    id_col="ImageID"
)
vindr_png_train = PNGFolderDataset(img_dir="/home/varrel/imaging/all_images/VinDr_dataset/images_all/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/vin_train_split.csv",
    id_col="image_id"
)
vindr_png_selection = PNGFolderDataset(img_dir="/home/varrel/imaging/all_images/VinDr_dataset/images_all/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/vin_selection_split.csv",
    id_col="image_id"
)
openi_png_train = PNGFolderDataset(img_dir="/home/varrel/imaging/all_images/Openi_dataset/images_all/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/openi_train_split_filtered.csv",
    id_col="filename"
)
openi_png_selection = PNGFolderDataset(img_dir="/home/varrel/imaging/all_images/Openi_dataset/images_all/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/openi_selection_split_filtered.csv",
    id_col="filename"
)

# Relabel PNG Datasets
xrv.datasets.relabel_dataset(classes, openi_png_train)
xrv.datasets.relabel_dataset(classes, openi_png_selection)
xrv.datasets.relabel_dataset(classes, pc_png_train)
xrv.datasets.relabel_dataset(classes, pc_png_selection)
xrv.datasets.relabel_dataset(classes, vindr_png_train)
xrv.datasets.relabel_dataset(classes, vindr_png_selection)

# 5. Merge (Now safe because EVERYTHING is [0, 1] numpy)
dmerge_train = xrv.datasets.MergeDataset([d_nih_train, d_chex_train, d_mimic_train, pc_png_train, openi_png_train, vindr_png_train])
dmerge_selection = xrv.datasets.MergeDataset([d_nih_selection, d_chex_selection, d_mimic_selection, pc_png_selection, openi_png_selection, vindr_png_selection])

print(f"Training samples: {len(dmerge_train)}, Selection samples: {len(dmerge_selection)}")

# ---- Load precomputed weights ----
sample_weights = np.load("/home/varrel/GitHub/CXR-Analysis/model/multilabel_train_weights.npy")
pos_weight_value = np.load("/home/varrel/GitHub/CXR-Analysis/model/multilabel_pos_weight.npy")[0]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = torch.utils.data.DataLoader(dmerge_train, batch_size=16, sampler=sampler, num_workers=8, collate_fn=safe_collate)
selection_loader = torch.utils.data.DataLoader(dmerge_selection, batch_size=16, shuffle=False, num_workers=8, collate_fn=safe_collate)

# Baseline Swin-B model
model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
in_features = model.head.in_features
model.head = nn.Linear(in_features, 6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)

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
    "val_f1": [],
    "per_class_acc": [],
    "per_class_f1": [],
    "per_class_rocauc": []
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
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", mininterval=300.0):
        inputs = batch["img"].to(device)
        
        labels = batch["lab"].to(device).float() 

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(inputs) # Removed squeeze(1), usually not needed for [N, 6] output
            
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
            
            labels = batch["lab"].to(device).float()

            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                outputs = torch.clamp(outputs, min=-20, max=20)
                labels = torch.nan_to_num(labels, nan=0.0)
                loss = criterion(outputs, labels)

            val_loss += loss.item()

            # sanitize predictions
            probs = torch.sigmoid(outputs).cpu().numpy()
            probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

            preds = (probs > 0.5).astype(int)
            
            # Extend lists (this creates a list of arrays)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.int().cpu().numpy())

    avg_val_loss = val_loss / len(selection_loader)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    try:
        val_rocauc = roc_auc_score(all_labels, all_probs, average="macro")
    except ValueError:
        val_rocauc = float("nan")

    # ---- Per-class metrics ----
    per_class_acc = {}
    per_class_f1 = {}
    per_class_rocauc = {}
    
    # Now this slicing will work because they are numpy arrays
    for i, cls in enumerate(classes):
        per_class_acc[cls] = accuracy_score(all_labels[:, i], all_preds[:, i])
    
        try:
            per_class_f1[cls] = f1_score(all_labels[:, i], all_preds[:, i])
        except ValueError:
            per_class_f1[cls] = float("nan")
    
        try:
            per_class_rocauc[cls] = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            per_class_rocauc[cls] = float("nan")

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
        'per_class_acc': per_class_acc,
        'per_class_f1': per_class_f1,
        'per_class_rocauc': per_class_rocauc,
        'pos_weight': pos_weight_value,
    }, f"/home/varrel/image_model_dict/swinb_multilabel_model_epoch{epoch+1}.pth")

    # ---- Append metrics ----
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(val_acc)
    history["val_rocauc"].append(val_rocauc)
    history["val_f1"].append(val_f1)
    history["per_class_acc"].append(per_class_acc)
    history["per_class_f1"].append(per_class_f1)
    history["per_class_rocauc"].append(per_class_rocauc)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {val_acc:.4f}, "
          f"Val ROC AUC: {val_rocauc:.4f}, "
          f"Val F1: {val_f1:.4f}, "
          f"Time: {epoch_time:.2f} sec")

    # ---- Early stopping check (F1-based) ----
    if val_f1 > best_val_f1:
        torch.save(model.state_dict(), "/home/varrel/image_model_dict/swinb_multilabel_best_model.pth")
        best_val_f1 = val_f1
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1} (F1)")
            break

# ---- Save history JSON ----
with open("/home/varrel/image_model_dict/swinb_multilabel_training_history.json", "w") as f:
    json.dump(history, f)

torch.cuda.empty_cache()
gc.collect()