import numpy as np
import torchxrayvision as xrv
import torch
import gc
from tqdm import tqdm
import concurrent.futures

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from util.PNGDataset import PNGFolderDataset

"""--- Load Processed Datasets ---"""
d_nih_train = xrv.datasets.NIH_Dataset(imgpath="/home/varrel/imaging/all_images/NIH_dataset/images_all/preprocess7",
                                       csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/nih_train_split_xrv.csv",
                                       views=["PA","AP"], unique_patients=False)
d_chex_train = xrv.datasets.CheX_Dataset(imgpath="/home/varrel/imaging/all_images/CheX_dataset/images_all/CheXpert-v1.0/preprocess7",
                                         csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/train_chex_train_split_xrv.csv",
                                         views=["PA","AP"], unique_patients=False)
d_mimic_train = xrv.datasets.MIMIC_Dataset(imgpath="/home/varrel/imaging/all_images/MIMIC_dataset/images_all/preprocess7",
                                     csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/mimic_train_split_xrv.csv",
                                     metacsvpath="/home/varrel/imaging/all_images/MIMIC_dataset/mimic-cxr-2.0.0-metadata.csv",
                                     views=["PA","AP"], unique_patients=False)

pc_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/PC_dataset/images_all/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/pc_train_split_filtered.csv",
    id_col="ImageID"
)

vindr_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/VinDr_dataset/images_all/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/vin_train_split.csv",
    id_col="image_id"
)

openi_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/Openi_dataset/images_all/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/openi_train_split_filtered.csv",
    id_col="filename"
)

classes = ['Edema','Consolidation','Pleural Effusion','Cardiomegaly','Atelectasis','Lung Opacity']
d_mimic_train.pathologies = [
    "Pleural Effusion" if p == "Effusion" else p
    for p in d_mimic_train.pathologies
]
d_chex_train.pathologies = [
    "Pleural Effusion" if p == "Effusion" else p
    for p in d_chex_train.pathologies
]
d_nih_train.pathologies = [
    "Pleural Effusion" if p == "Effusion" else p
    for p in d_nih_train.pathologies
]
d_nih_train.pathologies = [
    "Lung Opacity" if p == "Infiltration" else p
    for p in d_nih_train.pathologies
]
xrv.datasets.relabel_dataset(classes, openi_png_train)
xrv.datasets.relabel_dataset(classes, d_nih_train)
xrv.datasets.relabel_dataset(classes, d_chex_train)
xrv.datasets.relabel_dataset(classes, d_mimic_train)
xrv.datasets.relabel_dataset(classes, pc_png_train)
xrv.datasets.relabel_dataset(classes, vindr_png_train)

dmerge = xrv.datasets.MergeDataset([
    openi_png_train, d_nih_train, d_chex_train,
    d_mimic_train, pc_png_train, vindr_png_train
])

gc.collect()
torch.cuda.empty_cache()

# ---- Compute and Save pos_weight for all 6 classes ----
labels = np.zeros((len(dmerge), len(classes)), dtype=int)

for idx in tqdm(range(len(dmerge)), total=len(dmerge), desc="Extracting labels"):
    sample = dmerge[idx]
    lab = torch.nan_to_num(torch.tensor(sample["lab"]), nan=0.0).numpy().astype(int)
    labels[idx] = lab

# Count positives and negatives per class
num_pos = labels.sum(axis=0)
num_neg = labels.shape[0] - num_pos

# Compute pos_weight = (# negatives / # positives) for each class
pos_weight_values = np.array([
    num_neg[i] / max(num_pos[i], 1) for i in range(len(classes))
])

# Save for reuse
np.save("multilabel_pos_weight.npy", pos_weight_values)

print("Saved pos_weight vector:", pos_weight_values)
for cls, pw, pos, neg in zip(classes, pos_weight_values, num_pos, num_neg):
    print(f"{cls}: pos_weight={pw:.4f} (pos={pos}, neg={neg})")

print(f"labels shape = {labels.shape}, dataset length = {len(dmerge)}")