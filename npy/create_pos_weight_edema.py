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
d_rsna_train = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath="/home/varrel/imaging/all_images/RSNA_dataset/stage_2_train_images_jpg/preprocess2/",
                                                   csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/rsna_train_split_xrv.csv",
                                                   views=["PA","AP"])
d_nih_train = xrv.datasets.NIH_Dataset(imgpath="/home/varrel/imaging/all_images/NIH_dataset/images_all/preprocess3",
                                       csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/nih_train_split_xrv.csv",
                                       views=["PA","AP"], unique_patients=False)
d_chex_train = xrv.datasets.CheX_Dataset(imgpath="/home/varrel/imaging/all_images/CheX_dataset/images_all/CheXpert-v1.0/preprocess2",
                                         csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/train_chex_train_split_xrv.csv",
                                         views=["PA","AP"], unique_patients=False)
d_mimic_train = xrv.datasets.MIMIC_Dataset(imgpath="/home/varrel/imaging/all_images/MIMIC_dataset/images_all/preprocess2",
                                     csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/mimic_train_split_xrv.csv",
                                     metacsvpath="/home/varrel/imaging/all_images/MIMIC_dataset/mimic-cxr-2.0.0-metadata.csv",
                                     views=["PA","AP"], unique_patients=False)

pc_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/PC_dataset/images_all/preprocess2",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/pc_train_split_filtered.csv",
    id_col="ImageID"
)

vindr_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/VinDr_dataset/images_all/preprocess2",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/vin_train_split.csv",
    id_col="image_id"
)

siim_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/SIIM_dataset/flat_png",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/siim_train_split.csv",
    id_col="ImageId"
)

pp_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/PP_dataset/images_all/flat_png/train",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/pp_train_split.csv",
    id_col="image_id"
)

openi_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/Openi_dataset/images_all/preprocess2",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/openi_train_split.csv",
    id_col="filename"
)

ox_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/OX_dataset/ox_check/train/preprocess2",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/ox_train_split.csv",
    id_col="image_name"
)

classes = ['Edema']
xrv.datasets.relabel_dataset(classes, openi_png_train)
xrv.datasets.relabel_dataset(classes, d_nih_train)
xrv.datasets.relabel_dataset(classes, d_chex_train)
xrv.datasets.relabel_dataset(classes, d_mimic_train)
xrv.datasets.relabel_dataset(classes, pc_png_train)
xrv.datasets.relabel_dataset(classes, vindr_png_train)
#xrv.datasets.relabel_dataset(classes, siim_png_train)
#xrv.datasets.relabel_dataset(classes, ox_png_train)
#xrv.datasets.relabel_dataset(classes, d_rsna_train)
#xrv.datasets.relabel_dataset(classes, pp_png_train)

dmerge = xrv.datasets.MergeDataset([
    openi_png_train, d_chex_train, d_nih_train,
    d_mimic_train, pc_png_train, vindr_png_train
])

gc.collect()
torch.cuda.empty_cache()

# ---- Compute and Save pos_weight for Edema ----
pathology_idx = dmerge.pathologies.index("Edema")

def get_label(idx):
    try:
        sample = dmerge[idx]
        lab = sample["lab"]
        val = torch.nan_to_num(torch.tensor(lab[pathology_idx]), nan=0.0).item()
        return int(val)
    except Exception:
        return 0  # treat errors as negative, but keep length consistent

# Preallocate labels array
labels = np.zeros(len(dmerge), dtype=int)

for idx in tqdm(range(len(dmerge)), total=len(dmerge), desc="Extracting labels"):
    labels[idx] = get_label(idx)

# Count positives and negatives
num_pos = np.sum(labels == 1)
num_neg = np.sum(labels == 0)

# Compute pos_weight = (# negatives / # positives)
pos_weight_value = num_neg / max(num_pos, 1)   # avoid divide by zero

# Save for reuse
np.save("edema_pos_weight.npy", np.array([pos_weight_value]))
print(f"Saved pos_weight: {pos_weight_value:.4f} (neg={num_neg}, pos={num_pos})")
print(f"labels length = {len(labels)}, dataset length = {len(dmerge)}")