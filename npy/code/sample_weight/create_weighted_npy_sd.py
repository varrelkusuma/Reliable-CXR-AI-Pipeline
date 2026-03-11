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

openi_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/Openi_dataset/images_all/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/openi_train_split_filtered.csv",
    id_col="filename"
)

ox_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/OX_dataset/ox_check/train/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/ox_train_split.csv",
    id_col="image_name"
)

classes = ['Support Devices']
ox_png_train.pathologies = [
    "Support Devices" if p == "Support Device" else p
    for p in ox_png_train.pathologies
]
xrv.datasets.relabel_dataset(classes, openi_png_train)
xrv.datasets.relabel_dataset(classes, d_chex_train)
xrv.datasets.relabel_dataset(classes, d_mimic_train)
xrv.datasets.relabel_dataset(classes, pc_png_train)
xrv.datasets.relabel_dataset(classes, ox_png_train)

dmerge = xrv.datasets.MergeDataset([
    openi_png_train, d_chex_train,
    d_mimic_train, pc_png_train, ox_png_train
])

gc.collect()
torch.cuda.empty_cache()

# --- Create Weighted Sampler ---
pathology_idx = dmerge.pathologies.index("Support Devices")

def get_label(idx):
    try:
        sample = dmerge[idx]
        lab = sample["lab"]
        return int(lab[pathology_idx])
    except Exception:
        return 0  # treat errors as negative

# Step 1: Preallocate labels array to exact dataset length
labels = np.zeros(len(dmerge), dtype=int)

with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    for idx, lab in enumerate(tqdm(executor.map(get_label, range(len(dmerge))), total=len(dmerge))):
        labels[idx] = 0 if lab is None else lab

# Step 2: Compute class weights
class_counts = np.bincount(labels, minlength=2)
class_weights = np.where(class_counts > 0, 1. / class_counts, 0)

# Step 3: Assign sample weights
sample_weights = np.array([class_weights[label] for label in labels])

# Step 5: Save for reuse
np.save("supportdevice_train_weights.npy", sample_weights)
print(f"Saved supportdevice_train_weights.npy (len={len(sample_weights)})")