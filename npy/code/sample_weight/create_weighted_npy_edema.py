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

classes = ['Edema']
xrv.datasets.relabel_dataset(classes, openi_png_train)
xrv.datasets.relabel_dataset(classes, d_nih_train)
xrv.datasets.relabel_dataset(classes, d_chex_train)
xrv.datasets.relabel_dataset(classes, d_mimic_train)
xrv.datasets.relabel_dataset(classes, pc_png_train)
xrv.datasets.relabel_dataset(classes, vindr_png_train)

dmerge = xrv.datasets.MergeDataset([
    openi_png_train, d_chex_train, d_nih_train,
    d_mimic_train, pc_png_train, vindr_png_train
])

gc.collect()
torch.cuda.empty_cache()

# --- Create Weighted Sampler ---
pathology_idx = dmerge.pathologies.index("Edema")

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
np.save("edema_train_weights.npy", sample_weights)
print(f"Saved edema_train_weights.npy (len={len(sample_weights)})")