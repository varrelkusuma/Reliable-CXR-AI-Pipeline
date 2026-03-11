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
d_rsna_train = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath="/home/varrel/imaging/all_images/RSNA_dataset/stage_2_train_images_jpg/preprocess7/",
                                                   csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/rsna_train_split_xrv.csv",
                                                   views=["PA","AP"])
d_nih_train = xrv.datasets.NIH_Dataset(imgpath="/home/varrel/imaging/all_images/NIH_dataset/images_all/preprocess7",
                                       csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/nih_train_split_xrv.csv",
                                       views=["PA","AP"], unique_patients=False)
d_chex_train = xrv.datasets.CheX_Dataset(imgpath="/home/varrel/imaging/all_images/CheX_dataset/images_all/CheXpert-v1.0/preprocess7",
                                         csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/train_chex_train_split_xrv.csv",
                                         views=["PA","AP"], unique_patients=False)
d_mimic_train = xrv.datasets.MIMIC_Dataset(imgpath="/home/varrel/imaging/all_images/MIMIC_dataset/images_all/preprocess7",
                                     csvpath="/home/varrel/GitHub/CXR-Analysis/final_csv/mimic_train_uones_fixed.csv",
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

pp_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/PP_dataset/images_all/flat_png/train",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/pp_train_split.csv",
    id_col="image_id"
)

openi_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/Openi_dataset/images_all/preprocess7",
    csv_path="/home/varrel/GitHub/CXR-Analysis/final_csv/openi_train_split_filtered.csv",
    id_col="filename"
)

classes = ['Pneumonia']
xrv.datasets.relabel_dataset(classes, openi_png_train)
xrv.datasets.relabel_dataset(classes, d_nih_train)
xrv.datasets.relabel_dataset(classes, d_chex_train)
xrv.datasets.relabel_dataset(classes, d_mimic_train)
xrv.datasets.relabel_dataset(classes, pc_png_train)
xrv.datasets.relabel_dataset(classes, vindr_png_train)
xrv.datasets.relabel_dataset(classes, d_rsna_train)
xrv.datasets.relabel_dataset(classes, pp_png_train)


dmerge = xrv.datasets.MergeDataset([
    openi_png_train, d_nih_train, d_chex_train, d_rsna_train,
    d_mimic_train, pc_png_train, vindr_png_train, pp_png_train
])

gc.collect()
torch.cuda.empty_cache()

# --- Create Weighted Sampler ---
pathology_idx = dmerge.pathologies.index("Pneumonia")

def get_label(idx):
    """
    Fetches label and applies U-Ones policy:
    NaN (Uncertain) -> 1 (Positive)
    """
    try:
        sample = dmerge[idx]
        lab = sample["lab"][pathology_idx]
        
        # === U-ONES LOGIC IS HERE ===
        if np.isnan(lab):
            return 1  # Treat NaN as Positive
            
        return int(lab)
        
    except Exception as e:
        # It is safer to print errors during debug, 
        # but returning 0 for corrupt data is a standard fallback
        return 0

# Step 1: Preallocate labels array to exact dataset length
print("Extracting labels with U-Ones policy...")
labels = np.zeros(len(dmerge), dtype=int)

with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    results = list(tqdm(executor.map(get_label, range(len(dmerge))), total=len(dmerge)))
    
labels = np.array(results)

# Step 2: Compute class weights
class_counts = np.bincount(labels, minlength=2)
n_neg = class_counts[0]
n_pos = class_counts[1]

print(f"Dataset Stats (U-Ones): Neg={n_neg}, Pos={n_pos}")
print(f"Prevalence: {n_pos / (n_pos + n_neg):.2%}")

# Step 3: Assign sample weights
class_weights = np.where(class_counts > 0, 1. / class_counts, 0)
sample_weights = np.array([class_weights[label] for label in labels])

pos_weight_value = n_neg / n_pos

# Step 5: Save for reuse
np.save("pneumonia_train_weights_ones.npy", sample_weights)
np.save("pneumonia_pos_weight_uones.npy", np.array([pos_weight_value]))

print(f"Saved 'pneumonia_train_weights_ones.npy' (len={len(sample_weights)})")
print(f"Saved 'pneumonia_pos_weight_uones.npy' (value={pos_weight_value:.4f})")