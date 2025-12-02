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
                                                   csvpath="/home/varrel/GitHub/CXR-Analysis/data/rsna_train_split.csv",
                                                   views=["PA","AP"])
d_chex_train = xrv.datasets.CheX_Dataset(imgpath="/home/varrel/imaging/all_images/CheX_dataset/images_all/CheXpert-v1.0/preprocess2",
                                         csvpath="/home/varrel/GitHub/CXR-Analysis/data/chex_train_split.csv",
                                         views=["PA","AP"], unique_patients=False)
d_nih_train = xrv.datasets.NIH_Dataset(imgpath="/home/varrel/imaging/all_images/NIH_dataset/images_all/preprocess3",
                                       csvpath="/home/varrel/GitHub/CXR-Analysis/images/image_load/nih_train_split_xrv.csv",
                                       views=["PA","AP"], unique_patients=False)
d_mimic_train = xrv.datasets.MIMIC_Dataset(imgpath="/home/varrel/imaging/all_images/MIMIC_dataset/images_all/preprocess2",
                                     csvpath="/home/varrel/GitHub/CXR-Analysis/data/mimic_train_split.csv",
                                     metacsvpath="/home/varrel/imaging/all_images/MIMIC_dataset/mimic-cxr-2.0.0-metadata.csv",
                                     views=["PA","AP"], unique_patients=False)
d_pc_train = xrv.datasets.PC_Dataset(imgpath="/home/varrel/imaging/all_images/PC_dataset/images_all/preprocess2",
                                     csvpath="/home/varrel/GitHub/CXR-Analysis/images/image_load/pc_train_split_xrv.csv",
                                     views=["PA","AP"], unique_patients=False)

vindr_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/VinDr_dataset/images_all/preprocess2",
    csv_path="/home/varrel/GitHub/CXR-Analysis/data/vin_train_split.csv",
    id_col="image_id"
)

pp_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/PP_dataset/images_all/flat_png/train",
    csv_path="/home/varrel/GitHub/CXR-Analysis/images/image_load/pp_train_split_xrv.csv",
    id_col="image_id"
)

openi_png_train = PNGFolderDataset(
    img_dir="/home/varrel/imaging/all_images/Openi_dataset/images_all/preprocess2",
    csv_path="/home/varrel/GitHub/CXR-Analysis/images/image_load/openi_train_split_xrv.csv",
    id_col="filename"
)

classes = ['Atelectasis']
#xrv.datasets.relabel_dataset(classes, d_rsna_train)
xrv.datasets.relabel_dataset(classes, d_chex_train)
xrv.datasets.relabel_dataset(classes, d_nih_train)
xrv.datasets.relabel_dataset(classes, d_mimic_train)
xrv.datasets.relabel_dataset(classes, d_pc_train)
xrv.datasets.relabel_dataset(classes, openi_png_train)
#xrv.datasets.relabel_dataset(classes, pp_png_train)
xrv.datasets.relabel_dataset(classes, vindr_png_train)

dmerge = xrv.datasets.MergeDataset([
    d_nih_train, d_chex_train, d_mimic_train,
    d_pc_train, openi_png_train, vindr_png_train
])

gc.collect()
torch.cuda.empty_cache()

# --- Create Weighted Sampler ---
pathology_idx = dmerge.pathologies.index("Atelectasis")

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

# Step 4: Handle length mismatch
if len(sample_weights) != len(dmerge):
    print(f"Length mismatch detected: weights={len(sample_weights)}, dataset={len(dmerge)}")
    sample_weights = sample_weights[:len(dmerge)]
assert len(sample_weights) == len(dmerge)

# Step 5: Save for reuse
np.save("atelectasis_train_weights.npy", sample_weights)
print(f"Weighted sampler saved as atelectasis_train_weights.npy (len={len(sample_weights)})")