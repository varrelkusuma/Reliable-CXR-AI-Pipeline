import os
import cv2
import torch
import numpy as np
import pandas as pd
import torchxrayvision as xrv
from torch.utils.data import Dataset

class PNGFolderDataset(Dataset):
    def __init__(self, img_dir, csv_path=None, transform=None, data_aug=None, id_col="filename", resize_to=224):
        self.img_dir = img_dir
        self.transform = transform
        self.data_aug = data_aug
        self.resize_to = resize_to

        self.labels = []
        self.pathologies = []
        self.files = []
        self.file_map = {}

        valid_exts = (".png", ".jpg", ".jpeg")

        if csv_path is not None:
            df = pd.read_csv(csv_path)
            # Normalize filenames for matching
            df[id_col] = df[id_col].astype(str).str.strip().str.lower()
            
            # Identify pathology columns (all columns except filename)
            self.pathologies = [c for c in df.columns if c != id_col]
            df[self.pathologies] = df[self.pathologies].apply(pd.to_numeric, errors="coerce")

            # Build mapping from CSV
            fname_to_row = {row[id_col]: row for _, row in df.iterrows()}

            # Collect all files with valid extensions
            for f in os.listdir(img_dir):
                if f.lower().endswith(valid_exts):
                    base_lower = os.path.splitext(f)[0].lower()
                    if base_lower in fname_to_row:
                        row = fname_to_row[base_lower]
                        self.files.append(base_lower)
                        self.file_map[base_lower] = f
                        self.labels.append([row[p] for p in self.pathologies])

            self.labels = np.array(self.labels, dtype=float)

            # Create a .csv DataFrame attribute for MergeDataset compatibility
            self.csv = pd.DataFrame(self.labels, columns=self.pathologies)
            self.csv.insert(0, id_col, self.files)
        else:
            # No CSV: keep all files, no labels
            for f in os.listdir(img_dir):
                if f.lower().endswith(valid_exts):
                    base_lower = os.path.splitext(f)[0].lower()
                    self.files.append(base_lower)
                    self.file_map[base_lower] = f
            self.labels = np.zeros((len(self.files), 0))
            self.csv = pd.DataFrame({id_col: self.files})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fbase = self.files[idx]
        actual_fname = self.file_map[fbase]
        fpath = os.path.join(self.img_dir, actual_fname)

        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {fpath}")
        
        if self.resize_to is not None:
            img = cv2.resize(img, (self.resize_to, self.resize_to), interpolation=cv2.INTER_LINEAR)

        # ---- FIX 1: Normalize to [0, 1] Standard ----
        # Old: img = xrv.datasets.normalize(img, 255)  <-- This was [-1024, 1024]
        # New: Simple scaling to 0-1
        img = img.astype(np.float32) / 255.0
        
        # Add channel dimension: (H, W) -> (1, H, W)
        img = np.expand_dims(img, 0)

        # Apply augmentation (assuming data_aug expects/returns numpy)
        if self.data_aug is not None:
            img = self.data_aug(img)

        # Apply transforms (resize, crop, etc.)
        if self.transform is not None:
            img = self.transform(img)

        # ---- FIX 2: Return NumPy, NOT Tensor ----
        # This prevents collisions with XRV datasets in MergeDataset
        # The DataLoader will handle the Tensor conversion automatically.
        # img = torch.tensor(img).float() <--- DELETED
        
        lab = self.labels[idx]

        return {"img": img, "lab": lab, "idx": idx}
    
class XRVCorrectionDataset(torch.utils.data.Dataset):
    """
    Wrapper to force XRV datasets (typically -1024 to 1024) 
    into the standard [0, 1] range.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.csv = dataset.csv # Preserve metadata for MergeDataset
        self.pathologies = dataset.pathologies
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample['img'] # numpy array, usually -1024 to 1024
        
        # 1. Rescale [-1024, 1024] -> [0, 1]
        # (img + 1024) / 2048 is the exact mapping XRV uses
        img = (img + 1024) / 2048.0
        
        # 2. Clip to be safe (remove outliers)
        img = np.clip(img, 0, 1)
        
        sample['img'] = img
        return sample