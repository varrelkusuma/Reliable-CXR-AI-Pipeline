import os
import cv2
import torch
import numpy as np
import pandas as pd
import torchxrayvision as xrv
from torch.utils.data import Dataset

class PNGFolderDataset(Dataset):
    def __init__(self, img_dir, csv_path=None, transform=None, id_col="filename"):
        self.img_dir = img_dir
        self.transform = transform

        self.labels = []
        self.pathologies = []
        self.files = []
        self.file_map = {}

        valid_exts = (".png", ".jpg", ".jpeg")

        if csv_path is not None:
            df = pd.read_csv(csv_path)
            df[id_col] = df[id_col].astype(str).str.strip().str.lower()
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
                        self.file_map[base_lower] = f  # store actual filename
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
        actual_fname = self.file_map[fbase]  # get real filename with case + ext
        fpath = os.path.join(self.img_dir, actual_fname)

        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {fpath}")

        img = xrv.datasets.normalize(img, 255)
        img = np.expand_dims(img, 0)

        if self.transform:
            img = self.transform(img)

        img = torch.tensor(img).float()
        lab = self.labels[idx]

        return {"img": img, "lab": lab, "idx": idx}