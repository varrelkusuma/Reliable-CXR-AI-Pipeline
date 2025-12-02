import os
import torch
import torchvision
import torchxrayvision as xrv
import numpy as np
import cv2
import gc
from tqdm import tqdm

# Paths
input_dir = "/home/varrel/imaging/all_images/MIMIC_dataset/images_all"
output_dir = os.path.join(input_dir, "preprocess2")
os.makedirs(output_dir, exist_ok=True)

# Device setup → force GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model (load once globally, on GPU)
seg_model = xrv.baseline_models.chestx_det.PSPNet()
seg_model = seg_model.to(device)
seg_model.eval()

# Transforms (XRV native 512x512)
transforms = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(512)
])

def process_file(fpath):
    if not fpath.lower().endswith((".png", ".jpg", ".jpeg")):
        return f"Skipped {fpath}"

    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return f"Could not load {fpath}"

    # --- Step 1: Normalize for PSPNet ---
    img = xrv.datasets.normalize(img, 255)
    img = np.expand_dims(img, 0)  # (1, H, W)

    img_t = transforms(img)
    img_t = torch.tensor(img_t).unsqueeze(0).float().to(device)

    # --- Step 2: Segmentation ---
    with torch.no_grad():
        output = seg_model(img_t)
        mask = torch.sigmoid(output)
        mask = (mask > 0.5).float()

    mask4 = mask[0, 4]
    mask5 = mask[0, 5]
    combined_mask = torch.logical_or(mask4.bool(), mask5.bool()).float()

    filtered_img = img_t[0,0] * combined_mask
    filtered_img[combined_mask == 0] = -1024
    filtered_np = filtered_img.detach().cpu().numpy()
    filtered_np = np.clip(filtered_np, -1024, 1024)

    # --- Step 3: Rescale to 0–255 for visualization ---
    filtered_np = ((filtered_np + 1024) / 2048 * 255).astype(np.uint8)

    # --- Step 4: Apply CLAHE (post-processing only) ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    filtered_np = clahe.apply(filtered_np)

    # Downsample to 224x224 for saving
    filtered_np = cv2.resize(filtered_np, (224, 224), interpolation=cv2.INTER_AREA)

    # Save, preserving nested structure
    rel_path = os.path.relpath(fpath, input_dir)
    out_path = os.path.join(output_dir, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    success = cv2.imwrite(out_path, filtered_np)

    # Cleanup
    del img_t, output, mask, filtered_img
    torch.cuda.empty_cache()
    gc.collect()

    if not success:
        return f"Failed to save {rel_path}"
    return f"Processed {rel_path}"

if __name__ == "__main__":
    for root, dirs, files in os.walk(input_dir):
        if root.startswith(output_dir):
            continue
        for fname in files:
            fpath = os.path.join(root, fname)
            msg = process_file(fpath)
            tqdm.write(msg)

    print("Processing complete.")