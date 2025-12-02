import os
import torch
import torchvision
import torchxrayvision as xrv
import numpy as np
import cv2
import gc
from tqdm import tqdm

# Paths
input_dir = "/home/varrel/test_images/test_dataset/PC_dataset/images_all"
output_dir = os.path.join(input_dir, "preprocess2")
os.makedirs(output_dir, exist_ok=True)

# Device setup → force GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model (load once globally, on GPU)
seg_model = xrv.baseline_models.chestx_det.PSPNet()
seg_model = seg_model.to(device)
seg_model.eval()

# Transforms → native 512x512
transforms = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(512)
])

def process_file(fname):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        return f"Skipped {fname}"

    fpath = os.path.join(input_dir, fname)
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return f"Could not load {fname}"

    # --- Step 1: Normalize for PSPNet ---
    img = xrv.datasets.normalize(img, 255)
    img = np.expand_dims(img, 0)  # (1, H, W)

    # Apply transforms → now (1, 512, 512)
    img_t = transforms(img)
    img_t = torch.tensor(img_t).unsqueeze(0).float().to(device)  # (1,1,512,512) on GPU

    # --- Step 2: Segmentation ---
    with torch.no_grad():
        output = seg_model(img_t)
        mask = torch.sigmoid(output)
        mask = (mask > 0.5).float()

    # Combine masks 4 & 5
    mask4 = mask[0, 4]
    mask5 = mask[0, 5]
    combined_mask = torch.logical_or(mask4.bool(), mask5.bool()).float()

    # Apply mask
    filtered_img = img_t[0,0] * combined_mask
    filtered_img[combined_mask == 0] = -1024
    filtered_np = filtered_img.detach().cpu().numpy()

    # Clip to valid range [-1024, 1024]
    filtered_np = np.clip(filtered_np, -1024, 1024)

    # --- Step 3: Rescale to 0–255 for visualization ---
    filtered_np = ((filtered_np + 1024) / 2048 * 255).astype(np.uint8)

    # --- Step 4: Apply CLAHE (post-processing only) ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    filtered_np = clahe.apply(filtered_np)

    # Downsample to 224x224 for saving
    filtered_np = cv2.resize(filtered_np, (224, 224), interpolation=cv2.INTER_AREA)

    # Save
    out_path = os.path.join(output_dir, fname)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    success = cv2.imwrite(out_path, filtered_np)

    # Cleanup
    del img_t, output, mask, filtered_img
    torch.cuda.empty_cache()
    gc.collect()

    if not success:
        return f"Failed to save {fname}"
    return f"Processed {fname}"

if __name__ == "__main__":
    fnames = os.listdir(input_dir)

    # Single-process loop with progress bar
    for fname in tqdm(fnames, desc="Processing images", disable=None, ncols=80):
        msg = process_file(fname)
        tqdm.write(msg)

    print("Processing complete.")