import os
import torch
import torchvision
import torchxrayvision as xrv
import numpy as np
import cv2
import gc
from tqdm import tqdm

# Paths
input_dir = "/home/varrel/test_images/OX_dataset/images_all/dev/train"
output_dir = os.path.join(input_dir, "preprocess7")
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

def mask_bounding_box(mask, pad_fraction=0.01, H=512, W=512):
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy().astype(bool)
    else:
        mask_np = mask.astype(bool)

    ys, xs = np.where(mask_np)
    if ys.size == 0 or xs.size == 0:
        return 0, W-1, 0, H-1

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    box_w = x_max - x_min + 1
    box_h = y_max - y_min + 1

    pad_x = int(pad_fraction * box_w)
    pad_y = int(pad_fraction * box_h)

    x_min = max(0, x_min - pad_x)
    x_max = min(W - 1, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(H - 1, y_max + pad_y)

    return x_min, x_max, y_min, y_max

def fill_mask(submask):
    H_sub, W_sub = submask.shape
    filled = submask.copy()

    # horizontal fill per row
    rows_with_true = filled.any(axis=1)
    if rows_with_true.any():
        left = np.where(rows_with_true, filled.argmax(axis=1), W_sub)
        right = np.where(rows_with_true, W_sub - np.fliplr(filled).argmax(axis=1) - 1, -1)
        for r in np.where(rows_with_true)[0]:
            filled[r, left[r]:right[r]+1] = True

    # vertical fill per column
    cols_with_true = filled.any(axis=0)
    if cols_with_true.any():
        top = np.where(cols_with_true, filled.argmax(axis=0), H_sub)
        bottom = np.where(cols_with_true, H_sub - np.flipud(filled).argmax(axis=0) - 1, -1)
        for c in np.where(cols_with_true)[0]:
            filled[top[c]:bottom[c]+1, c] = True

    return filled

def process_file(fname, pad_fraction=0.01, target_lung_fraction=0.90):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        return f"Skipped {fname}"

    fpath = os.path.join(input_dir, fname)
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return f"Could not load {fname}"

    # --- Step 1: Normalize for PSPNet ---
    img_norm = xrv.datasets.normalize(img, 255)
    img_norm = np.expand_dims(img_norm, 0)  # (1, H, W)

    # Apply transforms → now (1, 512, 512)
    img_t = transforms(img_norm)
    img_t = torch.tensor(img_t).unsqueeze(0).float().to(device)

    # --- Step 2: Segmentation ---
    with torch.no_grad():
        output = seg_model(img_t)
        prob = torch.sigmoid(output)
        mask = (prob > 0.5).float()

    C = mask.shape[1]

    # Extract relevant masks (if available). Indices based on PSPNet targets:
    # 4 = Left Lung, 5 = Right Lung, 8 = Heart, 10 = Facies Diaphragmatica (abdomen), 11 = Mediastinum
    # We'll combine these into a single composite mask for cropping/masking.
    def get_mask(idx):
        if idx < C:
            return mask[0, idx].cpu().numpy().astype(bool)
        else:
            return np.zeros((512, 512), dtype=bool)

    mask_left = get_mask(4)
    mask_right = get_mask(5)
    mask_heart = get_mask(8)

    # Composite mask: include lungs and optionally heart/mediastinum/abdomen
    # This keeps a single boolean ROI used for cropping and masking.
    composite = np.logical_or(mask_left, mask_right)
    composite = np.logical_or(composite, mask_heart)

    # Fill small gaps inside the mask (so bbox is contiguous)
    composite_filled = fill_mask(composite)

    # Compute bbox on 512x512 mask
    Hm, Wm = composite_filled.shape
    x_min, x_max, y_min, y_max = mask_bounding_box(composite_filled, pad_fraction=pad_fraction, H=Hm, W=Wm)

    # Optionally expand bbox until lung fraction inside crop >= target_lung_fraction
    def lung_fraction_in_bbox(mask_bool, xmin, xmax, ymin, ymax):
        crop = mask_bool[ymin:ymax+1, xmin:xmax+1]
        if crop.size == 0:
            return 0.0
        return float(crop.sum()) / float(crop.size)

    cur_xmin, cur_xmax, cur_ymin, cur_ymax = x_min, x_max, y_min, y_max
    lung_frac = lung_fraction_in_bbox(composite_filled, cur_xmin, cur_xmax, cur_ymin, cur_ymax)
    expand_step = 5
    iter_count = 0
    while lung_frac < target_lung_fraction and iter_count < 40:
        cur_xmin = max(0, cur_xmin - expand_step)
        cur_xmax = min(Wm - 1, cur_xmax + expand_step)
        cur_ymin = max(0, cur_ymin - expand_step)
        cur_ymax = min(Hm - 1, cur_ymax + expand_step)
        new_frac = lung_fraction_in_bbox(composite_filled, cur_xmin, cur_xmax, cur_ymin, cur_ymax)
        if new_frac <= lung_frac:
            break
        lung_frac = new_frac
        iter_count += 1

    # --- Step 3: Apply composite mask to the 512x512 image tensor and convert to numpy ---
    filtered_img = img_t[0, 0].cpu().numpy() * composite_filled.astype(float)
    filtered_img[~composite_filled] = -1024.0

    # Clip to valid range [-1024, 1024]
    filtered_img = np.clip(filtered_img, -1024, 1024)

    # Rescale to 0–255 for visualization (uint8)
    filtered_np = ((filtered_img + 1024) / 2048 * 255).astype(np.uint8)

    # --- Step 4: Crop to bbox (in 512x512 coordinates) to focus on ROI ---
    cropped = filtered_np[cur_ymin:cur_ymax+1, cur_xmin:cur_xmax+1]
    if cropped.size == 0:
        cropped = filtered_np

    # --- Step 5: Apply CLAHE (post-processing only) ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cropped = clahe.apply(cropped)

    # Downsample to 224x224 for saving (preserve aspect by resizing to square)
    final_img = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)

    # Save single final image (overwrites/creates one file per input)
    fname = os.path.basename(fpath)
    out_path = os.path.join(output_dir, fname)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    success = cv2.imwrite(out_path, final_img)

    # Cleanup
    del img_t, output, mask, filtered_img
    torch.cuda.empty_cache()
    gc.collect()

    if not success:
        return f"Failed to save {fname}"
    return f"Processed {fname} (lung_frac_in_crop={lung_frac:.3f})"

if __name__ == "__main__":
    for root, dirs, files in os.walk(input_dir):
        if any(part.startswith("preprocess") for part in root.split(os.sep)):
            continue
        for fname in files:
            fpath = os.path.join(root, fname)
            msg = process_file(fpath, pad_fraction=0.01, target_lung_fraction=0.90)
            tqdm.write(msg)

    print("Processing complete.")