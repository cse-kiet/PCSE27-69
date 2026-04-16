import os
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.ndimage import uniform_filter
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR = "data/raw2"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# MODIS LST scale factor: raw values are in Kelvin × 50, invalid = 0
SCALE_FACTOR = 0.02  # multiply raw DN to get Kelvin
INVALID_VALUE = 0  # pixels with this value are cloud/missing
CELSIUS_OFFSET = 273.15  # convert Kelvin → Celsius

# Spatial crop: we'll standardize all tiles to this grid size
# (MODIS tiles vary slightly at edges — we crop to a safe common size)
CROP_H, CROP_W = 200, 200
# ──────────────────────────────────────────────────────────────────────────────


def read_lst_from_hdf(filepath):
    """Extract daytime LST layer from a MOD11A2 HDF4 file."""
    hdf = SD(filepath, SDC.READ)

    # 'LST_Day_1km' is the daytime land surface temperature band
    lst_raw = hdf.select("LST_Day_1km").get()
    qc_raw = hdf.select("QC_Day").get()  # quality control flags
    hdf.end()

    # Mask: 0 = fill/cloud, QC bits 0-1 != 00 = poor quality
    valid_mask = (lst_raw != INVALID_VALUE) & ((qc_raw & 0b11) == 0)

    # Scale to Celsius
    lst_celsius = np.where(valid_mask, lst_raw * SCALE_FACTOR - CELSIUS_OFFSET, np.nan)
    return lst_celsius


def gap_fill(arr):
    """Fill NaN pixels with a spatial average of their neighbors."""
    # Replace NaN with 0 temporarily, smooth, then restore valid pixels
    filled = np.where(np.isnan(arr), 0.0, arr)
    weights = np.where(np.isnan(arr), 0.0, 1.0)
    smooth_vals = uniform_filter(filled, size=5)
    smooth_weights = uniform_filter(weights, size=5)
    # Avoid division by zero in areas with no valid neighbors
    with np.errstate(invalid="ignore"):
        interpolated = np.where(
            smooth_weights > 0,
            smooth_vals / smooth_weights,
            0.0,  # fallback for completely isolated gaps
        )
    return np.where(np.isnan(arr), interpolated, arr)


def normalize(stack):
    """Z-score normalize across the whole time series per pixel."""
    mean = np.nanmean(stack, axis=0, keepdims=True)
    std = np.nanstd(stack, axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)  # avoid divide-by-zero
    return (stack - mean) / std, mean, std


def main():
    # Separate files by tile ID
    all_files = sorted(
        [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith(".hdf")]
    )

    # Group files by their date (characters 9-16 in MODIS filename encode the date)
    from collections import defaultdict

    date_groups = defaultdict(list)
    for path in all_files:
        fname = os.path.basename(path)
        # MODIS filename format: MOD11A2.AYYYYDDD.hXXvYY.061.*.hdf
        # The date token is the 2nd dot-separated field e.g. A2018009
        date_token = fname.split(".")[1]  # e.g. "A2018009"
        date_groups[date_token].append(path)

    print(f"Found {len(date_groups)} unique dates across {len(all_files)} HDF files.")

    frames = []
    for date_token in tqdm(sorted(date_groups.keys())):
        paths = sorted(date_groups[date_token])  # sort so h17 always before h18
        tiles = []
        for path in paths:
            try:
                lst = read_lst_from_hdf(path)
                lst = gap_fill(lst)
                # Crop each tile to standard height/width
                h, w = lst.shape
                ch = (h - CROP_H) // 2
                cw = (w - CROP_W) // 2
                lst = lst[ch : ch + CROP_H, cw : cw + CROP_W]
                tiles.append(lst)
            except Exception as e:
                print(f"  Skipping {os.path.basename(path)}: {e}")

        if len(tiles) == 2:
            # Stitch h17 and h18 side by side → (CROP_H, CROP_W * 2)
            combined = np.concatenate(tiles, axis=1)
            frames.append(combined)
        elif len(tiles) == 1:
            # Only one tile available for this date — use it as-is
            frames.append(tiles[0])
        # Skip dates where both tiles failed

    if not frames:
        print("No frames could be read. Check your HDF files.")
        return

    stack = np.stack(frames, axis=0).astype(np.float32)
    print(f"Raw stack shape: {stack.shape}  (timesteps × height × width)")

    # Save raw mean and std BEFORE normalization (in Celsius)
    raw_mean = np.nanmean(stack, axis=0, keepdims=True).astype(np.float32)
    raw_std  = np.nanstd(stack,  axis=0, keepdims=True).astype(np.float32)

    np.save(os.path.join(PROCESSED_DIR, "lst_mean_celsius.npy"), raw_mean)
    np.save(os.path.join(PROCESSED_DIR, "lst_std_celsius.npy"),  raw_std)

    stack_norm, mean, std = normalize(stack)

    np.save(os.path.join(PROCESSED_DIR, "lst_stack.npy"), stack_norm)
    np.save(os.path.join(PROCESSED_DIR, "lst_mean.npy"), mean)
    np.save(os.path.join(PROCESSED_DIR, "lst_std.npy"), std)

    print(f"Saved normalized tensor to {PROCESSED_DIR}/lst_stack.npy")
    print(
        f"Shape: {stack_norm.shape}, Min: {stack_norm.min():.2f}, Max: {stack_norm.max():.2f}"
    )


if __name__ == "__main__":
    main()
