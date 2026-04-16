import os
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.ndimage import uniform_filter
from tqdm import tqdm

RAW_DIR = "data/raw2"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

SCALE_FACTOR = 0.02
INVALID_VALUE = 0
CELSIUS_OFFSET = 273.15

CROP_H, CROP_W = 200, 200


def read_lst_from_hdf(filepath):
    """Extract daytime LST layer from a MOD11A2 HDF4 file."""
    hdf = SD(filepath, SDC.READ)

    lst_raw = hdf.select("LST_Day_1km").get()
    qc_raw = hdf.select("QC_Day").get()
    hdf.end()

    valid_mask = (lst_raw != INVALID_VALUE) & ((qc_raw & 0b11) == 0)

    lst_celsius = np.where(valid_mask, lst_raw * SCALE_FACTOR - CELSIUS_OFFSET, np.nan)
    return lst_celsius


def gap_fill(arr):
    """Fill NaN pixels with a spatial average of their neighbors."""
    filled = np.where(np.isnan(arr), 0.0, arr)
    weights = np.where(np.isnan(arr), 0.0, 1.0)
    smooth_vals = uniform_filter(filled, size=5)
    smooth_weights = uniform_filter(weights, size=5)
    with np.errstate(invalid="ignore"):
        interpolated = np.where(
            smooth_weights > 0,
            smooth_vals / smooth_weights,
            0.0,
        )
    return np.where(np.isnan(arr), interpolated, arr)


def normalize(stack):
    """Z-score normalize across the whole time series per pixel."""
    mean = np.nanmean(stack, axis=0, keepdims=True)
    std = np.nanstd(stack, axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (stack - mean) / std, mean, std


def main():
    all_files = sorted(
        [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith(".hdf")]
    )

    from collections import defaultdict

    date_groups = defaultdict(list)
    for path in all_files:
        fname = os.path.basename(path)
        date_token = fname.split(".")[1]
        date_groups[date_token].append(path)

    print(f"Found {len(date_groups)} unique dates across {len(all_files)} HDF files.")

    frames = []
    for date_token in tqdm(sorted(date_groups.keys())):
        paths = sorted(date_groups[date_token])
        tiles = []
        for path in paths:
            try:
                lst = read_lst_from_hdf(path)
                lst = gap_fill(lst)
                h, w = lst.shape
                ch = (h - CROP_H) // 2
                cw = (w - CROP_W) // 2
                lst = lst[ch : ch + CROP_H, cw : cw + CROP_W]
                tiles.append(lst)
            except Exception as e:
                print(f"  Skipping {os.path.basename(path)}: {e}")

        if len(tiles) == 2:
            combined = np.concatenate(tiles, axis=1)
            frames.append(combined)
        elif len(tiles) == 1:
            frames.append(tiles[0])

    if not frames:
        print("No frames could be read. Check your HDF files.")
        return

    stack = np.stack(frames, axis=0).astype(np.float32)
    print(f"Raw stack shape: {stack.shape}  (timesteps × height × width)")

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
