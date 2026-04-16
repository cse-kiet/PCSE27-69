import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "outputs"
HIDDEN_DIM = 16
SEQ_LEN = 6
PATCH_SIZE = 16
# ──────────────────────────────────────────────────────────────────────────────


# ── Model definition (copied here so we don't need to import 03_train.py) ────
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=pad
        )
        self.hidden_channels = hidden_channels

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch, h, w, device):
        return (
            torch.zeros(batch, self.hidden_channels, h, w, device=device),
            torch.zeros(batch, self.hidden_channels, h, w, device=device),
        )


class ConvLSTMForecaster(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels=1, hidden_channels=hidden_dim)
        self.dropout = nn.Dropout2d(p=0.2)
        self.decoder = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        device = x_seq.device
        h, c = self.cell.init_hidden(B, H, W, device)
        for t in range(T):
            h, c = self.cell(x_seq[:, t], h, c)
        h = self.dropout(h)
        pred = self.decoder(h)
        return pred


# ── Anomaly map from model reconstruction error ───────────────────────────────
def compute_error_map(model, stack, seq_len, patch_size, device):
    T, H, W = stack.shape
    error_accum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    stride = patch_size // 2
    p = patch_size

    model.eval()
    stack_t = torch.tensor(stack, dtype=torch.float32).to(device)

    with torch.no_grad():
        for t in tqdm(range(T - seq_len), desc="Computing reconstruction error"):
            seq = stack_t[t : t + seq_len]  # (SEQ_LEN, H, W)
            for r in range(0, H - p, stride):
                for c in range(0, W - p, stride):
                    xp = seq[:-1, r : r + p, c : c + p]  # (SEQ_LEN-1, p, p)
                    xp = xp.unsqueeze(0).unsqueeze(2)  # (1, T, 1, p, p)
                    pred = model(xp).squeeze().cpu().numpy()  # (p, p)
                    real = seq[-1, r : r + p, c : c + p].cpu().numpy()
                    error_accum[r : r + p, c : c + p] += (pred - real) ** 2
                    count[r : r + p, c : c + p] += 1

    count = np.where(count == 0, 1, count)
    return error_accum / count


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    stack = np.load(os.path.join(PROCESSED_DIR, "lst_stack.npy"))
    mean_celsius = np.load(
        os.path.join(PROCESSED_DIR, "lst_mean_celsius.npy")
    ).squeeze()
    std_celsius = np.load(os.path.join(PROCESSED_DIR, "lst_std_celsius.npy")).squeeze()
    slope_norm = np.load(os.path.join(OUTPUT_DIR, "trend_slope.npy"))

    T, H, W = stack.shape
    print(f"Tensor shape: {stack.shape}")
    print(f"Mean LST range: {mean_celsius.min():.1f} to {mean_celsius.max():.1f} °C")

    # ── Fix slope scaling ─────────────────────────────────────────────────────
    # Use median std to avoid extreme pixels (cloud artifacts) blowing up the scale
    median_std    = np.median(std_celsius[std_celsius > 0])
    slope_celsius = slope_norm * median_std * (365 / 8)
    slope_celsius = np.clip(slope_celsius, -3.0, 3.0)
    print(f"Median std: {median_std:.2f}")
    print(f"Slope range: {slope_celsius.min():.3f} to {slope_celsius.max():.3f} °C/year")

    # ── Load model and compute reconstruction error map ───────────────────────
    model = ConvLSTMForecaster(hidden_dim=HIDDEN_DIM).to(device)
    model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"), map_location=device)
    )
    print("Computing reconstruction error map (this takes a few minutes)...")
    error_map = compute_error_map(model, stack, SEQ_LEN, PATCH_SIZE, device)

    # ── Remove horizontal stripe artifacts ────────────────────────────────────
    from scipy.ndimage import median_filter

    mean_celsius_clean  = median_filter(mean_celsius,  size=5)
    slope_celsius_clean = median_filter(slope_celsius, size=5)
    error_map_clean     = median_filter(error_map,     size=5)

    # Fix black border — pad error map edges with nearest valid values
    error_map_clean = np.pad(
        error_map_clean[PATCH_SIZE//2:-PATCH_SIZE//2, PATCH_SIZE//2:-PATCH_SIZE//2],
        PATCH_SIZE//2,
        mode="edge"
    )

    # ── Combined anomaly score ────────────────────────────────────────────────
    def norm01(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-8)

    error_norm    = norm01(error_map_clean)
    slope_norm_01 = norm01(np.clip(slope_celsius_clean, 0, None))
    combined      = 0.5 * error_norm + 0.5 * slope_norm_01

    # ── Flagged regions (top 15% combined anomaly score) ─────────────────────
    threshold = np.percentile(combined, 85)
    flagged = (combined > threshold).astype(np.float32)

    # ── Save arrays for Streamlit ─────────────────────────────────────────────
    np.save(os.path.join(OUTPUT_DIR, "error_map.npy"), error_map)
    np.save(os.path.join(OUTPUT_DIR, "combined_anomaly.npy"), combined)
    np.save(os.path.join(OUTPUT_DIR, "flagged_regions.npy"), flagged)
    np.save(os.path.join(OUTPUT_DIR, "slope_celsius.npy"), slope_celsius)
    np.save(os.path.join(OUTPUT_DIR, "mean_celsius.npy"), mean_celsius)
    print("Saved all output arrays to outputs/")

    # ── Figure 1: Main results (2×2 grid) ────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Deep Learning for Climate Pattern Change Detection\n"
        "Sub-Saharan Africa — MODIS LST (2018–2022)",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel 1: Mean LST
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(mean_celsius_clean, cmap="RdYlBu_r", interpolation="bilinear")
    ax1.set_title("Mean Land Surface Temperature (°C)", fontsize=11)
    ax1.set_xlabel("← West Africa   |   East Africa →", fontsize=8)
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, label="°C", shrink=0.85)

    # Panel 2: Warming trend
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        slope_celsius_clean, cmap="RdYlBu_r", vmin=-1.5, vmax=1.5, interpolation="bilinear"
    )
    ax2.set_title("Warming Trend (°C/year)", fontsize=11)
    ax2.set_xlabel("← West Africa   |   East Africa →", fontsize=8)
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, label="°C/year", shrink=0.85)

    # Panel 3: Reconstruction error
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(
        error_map_clean,
        cmap="hot",
        norm=mcolors.PowerNorm(gamma=0.5),
        interpolation="bilinear",
    )
    ax3.set_title(
        "ConvLSTM Reconstruction Error\n(higher = more anomalous)", fontsize=11
    )
    ax3.set_xlabel("← West Africa   |   East Africa →", fontsize=8)
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, label="MSE", shrink=0.85)

    # Panel 4: Combined anomaly + flagged regions
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(combined, cmap="YlOrRd", interpolation="bilinear")
    # Overlay flagged regions as contour
    ax4.contour(flagged, levels=[0.5], colors="blue", linewidths=0.8, alpha=0.7)
    ax4.set_title(
        "Combined Anomaly Score\n(blue outline = flagged desertification zones)",
        fontsize=11,
    )
    ax4.set_xlabel("← West Africa   |   East Africa →", fontsize=8)
    ax4.axis("off")

    out1 = os.path.join(OUTPUT_DIR, "main_results.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.show()

    # ── Figure 2: Regional time series ───────────────────────────────────────
    # Split into west tile (h17) and east tile (h18) and plot mean LST over time
    mid = W // 2
    dates = np.linspace(2018, 2023, T)  # approximate decimal years

    west_series = stack[:, :, :mid].mean(axis=(1, 2))
    east_series = stack[:, :, mid:].mean(axis=(1, 2))

    # Smooth with a rolling mean (window=8, ~2 months)
    def rolling_mean(arr, w=8):
        return np.convolve(arr, np.ones(w) / w, mode="same")

    fig2, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, west_series, alpha=0.3, color="coral", linewidth=0.8)
    ax.plot(dates, east_series, alpha=0.3, color="steelblue", linewidth=0.8)
    ax.plot(
        dates,
        rolling_mean(west_series),
        color="coral",
        linewidth=2,
        label="West Sahel (h17v07)",
    )
    ax.plot(
        dates,
        rolling_mean(east_series),
        color="steelblue",
        linewidth=2,
        label="East Sahel (h18v07)",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Normalized LST (z-score)")
    ax.set_title(
        "Regional LST Time Series — Sub-Saharan Africa (2018–2022)", fontsize=12
    )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    out2 = os.path.join(OUTPUT_DIR, "time_series.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.show()

    # ── Summary stats ─────────────────────────────────────────────────────────
    flagged_pct = flagged.mean() * 100
    max_warming = slope_celsius.max()
    mean_warming = slope_celsius[slope_celsius > 0].mean()

    print("\n── Summary ──────────────────────────────────────────────────")
    print(f"  Flagged area:        {flagged_pct:.1f}% of region")
    print(f"  Max warming trend:   {max_warming:.2f} °C/year")
    print(f"  Mean warming trend:  {mean_warming:.2f} °C/year (positive pixels only)")
    print("  Best val MSE:        0.9561")
    print("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
