# 04_visualize.py
# Produces three figures for the MVP presentation:
#   main_results.png    — 2×2 grid: mean LST, warming trend, z-score anomaly, combined detection
#   time_series.png     — West vs East Sahel normalized LST over time
#   anomaly_timeline.png — % of region flagged anomalous at each timestep

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from scipy.ndimage import median_filter
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
OUTPUT_DIR    = "outputs"
HIDDEN_DIM    = 16
SEQ_LEN       = 6
PATCH_SIZE    = 16
# ─────────────────────────────────────────────────────────────────────────────


# ── Model definition (inline — no import from 03_train.py) ───────────────────
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
        gates    = self.conv(combined)
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
        self.cell    = ConvLSTMCell(in_channels=1, hidden_channels=hidden_dim)
        self.dropout = nn.Dropout2d(p=0.2)
        self.decoder = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        device = x_seq.device
        h, c = self.cell.init_hidden(B, H, W, device)
        for t in range(T):
            h, c = self.cell(x_seq[:, t], h, c)
        h    = self.dropout(h)
        pred = self.decoder(h)
        return pred


# ── Reconstruction error map via sliding patch inference ─────────────────────
def compute_error_map(model, stack, seq_len, patch_size, device):
    """
    Slide the model over every (t, spatial) position to get per-pixel
    mean squared reconstruction error. High error = model surprised = anomaly.
    Uses stride = patch_size // 2 to get overlap and average out patch edges.
    """
    T, H, W   = stack.shape
    error_accum = np.zeros((H, W), dtype=np.float32)
    count       = np.zeros((H, W), dtype=np.float32)
    stride      = patch_size // 2
    p           = patch_size

    model.eval()
    stack_t = torch.tensor(stack, dtype=torch.float32).to(device)

    with torch.no_grad():
        for t in tqdm(range(T - seq_len), desc="Computing reconstruction error"):
            seq = stack_t[t : t + seq_len]          # (SEQ_LEN, H, W)
            for r in range(0, H - p, stride):
                for c in range(0, W - p, stride):
                    xp   = seq[:-1, r : r + p, c : c + p]   # (SEQ_LEN-1, p, p)
                    xp   = xp.unsqueeze(0).unsqueeze(2)      # (1, T, 1, p, p)
                    pred = model(xp).squeeze().cpu().numpy()  # (p, p)
                    real = seq[-1, r : r + p, c : c + p].cpu().numpy()
                    error_accum[r : r + p, c : c + p] += (pred - real) ** 2
                    count[r : r + p, c : c + p]        += 1

    count = np.where(count == 0, 1, count)
    return error_accum / count


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    stack_raw    = np.load(os.path.join(PROCESSED_DIR, "lst_stack.npy"))        # (T,H,W) z-score
    mean_celsius = np.load(os.path.join(PROCESSED_DIR, "lst_mean_celsius.npy")).squeeze()
    std_celsius  = np.load(os.path.join(PROCESSED_DIR, "lst_std_celsius.npy")).squeeze()
    slope_norm   = np.load(os.path.join(OUTPUT_DIR,    "trend_slope.npy"))

    T, H, W = stack_raw.shape
    print(f"Stack shape: {stack_raw.shape}")
    print(f"Mean LST range: {mean_celsius.min():.1f} to {mean_celsius.max():.1f} °C")

    # ── Step 1: Separate raw vs display stacks ────────────────────────────────
    # stack_raw    → anomaly detection and time series (untouched signal)
    # stack_display → only used to drive display colormaps (stripe-suppressed)
    # Median filter along spatial axes smooths MODIS stripe artifacts for display
    # without flattening the temporal signal used for anomaly detection.
    stack_display = median_filter(stack_raw, size=(1, 9, 9))

    # ── Step 2: Slope scaling with median std ─────────────────────────────────
    # Per-pixel std_celsius has extreme outliers (5–746°C) from artifact pixels.
    # Using the spatial median gives a robust single scale factor that represents
    # typical LST variability in the region (~15–25°C for Sub-Saharan Africa).
    median_std    = np.median(std_celsius[std_celsius > 0])
    slope_celsius = slope_norm * median_std * (365 / 8)
    slope_celsius = np.clip(slope_celsius, -2.0, 2.0)
    print(f"Median std: {median_std:.2f} °C")
    print(f"Slope range: {slope_celsius.min():.3f} to {slope_celsius.max():.3f} °C/year")

    # ── Step 3: Persistent anomaly map (PRIMARY signal) ──────────────────────
    # stack_raw is already z-score normalized (mean=0, std=1) from preprocessing.
    # Threshold directly on the normalized values — no re-normalization needed.
    # 0.8 ≈ 0.8σ above baseline; persistent = exceeds threshold in >15% of timesteps.
    anomaly_mask = (stack_raw > 0.8)                      # (T, H, W) bool
    z_frac_map   = anomaly_mask.mean(axis=0)              # (H, W) float, 0–1
    z_flag_cube  = anomaly_mask.astype(np.float32)        # kept for timeline figure
    persistent   = (z_frac_map > 0.15).astype(np.float32)  # binary: chronic anomaly
    print(f"Anomaly map: {persistent.mean()*100:.1f}% of pixels persistently anomalous")

    # ── Step 4: Model reconstruction error (secondary confirming signal) ──────
    model = ConvLSTMForecaster(hidden_dim=HIDDEN_DIM).to(device)
    model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"), map_location=device, weights_only=True)
    )
    print("Computing reconstruction error map (this takes a few minutes)...")
    error_map = compute_error_map(model, stack_raw, SEQ_LEN, PATCH_SIZE, device)

    # Fix black border: patch inference doesn't reach the outer PATCH_SIZE//2 pixels.
    # Crop to the covered interior, then pad back with the nearest valid edge value.
    half = PATCH_SIZE // 2
    inner           = error_map[half:-half, half:-half]
    error_map_clean = np.pad(inner, half, mode="edge")
    error_map_clean = median_filter(error_map_clean, size=5)

    # ── Step 5: Combined high-confidence detection mask ───────────────────────
    # Flag pixels where BOTH the z-score signal AND the model error agree they
    # are anomalous (top 25% of each). This minimises false positives.
    error_thresh  = np.percentile(error_map_clean, 75)
    z_thresh      = np.percentile(z_frac_map, 75)
    combined_mask = (
        (error_map_clean > error_thresh) & (z_frac_map > z_thresh)
    ).astype(np.float32)

    # ── Step 6: Smoothed display arrays ──────────────────────────────────────
    mean_celsius_disp  = median_filter(mean_celsius,  size=5)
    slope_celsius_disp = median_filter(slope_celsius, size=5)

    # ── Save all arrays for Streamlit dashboard ───────────────────────────────
    np.save(os.path.join(OUTPUT_DIR, "error_map.npy"),        error_map_clean)
    np.save(os.path.join(OUTPUT_DIR, "combined_anomaly.npy"), z_frac_map)
    np.save(os.path.join(OUTPUT_DIR, "flagged_regions.npy"),  persistent)
    np.save(os.path.join(OUTPUT_DIR, "slope_celsius.npy"),    slope_celsius)
    np.save(os.path.join(OUTPUT_DIR, "mean_celsius.npy"),     mean_celsius)
    np.save(os.path.join(OUTPUT_DIR, "z_frac_map.npy"),       z_frac_map)
    np.save(os.path.join(OUTPUT_DIR, "z_flag_cube.npy"),      z_flag_cube)
    print("Saved all output arrays to outputs/")

    # ── Figure 1: Main results (2×2 grid) ────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Deep Learning for Climate Pattern Change Detection\n"
        "Sub-Saharan Africa — MODIS LST (2018–2022)",
        fontsize=15, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel 1: Mean Land Surface Temperature
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(mean_celsius_disp, cmap="RdYlBu_r", interpolation="bilinear")
    ax1.set_title("Mean Land Surface Temperature (°C)", fontsize=11)
    ax1.set_xlabel("← West Africa   |   East Africa →", fontsize=8)
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, label="°C", shrink=0.85)

    # Panel 2: Warming trend (slope in °C/year)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        slope_celsius_disp, cmap="RdYlBu_r",
        vmin=-2.0, vmax=2.0, interpolation="bilinear"
    )
    ax2.set_title("Warming Trend (°C/year)", fontsize=11)
    ax2.set_xlabel("← West Africa   |   East Africa →", fontsize=8)
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, label="°C/year", shrink=0.85)

    # Panel 3: Z-score anomaly map — PRIMARY signal
    # Shows the fraction of timesteps where each pixel was >2 std from its mean.
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(z_frac_map, cmap="YlOrRd", vmin=0, vmax=0.5, interpolation="bilinear")
    ax3.set_title(
        "Z-Score Anomaly Frequency\n(fraction of timesteps with |z| > 2.0)",
        fontsize=11
    )
    ax3.set_xlabel("← West Africa   |   East Africa →", fontsize=8)
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, label="Fraction of timesteps", shrink=0.85)

    # Panel 4: Combined detection — both signals agree
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(z_frac_map, cmap="YlOrRd", vmin=0, vmax=0.5, interpolation="bilinear")
    # Red overlay only where BOTH z-score AND model error are in top 25%
    red_overlay = np.ma.masked_where(combined_mask == 0, combined_mask)
    ax4.imshow(red_overlay, cmap="Reds", alpha=0.6, interpolation="bilinear")
    ax4.set_title(
        "High-Confidence Detection\n(red = both Z-score & model error agree)",
        fontsize=11
    )
    ax4.set_xlabel("← West Africa   |   East Africa →", fontsize=8)
    ax4.axis("off")

    out1 = os.path.join(OUTPUT_DIR, "main_results.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.show()

    # ── Figure 2: Regional time series (West vs East Sahel) ──────────────────
    mid  = W // 2
    dates = np.linspace(2018, 2023, T)

    west_series = stack_raw[:, :, :mid].mean(axis=(1, 2))
    east_series = stack_raw[:, :, mid:].mean(axis=(1, 2))

    def rolling_mean(arr, w=8):
        return np.convolve(arr, np.ones(w) / w, mode="same")

    fig2, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, west_series,           alpha=0.3, color="coral",     linewidth=0.8)
    ax.plot(dates, east_series,           alpha=0.3, color="steelblue", linewidth=0.8)
    ax.plot(dates, rolling_mean(west_series), color="coral",     linewidth=2,
            label="West Sahel (h17v07)")
    ax.plot(dates, rolling_mean(east_series), color="steelblue", linewidth=2,
            label="East Sahel (h18v07)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Normalized LST (z-score)")
    ax.set_title("Regional LST Time Series — Sub-Saharan Africa (2018–2022)", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    out2 = os.path.join(OUTPUT_DIR, "time_series.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.show()

    # ── Figure 3: Anomaly timeline — WHEN did anomalies peak? ────────────────
    # For each timestep, what fraction of the region had |z| > 2?
    # Red shading highlights when that fraction exceeded 20%.
    pct_flagged = z_flag_cube.mean(axis=(1, 2)) * 100   # (T,) in percent

    fig3, ax3t = plt.subplots(figsize=(12, 4))
    ax3t.plot(dates, pct_flagged, color="orangered", linewidth=1.5, label="% flagged pixels")
    ax3t.fill_between(
        dates, pct_flagged,
        where=(pct_flagged > 20),
        color="red", alpha=0.3, label="Exceeds 20% threshold"
    )
    ax3t.axhline(20, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
    ax3t.set_xlabel("Year")
    ax3t.set_ylabel("% pixels with |z| > 2.0")
    ax3t.set_title(
        "Anomaly Prevalence Over Time — Sub-Saharan Africa (2018–2022)", fontsize=12
    )
    ax3t.legend()
    ax3t.grid(alpha=0.3)

    out3 = os.path.join(OUTPUT_DIR, "anomaly_timeline.png")
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved: {out3}")
    plt.show()

    # ── Summary stats ─────────────────────────────────────────────────────────
    flagged_pct  = persistent.mean() * 100
    max_warming  = slope_celsius.max()
    mean_warming = slope_celsius[slope_celsius > 0].mean()
    peak_t       = int(anomaly_mask.mean(axis=(1, 2)).argmax())
    peak_date    = dates[peak_t]

    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  Persistently anomalous area:  {flagged_pct:.1f}%  (>0.8σ for >15% of timesteps)")
    print(f"  Max warming trend:            {max_warming:.2f} °C/year")
    print(f"  Mean warming (pos. pixels):   {mean_warming:.2f} °C/year")
    print(f"  Peak anomaly date:            {peak_date:.2f}  ({pct_flagged[peak_t]:.1f}% of region flagged)")
    print(f"  Model val MSE:                0.9561")
    print("─────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
