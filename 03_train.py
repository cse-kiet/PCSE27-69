import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 6  # shorter sequence — more samples from small dataset
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
PATCH_SIZE = 16  # smaller patches — more samples, less overfitting
HIDDEN_DIM = 16  # smaller model — right-sized for our dataset
# ──────────────────────────────────────────────────────────────────────────────


# ── Dataset ───────────────────────────────────────────────────────────────────
class LSTDataset(Dataset):
    def __init__(self, stack, seq_len, patch_size, device):
        self.stack = torch.tensor(stack, dtype=torch.float32).to(device)
        self.seq_len = seq_len
        self.patch_size = patch_size
        T, H, W = stack.shape
        self.time_starts = list(range(T - seq_len))
        self.patches = [
            (r, c)
            for r in range(0, H - patch_size, patch_size // 4)
            for c in range(0, W - patch_size, patch_size // 4)
        ]

    def __len__(self):
        return len(self.time_starts) * len(self.patches)

    def __getitem__(self, idx):
        t_idx = idx // len(self.patches)
        p_idx = idx % len(self.patches)
        t = self.time_starts[t_idx]
        r, c = self.patches[p_idx]
        p = self.patch_size

        seq = self.stack[t : t + self.seq_len, r : r + p, c : c + p]
        seq = seq.unsqueeze(1)
        x = seq[:-1]
        y = seq[-1]
        return x, y


# ── Model ─────────────────────────────────────────────────────────────────────
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


# ── Early stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ── Training ──────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    stack = np.load(os.path.join(PROCESSED_DIR, "lst_stack.npy"))
    print(f"Loaded tensor: {stack.shape}")

    split = int(stack.shape[0] * 0.8)
    train_arr = stack[:split]
    val_arr = stack[split:]

    train_ds = LSTDataset(train_arr, SEQ_LEN, PATCH_SIZE, device)
    val_ds = LSTDataset(val_arr, SEQ_LEN, PATCH_SIZE, device)

    print(f"Train patches: {len(train_ds):,} | Val patches: {len(val_ds):,}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = ConvLSTMForecaster(hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=7, min_delta=0.001)
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS} [train]", leave=False):
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                val_loss += criterion(model(x), y).item()

        train_loss /= len(train_dl)
        val_loss /= len(val_dl)
        print(
            f"Epoch {epoch:3d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))

        if early_stop(val_loss):
            print(f"\nEarly stopping at epoch {epoch}. Best val MSE: {best_val:.4f}")
            break

    print(f"\nTraining complete. Best val MSE: {best_val:.4f}")
    print(f"Model saved to outputs/best_model.pt")

    # ── Statistical anomaly map ───────────────────────────────────────────────
    print("\nGenerating statistical anomaly map from tensor...")
    mean_lst = np.load(os.path.join(PROCESSED_DIR, "lst_mean.npy")).squeeze()
    std_lst  = np.load(os.path.join(PROCESSED_DIR, "lst_std.npy")).squeeze()

    T, H, W  = stack.shape
    x_time   = np.arange(T, dtype=np.float32)
    x_time  -= x_time.mean()

    # Vectorized linear regression across all pixels
    stack_f  = stack.reshape(T, -1)
    slope    = (x_time @ stack_f) / (x_time @ x_time)
    slope    = slope.reshape(H, W)

    # Convert slope from normalized units back to Celsius per timestep
    # Each timestep is 8 days, so multiply by (365/8) to get °C per year
    slope_celsius = slope * std_lst * (365 / 8)

    np.save(os.path.join(OUTPUT_DIR, "trend_slope.npy"),         slope)
    np.save(os.path.join(OUTPUT_DIR, "trend_slope_celsius.npy"), slope_celsius)

    print("Saved trend slope map to outputs/trend_slope.npy")
    print(f"Warming trend range: {slope_celsius.min():.2f} to {slope_celsius.max():.2f} °C/year")


if __name__ == "__main__":
    main()
