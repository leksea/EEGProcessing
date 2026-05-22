"""
hms_model.py  —  Multimodal CNN for HMS-HBAC
Alexandra Yakovleva, 2026
=============================================
Two-branch architecture:
  · 1D CNN   — EEG + EKG time series   (batch, 17, 5000)
  · 2D CNN   — Spectrogram image       (batch, 6, 100, 300)

Both branches produce a 128-d embedding.  Embeddings are
concatenated and passed through a shared fusion head that
outputs a 6-class probability distribution.

Loss: KL divergence against soft annotator-vote labels.

Usage
-----
    from hms_model import HMSModel, HMSLoss, HMSDataset

    model = HMSModel()
    loss_fn = HMSLoss()

    # Single forward pass
    eeg  = torch.randn(4, 17, 5000)   # (batch, channels, time)
    spec = torch.randn(4, 6, 100, 300) # (batch, chains, freq, time)
    out  = model(eeg, spec)            # (4, 6) — softmax probabilities

    # Training step
    labels = torch.softmax(torch.randn(4, 6), dim=-1)  # soft targets
    loss   = loss_fn(out, labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os


# ── Constants ────────────────────────────────────────────────────────────────

N_CLASSES    = 6          # seizure, lpd, gpd, lrda, grda, other
EEG_CHANNELS = 17         # 16 bipolar + 1 EKG
EEG_LENGTH   = 5000       # 50 s × 100 Hz
SPEC_FREQ    = 50         # frequency bins
SPEC_TIME    = 150        # time steps (10-min window)
SPEC_CHAINS  = 6          # LL, LP, RP, RL, LL-RL, LP-RP
EMBED_DIM    = 128        # shared embedding size


# ══════════════════════════════════════════════════════════════════════════════
#  1D CNN BRANCH — EEG + EKG
# ══════════════════════════════════════════════════════════════════════════════

class SEBlock1d(nn.Module):
    """Squeeze-and-Excitation channel attention for 1-D feature maps."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        w = self.fc(x).unsqueeze(-1)   # (B, C, 1)
        return x * w


class ResBlock1d(nn.Module):
    """
    Residual block for 1-D temporal signals.
    Downsamples by stride=2 on the first conv when stride > 1.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 7, stride: int = 2):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                      padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.se  = SEBlock1d(out_ch)
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) * self.se(self.conv(x))
                        + self.skip(x))

    # Cleaner version (avoids double conv call)
    def forward(self, x):
        h = self.conv(x)
        h = self.se(h) * h
        return self.act(h + self.skip(x))


class EEGBranch(nn.Module):
    """
    1-D CNN for EEG + EKG time series.

    Input  : (B, 17, 5000)  — 16 bipolar channels + 1 EKG
    Output : (B, EMBED_DIM) — 128-d embedding

    Architecture
    ------------
    1. Channel projection   — Conv1d 17→32, k=1  (mix channels)
    2. Multi-scale stem     — three parallel Conv1d with k=7/15/31
                              concatenate → project 96→64
    3. Residual blocks × 3  — 64→64→128→128, stride=2 each
    4. SE channel attention  — reweight 128 channels
    5. Adaptive avg pool + flatten → Linear 128*4→EMBED_DIM
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()

        # Step 1: mix across channels (pointwise in time)
        self.channel_proj = nn.Sequential(
            nn.Conv1d(EEG_CHANNELS, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # Step 2: multi-scale temporal feature extraction
        self.scale7  = self._conv_bn(32, 32, kernel=7)
        self.scale15 = self._conv_bn(32, 32, kernel=15)
        self.scale31 = self._conv_bn(32, 32, kernel=31)
        self.merge   = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Step 3: residual blocks with downsampling (stride=2 each)
        self.res_blocks = nn.Sequential(
            ResBlock1d(64,  64,  kernel=7, stride=2),   # T/2
            ResBlock1d(64,  128, kernel=7, stride=2),   # T/4
            ResBlock1d(128, 128, kernel=7, stride=2),   # T/8
        )

        # Step 4: channel attention
        self.se = SEBlock1d(128)

        # Step 5: pool and embed
        self.pool = nn.AdaptiveAvgPool1d(5)   # → (B, 125, 5)
        self.head = nn.Sequential(
            nn.Flatten(),                      # → (B, 640)
            nn.Linear(640, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    @staticmethod
    def _conv_bn(in_ch, out_ch, kernel):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, 17, T)
        x = self.channel_proj(x)                          # (B, 32, T)
        x = self.merge(torch.cat([
            self.scale7(x),
            self.scale15(x),
            self.scale31(x),
        ], dim=1))                                         # (B, 64, T)
        x = self.res_blocks(x)                            # (B, 128, T/8)
        x = self.se(x) * x                                # (B, 128, T/8)
        x = self.pool(x)                                   # (B, 128, 5)
        return self.head(x)                                # (B, embed_dim)

# ══════════════════════════════════════════════════════════════════════════════
#  2D CNN BRANCH — Spectrogram
# ══════════════════════════════════════════════════════════════════════════════

class DepthwiseSepConv2d(nn.Module):
    """Depthwise-separable 2-D convolution (cheaper than standard Conv2d)."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride=stride,
                            padding=pad, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class InvertedResidual2d(nn.Module):
    """
    MobileNetV2-style inverted residual block.
    expand_ratio × in_ch intermediate channels, then project back.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, expand_ratio: int = 6):
        super().__init__()
        mid = in_ch * expand_ratio
        self.use_skip = (stride == 1 and in_ch == out_ch)
        self.conv = nn.Sequential(
            # Expand
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            # Depthwise
            nn.Conv2d(mid, mid, 3, stride=stride,
                      padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            # Project
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        h = self.conv(x)
        return h + x if self.use_skip else h


class SpatialAttention2d(nn.Module):
    """
    CBAM-style spatial attention.
    Computes attention weights over the (H, W) spatial dimensions.
    """

    def __init__(self, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        avg = x.mean(dim=1, keepdim=True)    # (B, 1, H, W)
        mx  = x.amax(dim=1, keepdim=True)    # (B, 1, H, W)
        w   = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class SpectrogramBranch(nn.Module):
    """
    2-D CNN for spectrogram images.

    Input  : (B, 6, 100, 300)  — 6 chains × freq × time
    Output : (B, EMBED_DIM)    — 128-d embedding

    Architecture
    ------------
    1. Stem            — Conv2d 6→32, 3×3, s=2  →  (B, 32, 50, 150)
    2. DS blocks × 2   — depthwise-separable, s=2 each
                         32→64→128             →  (B, 128, 12, 37)
    3. IR blocks × 4   — inverted residual, expand=6
                         128→192→256, s=2 last  →  (B, 256, 6, 18)
    4. Spatial attn    — CBAM spatial gate
    5. Global avg pool + embed → Linear 256→EMBED_DIM
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()

        # Step 1: stem
        self.stem = nn.Sequential(
            nn.Conv2d(SPEC_CHAINS, 32, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Step 2: depthwise-separable blocks
        self.ds_blocks = nn.Sequential(
            DepthwiseSepConv2d(32,  64,  stride=2),
            DepthwiseSepConv2d(64,  128, stride=2),
        )

        # Step 3: inverted residual blocks
        self.ir_blocks = nn.Sequential(
            InvertedResidual2d(128, 128, stride=1, expand_ratio=6),
            InvertedResidual2d(128, 192, stride=2, expand_ratio=6),
            InvertedResidual2d(192, 192, stride=1, expand_ratio=6),
            InvertedResidual2d(192, 256, stride=1, expand_ratio=6),
        )

        # Step 4: spatial attention
        self.spatial_attn = SpatialAttention2d(kernel=7)

        # Step 5: pool and embed
        self.pool = nn.AdaptiveAvgPool2d(1)   # → (B, 256, 1, 1)
        self.head = nn.Sequential(
            nn.Flatten(),                      # → (B, 256)
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (B, 6, F, T)
        x = self.stem(x)           # (B, 32,  50, 150)
        x = self.ds_blocks(x)      # (B, 128, 12,  37)
        x = self.ir_blocks(x)      # (B, 256,  6,  18)
        x = self.spatial_attn(x)   # (B, 256,  6,  18)
        x = self.pool(x)           # (B, 256,  1,   1)
        return self.head(x)        # (B, embed_dim)


# ══════════════════════════════════════════════════════════════════════════════
#  FUSION + CLASSIFICATION HEAD
# ══════════════════════════════════════════════════════════════════════════════

class FusionHead(nn.Module):
    """
    Concatenate both embeddings and classify.

    Input  : eeg_emb (B, 128) + spec_emb (B, 128) → concat (B, 256)
    Output : (B, 6) softmax probabilities
    """

    def __init__(self, embed_dim: int = EMBED_DIM,
                 n_classes: int = N_CLASSES,
                 dropout: float = 0.3):
        super().__init__()
        fused = embed_dim * 2   # 256
        self.net = nn.Sequential(
            nn.Linear(fused, fused // 2),   # 256 → 128
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused // 2, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, n_classes),
        )

    def forward(self, eeg_emb, spec_emb):
        x = torch.cat([eeg_emb, spec_emb], dim=-1)
        return F.softmax(self.net(x), dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
#  FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════

class HMSModel(nn.Module):
    """
    Multimodal HMS harmful brain activity classifier.

    Parameters
    ----------
    embed_dim : int   — shared embedding dimension (default 128)
    dropout   : float — dropout rate in fusion head (default 0.3)

    Inputs
    ------
    eeg  : (B, 17, 5000)   — 16 bipolar EEG + 1 EKG, 100 Hz, 50 s
    spec : (B, 6, 100, 300) — LL, LP, RP, RL, LL-RL, LP-RP chains

    Output
    ------
    (B, 6)  — softmax probability distribution over 6 classes:
               seizure · lpd · gpd · lrda · grda · other
    """

    def __init__(self,
                 embed_dim : int   = EMBED_DIM,
                 dropout   : float = 0.3):
        super().__init__()
        self.eeg_branch  = EEGBranch(embed_dim)
        self.spec_branch = SpectrogramBranch(embed_dim)
        self.fusion      = FusionHead(embed_dim, N_CLASSES, dropout)

    def forward(self, eeg: torch.Tensor, spec: torch.Tensor):
        eeg_emb  = self.eeg_branch(eeg)    # (B, 128)
        spec_emb = self.spec_branch(spec)  # (B, 128)
        return self.fusion(eeg_emb, spec_emb)

    def encode(self, eeg: torch.Tensor, spec: torch.Tensor):
        """Return concatenated embedding without classifying — for analysis."""
        return torch.cat([
            self.eeg_branch(eeg),
            self.spec_branch(spec),
        ], dim=-1)   # (B, 256)


# ══════════════════════════════════════════════════════════════════════════════
#  LOSS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

class HMSLoss(nn.Module):
    """
    KL divergence loss against soft annotator-vote labels.

    target is a probability distribution derived from raw vote counts:
        target[i] = votes[i] / votes.sum()

    KLDiv expects log-probabilities for the input, so we apply log() here.
    reduction='batchmean' divides by batch size (standard KL convention).
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # pred   : (B, 6)  softmax probabilities (already >= 0, sum to 1)
        # target : (B, 6)  soft labels (already normalised vote fractions)
        if self.label_smoothing > 0:
            target = (1 - self.label_smoothing) * target \
                     + self.label_smoothing / N_CLASSES
        return self.kl(torch.log(pred.clamp(min=1e-7)), target)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

LABEL_COLS = ["seizure_vote","lpd_vote","gpd_vote",
              "lrda_vote","grda_vote","other_vote"]


class HMSDataset(Dataset):
    """
    PyTorch Dataset for HMS-HBAC.

    Expects:
        processed_dir     — directory with {eeg_id}_{offset}.npy (z-scored EEG)
        processed_raw_dir — directory with {eeg_id}_{offset}.npy (raw EEG for EKG)
        spec_dir          — directory with {eeg_id}_{offset}_spec.npy
        df                — train.csv DataFrame (already split into train/val)

    Returns per sample:
        eeg    : (17, 5000) float32  — 16 bipolar + EKG appended as ch 16
        spec   : (6, 100, 300) float32
        target : (6,) float32  — normalised vote fractions
    """

    def __init__(self,
                 df,
                 processed_dir:     str,
                 processed_raw_dir: str,
                 spec_dir:          str,
                 eeg_dir:           str  = None,
                 augment:           bool = False):
        self.df                = df.reset_index(drop=True)
        self.processed_dir     = processed_dir
        self.processed_raw_dir = processed_raw_dir
        self.spec_dir          = spec_dir
        self.eeg_dir           = eeg_dir
        self.augment           = augment
        votes        = df[LABEL_COLS].values.astype(np.float32)
        total        = votes.sum(axis=1, keepdims=True).clip(min=1)
        self.targets = votes / total

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        eid    = int(row["eeg_id"])
        offset = int(row["eeg_label_offset_seconds"])
        stem   = f"{eid}_{offset}"

        # ── EEG (16 bipolar, z-scored) ────────────────────────
        eeg = np.load(
            os.path.join(self.processed_dir, f"{stem}.npy")
        ).T.astype(np.float32)                  # (16, T)

        # ── EKG from raw parquet ──────────────────────────────
        ekg = np.zeros((1, eeg.shape[1]), dtype=np.float32)
        if self.eeg_dir:
            ekg_path = os.path.join(self.eeg_dir, f"{eid}.parquet")
            if os.path.exists(ekg_path):
                try:
                    raw_df  = pd.read_parquet(ekg_path,
                                              columns=["EKG"])
                    t0      = offset * 200
                    t1      = t0 + 50 * 200
                    ekg_raw = raw_df["EKG"].values[t0:t1]\
                                  .astype(np.float32)
                    ekg_raw = ekg_raw[::2]            # 200→100 Hz
                    mu      = np.nanmean(ekg_raw)
                    std     = np.nanstd(ekg_raw) + 1e-8
                    ekg_raw = (np.nan_to_num(ekg_raw) - mu) / std
                    T       = eeg.shape[1]
                    n       = min(len(ekg_raw), T)
                    ekg[0, :n] = ekg_raw[:n]
                except Exception:
                    pass

        # ── Concat → (17, T) ──────────────────────────────────
        eeg = np.concatenate([eeg, ekg], axis=0)
        T   = eeg.shape[1]
        if T < EEG_LENGTH:
            eeg = np.pad(eeg, ((0,0),(0, EEG_LENGTH-T)))
        else:
            eeg = eeg[:, :EEG_LENGTH]

        # ── Spectrogram ───────────────────────────────────────
        spec_path = os.path.join(self.spec_dir, f"{stem}_spec.npy")

        if os.path.exists(spec_path):
            spec = np.load(spec_path).astype(np.float32)
            spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)

            # Ensure shape is (SPEC_CHAINS, SPEC_FREQ, SPEC_TIME)
            if spec.ndim == 3:
                if spec.shape[0] == SPEC_CHAINS:
                    pass  # already (C, F, T) — no transpose needed
                elif spec.shape[2] == SPEC_CHAINS:
                    spec = spec.transpose(2, 0, 1)  # (F,T,C)→(C,F,T)
                else:
                    spec = np.zeros((SPEC_CHAINS, SPEC_FREQ, SPEC_TIME),
                                    dtype=np.float32)
            else:
                spec = np.zeros((SPEC_CHAINS, SPEC_FREQ, SPEC_TIME),
                                dtype=np.float32)

            # Pad or trim to exact expected size
            C, F, T = spec.shape
            if F != SPEC_FREQ or T != SPEC_TIME:
                out = np.zeros((SPEC_CHAINS, SPEC_FREQ, SPEC_TIME),
                               dtype=np.float32)
                f_min = min(F, SPEC_FREQ)
                t_min = min(T, SPEC_TIME)
                out[:, :f_min, :t_min] = spec[:, :f_min, :t_min]
                spec = out

            # Final safety clamp
            spec = np.clip(spec, -100.0, 100.0)

        else:
            spec = np.zeros((SPEC_CHAINS, SPEC_FREQ, SPEC_TIME),
                            dtype=np.float32)
        spec = self._pad_or_trim_2d(spec, SPEC_FREQ, SPEC_TIME)

        if self.augment:
            eeg, spec = self._augment(eeg, spec)

        return (torch.from_numpy(eeg),
                torch.from_numpy(spec),
                torch.from_numpy(self.targets[idx]))

    @staticmethod
    def _pad_or_trim_2d(spec, freq, time):
        C, F, T = spec.shape
        if F < freq:
            spec = np.pad(spec, ((0,0),(0,freq-F),(0,0)))
        else:
            spec = spec[:, :freq, :]
        if T < time:
            spec = np.pad(spec, ((0,0),(0,0),(0,time-T)))
        else:
            spec = spec[:, :, :time]
        return spec

    @staticmethod
    def _augment(eeg, spec):
        if np.random.rand() < 0.5:
            eeg = -eeg
        eeg += np.random.normal(0, 0.01, eeg.shape).astype(np.float32)
        return eeg, spec

# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    model.train()
    total_loss = 0.0
    for eeg, spec, target in loader:
        eeg    = eeg.to(device)
        spec   = spec.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                pred = model(eeg, spec)
                loss = loss_fn(pred, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(eeg, spec)
            loss = loss_fn(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    for eeg, spec, target in loader:
        pred = model(eeg.to(device), spec.to(device))
        total_loss += loss_fn(pred, target.to(device)).item()
    return total_loss / len(loader)


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model   = HMSModel(embed_dim=128, dropout=0.3).to(device)
    loss_fn = HMSLoss()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_eeg    = sum(p.numel() for p in model.eeg_branch.parameters())
    n_spec   = sum(p.numel() for p in model.spec_branch.parameters())
    n_fusion = sum(p.numel() for p in model.fusion.parameters())
    print(f"\nParameters:")
    print(f"  EEG branch  : {n_eeg:>10,}")
    print(f"  Spec branch : {n_spec:>10,}")
    print(f"  Fusion head : {n_fusion:>10,}")
    print(f"  Total       : {n_params:>10,}")

    # Forward pass timing
    B   = 4
    eeg = torch.randn(B, EEG_CHANNELS, EEG_LENGTH).to(device)
    spec= torch.randn(B, SPEC_CHAINS, SPEC_FREQ, SPEC_TIME).to(device)
    tgt = torch.softmax(torch.randn(B, N_CLASSES), dim=-1).to(device)

    t0  = time.time()
    out = model(eeg, spec)
    dt  = (time.time() - t0) * 1000

    loss = loss_fn(out, tgt)

    print(f"\nForward pass ({B} samples): {dt:.1f} ms")
    print(f"Output shape : {tuple(out.shape)}")
    print(f"Output sum   : {out.sum(dim=-1).tolist()}  (should be ~1.0)")
    print(f"Loss         : {loss.item():.4f}")

    # Intermediate shapes
    model.eval()
    with torch.no_grad():
        x = eeg
        x = model.eeg_branch.channel_proj(x)
        print(f"\nEEG branch shapes:")
        print(f"  after channel_proj : {tuple(x.shape)}")
        x = model.eeg_branch.merge(torch.cat([
            model.eeg_branch.scale7(x),
            model.eeg_branch.scale15(x),
            model.eeg_branch.scale31(x),
        ], dim=1))
        print(f"  after multi-scale  : {tuple(x.shape)}")
        x = model.eeg_branch.res_blocks(x)
        print(f"  after res_blocks   : {tuple(x.shape)}")
        x = model.eeg_branch.pool(x)
        print(f"  after pool         : {tuple(x.shape)}")

        s = spec
        s = model.spec_branch.stem(s)
        print(f"\nSpec branch shapes:")
        print(f"  after stem         : {tuple(s.shape)}")
        s = model.spec_branch.ds_blocks(s)
        print(f"  after ds_blocks    : {tuple(s.shape)}")
        s = model.spec_branch.ir_blocks(s)
        print(f"  after ir_blocks    : {tuple(s.shape)}")
        s = model.spec_branch.pool(s)
        print(f"  after pool         : {tuple(s.shape)}")
