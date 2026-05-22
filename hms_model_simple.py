"""
hms_model_simple.py  —  Simplified multimodal CNN for HMS-HBAC
Alexandra Yakovleva, 2026
==============================================================
Lightweight versions of EEGBranch, SpectrogramBranch, and HMSModel.
Roughly 4× fewer parameters than the full model (~0.57M vs ~2.1M).

Reductions vs full model
------------------------
EEG branch  : removed multi-scale stem, 2 res blocks (was 3),
              channels halved (32→64 was 64→128), no top-level SE
Spec branch : 2 IR blocks (was 4), expand_ratio=4 (was 6),
              channels capped at 128 (was 256), no CBAM attention
Fusion head : one fewer Dense layer (256→64→6 was 256→128→64→6)

Public API
----------
    SimpleEEGBranch(embed_dim)          -> (B, embed_dim)
    SimpleSpectrogramBranch(embed_dim)  -> (B, embed_dim)
    SimpleEEGOnlyModel()                -> (B, 6)
    SimpleSpecOnlyModel()               -> (B, 6)
    SimpleHMSModel()                    -> (B, 6)
    HMSLoss(label_smoothing)
    count_parameters(model)
"""

from __future__ import annotations
import torch.nn as nn
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants  (must match your dataset / pipeline config)
# ---------------------------------------------------------------------------

N_CLASSES    = 6
EEG_CHANNELS = 17       # 16 bipolar + 1 EKG
EEG_LENGTH   = 5000     # 50 s × 100 Hz
SPEC_CHAINS  = 6        # LL, LP, RP, RL, LL-RL, LP-RP
SPEC_FREQ    = 50
SPEC_TIME    = 150
EMBED_DIM    = 128


# ===========================================================================
#  Shared building blocks
# ===========================================================================

class SEBlock1d(nn.Module):
    """Squeeze-and-Excitation channel attention (1-D)."""
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
        return x * self.fc(x).unsqueeze(-1)


class ResBlock1d(nn.Module):
    """Residual block with optional stride and SE attention."""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 7, stride: int = 1):
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
        self.se   = SEBlock1d(out_ch)
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        )
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.se(self.conv(x))
        return self.act(h + self.skip(x))


class DepthwiseSepConv2d(nn.Module):
    """Depthwise-separable 2-D convolution."""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, stride: int = 1):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, kernel, stride=stride,
                             padding=kernel//2, groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class InvertedResidual2d(nn.Module):
    """MobileNetV2-style inverted residual block."""
    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, expand_ratio: int = 4):
        super().__init__()
        mid = in_ch * expand_ratio
        self.use_skip = (stride == 1 and in_ch == out_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
            nn.Conv2d(mid, mid, 3, stride=stride,
                      padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        h = self.conv(x)
        return h + x if self.use_skip else h


# ===========================================================================
#  1-D EEG branch  (simplified)
# ===========================================================================

class SimpleEEGBranch(nn.Module):
    """
    Simplified 1-D CNN for EEG + EKG.

    Architecture
    ------------
    Stem      : Conv1d 17→32, k=7  (single scale)
    Res block 1: 32→64, stride=2   → T/2
    Res block 2: 64→64, stride=2   → T/4
    Pool       : AdaptiveAvgPool1d(5)  — T/4 must be divisible by 5
    Head       : Linear 320→embed_dim

    Input : (B, 17, 5000)
    Output: (B, embed_dim)
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()

        # Single-scale stem (replaces parallel k=7/15/31)
        self.stem = nn.Sequential(
            nn.Conv1d(EEG_CHANNELS, 32, kernel_size=7,
                      padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # Two residual blocks (was three)
        self.res_blocks = nn.Sequential(
            ResBlock1d(32, 64, kernel=7, stride=2),  # (B, 64, 2500)
            ResBlock1d(64, 64, kernel=7, stride=2),  # (B, 64, 1250)
        )

        # Pool: 1250 / 5 = 250  ✓  (MPS-safe)
        self.pool = nn.AdaptiveAvgPool1d(5)

        self.head = nn.Sequential(
            nn.Flatten(),                          # (B, 320)
            nn.Linear(64 * 5, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (B, 17, T)
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.pool(x)
        return self.head(x)


# ===========================================================================
#  2-D Spectrogram branch  (simplified)
# ===========================================================================


class SimpleSpectrogramBranch(nn.Module):
    """
    Simplified 2-D CNN for spectrograms.

    Architecture
    ------------
    Stem        : Conv2d 6→32, 3×3, stride=2
    DS block 1  : 32→64,  stride=2
    DS block 2  : 64→128, stride=2
    IR block 1  : 128→128, expand=2, stride=1
    IR block 2  : 128→128, expand=2, stride=2
    Global pool : AdaptiveAvgPool2d(1)
    Head        : Linear 128→embed_dim

    Input : (B, 6, 100, 300)
    Output: (B, embed_dim)
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()

        # Add weight initialisation to stem conv
        self.stem = nn.Sequential(
            nn.Conv2d(SPEC_CHAINS, 32, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Kaiming init — prevents large activations at layer 1
        nn.init.kaiming_normal_(
            self.stem[0].weight, mode="fan_out",
            nonlinearity="relu")

        # Two DS blocks (unchanged — efficient and necessary)
        self.ds_blocks = nn.Sequential(
            DepthwiseSepConv2d(32,  64,  stride=2),
            DepthwiseSepConv2d(64,  128, stride=2),
        )

        # Two IR blocks with expand_ratio=4 (was 4 blocks, ratio=6)
        # Caps at 128 channels (was 256)
        # Reduce expand_ratio 4 → 2 — less explosive intermediate activations
        self.ir_blocks = nn.Sequential(
            InvertedResidual2d(128, 128, stride=1, expand_ratio=2),
            InvertedResidual2d(128, 128, stride=2, expand_ratio=2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),                          # (B, 128)
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 6, F, T)
        x = self.stem(x)
        x = self.ds_blocks(x)
        x = self.ir_blocks(x)
        x = self.pool(x)
        return self.head(x)



# ===========================================================================
#  Single-branch wrappers (for staged training)
# ===========================================================================

class SimpleEEGOnlyModel(nn.Module):
    """EEG branch + classification head.  spec input is ignored."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.branch = SimpleEEGBranch(embed_dim)
        self.head   = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, eeg, spec=None):
        return F.softmax(self.head(self.branch(eeg)), dim=-1)


class SimpleSpecOnlyModel(nn.Module):
    """Spectrogram branch + classification head.  eeg input is ignored."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.branch = SimpleSpectrogramBranch(embed_dim)
        self.head   = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, eeg=None, spec=None):
        return F.softmax(self.head(self.branch(spec)), dim=-1)


# ===========================================================================
#  Combined model
# ===========================================================================

class SimpleHMSModel(nn.Module):
    """
    Simplified multimodal HMS model.

    Both branches produce embed_dim-d vectors that are concatenated
    and classified by a two-layer MLP.

    Input : eeg  (B, 17, 5000)
            spec (B, 6, 100, 300)
    Output: (B, 6)  softmax probabilities
    """

    def __init__(self,
                 embed_dim : int   = EMBED_DIM,
                 dropout   : float = 0.5):
        super().__init__()
        self.eeg_branch  = SimpleEEGBranch(embed_dim)
        self.spec_branch = SimpleSpectrogramBranch(embed_dim)

        # Simpler fusion: 256 → 64 → 6  (was 256 → 128 → 64 → 6)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, eeg: torch.Tensor, spec: torch.Tensor):
        eeg_emb  = self.eeg_branch(eeg)
        spec_emb = self.spec_branch(spec)
        return F.softmax(
            self.fusion(torch.cat([eeg_emb, spec_emb], dim=-1)),
            dim=-1)

    def encode(self, eeg: torch.Tensor, spec: torch.Tensor):
        """Return 256-d concatenated embedding (no classification)."""
        return torch.cat([
            self.eeg_branch(eeg),
            self.spec_branch(spec),
        ], dim=-1)


# ===========================================================================
#  Loss
# ===========================================================================

class HMSLoss(nn.Module):
    """
    KL divergence loss against soft annotator-vote labels.
    target must be a normalised probability distribution (sums to 1).
    """

    def __init__(self, label_smoothing: float = 0.05):
        super().__init__()
        self.eps = label_smoothing
        self.kl  = nn.KLDivLoss(reduction="batchmean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        if self.eps > 0:
            target = (1 - self.eps) * target + self.eps / N_CLASSES
        return self.kl(torch.log(pred.clamp(min=1e-7)), target)


# ===========================================================================
#  Utility
# ===========================================================================

def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ===========================================================================
#  Sanity check
# ===========================================================================

if __name__ == "__main__":
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}\n")

    eeg  = torch.randn(4, EEG_CHANNELS, EEG_LENGTH).to(device)
    spec = torch.randn(4, SPEC_CHAINS,  SPEC_FREQ, SPEC_TIME).to(device)
    tgt  = torch.softmax(torch.randn(4, N_CLASSES), dim=-1).to(device)

    models = {
        "SimpleEEGOnlyModel" : SimpleEEGOnlyModel().to(device),
        "SimpleSpecOnlyModel": SimpleSpecOnlyModel().to(device),
        "SimpleHMSModel"     : SimpleHMSModel().to(device),
    }

    loss_fn = HMSLoss()

    for name, model in models.items():
        out  = model(eeg, spec)
        loss = loss_fn(out, tgt)
        p    = count_parameters(model)
        print(f"{name}")
        print(f"  output : {tuple(out.shape)}")
        print(f"  sum    : {out.sum(dim=-1).tolist()}")
        print(f"  loss   : {loss.item():.4f}")
        print(f"  params : {p['total']:,}  ({p['total']/1e6:.3f}M)\n")