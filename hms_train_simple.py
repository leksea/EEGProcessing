"""
hms_train_simple.py  —  Training script for simplified HMS models
=================================================================
Run as a notebook cell or standalone script.
Trains three models in sequence:
  1. SimpleEEGOnlyModel  — EEG + EKG branch alone
  2. SimpleSpecOnlyModel — Spectrogram branch alone
  3. SimpleHMSModel      — Combined (branches pretrained from stages 1&2)

Produces three plots:
  train_curves_simple.png    — train vs val loss, all three models
  lr_search_simple.png       — validation loss for 4 learning rates
  bs_search_simple.png       — validation loss for 4 batch sizes
"""

import os, shutil, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display

# ── Make sure scripts are on path ────────────────────────────────────────────
for path in [TMP, SCRIPTS_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

for script in ["hms_model_simple.py", "hms_model.py"]:
    src = os.path.join(SCRIPTS_DIR, script)
    dst = os.path.join(TMP, script)
    if os.path.exists(src):
        shutil.copy(src, dst)

sys.modules.pop("hms_model_simple", None)
from hms_model_simple import (
    SimpleEEGOnlyModel, SimpleSpecOnlyModel, SimpleHMSModel,
    HMSLoss, count_parameters,
    EMBED_DIM, N_CLASSES,
)
from hms_model import HMSDataset   # dataset stays the same

# ── Device ───────────────────────────────────────────────────────────────────
device = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Device : {device}")

# ════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT PARAMETERS  ← edit here
# ════════════════════════════════════════════════════════════════════════════

LR_DEFAULT    = 1e-3
BATCH_DEFAULT = 16
EPOCHS_EEG    = 10      # single-branch epochs
EPOCHS_SPEC   = 10
EPOCHS_FUSE   = 5       # fusion head only (branches frozen)
EPOCHS_FULL   = 15      # full fine-tune

LR_SEARCH_VALUES = [1e-2, 1e-3, 5e-4, 1e-4]
BS_SEARCH_VALUES = [8, 16, 32, 64]

WEIGHT_DECAY   = 1e-4
LABEL_SMOOTHING= 0.05
GRAD_CLIP      = 1.0

BG = "#0e0e0e"   # plot background

# ════════════════════════════════════════════════════════════════════════════
#  DataLoaders
# ════════════════════════════════════════════════════════════════════════════

train_dataset = HMSDataset(
    df               = df_train,
    processed_dir    = PROCESSED_DIR,
    processed_raw_dir= PROCESSED_DIR_RAW,
    spec_dir         = PROCESSED_DIR,
    eeg_dir          = EEG_DIR,
    augment          = True,
)
val_dataset = HMSDataset(
    df               = df_val,
    processed_dir    = PROCESSED_DIR,
    processed_raw_dir= PROCESSED_DIR_RAW,
    spec_dir         = PROCESSED_DIR,
    eeg_dir          = EEG_DIR,
    augment          = False,
)
test_dataset = HMSDataset(
    df               = df_test,
    processed_dir    = PROCESSED_DIR,
    processed_raw_dir= PROCESSED_DIR_RAW,
    spec_dir         = PROCESSED_DIR,
    eeg_dir          = EEG_DIR,
    augment          = False,
)

def make_loaders(batch_size=BATCH_DEFAULT):
    return (
        DataLoader(train_dataset, batch_size=batch_size,
                   shuffle=True,  num_workers=0, pin_memory=False),
        DataLoader(val_dataset,   batch_size=batch_size,
                   shuffle=False, num_workers=0, pin_memory=False),
        DataLoader(test_dataset,  batch_size=batch_size,
                   shuffle=False, num_workers=0, pin_memory=False),
    )

train_loader, val_loader, test_loader = make_loaders(BATCH_DEFAULT)
print(f"Train batches : {len(train_loader):,}")
print(f"Val batches   : {len(val_loader):,}")
print(f"Test batches  : {len(test_loader):,}")


# ════════════════════════════════════════════════════════════════════════════
#  Core training function
# ════════════════════════════════════════════════════════════════════════════

def run_experiment(
    model,
    name,
    epochs         = 10,
    lr             = LR_DEFAULT,
    batch_size     = BATCH_DEFAULT,
    save           = True,
    tr_loader      = None,
    va_loader      = None,
):
    """
    Train and validate a model.  Returns history dict.

    Parameters
    ----------
    model      : nn.Module
    name       : str    — used for checkpoint filename and print prefix
    epochs     : int
    lr         : float  — initial learning rate
    batch_size : int    — ignored if tr_loader/va_loader provided directly
    save       : bool   — save best checkpoint to OUT_DIR
    tr_loader  : optional DataLoader override
    va_loader  : optional DataLoader override
    """
    _tr = tr_loader or train_loader
    _va = va_loader or val_loader

    optim   = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=epochs, eta_min=1e-5)
    loss_fn = HMSLoss(label_smoothing=LABEL_SMOOTHING)

    history = {"train": [], "val": [], "lr": []}
    best_val  = float("inf")
    ckpt_path = os.path.join(
        OUT_DIR, f"{name.replace(' ','-').replace('=','')}_best.pt")

    for epoch in range(1, epochs + 1):

        # ── Train ────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for eeg, spec, target in _tr:
            optim.zero_grad()
            pred = model(eeg.to(device), spec.to(device))
            loss = loss_fn(pred, target.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP)
            optim.step()
            tr_loss += loss.item()
        tr_loss /= len(_tr)

        # ── Validate ─────────────────────────────────────────
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for eeg, spec, target in _va:
                pred    = model(eeg.to(device), spec.to(device))
                va_loss += loss_fn(
                    pred, target.to(device)).item()
        va_loss /= len(_va)
        sched.step()

        history["train"].append(tr_loss)
        history["val"].append(va_loss)
        history["lr"].append(optim.param_groups[0]["lr"])

        marker = ""
        if va_loss < best_val:
            best_val = va_loss
            if save:
                torch.save(model.state_dict(), ckpt_path)
                marker = "  ← saved"

        print(f"[{name}] epoch {epoch:3d}/{epochs}  "
              f"train={tr_loss:.4f}  val={va_loss:.4f}"
              f"  lr={history['lr'][-1]:.2e}{marker}")

    history["best_val"]  = best_val
    history["ckpt_path"] = ckpt_path if save else None
    print(f"  best val = {best_val:.4f}\n")
    return history


# ════════════════════════════════════════════════════════════════════════════
#  Evaluation helper
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, label="Test"):
    model.eval()
    loss_fn  = HMSLoss()
    total    = 0.0
    for eeg, spec, target in loader:
        pred   = model(eeg.to(device), spec.to(device))
        total += loss_fn(pred, target.to(device)).item()
    kl = total / len(loader)
    print(f"  {label} KL divergence : {kl:.4f}")
    return kl


# ════════════════════════════════════════════════════════════════════════════
#  Stage 1 — EEG-only model
# ════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("STAGE 1  —  EEG-only branch")
print("=" * 60)

eeg_model = SimpleEEGOnlyModel().to(device)
p = count_parameters(eeg_model)
print(f"Parameters: {p['total']:,}  ({p['total']/1e6:.3f}M)\n")

hist_eeg  = run_experiment(
    eeg_model, "EEG-only",
    epochs=EPOCHS_EEG, lr=LR_DEFAULT,
)
evaluate(eeg_model, test_loader, "EEG-only test")


# ════════════════════════════════════════════════════════════════════════════
#  Stage 2 — Spectrogram-only model
# ════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("STAGE 2  —  Spectrogram-only branch")
print("=" * 60)

spec_model = SimpleSpecOnlyModel().to(device)
p = count_parameters(spec_model)
print(f"Parameters: {p['total']:,}  ({p['total']/1e6:.3f}M)\n")

hist_spec = run_experiment(
    spec_model, "Spec-only",
    epochs=EPOCHS_SPEC, lr=LR_DEFAULT,
)
evaluate(spec_model, test_loader, "Spec-only test")


# ════════════════════════════════════════════════════════════════════════════
#  Stage 3 — Combined model
# ════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("STAGE 3  —  Combined model")
print("=" * 60)

full_model = SimpleHMSModel().to(device)
p = count_parameters(full_model)
print(f"Parameters: {p['total']:,}  ({p['total']/1e6:.3f}M)\n")

# Load pretrained branch weights
full_model.eeg_branch.load_state_dict(
    torch.load(hist_eeg["ckpt_path"],  map_location=device))
full_model.spec_branch.load_state_dict(
    torch.load(hist_spec["ckpt_path"], map_location=device))
print("Pretrained branch weights loaded.\n")

# Step A: freeze branches, train fusion only
for p in full_model.eeg_branch.parameters():  p.requires_grad = False
for p in full_model.spec_branch.parameters(): p.requires_grad = False
print(f"Branches frozen — training fusion head only ({EPOCHS_FUSE} epochs)")
hist_fuse = run_experiment(
    full_model, "full-frozen",
    epochs=EPOCHS_FUSE, lr=LR_DEFAULT,
)

# Step B: unfreeze and fine-tune all layers with lower lr
for p in full_model.parameters(): p.requires_grad = True
print(f"Branches unfrozen — fine-tuning all layers ({EPOCHS_FULL} epochs)")
hist_ft   = run_experiment(
    full_model, "full-finetune",
    epochs=EPOCHS_FULL, lr=1e-4,
)

# Merge histories for plotting
hist_full = {
    "train"    : hist_fuse["train"] + hist_ft["train"],
    "val"      : hist_fuse["val"]   + hist_ft["val"],
    "lr"       : hist_fuse["lr"]    + hist_ft["lr"],
    "best_val" : hist_ft["best_val"],
    "ckpt_path": hist_ft["ckpt_path"],
}

evaluate(full_model, test_loader, "Combined model test")


# ════════════════════════════════════════════════════════════════════════════
#  Hyperparameter searches  (EEG branch, 10 epochs each)
# ════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("HYPERPARAMETER SEARCH — learning rate")
print("=" * 60)

lr_histories = {}
for lr in LR_SEARCH_VALUES:
    h = run_experiment(
        SimpleEEGOnlyModel().to(device),
        f"lr={lr}", epochs=10, lr=lr, save=False,
    )
    lr_histories[lr] = h

print("=" * 60)
print("HYPERPARAMETER SEARCH — batch size")
print("=" * 60)

bs_histories = {}
for bs in BS_SEARCH_VALUES:
    tr_l, va_l, _ = make_loaders(bs)
    h = run_experiment(
        SimpleEEGOnlyModel().to(device),
        f"bs={bs}", epochs=10, lr=LR_DEFAULT, save=False,
        tr_loader=tr_l, va_loader=va_l,
    )
    bs_histories[bs] = h


# ════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ════════════════════════════════════════════════════════════════════════════

def ax_style(ax, title="", color="white", ylabel="KL divergence"):
    ax.set_facecolor(BG)
    ax.set_title(title, color=color, fontsize=9, pad=5)
    ax.set_xlabel("Epoch", color="lightgray", fontsize=8)
    ax.set_ylabel(ylabel, color="lightgray", fontsize=8)
    ax.tick_params(colors="lightgray", labelsize=7)
    ax.legend(fontsize=8, labelcolor="white", framealpha=0.15)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")


# ── Plot 1: train vs val — all three models ──────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor=BG)
fig.suptitle("Training vs validation KL divergence  (simple models)",
             color="white", fontsize=11, y=1.01)

for ax, hist, title, color in zip(
    axes,
    [hist_eeg, hist_spec, hist_full],
    ["EEG-only", "Spectrogram-only", "Combined model"],
    ["#D85A30",  "#378ADD",          "#1D9E75"],
):
    ep = range(1, len(hist["train"]) + 1)
    ax.plot(ep, hist["train"], color=color, lw=2, label="Train")
    ax.plot(ep, hist["val"],   color=color, lw=2,
            ls="--", alpha=0.8, label="Validation")
    ax.fill_between(ep, hist["train"], hist["val"],
                    color=color, alpha=0.08)
    best_e = int(np.argmin(hist["val"])) + 1
    ax.axvline(best_e, color="white", lw=0.8, ls=":", alpha=0.5)
    ax.text(best_e + 0.3,
            float(np.min(hist["val"])) + 0.003,
            f"best\nep {best_e}",
            color="white", fontsize=7)
    # Combined model: mark where freeze ends
    if "full" in title.lower() and EPOCHS_FUSE > 0:
        ax.axvline(EPOCHS_FUSE, color="#e8a020",
                   lw=1, ls="--", alpha=0.6)
        ax.text(EPOCHS_FUSE + 0.2, ax.get_ylim()[1] * 0.95,
                "unfreeze", color="#e8a020", fontsize=6)
    ax_style(ax, title, color)

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "train_curves_simple.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

# ── Plot 2: LR search ────────────────────────────────────────────────────

colors_lr = ["#e84a4a","#1D9E75","#378ADD","#888888"]
fig, ax   = plt.subplots(figsize=(8, 4.5), facecolor=BG)
for (lr, hist), c in zip(lr_histories.items(), colors_lr):
    chosen = (lr == LR_DEFAULT)
    ep     = range(1, len(hist["val"]) + 1)
    label  = f"lr={lr}" + ("  ← chosen" if chosen else "")
    ax.plot(ep, hist["val"],
            color=c, lw=2.5 if chosen else 1.5,
            alpha=1.0 if chosen else 0.6,
            label=label)
ax_style(ax, "Validation loss — learning rate search")
plt.tight_layout()
p2 = os.path.join(OUT_DIR, "lr_search_simple.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

# ── Plot 3: Batch size search ─────────────────────────────────────────────

colors_bs = ["#e84a4a","#1D9E75","#378ADD","#888888"]
fig, ax   = plt.subplots(figsize=(8, 4.5), facecolor=BG)
for (bs, hist), c in zip(bs_histories.items(), colors_bs):
    chosen = (bs == BATCH_DEFAULT)
    ep     = range(1, len(hist["val"]) + 1)
    label  = f"bs={bs}" + ("  ← chosen" if chosen else "")
    ax.plot(ep, hist["val"],
            color=c, lw=2.5 if chosen else 1.5,
            alpha=1.0 if chosen else 0.6,
            label=label)
ax_style(ax, "Validation loss — batch size comparison")
plt.tight_layout()
p3 = os.path.join(OUT_DIR, "bs_search_simple.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

# ── Display all plots ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  EEG-only      best val KL : {hist_eeg['best_val']:.4f}")
print(f"  Spec-only     best val KL : {hist_spec['best_val']:.4f}")
print(f"  Combined      best val KL : {hist_full['best_val']:.4f}")
print(f"\nPlots saved to: {OUT_DIR}")

for p in [p1, p2, p3]:
    display(Image(p, width=950))