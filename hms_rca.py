"""
hms_rca.py  —  Reliable Components Analysis for HMS-HBAC EEG
Alexandra Yakovleva, 2026
==============================================================
Implements Reliable Components Analysis (RCA) as described in:

  Dmochowski J.P., Sajda P., Dias J., Parra L.C. (2012).
  "Correlated components of ongoing EEG point to emotionally
  reactive brain networks during free-viewing of natural movies."
  Frontiers in Human Neuroscience, 6:112.

RCA finds spatial filters (electrode weights) that maximise the
inter-trial covariance — the signal shared reliably across repetitions
of the same condition — relative to the total signal covariance.

A component is "reliable" if it produces similar activation patterns
across all trials of the same class.  Patient-specific noise and
artifacts are unreliable and are suppressed.

Mathematical formulation
------------------------
For a set of N trials, each of shape (T x C):

    R_between = (1/N²) Σ_{i≠j} X_i.T @ X_j    # inter-trial covariance
    R_total   = (1/N)  Σ_i    X_i.T @ X_i      # total covariance

Solve the generalised eigenvalue problem:
    R_between · W = Λ · R_total · W

Eigenvalues λ ∈ [0, 1] are reliability scores.
Eigenvectors W are the spatial filters (C x K).
Component activations: A_i = X_i @ W  →  (T x K) per trial.

Pipeline integration
--------------------
This module consumes the .npy files produced by hms_pipeline.py.
Each file is one trial (50-second EEG window, T x 16 bipolar channels).

Workflow
--------
  1. Load trials from processed .npy files, grouped by class label
  2. (Optional) sub-group seizure/LPD/LRDA by laterality, flip right→left
  3. Compute R_between and R_total for each group
  4. Solve generalised eigenvalue problem → spatial filters W, scores λ
  5. Project all trials onto top-K components → (N x K) score matrix
  6. Run UMAP on score matrix for 2-D visualisation
  7. Export filters, projections, and UMAP embedding

Usage — library
---------------
    from hms_rca import RCAConfig, RCAGroup, run_rca

    groups = RCAGroup.from_pipeline_output(
        processed_dir = "/data/processed/",
        train_csv     = "/data/train.csv",
        min_votes     = 5,     # only high-confidence labels
        min_trials    = 20,    # skip groups with too few trials
    )
    results = run_rca(groups, RCAConfig(n_components=6))
    results["gpd"].plot_topomap()
    results["gpd"].plot_umap()

Usage — CLI
-----------
    python hms_rca.py \\
        --processed_dir /data/processed/ \\
        --train_csv     /data/train.csv  \\
        --out_dir       /data/rca/       \\
        --min_votes     5                \\
        --n_components  6                \\
        --umap
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import eigh


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_COLS = [
    "seizure_vote", "lpd_vote", "gpd_vote",
    "lrda_vote", "grda_vote", "other_vote",
]
LABEL_NAMES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]

BIPOLAR_CHAINS = {
    "LL": [("Fp1","F7"),("F7","T3"),("T3","T5"),("T5","O1")],
    "LP": [("Fp1","F3"),("F3","C3"),("C3","P3"),("P3","O1")],
    "RP": [("Fp2","F4"),("F4","C4"),("C4","P4"),("P4","O2")],
    "RL": [("Fp2","F8"),("F8","T4"),("T4","T6"),("T6","O2")],
}
BIPOLAR_NAMES = [
    f"{a}-{c}" for pairs in BIPOLAR_CHAINS.values() for a, c in pairs]

# Channel indices for left/right hemisphere in bipolar layout
# LL: channels 0-3, LP: channels 4-7, RP: channels 8-11, RL: channels 12-15
LL_IDX = list(range(0, 4))
LP_IDX = list(range(4, 8))
RP_IDX = list(range(8, 12))
RL_IDX = list(range(12, 16))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RCAConfig:
    """
    Parameters for the RCA computation.

    n_components    : int   — number of reliable components to retain (default 6)
    n_reg           : int   — regularisation parameter K: number of pooled
                              autocovariance eigenvectors used to invert Rpool.
                              Equivalent to nReg in rcaRun.m. Typical 5–15.
                              Must be >= n_components.  (default 6)
    min_reliability : float — minimum eigenvalue to keep a component
                              (default 0.0 = keep top n_components regardless)
    time_avg        : bool  — average over time before computing covariance
                              (False = use full time series, default)
    baseline        : bool  — subtract per-channel mean of a pre-event
                              baseline period from each trial before RCA.
                              Removes DC offset / electrode drift differences
                              between patients.  (default True)
    baseline_end_sec: float — end of the baseline period in seconds within
                              the 50-s window.  Default 10.0 (first 10 s,
                              before the labeled epoch at t=20–30 s).
    baseline_sfreq  : int   — sampling rate of the processed arrays in Hz.
                              Must match target_sfreq in PipelineConfig.
                              Default 100 (after resampling).
    standardise     : bool  — z-score each trial before computing covariance
                              reduces amplitude differences between patients
    flip_lateralised: bool  — flip right-hemisphere lateralised patterns
                              (LPD/LRDA) to canonical left orientation
                              before computing RCA  (default True)
    umap_n_neighbors: int   — UMAP neighbours parameter (default 15)
    umap_min_dist   : float — UMAP min_dist parameter (default 0.1)
    umap_n_components: int  — UMAP output dimensions (default 2)
    random_state    : int   — random seed for UMAP (default 42)
    """
    n_components     : int   = 6
    n_reg            : int   = 6     # regularisation: number of pooled autocovariance
                                     # eigenvectors to keep (K in rcaTrain.m).
                                     # Typical values 5–15 for EEG. Must be ≥ n_components.
    min_reliability  : float = 0.0
    time_avg         : bool  = False
    baseline         : bool  = True
    baseline_end_sec : float = 10.0
    baseline_sfreq   : int   = 100
    standardise      : bool  = True  # divide by per-channel std after baseline
    flip_lateralised : bool  = True
    umap_n_neighbors : int   = 15
    umap_min_dist    : float = 0.1
    umap_n_components: int   = 2
    random_state     : int   = 42

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Trial container
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    """One 50-second EEG window = one RCA trial."""
    eeg_id      : int
    offset_sec  : float
    patient_id  : int
    label       : str          # dominant label for this trial
    label_dist  : np.ndarray   # full 6-class probability distribution
    is_right    : bool         # True if lateralised to right hemisphere
    data        : np.ndarray   # float32  (T x C) — EEG only, NO EKG
    ekg         : Optional[np.ndarray] = None  # float32  (T,) or None


# ---------------------------------------------------------------------------
# RCA group — all trials for one condition
# ---------------------------------------------------------------------------

@dataclass
class RCAGroup:
    """
    A set of trials sharing the same condition (class + laterality).

    Attributes
    ----------
    name    : str         — e.g. "gpd", "lpd_left", "seizure_left"
    trials  : list[Trial]
    """
    name   : str
    trials : List[Trial]

    def __len__(self):
        return len(self.trials)

    # ------------------------------------------------------------------
    @classmethod
    def from_pipeline_output(
        cls,
        processed_dir    : str,
        train_csv        : str,
        eeg_dir          : Optional[str] = None,
        min_votes        : int   = 3,
        min_trials       : int   = 10,
        max_trials       : Optional[int] = None,
        flip_lateralised : bool  = True,
        verbose          : bool  = True,
    ) -> Dict[str, "RCAGroup"]:
        """
        Load processed .npy files from hms_pipeline and group by label.

        Parameters
        ----------
        processed_dir   : directory containing {eeg_id}_{offset}.npy files
        train_csv       : path to train.csv (for label votes + patient_id)
        eeg_dir         : directory containing raw .parquet EEG files
                          (optional). When provided, the EKG channel is
                          loaded from the parquet and attached to each Trial
                          so it can be reattached after RCA projection.
        min_votes       : minimum votes for the dominant class to include
                          a trial (filters out ambiguous/low-confidence rows)
        min_trials      : discard groups with fewer than this many trials
        max_trials      : cap per group (None = no cap; useful for balancing)
        flip_lateralised: flip right-hemisphere LPD/LRDA to canonical left
        verbose         : print loading summary

        Returns
        -------
        dict mapping group name → RCAGroup
        """
        df = pd.read_csv(train_csv)

        # Compute dominant label and its vote count
        vote_mat = df[LABEL_COLS].values.astype(float)
        total    = vote_mat.sum(axis=1, keepdims=True).clip(min=1)
        prob_mat = vote_mat / total

        dom_idx   = vote_mat.argmax(axis=1)
        dom_votes = vote_mat.max(axis=1)
        dom_label = [LABEL_NAMES[i] for i in dom_idx]

        df["_label"]      = dom_label
        df["_dom_votes"]  = dom_votes
        df["_prob_mat"]   = list(prob_mat)

        # Filter by minimum vote confidence
        df = df[df["_dom_votes"] >= min_votes].copy()

        groups: Dict[str, List[Trial]] = {}

        n_loaded = n_missing = n_skipped = 0

        for _, row in df.iterrows():
            eid    = int(row["eeg_id"])
            offset = float(row["eeg_label_offset_seconds"])
            label  = row["_label"]
            pid    = int(row.get("patient_id", -1))
            prob   = np.array(row["_prob_mat"], dtype=np.float32)

            if label == "other":
                n_skipped += 1
                continue

            # Load .npy
            stem = f"{eid}_{int(offset)}"
            npy  = os.path.join(processed_dir, f"{stem}.npy")
            if not os.path.exists(npy):
                n_missing += 1
                continue

            data = np.load(npy).astype(np.float32)  # (T, C)

            # Load EKG from raw parquet if eeg_dir supplied
            ekg_signal = None
            if eeg_dir is not None:
                ekg_path = os.path.join(eeg_dir, f"{eid}.parquet")
                if os.path.exists(ekg_path):
                    try:
                        raw_df   = pd.read_parquet(ekg_path)
                        sfreq_   = 200
                        t_start  = int(offset * sfreq_)
                        t_end    = t_start + 50 * sfreq_
                        if "EKG" in raw_df.columns:
                            ekg_col  = raw_df["EKG"].values[t_start:t_end]
                            ekg_signal = np.nan_to_num(
                                ekg_col.astype(np.float32))
                    except Exception as _ekg_ex:
                        warnings.warn(
                            f"[RCA] EKG load failed for {eid}: {_ekg_ex}")

            # Detect laterality for LPD / LRDA
            is_right = False
            if label in ("lpd", "lrda") and flip_lateralised:
                is_right = _detect_right_hemisphere(data)

            # Canonical orientation: flip right to left
            if is_right:
                data = _flip_hemispheres(data)

            # Group name
            if label in ("lpd", "lrda"):
                side     = "right" if is_right else "left"
                grp_name = f"{label}_{side}"
            else:
                grp_name = label

            if grp_name not in groups:
                groups[grp_name] = []

            if max_trials and len(groups[grp_name]) >= max_trials:
                continue

            groups[grp_name].append(Trial(
                eeg_id     = eid,
                offset_sec = offset,
                patient_id = pid,
                label      = label,
                label_dist = prob,
                is_right   = is_right,
                data       = data,
                ekg        = ekg_signal,
            ))
            n_loaded += 1

        # Filter small groups
        groups = {k: RCAGroup(name=k, trials=v)
                  for k, v in groups.items()
                  if len(v) >= min_trials}

        if verbose:
            print(f"[RCA] Loaded {n_loaded} trials  "
                  f"({n_missing} missing npy, {n_skipped} 'other' skipped)")
            for name, g in sorted(groups.items()):
                print(f"  {name:<20} {len(g):>5} trials")

        return groups


# ---------------------------------------------------------------------------
# Hemisphere detection and flipping
# ---------------------------------------------------------------------------

def _detect_right_hemisphere(data: np.ndarray) -> bool:
    """
    Determine if a lateralised pattern is right-hemisphere dominant.

    Compares RMS power in LL+LP channels vs RP+RL channels.
    Returns True if right side has more power.
    """
    left_power  = float(np.sqrt(np.mean(
        data[:, LL_IDX + LP_IDX] ** 2)))
    right_power = float(np.sqrt(np.mean(
        data[:, RP_IDX + RL_IDX] ** 2)))
    return right_power > left_power


def _flip_hemispheres(data: np.ndarray) -> np.ndarray:
    """
    Flip left↔right hemisphere channels to canonical left-dominant orientation.
    Swaps LL↔RL and LP↔RP channel blocks.
    """
    flipped = data.copy()
    flipped[:, LL_IDX] = data[:, RL_IDX]
    flipped[:, RL_IDX] = data[:, LL_IDX]
    flipped[:, LP_IDX] = data[:, RP_IDX]
    flipped[:, RP_IDX] = data[:, LP_IDX]
    return flipped


# ---------------------------------------------------------------------------
# Core RCA computation
# ---------------------------------------------------------------------------

def compute_rca(
    trials  : List[np.ndarray],   # each (T x C)
    cfg     : RCAConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reliable Components Analysis — Python port of Dmochowski et al. (2012).

    Matches rcaRun.m / preComputeRcaCovariancesLoop.m / rcaTrain.m exactly.

    Covariance computation (preComputeRcaCovariancesLoop.m)
    --------------------------------------------------------
    1. Global mean centering: subtract the grand mean per channel
       (mean across ALL timepoints × ALL trials) — matches MATLAB's thisMu.
    2. Efficient cross-trial covariance using the identity:
         sumXY = S.T @ S  -  sumXX
       where S = sum of all mean-centred trials (T×C), and
       sumXX = sum of per-trial auto-covariances (C×C).
       This avoids enumerating all N*(N-1) pairs explicitly.
    3. Normalise:
         Rxx = sumXX / (N * T)          — mean within-trial autocovariance
         Rxy = sumXY / (N*(N-1) * T)    — mean cross-trial covariance

    Spatial filter training (rcaTrain.m)
    -------------------------------------
    4. Pooled autocovariance: Rpool = 0.5*(Rxx + Ryy) = Rxx (symmetric pairs)
    5. Eigendecompose Rpool: [Vpool, dPool] = eig(Rxx + Ryy)
       Keep top K (n_reg) eigenvectors.
    6. Regularised inverse applied to symmetrised cross-covariance:
         Rw = Vk @ diag(1/dk) @ Vk.T @ (Rxy + Rxy.T)
    7. Eigendecompose Rw, sort by |eigenvalue| descending → W filters.
    8. Forward model: A = Rpool @ Wsub @ inv(Wsub.T @ Rpool @ Wsub)

    Parameters
    ----------
    trials : list of (T x C) float32 arrays
    cfg    : RCAConfig

    Returns
    -------
    W       : (C x n_components) spatial filters
    lambdas : (n_components,) reliability scores (eigenvalues of Rw)
    A_fwd   : (C x n_components) forward models
    """
    from scipy.linalg import eigh, eig as scipy_eig, inv as scipy_inv

    n_trials = len(trials)
    if n_trials < 2:
        raise ValueError(f"Need at least 2 trials, got {n_trials}")

    n_ch      = trials[0].shape[1]
    T         = trials[0].shape[0]
    n_baseline= int(cfg.baseline_end_sec * cfg.baseline_sfreq)
    n_reg     = min(cfg.n_reg, n_ch)

    if n_reg < cfg.n_components:
        raise ValueError(
            f"n_reg ({n_reg}) must be >= n_components ({cfg.n_components}). "
            "Increase n_reg in RCAConfig.")

    # ── Step 1: stack and preprocess ─────────────────────────────────
    # Shape: (N, T, C)
    X = np.stack([t.astype(np.float64) for t in trials], axis=0)

    # Step 1a — baseline subtraction (per trial, pre-event period)
    if cfg.baseline and n_baseline > 0 and n_baseline < T:
        bm  = X[:, :n_baseline, :].mean(axis=1, keepdims=True)  # (N,1,C)
        X   = X - bm

    # Step 1b — global mean centering (matches MATLAB thisMu)
    # Subtract grand mean over ALL timepoints and ALL trials per channel
    grand_mean = X.mean(axis=(0, 1), keepdims=True)   # (1, 1, C)
    X          = X - grand_mean

    # Step 1c — scale by per-channel std (optional, matches standardise flag)
    if cfg.standardise:
        std = X.std(axis=(0, 1), keepdims=True) + 1e-8   # (1, 1, C)
        X   = X / std

    # ── Step 2: efficient covariance computation ──────────────────────
    # sumXX = Σ_n X[n].T @ X[n]  (C×C)
    # Using reshape: equivalent to Xflat.T @ Xflat per trial summed
    Xflat  = X.reshape(n_trials * T, n_ch)   # (N*T, C)
    # But we need per-trial sum, NOT mixing timepoints across trials
    # Correct efficient form: einsum over n,t
    # sumXX[c,d] = Σ_{n,t} X[n,t,c] * X[n,t,d]
    # This equals Xflat.T @ Xflat  — valid because sum factorises
    sumXX  = Xflat.T @ Xflat                  # (C, C)

    # S = sum of all trials at each timepoint: (T, C)
    S      = X.sum(axis=0)                    # (T, C)

    # sumXY = Σ_{n≠m} X[n].T @ X[m] = S.T @ S - sumXX
    sumXY  = S.T @ S - sumXX                  # (C, C)

    # Normalise  (no NaN handling — data is clean after imputation)
    Rxx    = sumXX / (n_trials * T)           # mean within-trial autocovariance
    Rxy    = sumXY / (n_trials * (n_trials - 1) * T)  # mean cross-trial covariance

    # By construction with symmetric pairs: Ryy = Rxx
    Ryy    = Rxx.copy()

    # ── Step 3: rcaTrain ─────────────────────────────────────────────

    # Pooled autocovariance
    Rpool  = 0.5 * (Rxx + Ryy)               # = Rxx

    # Eigendecompose Rxx + Ryy = 2*Rpool
    dpool_all, Vpool_all = eigh(Rxx + Ryy)   # ascending order
    # Keep top K eigenvectors (largest eigenvalues)
    Vk     = Vpool_all[:, -n_reg:]            # (C, K)
    dk     = dpool_all[-n_reg:]               # (K,)
    dk     = np.maximum(dk, 1e-10)            # avoid division by zero

    # Regularised inverse applied to symmetrised cross-covariance:
    # Rw = Vk @ diag(1/dk) @ Vk.T @ (Rxy + Rxy.T)
    Rxy_sym = Rxy + Rxy.T
    Rw      = Vk @ np.diag(1.0 / dk) @ Vk.T @ Rxy_sym  # (C, C)

    # Eigendecompose Rw
    dgen, Vgen = np.linalg.eig(Rw)

    # Sort by |eigenvalue| descending (matches MATLAB sort(abs(dGen)))
    sort_idx = np.argsort(np.abs(dgen))[::-1]
    dgen     = dgen[sort_idx]
    Vgen     = Vgen[:, sort_idx]

    # Take top n_components
    k        = min(cfg.n_components, n_ch)
    W        = np.real(Vgen[:, :k])           # (C, k) — discard small imaginary
    lambdas  = np.real(dgen[:k])

    # ── Step 4: forward model ─────────────────────────────────────────
    # A = Rpool @ W @ inv(W.T @ Rpool @ W)  (Parra et al., 2005)
    try:
        A_fwd = Rpool @ W @ scipy_inv(W.T @ Rpool @ W)
    except Exception:
        A_fwd = W.copy()   # fallback

    # ── Step 5: activations (RMS per trial per component) ────────────
    activations = []
    for n in range(n_trials):
        act = X[n] @ W                        # (T, k)
        activations.append(
            act.mean(axis=0) if cfg.time_avg
            else np.sqrt(np.mean(act ** 2, axis=0)))
    A_act = np.stack(activations, axis=0)     # (N, k)

    return (W.astype(np.float32),
            lambdas.astype(np.float32),
            A_act.astype(np.float32),
            A_fwd.astype(np.float32))



# ---------------------------------------------------------------------------
# RCA result container
# ---------------------------------------------------------------------------

@dataclass
class RCAResult:
    """Output of RCA for one group (condition)."""
    group_name      : str
    n_trials        : int
    n_channels      : int
    channel_names   : List[str]

    # Core outputs
    W               : np.ndarray    # (C x K) spatial filters
    lambdas         : np.ndarray    # (K,) reliability scores (eigenvalues of Rw)
    activations     : np.ndarray    # (N x K) trial projections

    # Trial metadata
    trial_meta      : pd.DataFrame  # eeg_id, offset_sec, patient_id, label

    # Full projected signals with EKG reattached (populated by run_rca)
    # List of length N, each entry is (T x (K + 1)) if EKG present, else (T x K)
    projected_trials : Optional[List[np.ndarray]] = None

    # Forward model A = Rpool @ W @ inv(W.T @ Rpool @ W)
    forward_model   : Optional[np.ndarray] = None   # (C x K)

    # UMAP (populated by run_umap)
    umap_embedding  : Optional[np.ndarray] = None   # (N x 2)

    # Config
    config          : Optional[RCAConfig] = None

    def summary(self) -> str:
        cfg = self.config
        bl  = (f"t=0–{cfg.baseline_end_sec:.0f}s"
               if cfg and cfg.baseline else "none")
        std = "yes" if cfg and cfg.standardise else "no"
        lines = [
            f"RCA result — {self.group_name}",
            f"  Trials       : {self.n_trials}",
            f"  Channels     : {self.n_channels}",
            f"  Baseline     : {bl}",
            f"  Standardise  : {std}",
            f"  Components   : {len(self.lambdas)}",
            f"  Reliability  :",
        ]
        for i, lam in enumerate(self.lambdas):
            lines.append(f"    Component {i+1:02d}  λ = {lam:.4f}  "
                         f"{'█' * int(lam * 20)}")
        return "\n".join(lines)

    def plot_filters(
        self,
        output   : str = None,
        n_cols   : int = 3,
    ) -> str:
        """
        Plot spatial filter topomaps for each component.
        Requires MNE. Returns output path.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            import mne
            from mne.viz import plot_topomap
        except ImportError:
            raise ImportError("MNE required for topomap plotting.")

        k       = len(self.lambdas)
        n_rows  = int(np.ceil(k / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 3, n_rows * 3.2),
            facecolor="#0e0e0e",
        )
        axes = np.array(axes).flatten()

        # Build MNE info for bipolar or referential channels
        ch_names = self.channel_names
        try:
            info = mne.create_info(
                ch_names=ch_names, sfreq=200, ch_types="eeg")
            montage = mne.channels.make_standard_montage("standard_1020")
            info.set_montage(montage, match_case=False,
                             on_missing="ignore", verbose=False)
        except Exception:
            info = mne.create_info(
                ch_names=ch_names, sfreq=200, ch_types="eeg")

        for i in range(k):
            ax  = axes[i]
            ax.set_facecolor("#0e0e0e")
            w   = self.W[:, i]
            vmax = np.abs(w).max()

            try:
                plot_topomap(w, info, axes=ax, cmap="RdBu_r",
                             vlim=(-vmax, vmax), show=False,
                             contours=4, sensors=True)
            except Exception:
                ax.bar(range(len(w)), w, color="#7F77DD")

            ax.set_title(
                f"Component {i+1}\nλ = {self.lambdas[i]:.3f}",
                color="#aaaaaa", fontsize=9, pad=4)

        for j in range(k, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"RCA Spatial Filters — {self.group_name}",
            color="white", fontsize=11)
        fig.tight_layout()

        if output is None:
            output = f"rca_filters_{self.group_name}.png"
        fig.savefig(output, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved: {output}")
        return output

    def plot_umap(
        self,
        output       : str  = None,
        color_by     : str  = "patient",  # "patient" | "reliability"
    ) -> str:
        """Plot UMAP embedding coloured by patient or component reliability."""
        if self.umap_embedding is None:
            raise RuntimeError("Run run_umap() first.")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        emb = self.umap_embedding
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0e0e0e")
        ax.set_facecolor("#0e0e0e")

        if color_by == "patient":
            pids   = self.trial_meta["patient_id"].values
            upids  = np.unique(pids)
            cmap   = plt.cm.tab20
            for i, pid in enumerate(upids):
                mask = pids == pid
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           c=[cmap(i % 20)], s=12, alpha=0.7,
                           label=f"p{pid}" if len(upids) <= 10 else None)
            if len(upids) <= 10:
                ax.legend(fontsize=7, framealpha=0.2,
                          labelcolor="white")
            title_sfx = "coloured by patient"

        else:
            # Colour by RMS of first component activation
            c = self.activations[:, 0]
            sc = ax.scatter(emb[:, 0], emb[:, 1],
                            c=c, cmap="viridis", s=12, alpha=0.8)
            plt.colorbar(sc, ax=ax, label="Component 1 RMS",
                         fraction=0.03)
            title_sfx = "coloured by component 1 activation"

        ax.set_xlabel("UMAP 1", color="lightgray")
        ax.set_ylabel("UMAP 2", color="lightgray")
        ax.tick_params(colors="lightgray")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.set_title(
            f"UMAP — {self.group_name}  ({self.n_trials} trials)\n"
            + title_sfx,
            color="white", fontsize=10)

        if output is None:
            output = f"rca_umap_{self.group_name}.png"
        fig.savefig(output, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved: {output}")
        return output

    def save(self, out_dir: str):
        """Save filters, activations, metadata, and config to out_dir."""
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.join(out_dir, self.group_name)
        np.save(f"{stem}_W.npy",           self.W)
        np.save(f"{stem}_lambdas.npy",     self.lambdas)
        np.save(f"{stem}_activations.npy", self.activations)
        self.trial_meta.to_csv(f"{stem}_trials.csv", index=False)
        if self.umap_embedding is not None:
            np.save(f"{stem}_umap.npy", self.umap_embedding)
        meta = {
            "group_name"   : self.group_name,
            "n_trials"     : self.n_trials,
            "n_channels"   : self.n_channels,
            "channel_names": self.channel_names,
            "lambdas"      : self.lambdas.tolist(),
            "config"       : self.config.to_dict() if self.config else {},
        }
        with open(f"{stem}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[RCA] Saved {self.group_name} → {out_dir}")

    @classmethod
    def load(cls, out_dir: str, group_name: str) -> "RCAResult":
        """Load a previously saved RCAResult."""
        stem = os.path.join(out_dir, group_name)
        with open(f"{stem}_meta.json") as f:
            meta = json.load(f)
        umap_path = f"{stem}_umap.npy"
        return cls(
            group_name    = group_name,
            n_trials      = meta["n_trials"],
            n_channels    = meta["n_channels"],
            channel_names = meta["channel_names"],
            W             = np.load(f"{stem}_W.npy"),
            lambdas       = np.load(f"{stem}_lambdas.npy"),
            activations   = np.load(f"{stem}_activations.npy"),
            trial_meta    = pd.read_csv(f"{stem}_trials.csv"),
            umap_embedding= np.load(umap_path) if os.path.exists(umap_path)
                            else None,
            config        = RCAConfig(**meta["config"]) if meta.get("config")
                            else None,
        )


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def run_umap(
    result : RCAResult,
    cfg    : RCAConfig,
) -> RCAResult:
    """Fit UMAP on the RCA component activations and attach to result."""
    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError(
            "umap-learn required.  Install with:  pip install umap-learn")

    reducer = umap_lib.UMAP(
        n_neighbors  = cfg.umap_n_neighbors,
        min_dist     = cfg.umap_min_dist,
        n_components = cfg.umap_n_components,
        random_state = cfg.random_state,
        metric       = "euclidean",
    )
    embedding = reducer.fit_transform(result.activations)
    result.umap_embedding = embedding.astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_rca(
    groups         : Dict[str, "RCAGroup"],
    cfg            : RCAConfig,
    channel_names  : List[str] = None,
    compute_umap   : bool      = True,
    verbose        : bool      = True,
) -> Dict[str, RCAResult]:
    """
    Run RCA (and optionally UMAP) for every group.

    Parameters
    ----------
    groups        : dict of group_name → RCAGroup (from RCAGroup.from_pipeline_output)
    cfg           : RCAConfig
    channel_names : list of channel names (default BIPOLAR_NAMES)
    compute_umap  : run UMAP on component activations (default True)
    verbose       : print progress

    Returns
    -------
    dict of group_name → RCAResult
    """
    ch_names = channel_names or BIPOLAR_NAMES
    results  = {}

    for name, group in sorted(groups.items()):
        if verbose:
            print(f"\n[RCA] Processing group: {name}  "
                  f"({len(group)} trials) ...")

        # ── Step 1: strip EKG — RCA sees only EEG channels ──────────
        # trial.data is already (T x 16) bipolar EEG from the pipeline,
        # so no EKG column is present there. The EKG is held separately
        # in trial.ekg and will be reattached after projection.
        # Note: baseline and standardise are applied inside compute_rca
        # and again in the projection loop below — same cfg ensures consistency.
        trials_data = [t.data for t in group.trials]
        has_ekg     = any(t.ekg is not None for t in group.trials)

        if verbose and has_ekg:
            n_with_ekg = sum(1 for t in group.trials if t.ekg is not None)
            print(f"  EKG present in {n_with_ekg}/{len(group.trials)} trials "
                  f"— excluded from RCA, will be reattached after projection.")

        # ── Step 2: compute RCA on EEG-only data ─────────────────────
        try:
            W, lambdas, A, A_fwd = compute_rca(trials_data, cfg)
        except Exception as ex:
            warnings.warn(f"[RCA] {name} failed: {ex}")
            continue

        # ── Step 3: project raw signals through W, reattach EKG ──────
        # For each trial: X (T x C) @ W (C x K) → (T x K) component ts
        # Then append EKG as an extra column → (T x (K+1)) if EKG exists
        projected_trials = []
        n_bl = int(cfg.baseline_end_sec * cfg.baseline_sfreq)
        # Compute grand mean over all trials for consistent preprocessing
        all_data   = np.stack([t.data.astype(np.float64) for t in group.trials])
        if cfg.baseline and n_bl > 0 and n_bl < all_data.shape[1]:
            bm_all = all_data[:, :n_bl, :].mean(axis=1, keepdims=True)
            all_data = all_data - bm_all
        grand_mu  = all_data.mean(axis=(0,1), keepdims=True)
        grand_std = all_data.std(axis=(0,1),  keepdims=True) + 1e-8

        for i_trial, trial in enumerate(group.trials):
            X = all_data[i_trial].copy()
            X = X - grand_mu[0]
            if cfg.standardise:
                X = X / grand_std[0]

            proj = (X @ W).astype(np.float32)   # (T x K)

            if trial.ekg is not None:
                # Trim/pad EKG to match projection length
                T_proj = proj.shape[0]
                ekg    = trial.ekg[:T_proj].astype(np.float32)
                if len(ekg) < T_proj:
                    ekg = np.pad(ekg, (0, T_proj - len(ekg)))
                # Reattach: (T x K+1), last column is EKG
                proj = np.concatenate(
                    [proj, ekg[:, np.newaxis]], axis=1)

            projected_trials.append(proj)

        # Component names: Comp_1 … Comp_K [+ EKG]
        comp_names = [f"Comp_{k+1}" for k in range(W.shape[1])]
        if has_ekg:
            comp_names.append("EKG")

        meta = pd.DataFrame({
            "eeg_id"    : [t.eeg_id     for t in group.trials],
            "offset_sec": [t.offset_sec for t in group.trials],
            "patient_id": [t.patient_id for t in group.trials],
            "label"     : [t.label      for t in group.trials],
            "is_right"  : [t.is_right   for t in group.trials],
            "has_ekg"   : [t.ekg is not None for t in group.trials],
        })

        result = RCAResult(
            group_name       = name,
            n_trials         = len(group),
            n_channels       = W.shape[0],
            channel_names    = ch_names[:W.shape[0]],
            W                = W,
            lambdas          = lambdas,
            activations      = A,
            forward_model    = A_fwd,
            trial_meta       = meta,
            projected_trials = projected_trials,
            config           = cfg,
        )

        if verbose:
            print(result.summary())

        if compute_umap and len(group) >= cfg.umap_n_neighbors + 1:
            if verbose:
                print(f"  Running UMAP ...")
            result = run_umap(result, cfg)
        elif compute_umap:
            warnings.warn(
                f"[RCA] {name}: too few trials for UMAP "
                f"({len(group)} < {cfg.umap_n_neighbors + 1}). Skipping.")

        results[name] = result

    return results


# ---------------------------------------------------------------------------
# Cross-group comparison
# ---------------------------------------------------------------------------

def cross_group_reliability(
    results : Dict[str, RCAResult],
) -> pd.DataFrame:
    """
    Compare reliability scores across groups.
    Returns DataFrame with one row per group, columns = components.
    """
    rows = []
    for name, r in sorted(results.items()):
        row = {"group": name, "n_trials": r.n_trials}
        for i, lam in enumerate(r.lambdas):
            row[f"lambda_{i+1}"] = float(lam)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_cross_group_umap(
    results : Dict[str, RCAResult],
    output  : str = "rca_cross_group_umap.png",
) -> str:
    """
    Plot all groups in a single UMAP space.
    Each group's trials are projected onto its own RCA filters first,
    then all activations are concatenated and UMAP is run jointly.
    This allows direct comparison of the component structure across classes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError("pip install umap-learn")

    COLORS = {
        "seizure"   : "#e84a4a",
        "lpd_left"  : "#D85A30",
        "lpd_right" : "#f0a070",
        "gpd"       : "#e8a020",
        "lrda_left" : "#1D9E75",
        "lrda_right": "#7FD4B0",
        "grda"      : "#378ADD",
    }

    all_acts  = []
    all_labels= []
    all_groups= []

    for name, r in results.items():
        all_acts.append(r.activations)
        all_labels.extend([name] * r.n_trials)
        all_groups.append(name)

    # Pad activations to same number of components (fill with 0)
    max_k = max(a.shape[1] for a in all_acts)
    padded = []
    for a in all_acts:
        if a.shape[1] < max_k:
            pad = np.zeros((a.shape[0], max_k - a.shape[1]), dtype=np.float32)
            a   = np.hstack([a, pad])
        padded.append(a)

    X = np.vstack(padded)

    # Joint UMAP
    reducer = umap_lib.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2,
        random_state=42, metric="euclidean")
    emb = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0e0e0e")
    ax.set_facecolor("#0e0e0e")

    labels_arr = np.array(all_labels)
    for grp in sorted(set(all_labels)):
        mask  = labels_arr == grp
        color = COLORS.get(grp, "#888888")
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=color, s=14, alpha=0.75, label=grp)

    ax.legend(fontsize=9, framealpha=0.2, labelcolor="white",
              markerscale=2)
    ax.set_xlabel("UMAP 1", color="lightgray")
    ax.set_ylabel("UMAP 2", color="lightgray")
    ax.tick_params(colors="lightgray")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.set_title(
        "Cross-group UMAP — RCA component activations\n"
        "Each point = one 50-second EEG window",
        color="white", fontsize=11)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {output}")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Reliable Components Analysis for HMS-HBAC EEG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--processed_dir", required=True,
                   help="Directory with pipeline .npy output files")
    p.add_argument("--train_csv",     required=True)
    p.add_argument("--out_dir",       required=True,
                   help="Output directory for RCA results")
    p.add_argument("--n_components",  type=int,   default=6)
    p.add_argument("--min_votes",     type=int,   default=3,
                   help="Min dominant-class votes to include trial")
    p.add_argument("--min_trials",    type=int,   default=10,
                   help="Min trials per group")
    p.add_argument("--min_reliability", type=float, default=0.0,
                   help="Min eigenvalue to keep component")
    p.add_argument("--no_standardise", action="store_true",
                   help="Skip per-trial z-scoring")
    p.add_argument("--no_flip",       action="store_true",
                   help="Skip hemisphere flipping for LPD/LRDA")
    p.add_argument("--umap",          action="store_true", default=True,
                   help="Run UMAP on component activations (default on)")
    p.add_argument("--no_umap",       action="store_false", dest="umap")
    p.add_argument("--cross_umap",    action="store_true",
                   help="Also plot all groups in a single UMAP space")
    p.add_argument("--plot_filters",  action="store_true",
                   help="Plot spatial filter topomaps (requires MNE)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = RCAConfig(
        n_components      = args.n_components,
        min_reliability   = args.min_reliability,
        standardise       = not args.no_standardise,
        flip_lateralised  = not args.no_flip,
    )

    groups = RCAGroup.from_pipeline_output(
        processed_dir    = args.processed_dir,
        train_csv        = args.train_csv,
        min_votes        = args.min_votes,
        min_trials       = args.min_trials,
        flip_lateralised = not args.no_flip,
    )

    results = run_rca(groups, cfg, compute_umap=args.umap)

    os.makedirs(args.out_dir, exist_ok=True)

    for name, r in results.items():
        r.save(args.out_dir)
        if args.plot_filters:
            r.plot_filters(
                output=os.path.join(args.out_dir, f"rca_filters_{name}.png"))
        if args.umap and r.umap_embedding is not None:
            r.plot_umap(
                output=os.path.join(args.out_dir, f"rca_umap_{name}.png"))

    # Cross-group reliability table
    rel_df = cross_group_reliability(results)
    rel_path = os.path.join(args.out_dir, "reliability_scores.csv")
    rel_df.to_csv(rel_path, index=False)
    print(f"\nReliability scores:\n{rel_df.to_string(index=False)}")

    if args.cross_umap and len(results) >= 2:
        plot_cross_group_umap(
            results,
            output=os.path.join(args.out_dir, "rca_cross_group_umap.png"))


# ---------------------------------------------------------------------------
# Verification plot — topography map + component waveforms
# ---------------------------------------------------------------------------

def plot_rca_verification(
    result        : "RCAResult",
    group         : "RCAGroup",
    sfreq         : float = 200.0,
    components    : Optional[List[int]] = None,
    error_type    : str   = "sem",       # "sem" | "std" | "ci95"
    output        : str   = None,
    figsize_per_comp : tuple = (14, 4.5),
) -> str:
    """
    Verification plot for one RCA result.

    For each component produces a two-panel figure:

    TOP   — Topography (activation map)
            Each trial's EEG is projected through the spatial filter W[:,k]
            to give a scalar weight per channel.  The nanmean across trials
            is displayed as a hot-cool topomap on the standard 10-20 head.

    BOTTOM — Average component waveform ± error interval
            Project every trial through W[:,k] → (T,) time series.
            Plot the nanmean across trials as a thick line.
            Shade ± error (SEM, SD, or 95% CI) around it.
            Individual trial traces are plotted faintly behind the mean
            so you can see trial-to-trial variability.

    Parameters
    ----------
    result       : RCAResult from run_rca()
    group        : RCAGroup whose trials were used (for raw data access)
    sfreq        : sampling rate Hz (default 200)
    components   : list of 0-based component indices to plot
                   (None = plot all components in result)
    error_type   : "sem"   — standard error of the mean  (default)
                   "std"   — standard deviation
                   "ci95"  — 95% bootstrap confidence interval
    output       : output path prefix (component index appended automatically)
                   None = auto-name from group name
    figsize_per_comp : figure size per component (width, height)

    Returns
    -------
    list of output paths (one per component)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import TwoSlopeNorm

    try:
        import mne
        from mne.viz import plot_topomap
        HAS_MNE = True
    except ImportError:
        HAS_MNE = False
        warnings.warn("MNE not found — topomap will be a bar chart fallback.")

    W        = result.W.astype(np.float64)   # (C, K)
    lambdas  = result.lambdas
    n_comp   = W.shape[1]
    ch_names = result.channel_names

    comp_idx = components if components is not None else list(range(n_comp))

    # ── Build (N, T, K) full time-series activations ─────────────────
    # Re-project each trial through W to get full time series
    # (compute_rca stores only scalar summaries; we need the waveforms)
    cfg        = result.config or RCAConfig()
    trial_ts   = []   # list of (T, K) arrays — one per trial

    for trial in group.trials:
        X = trial.data.astype(np.float64)
        if cfg.standardise:
            mu  = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0,  keepdims=True) + 1e-8
            X   = (X - mu) / std
        act = X @ W          # (T, K)
        trial_ts.append(act)

    # Stack → (N, T, K)
    # Trials may have slightly different lengths — pad with NaN to max T
    T_max   = max(a.shape[0] for a in trial_ts)
    n_trials = len(trial_ts)
    ts_cube  = np.full((n_trials, T_max, n_comp), np.nan, dtype=np.float64)
    for i, act in enumerate(trial_ts):
        ts_cube[i, :act.shape[0], :] = act

    # Time axis in seconds relative to window start
    time_ax = np.arange(T_max) / sfreq

    # ── MNE info for topomap ─────────────────────────────────────────
    if HAS_MNE:
        # Bipolar channels don't map to standard montage positions —
        # use the mean position of each electrode pair as the sensor location
        montage = mne.channels.make_standard_montage("standard_1020")
        pos_dict = {n.lower(): np.array(d["r"])
                    for n, d in zip(montage.ch_names, montage.dig[3:])}

        # Compute 2-D positions for bipolar pairs
        from hms_pipeline import BIPOLAR_CHAINS as _BC
        bip_pos = []
        for pairs in _BC.values():
            for a, c in pairs:
                pa = pos_dict.get(a.lower(), np.zeros(3))
                pc = pos_dict.get(c.lower(), np.zeros(3))
                bip_pos.append((pa + pc) / 2.0)
        bip_pos = np.array(bip_pos)[:, :2]   # (16, 2) — x, y only

        # Build MNE info with custom sensor locations
        n_ch_actual = min(len(ch_names), W.shape[0])
        info = mne.create_info(
            ch_names = ch_names[:n_ch_actual],
            sfreq    = sfreq,
            ch_types = "eeg",
        )
        with info._unlock():
            for i in range(n_ch_actual):
                info["chs"][i]["loc"][:2] = bip_pos[i] \
                    if i < len(bip_pos) else np.zeros(2)

    # ── Plot each component ───────────────────────────────────────────
    saved_paths = []
    WAVEFORM_COLORS = [
        "#D85A30","#1D9E75","#7F77DD","#378ADD",
        "#e8a020","#e84a4a","#aaaaaa",
    ]

    for k in comp_idx:
        if k >= n_comp:
            warnings.warn(f"Component {k} out of range ({n_comp} total).")
            continue

        lam      = float(lambdas[k])
        w_k      = W[:, k]               # (C,) spatial filter for this comp
        ts_k     = ts_cube[:, :, k]      # (N, T) time series across trials

        # ── Error interval ────────────────────────────────────────────
        mean_ts = np.nanmean(ts_k, axis=0)     # (T,)
        n_valid = np.sum(~np.isnan(ts_k), axis=0).clip(min=1)

        if error_type == "std":
            err = np.nanstd(ts_k, axis=0)
            err_label = "±1 SD"

        elif error_type == "ci95":
            # Bootstrap 95% CI — sample with replacement 500 times
            rng_bs = np.random.default_rng(42)
            boot   = np.array([
                np.nanmean(ts_k[rng_bs.integers(0, n_trials, n_trials)], axis=0)
                for _ in range(500)
            ])
            lo = np.nanpercentile(boot, 2.5,  axis=0)
            hi = np.nanpercentile(boot, 97.5, axis=0)
            err_label = "95% CI"

        else:  # sem (default)
            err = np.nanstd(ts_k, axis=0) / np.sqrt(n_valid)
            err_label = "±SEM"

        # ── Figure layout ─────────────────────────────────────────────
        fig = plt.figure(
            figsize = figsize_per_comp,
            facecolor = "#0e0e0e",
        )
        # GridSpec: topomap on left (square), waveform on right (wide)
        gs = gridspec.GridSpec(
            1, 2,
            figure      = fig,
            width_ratios= [1, 3],
            wspace      = 0.08,
            left=0.06, right=0.97, top=0.88, bottom=0.14,
        )
        ax_topo = fig.add_subplot(gs[0])
        ax_wave = fig.add_subplot(gs[1])

        ax_topo.set_facecolor("#0e0e0e")
        ax_wave.set_facecolor("#0e0e0e")

        # ── TOP: Topography ───────────────────────────────────────────
        # Topography value per channel = filter weight w_k
        # We scale by mean absolute activation to weight by contribution
        mean_act = np.nanmean(np.abs(ts_k), axis=0).mean()
        topo_vals = w_k * mean_act    # scale for interpretability

        vmax = np.nanpercentile(np.abs(topo_vals), 99)
        vmax = vmax if vmax > 0 else 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        if HAS_MNE and len(topo_vals) >= 3:
            try:
                im, _ = plot_topomap(
                    topo_vals[:n_ch_actual],
                    info,
                    axes      = ax_topo,
                    cmap      = "RdBu_r",
                    vlim      = (-vmax, vmax),
                    show      = False,
                    contours  = 6,
                    sensors   = True,
                    names     = ch_names[:n_ch_actual],
                )
                plt.colorbar(im, ax=ax_topo, fraction=0.046, pad=0.04,
                             label="Weight × activation",
                             format="%.2f")
                ax_topo.set_title(
                    f"Topography\nComponent {k+1}  λ={lam:.3f}",
                    color="#aaaaaa", fontsize=9, pad=6)
            except Exception as e:
                _fallback_topo_bar(ax_topo, topo_vals, ch_names,
                                   k, lam, vmax)
        else:
            _fallback_topo_bar(ax_topo, topo_vals, ch_names, k, lam, vmax)

        # ── BOTTOM: Waveform ──────────────────────────────────────────
        wave_color = WAVEFORM_COLORS[k % len(WAVEFORM_COLORS)]

        # Individual trial traces (faint)
        for i in range(min(n_trials, 50)):   # cap at 50 to avoid overload
            ax_wave.plot(time_ax, ts_k[i], color=wave_color,
                         lw=0.3, alpha=0.12, zorder=1)

        # Mean waveform
        ax_wave.plot(time_ax, mean_ts, color=wave_color,
                     lw=2.0, alpha=0.95, zorder=4,
                     label=f"Mean (n={n_trials})")

        # Error shading
        if error_type == "ci95":
            ax_wave.fill_between(time_ax, lo, hi,
                                 color=wave_color, alpha=0.25, zorder=3,
                                 label=err_label)
        else:
            ax_wave.fill_between(time_ax, mean_ts - err, mean_ts + err,
                                 color=wave_color, alpha=0.25, zorder=3,
                                 label=err_label)

        # Zero line
        ax_wave.axhline(0, color="#444444", lw=0.6, zorder=2)

        # Second-tick grid
        for t_sec in np.arange(0, time_ax[-1], 1.0):
            ax_wave.axvline(t_sec, color="#1e1e1e", lw=0.5, zorder=0)

        # Epoch marker (label window: t=20–30 s in the 50-s window)
        epoch_lo = min(20.0, time_ax[-1])
        epoch_hi = min(30.0, time_ax[-1])
        if epoch_lo < epoch_hi:
            ax_wave.axvspan(epoch_lo, epoch_hi,
                            color="#e84a4a", alpha=0.07, zorder=0)
            ax_wave.axvline(epoch_lo, color="#e84a4a",
                            lw=0.8, ls="--", alpha=0.5)
            ax_wave.axvline(epoch_hi, color="#e84a4a",
                            lw=0.8, ls="--", alpha=0.5)

        ax_wave.set_xlabel("Time (s)", color="lightgray", fontsize=9)
        ax_wave.set_ylabel("Component activation (a.u.)",
                           color="lightgray", fontsize=9)
        ax_wave.tick_params(colors="lightgray", labelsize=8)
        for sp in ax_wave.spines.values():
            sp.set_edgecolor("#444")

        leg = ax_wave.legend(fontsize=8, framealpha=0.15,
                             labelcolor="white", loc="upper right")

        # Overall title
        fig.suptitle(
            f"RCA Verification  ·  {result.group_name}  ·  "
            f"Component {k+1} of {n_comp}  ·  λ = {lam:.4f}  ·  "
            f"n = {n_trials} trials",
            color="white", fontsize=10, y=0.98,
        )

        # Save
        if output is None:
            out_p = f"rca_verify_{result.group_name}_comp{k+1}.png"
        else:
            base, ext = os.path.splitext(output)
            out_p = f"{base}_comp{k+1}{ext or '.png'}"

        fig.savefig(out_p, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved: {out_p}")
        saved_paths.append(out_p)

    return saved_paths


def _fallback_topo_bar(ax, topo_vals, ch_names, k, lam, vmax):
    """Bar chart fallback when MNE topomap is unavailable."""
    colors = ["#e84a4a" if v > 0 else "#378ADD" for v in topo_vals]
    ax.barh(range(len(topo_vals)), topo_vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(topo_vals)))
    ax.set_yticklabels(ch_names, fontsize=6, color="lightgray")
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_xlim(-vmax * 1.1, vmax * 1.1)
    ax.set_title(f"Topography\nComponent {k+1}  λ={lam:.3f}",
                 color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="lightgray", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
