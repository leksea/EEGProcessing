"""
hms_plot_common.py  —  Shared constants, I/O and signal processing
Alexandra Yakovleva, 2026
===================================================================
Imported by hms_visualize.py.  No matplotlib dependency at import time.

Exports
-------
Constants
    SFREQ, WINDOW_SEC, EPOCH_START, EPOCH_END
    EEG_CHANNELS
    BIPOLAR_CHAINS, CHAIN_COLORS, CHAIN_FULL, CHAIN_LABELS
    CHAINS
    SPEC_TIME_RES, SPEC_DURATION, N_TIME_ROWS, N_FREQ_BINS, BANDS
    VOTE_COLORS

I/O helpers
    resolve_eeg_ids(train_csv, eeg_id, patient_id)
        -> (eeg_id, offset_sec, patient_id, row)

    resolve_spec_ids(train_csv, eeg_id, spectrogram_id, patient_id)
        -> (eeg_id, spectrogram_id, spec_label_offset_sec, patient_id)

    load_eeg_window(eeg_dir, eeg_id, offset_sec)    -> DataFrame
    load_spectrogram(spec_dir, spectrogram_id)       -> DataFrame
    parse_spectrogram(df)                            -> (dict, np.ndarray)

Signal processing
    bandpass(sig, lo, hi, fs, order)   -> np.ndarray
    notch_filter(sig, freq, fs, q)     -> np.ndarray
    compute_bipolar(window, pairs, lo, hi, apply_notch) -> list[np.ndarray]

Topomap helpers
    get_channel_metric(signal, metric) -> float
    build_mne_info()                   -> mne.Info

Annotation helpers
    build_label_votes(row)             -> dict
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ===========================================================================
# Constants
# ===========================================================================

# ── EEG recording ───────────────────────────────────────────────────────────
SFREQ        : int   = 200      # Hz — all HMS EEGs sampled at 200 Hz
WINDOW_SEC   : int   = 50       # seconds per label window
EPOCH_START  : float = 20.0     # labeled epoch starts at t=20 s
EPOCH_END    : float = 30.0     # labeled epoch ends   at t=30 s

# 19-channel referential montage present in every HMS parquet file
EEG_CHANNELS: List[str] = [
    "Fp1", "F3", "C3", "P3", "O1",
    "F7",  "T3", "T5",
    "Fz",  "Cz", "Pz",
    "Fp2", "F4", "C4", "P4", "O2",
    "F8",  "T4", "T6",
]

# Longitudinal bipolar double-banana montage (16 channels, 4 chains of 4)
BIPOLAR_CHAINS: Dict[str, List[Tuple[str, str]]] = {
    "LL": [("Fp1","F7"), ("F7","T3"), ("T3","T5"), ("T5","O1")],
    "LP": [("Fp1","F3"), ("F3","C3"), ("C3","P3"), ("P3","O1")],
    "RP": [("Fp2","F4"), ("F4","C4"), ("C4","P4"), ("P4","O2")],
    "RL": [("Fp2","F8"), ("F8","T4"), ("T4","T6"), ("T6","O2")],
}

CHAIN_COLORS: Dict[str, str] = {
    "LL": "#D85A30",   # coral / orange
    "LP": "#1D9E75",   # teal
    "RP": "#7F77DD",   # purple
    "RL": "#378ADD",   # blue
}

CHAIN_FULL: Dict[str, str] = {
    "LL": "Left Lateral",
    "LP": "Left Parasagittal",
    "RP": "Right Parasagittal",
    "RL": "Right Lateral",
}

CHAIN_LABELS: Dict[str, str] = {
    k: f"{k}  {v}" for k, v in CHAIN_FULL.items()
}

# Ordered list of chain names
CHAINS: List[str] = ["LL", "LP", "RP", "RL"]

# ── Spectrogram ─────────────────────────────────────────────────────────────
SPEC_TIME_RES : float = 2.0     # seconds per row
SPEC_DURATION : float = 600.0   # 10 minutes total
N_TIME_ROWS   : int   = 300     # 600 / 2
N_FREQ_BINS   : int   = 100     # per chain

# EEG frequency band boundaries for annotation (Hz)
BANDS: Dict[str, Tuple[float, float]] = {
    "δ": (0.5,  4.0),
    "θ": (4.0,  8.0),
    "α": (8.0,  12.0),
    "β": (12.0, 20.0),
}

# ── Annotator votes ──────────────────────────────────────────────────────────
VOTE_COLORS: Dict[str, str] = {
    "seizure": "#e84a4a",
    "lpd":     "#D85A30",
    "gpd":     "#e8a020",
    "lrda":    "#1D9E75",
    "grda":    "#378ADD",
    "other":   "#888888",
}

VOTE_COLS: List[str] = [
    "seizure_vote", "lpd_vote", "gpd_vote",
    "lrda_vote",    "grda_vote", "other_vote",
]

# ===========================================================================
# I/O helpers
# ===========================================================================

def resolve_eeg_ids(
    train_csv:  str,
    eeg_id:     Optional[int],
    patient_id: Optional[int],
) -> Tuple[int, int, int, pd.Series]:
    """
    Resolve (eeg_id, offset_sec, patient_id, row) from train.csv.
    Accepts either eeg_id or patient_id (auto-selects first eeg_id).
    """
    df = pd.read_csv(train_csv)
    if eeg_id is None and patient_id is None:
        raise ValueError("Provide eeg_id or patient_id.")
    if eeg_id is None:
        sub = df[df["patient_id"] == patient_id]
        if sub.empty:
            raise ValueError(f"patient_id {patient_id} not found in train.csv")
        eeg_id = int(sub["eeg_id"].iloc[0])
        print(f"Auto-selected eeg_id={eeg_id} for patient_id={patient_id}")
    row = df[df["eeg_id"] == eeg_id].iloc[0]
    return (
        int(row["eeg_id"]),
        int(row["eeg_label_offset_seconds"]),
        int(row.get("patient_id", -1)),
        row,
    )


def resolve_spec_ids(
    train_csv:      str,
    eeg_id:         Optional[int],
    spectrogram_id: Optional[int],
    patient_id:     Optional[int],
) -> Tuple[int, int, float, int]:
    """
    Resolve (eeg_id, spectrogram_id, spec_label_offset_sec, patient_id).
    Accepts any one of the three identifiers.
    """
    df = pd.read_csv(train_csv)
    if eeg_id is None and spectrogram_id is None and patient_id is None:
        raise ValueError(
            "Provide at least one of eeg_id, spectrogram_id, or patient_id.")
    if patient_id is not None and eeg_id is None and spectrogram_id is None:
        sub = df[df["patient_id"] == patient_id]
        if sub.empty:
            raise ValueError(f"patient_id {patient_id} not found in train.csv")
        eeg_id = int(sub["eeg_id"].iloc[0])
        print(f"Auto-selected eeg_id={eeg_id} for patient_id={patient_id}")
    row = (df[df["eeg_id"] == eeg_id].iloc[0] if eeg_id is not None
           else df[df["spectrogram_id"] == spectrogram_id].iloc[0])
    return (
        int(row["eeg_id"]),
        int(row["spectrogram_id"]),
        float(row["spectrogram_label_offset_seconds"]),
        int(row.get("patient_id", -1)),
    )


def load_eeg_window(
    eeg_dir:    str,
    eeg_id:     int,
    offset_sec: int,
) -> pd.DataFrame:
    """Load the 50-second EEG window starting at offset_sec."""
    path = os.path.join(eeg_dir, f"{eeg_id}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"EEG parquet not found: {path}")
    print(f"Loading: {path}")
    eeg    = pd.read_parquet(path)
    start  = int(offset_sec * SFREQ)
    end    = start + WINDOW_SEC * SFREQ
    window = eeg.iloc[start:end].reset_index(drop=True)
    print(f"  Window shape: {window.shape}")
    return window


def load_spectrogram(spec_dir: str, spectrogram_id: int) -> pd.DataFrame:
    """Load a spectrogram parquet file."""
    path = os.path.join(spec_dir, f"{spectrogram_id}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Spectrogram not found: {path}")
    print(f"Loading spectrogram: {path}")
    df = pd.read_parquet(path)
    print(f"  Shape: {df.shape}  "
          f"cols: {list(df.columns[:3])} … {list(df.columns[-2:])}")
    return df


def parse_spectrogram(
    df: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Split the wide spectrogram parquet into {chain: array(time, freq)}.
    Also extracts the frequency axis from column names.
    """
    chains_data: Dict[str, np.ndarray] = {}
    for chain in CHAINS:
        cols = [c for c in df.columns if c.startswith(f"{chain}_")]
        if not cols:
            warnings.warn(f"No columns for chain '{chain}' — skipping.")
            continue
        chains_data[chain] = df[cols].values.astype(np.float32)

    freqs = np.linspace(0.59, 20.0, N_FREQ_BINS)
    for chain in CHAINS:
        cols = [c for c in df.columns if c.startswith(f"{chain}_")]
        if cols:
            try:
                freqs = np.array([float(c.split("_")[1]) for c in cols])
            except (IndexError, ValueError):
                pass
            break

    return chains_data, freqs


# ===========================================================================
# Signal processing
# ===========================================================================

def bandpass(
    sig:   np.ndarray,
    lo:    float,
    hi:    float,
    fs:    float = SFREQ,
    order: int   = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    from scipy.signal import butter, sosfiltfilt
    nyq = fs / 2.0
    lo  = max(lo, 0.01)
    hi  = min(hi, nyq - 0.5)
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, sig)


def notch_filter(
    sig:  np.ndarray,
    freq: float = 60.0,
    fs:   float = SFREQ,
    q:    float = 30.0,
) -> np.ndarray:
    """Notch filter at power-line frequency."""
    from scipy.signal import iirnotch, lfilter
    b, a = iirnotch(freq / (fs / 2.0), q)
    return lfilter(b, a, sig)


def compute_bipolar(
    window:      pd.DataFrame,
    chain_pairs: List[Tuple[str, str]],
    lo:          float,
    hi:          float,
    apply_notch: bool,
) -> List[np.ndarray]:
    """
    Compute filtered bipolar difference signals for one chain.
    Returns one 1-D array per electrode pair.
    """
    sigs = []
    for anode, cathode in chain_pairs:
        a    = (window[anode].values.astype(float)
                if anode   in window.columns else np.zeros(len(window)))
        c    = (window[cathode].values.astype(float)
                if cathode in window.columns else np.zeros(len(window)))
        diff = np.nan_to_num(a - c)
        if lo > 0 or hi < SFREQ / 2:
            diff = bandpass(diff, lo, hi)
        if apply_notch:
            diff = notch_filter(diff)
        sigs.append(diff)
    return sigs


# ===========================================================================
# Topomap helpers
# ===========================================================================

def get_channel_metric(signal: np.ndarray, metric: str) -> float:
    """Collapse a 1-D time series to a scalar amplitude measure."""
    if metric == "rms":
        return float(np.sqrt(np.mean(signal ** 2)))
    elif metric == "mean":
        return float(np.mean(np.abs(signal)))
    elif metric == "std":
        return float(np.std(signal))
    elif metric == "peak":
        return float(np.max(np.abs(signal)))
    else:
        raise ValueError(f"Unknown metric '{metric}'. "
                         "Choose: rms | mean | std | peak")


def build_mne_info():
    """Create an MNE Info object with standard 10-20 positions."""
    try:
        import mne
    except ImportError:
        raise ImportError(
            "MNE is required for topomap generation.\n"
            "Install with:  pip install mne")
    info = mne.create_info(
        ch_names = EEG_CHANNELS,
        sfreq    = SFREQ,
        ch_types = "eeg",
    )
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, match_case=False, on_missing="ignore")
    return info


# ===========================================================================
# Annotation helpers
# ===========================================================================

def build_label_votes(row: pd.Series) -> Dict[str, int]:
    """Extract annotator vote counts from a train.csv row."""
    votes = {}
    for col in VOTE_COLS:
        if col in row.index:
            votes[col.replace("_vote", "")] = int(row[col])
    return votes
