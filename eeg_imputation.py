"""
eeg_imputation.py  —  EEG NaN imputation (montage-agnostic)
Alexandra Yakovleva, 2026
============================================================
Two-level imputation strategy:

Level 1 — Sample-level NaNs
    Linear interpolation for short gaps (≤ max_interp_gap_sec).
    Rolling-median fill for longer gaps.
    Forward/backward fill for any remaining boundary NaNs.

Level 2 — Channel-level NaNs (entire electrode dead / missing)
    Distance-weighted mean of k nearest neighbours in sensor space.
    Weights = 1/d² (inverse squared Euclidean distance in 3-D).
    Neighbour positions come from MNE standard montages or a custom
    position dict — so this works for 10-20 (19 ch), 64-ch, 128-ch,
    256-ch, or any custom layout.

Montage support
---------------
    "standard_1020"   — 19-channel clinical EEG  (default)
    "standard_1005"   — 10-05 extended (64 ch subset)
    "biosemi64"       — BioSemi 64-channel cap
    "biosemi128"      — BioSemi 128-channel cap
    "biosemi256"      — BioSemi 256-channel cap
    "GSN-HydroCel-128" — EGI 128-channel HydroCel
    custom dict       — {ch_name: np.ndarray([x,y,z])} in metres

Public API
----------
    impute_eeg_window(window, channels, sfreq, montage, k_neighbors,
                      max_interp_gap_sec, rolling_median_sec, verbose)
                      -> (DataFrame, NanReport)

    build_neighbor_graph(channels, montage, k) -> dict

    get_montage_positions(channels, montage)   -> dict

    report_nans(window, channels)              -> NanReport
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Type alias for montage specification
# ---------------------------------------------------------------------------

# A montage can be:
#   - a string naming an MNE standard montage ("standard_1020", "biosemi64" …)
#   - a dict mapping channel name → 3-D position array (metres)
MontageSpec = Union[str, Dict[str, np.ndarray]]

_DEFAULT_MONTAGE = "standard_1020"

# ---------------------------------------------------------------------------
# Montage position loading
# ---------------------------------------------------------------------------

def get_montage_positions(
    channels: List[str],
    montage:  MontageSpec = _DEFAULT_MONTAGE,
) -> Dict[str, np.ndarray]:
    """
    Return {channel_name: xyz_array} for the requested channels.

    Parameters
    ----------
    channels : list of str
        Electrode names to look up.
    montage : str or dict
        Named MNE standard montage  — any string accepted by
        mne.channels.make_standard_montage(), e.g.:
            "standard_1020"       19-ch clinical EEG
            "standard_1005"       10-05 extended
            "biosemi64"           BioSemi 64-ch
            "biosemi128"          BioSemi 128-ch
            "biosemi256"          BioSemi 256-ch
            "GSN-HydroCel-128"   EGI HydroCel 128-ch
        Custom dict — {ch_name: np.ndarray([x, y, z])} in metres.
        Channels not in the dict get equal-weight fallback treatment.

    Returns
    -------
    dict mapping channel name → (3,) float64 array (metres)
    """
    if isinstance(montage, dict):
        # Custom position dict — filter to requested channels
        out = {}
        for ch in channels:
            if ch in montage:
                out[ch] = np.asarray(montage[ch], dtype=float)
            else:
                warnings.warn(
                    f"Channel '{ch}' not in custom montage dict. "
                    "Equal-weight fallback will be used for imputation.")
        return out

    # Named MNE montage
    try:
        import mne
    except ImportError:
        raise ImportError(
            "MNE is required for named montage position lookup.\n"
            "Install with:  pip install mne\n"
            "Or provide a custom position dict instead."
        )

    try:
        mne_montage = mne.channels.make_standard_montage(montage)
    except ValueError as e:
        raise ValueError(
            f"Unknown MNE montage '{montage}'.\n"
            f"Valid options include: standard_1020, standard_1005, "
            f"biosemi64, biosemi128, biosemi256, GSN-HydroCel-128 ...\n"
            f"Original error: {e}"
        )

    # Build name→position map (case-insensitive)
    pos_all  = {}
    name_map = {n.lower(): n for n in mne_montage.ch_names}
    for dig, name in zip(mne_montage.dig[3:], mne_montage.ch_names):
        pos_all[name] = np.array(dig["r"], dtype=float)

    out = {}
    for ch in channels:
        canonical = name_map.get(ch.lower())
        if canonical:
            out[ch] = pos_all[canonical]
        else:
            warnings.warn(
                f"Channel '{ch}' not found in montage '{montage}'. "
                "Equal-weight fallback will be used for imputation."
            )
    return out


# ---------------------------------------------------------------------------
# Neighbour graph
# ---------------------------------------------------------------------------

def build_neighbor_graph(
    channels: List[str],
    montage:  MontageSpec = _DEFAULT_MONTAGE,
    k:        int         = 6,
) -> Dict[str, List[str]]:
    """
    Build a k-nearest-neighbour graph over 3-D electrode positions.

    Parameters
    ----------
    channels : list of str
        Electrode names to include.
    montage : str or dict
        Montage specification (see get_montage_positions).
    k : int
        Neighbours per electrode. Default 6 covers the immediate ring
        in 10-20; increase to 8-12 for denser 64/128-ch layouts.

    Returns
    -------
    dict mapping channel name → list of k nearest neighbour names
    (ordered nearest-first).  Channels with unknown positions are
    assigned all other channels as neighbours (equal-weight fallback).
    """
    pos      = get_montage_positions(channels, montage)
    with_pos = [ch for ch in channels if ch in pos]
    no_pos   = [ch for ch in channels if ch not in pos]

    # Compute pairwise distances for channels that have positions
    graph: Dict[str, List[str]] = {}

    if with_pos:
        ch_list = with_pos
        coords  = np.array([pos[c] for c in ch_list])   # (N, 3)
        diff    = coords[:, None, :] - coords[None, :, :]
        dist    = np.sqrt((diff ** 2).sum(axis=-1))       # (N, N)
        np.fill_diagonal(dist, np.inf)

        for i, ch in enumerate(ch_list):
            k_eff   = min(k, len(ch_list) - 1)
            order   = np.argsort(dist[i])[:k_eff]
            graph[ch] = [ch_list[j] for j in order]

    # Channels without positions → all others as neighbours
    for ch in no_pos:
        graph[ch] = [c for c in channels if c != ch][:k]

    return graph


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass
class NanReport:
    """Summary of NaN content before and after imputation."""
    total_samples:       int      = 0
    total_channels:      int      = 0
    montage:             str      = ""
    channels_with_nans:  List[str] = field(default_factory=list)
    channels_all_nan:    List[str] = field(default_factory=list)
    sample_nan_counts:   Dict[str, int] = field(default_factory=dict)
    # Filled after imputation
    sample_nans_filled:  Dict[str, int] = field(default_factory=dict)
    channel_nans_filled: List[str]      = field(default_factory=list)
    channels_unfillable: List[str]      = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"NaN report  ({self.total_samples} samples × "
            f"{self.total_channels} channels"
            + (f"  ·  montage: {self.montage}" if self.montage else "")
            + ")",
            f"  Channels with any NaN : {len(self.channels_with_nans)}"
            + (f"  {self.channels_with_nans}" if self.channels_with_nans else ""),
            f"  Channels entirely NaN : {len(self.channels_all_nan)}"
            + (f"  {self.channels_all_nan}" if self.channels_all_nan else ""),
        ]
        if self.sample_nans_filled:
            lines.append("  Sample-level NaNs filled:")
            for ch, n in self.sample_nans_filled.items():
                lines.append(
                    f"    {ch:8s}: {n} samples "
                    f"({100 * n / max(self.total_samples, 1):.1f}%)")
        if self.channel_nans_filled:
            lines.append(
                f"  Spatially imputed : {self.channel_nans_filled}")
        if self.channels_unfillable:
            lines.append(
                f"  Unfillable (zeroed): {self.channels_unfillable}")
        return "\n".join(lines)


def report_nans(
    window:   pd.DataFrame,
    channels: List[str],
) -> NanReport:
    """Inspect a window DataFrame and return a NanReport (no modification)."""
    r = NanReport(
        total_samples  = len(window),
        total_channels = len(channels),
    )
    for ch in channels:
        if ch not in window.columns:
            r.channels_all_nan.append(ch)
            continue
        col   = window[ch]
        n_nan = int(col.isna().sum())
        if n_nan == 0:
            continue
        r.channels_with_nans.append(ch)
        r.sample_nan_counts[ch] = n_nan
        if n_nan == len(window):
            r.channels_all_nan.append(ch)
    return r


# ---------------------------------------------------------------------------
# Level-1: sample-level temporal imputation
# ---------------------------------------------------------------------------

_DEFAULT_MAX_INTERP_GAP_SEC = 0.5    # gaps ≤ 0.5 s → linear interpolation
_DEFAULT_ROLLING_MEDIAN_SEC = 2.0    # rolling-median window for long gaps


def _fill_sample_nans(
    series:             pd.Series,
    sfreq:              float,
    max_interp_gap_sec: float = _DEFAULT_MAX_INTERP_GAP_SEC,
    rolling_median_sec: float = _DEFAULT_ROLLING_MEDIAN_SEC,
) -> Tuple[pd.Series, int]:
    """
    Fill NaNs in a single-channel time series.

    Strategy
    --------
    1. Identify contiguous NaN runs.
    2. Short runs (≤ max_interp_gap_sec × sfreq): linear interpolation.
    3. Long runs: rolling-median fill.
    4. Boundary NaNs: forward then backward fill.

    Returns
    -------
    filled_series, n_nan_before (int)
    """
    s             = series.copy().astype(float)
    n_nan_before  = int(s.isna().sum())
    if n_nan_before == 0:
        return s, 0

    max_gap_samp = int(max_interp_gap_sec * sfreq)
    rolling_win  = max(1, int(rolling_median_sec * sfreq))

    rolling_med  = s.rolling(
        window=rolling_win, center=True, min_periods=1).median()

    # Identify contiguous NaN blocks
    is_nan  = s.isna()
    changes = is_nan.astype(int).diff().fillna(0)
    starts  = list(s.index[changes == 1])
    ends    = list(s.index[changes == -1])

    if is_nan.iloc[0]:
        starts = [s.index[0]] + starts
    if is_nan.iloc[-1]:
        ends = ends + [s.index[-1] + 1]

    for start, end in zip(starts, ends):
        gap_len = end - start
        if gap_len > max_gap_samp:
            s.iloc[start:end] = rolling_med.iloc[start:end]
        # Short gaps: left as NaN for linear interpolation below

    s = s.interpolate(
        method="linear", limit=max_gap_samp, limit_direction="both")
    s = s.ffill().bfill()

    return s, n_nan_before


# ---------------------------------------------------------------------------
# Level-2: channel-level spatial imputation
# ---------------------------------------------------------------------------

def _fill_channel_nans(
    window:             pd.DataFrame,
    channels:           List[str],
    neighbor_graph:     Dict[str, List[str]],
    pos:                Dict[str, np.ndarray],
    sfreq:              float,
    max_interp_gap_sec: float,
    rolling_median_sec: float,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Replace all-NaN channels with distance-weighted neighbour mean.

    Returns
    -------
    window (modified), channels_spatially_imputed, channels_unfillable
    """
    spatially_filled: List[str] = []
    unfillable:       List[str] = []

    for ch in channels:
        col = (window[ch] if ch in window.columns
               else pd.Series([np.nan] * len(window),
                              index=window.index))
        if not col.isna().all():
            continue   # not a dead channel

        neighbours = neighbor_graph.get(ch, [])
        weights, signals = [], []

        for nb in neighbours:
            if nb not in window.columns:
                continue
            nb_col = window[nb]
            if nb_col.isna().all():
                continue   # skip also-dead neighbours

            nb_filled, _ = _fill_sample_nans(
                nb_col, sfreq, max_interp_gap_sec, rolling_median_sec)

            # Weight: 1/d² if both positions known, else equal weight
            if ch in pos and nb in pos:
                d = float(np.linalg.norm(pos[ch] - pos[nb]))
                w = 1.0 / (d ** 2 + 1e-12)
            else:
                w = 1.0

            weights.append(w)
            signals.append(nb_filled.values)

        if signals:
            w_arr  = np.array(weights)
            w_arr /= w_arr.sum()
            imputed = np.stack(signals, axis=0)    # (k, T)
            window[ch] = (w_arr[:, None] * imputed).sum(axis=0)
            spatially_filled.append(ch)
        else:
            window[ch] = 0.0
            unfillable.append(ch)
            warnings.warn(
                f"Channel '{ch}' has no valid neighbours — zeroed.")

    return window, spatially_filled, unfillable


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def impute_eeg_window(
    window:             pd.DataFrame,
    channels:           List[str],
    sfreq:              float       = 200.0,
    montage:            MontageSpec = _DEFAULT_MONTAGE,
    k_neighbors:        int         = 6,
    max_interp_gap_sec: float       = _DEFAULT_MAX_INTERP_GAP_SEC,
    rolling_median_sec: float       = _DEFAULT_ROLLING_MEDIAN_SEC,
    verbose:            bool        = True,
) -> Tuple[pd.DataFrame, NanReport]:
    """
    Impute NaNs in an EEG window DataFrame.

    Works for any channel layout — 10-20 (19 ch), 64-ch, 128-ch, or
    a custom set of electrodes — by specifying the `montage` parameter.

    Parameters
    ----------
    window : pd.DataFrame
        Raw EEG window.  Shape (T, C).  May contain NaN values.
    channels : list of str
        Channel names to process.  Must be a subset of window.columns
        (or will be added as all-NaN and spatially imputed).
    sfreq : float
        Sampling frequency in Hz.  Default 200 (HMS dataset).
    montage : str or dict
        Electrode position source.  Named MNE montage string or a custom
        dict {ch_name: np.ndarray([x, y, z])} in metres.
        Examples:
            "standard_1020"       19-ch clinical EEG  (default)
            "biosemi64"           BioSemi 64-ch
            "biosemi128"          BioSemi 128-ch
            {"Fp1": [0.0, 0.09, 0.0], ...}   custom
    k_neighbors : int
        Spatial neighbours for channel-level imputation.
        Recommended values:
            6   for 10-20  (default)
            8   for 64-ch
            12  for 128-ch or 256-ch
    max_interp_gap_sec : float
        Max gap filled by linear interpolation.  Default 0.5 s.
    rolling_median_sec : float
        Window for rolling-median fill of long gaps.  Default 2.0 s.
    verbose : bool
        Print NaN report.

    Returns
    -------
    window_filled : pd.DataFrame
    report        : NanReport
    """
    window = window.copy()

    # Ensure all requested channels exist in the DataFrame
    for ch in channels:
        if ch not in window.columns:
            window[ch] = np.nan

    # ── Pre-imputation report ────────────────────────────────────────
    report         = report_nans(window, channels)
    report.montage = str(montage) if isinstance(montage, str) else "custom"

    # ── Build spatial structures ─────────────────────────────────────
    pos            = get_montage_positions(channels, montage)
    neighbor_graph = build_neighbor_graph(channels, montage, k=k_neighbors)

    # ── Level 1: sample-level interpolation ─────────────────────────
    for ch in channels:
        if ch in report.channels_all_nan:
            continue   # handled at level 2
        if ch not in report.channels_with_nans:
            continue   # clean channel

        filled_series, n_filled = _fill_sample_nans(
            window[ch], sfreq, max_interp_gap_sec, rolling_median_sec)
        window[ch] = filled_series
        report.sample_nans_filled[ch] = n_filled

    # ── Level 2: spatial imputation for dead channels ────────────────
    if report.channels_all_nan:
        window, spatially_filled, unfillable = _fill_channel_nans(
            window, channels, neighbor_graph, pos,
            sfreq, max_interp_gap_sec, rolling_median_sec,
        )
        report.channel_nans_filled = spatially_filled
        report.channels_unfillable = unfillable

    # ── Safety net: zero any residual NaNs ───────────────────────────
    for ch in channels:
        if window[ch].isna().any():
            window[ch] = window[ch].fillna(0.0)

    if verbose:
        print(report)

    return window, report


# ---------------------------------------------------------------------------
# Convenience: list supported named montages
# ---------------------------------------------------------------------------

SUPPORTED_MONTAGES = [
    "standard_1020",        # 19-ch clinical
    "standard_1005",        # 10-05 extended
    "biosemi16",
    "biosemi32",
    "biosemi64",
    "biosemi128",
    "biosemi256",
    "easycap-M1",
    "GSN-HydroCel-32",
    "GSN-HydroCel-64_1.0",
    "GSN-HydroCel-128",
    "GSN-HydroCel-256",
]

def list_montages() -> List[str]:
    """Return a list of commonly supported MNE montage names."""
    return SUPPORTED_MONTAGES