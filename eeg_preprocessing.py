"""
eeg_preprocessing.py  —  EEG preprocessing entry point
Alexandra Yakovleva, 2026
=======================================================
Thin orchestration layer that composes eeg_imputation and eeg_cardiac
into a single callable for use by hms_pipeline.py (and any other pipeline).

Public API
----------
    preprocess_eeg_window(
        window, channels, sfreq, montage,
        impute, k_neighbors, max_interp_gap_sec, rolling_median_sec,
        cardiac_method, ekg_col, epoch_ms, robust,
        verbose
    ) -> (DataFrame, PreprocessReport)

    preprocess_eeg_from_parquet(
        eeg_path, channels, offset_sec, window_sec, sfreq,
        montage, impute, k_neighbors,
        cardiac_method, ekg_col, n_ica_components, max_fit_sec,
        l_freq, h_freq, ica_threshold,
        verbose
    ) -> (DataFrame, PreprocessReport)

Montage examples
----------------
    "standard_1020"   19-ch clinical EEG  (HMS dataset)
    "biosemi64"       BioSemi 64-ch
    "biosemi128"      BioSemi 128-ch
    {"Fp1": [x,y,z], ...}   custom dict (metres)

Cardiac methods
---------------
    "none"      skip cardiac removal
    "template"  QRS average-artifact subtraction  (fast, no MNE)
    "ica"       FastICA on full recording          (best quality)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from eeg_nan_sub import (
    impute_eeg_window,
    NanReport,
    MontageSpec,
    _DEFAULT_MONTAGE,
    _DEFAULT_MAX_INTERP_GAP_SEC,
    _DEFAULT_ROLLING_MEDIAN_SEC,
)
from eeg_cardiac import (
    remove_cardiac_template,
    remove_cardiac_ica,
)

# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreprocessReport:
    """Combined report for one preprocessing run."""
    eeg_path:       str = ""
    offset_sec:     float = 0.0
    window_sec:     float = 50.0
    n_channels:     int   = 0
    sfreq:          float = 200.0
    montage:        str   = ""
    cardiac_method: str   = "none"
    imputation:     Optional[NanReport] = None
    cardiac_ok:     bool  = True
    cardiac_msg:    str   = ""
    elapsed_sec:    float = 0.0

    def __str__(self) -> str:
        lines = [
            "PreprocessReport",
            f"  file         : {self.eeg_path or '(in-memory)'}",
            f"  offset       : {self.offset_sec:.1f} s",
            f"  window       : {self.window_sec:.1f} s",
            f"  channels     : {self.n_channels}",
            f"  sfreq        : {self.sfreq} Hz",
            f"  montage      : {self.montage}",
            f"  cardiac      : {self.cardiac_method}"
            + ("  ✓" if self.cardiac_ok else f"  ✗  {self.cardiac_msg}"),
        ]
        if self.imputation:
            lines.append("  imputation   :")
            for line in str(self.imputation).split("\n"):
                lines.append("    " + line)
        lines.append(f"  elapsed      : {self.elapsed_sec:.2f} s")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def preprocess_eeg_window(
    window:             pd.DataFrame,
    channels:           List[str],
    sfreq:              float       = 200.0,
    montage:            MontageSpec = _DEFAULT_MONTAGE,
    impute:             bool        = True,
    k_neighbors:        int         = 6,
    max_interp_gap_sec: float       = _DEFAULT_MAX_INTERP_GAP_SEC,
    rolling_median_sec: float       = _DEFAULT_ROLLING_MEDIAN_SEC,
    cardiac_method:     str         = "none",
    ekg_col:            str         = "EKG",
    epoch_ms:           float       = 300.0,
    robust_template:    bool        = True,
    verbose:            bool        = True,
) -> Tuple[pd.DataFrame, PreprocessReport]:
    """
    Preprocess an in-memory EEG window DataFrame.

    Use this when the window is already loaded (e.g. from hms_pipeline.py).
    For ICA cardiac removal, use preprocess_eeg_from_parquet instead —
    ICA needs the full recording, not just the 50-second window.

    Parameters
    ----------
    window   : pd.DataFrame
        Raw EEG window, shape (T, C).  May contain NaN values.
    channels : list of str
        EEG channel names to process.
    sfreq    : float
        Sampling rate in Hz (default 200).
    montage  : str or dict
        Electrode montage for spatial NaN imputation.
        Default "standard_1020" (19-ch clinical EEG).
        Other options: "biosemi64", "biosemi128", custom dict.
    impute   : bool
        Run NaN imputation (default True).
    k_neighbors : int
        Spatial neighbours for channel-level imputation.
        Recommended: 6 for 10-20, 8 for 64-ch, 12 for 128-ch.
    max_interp_gap_sec : float
        Max gap for linear interpolation (default 0.5 s).
    rolling_median_sec : float
        Rolling-median window for long gaps (default 2.0 s).
    cardiac_method : str
        "none" | "template" | "ica"
        Note: "ica" is not supported here — use preprocess_eeg_from_parquet.
    ekg_col  : str
        EKG column name in window (default "EKG").
    epoch_ms : float
        Half-epoch for template subtraction in ms (default 300).
    robust_template : bool
        Use median template (True) or mean (False).
    verbose  : bool
        Print progress.

    Returns
    -------
    window_clean : pd.DataFrame
    report       : PreprocessReport
    """
    import time
    t0     = time.time()
    report = PreprocessReport(
        window_sec      = len(window) / sfreq,
        n_channels      = len(channels),
        sfreq           = sfreq,
        montage         = str(montage) if isinstance(montage, str) else "custom",
        cardiac_method  = cardiac_method,
    )

    window = window.copy()

    # ── Imputation ───────────────────────────────────────────────────
    if impute:
        window, nan_report = impute_eeg_window(
            window, channels,
            sfreq              = sfreq,
            montage            = montage,
            k_neighbors        = k_neighbors,
            max_interp_gap_sec = max_interp_gap_sec,
            rolling_median_sec = rolling_median_sec,
            verbose            = verbose,
        )
        report.imputation = nan_report

    # ── Cardiac removal ──────────────────────────────────────────────
    method = cardiac_method.lower().strip()
    if method == "template":
        try:
            window = remove_cardiac_template(
                window, channels,
                sfreq    = sfreq,
                ekg_col  = ekg_col,
                epoch_ms = epoch_ms,
                robust   = robust_template,
                verbose  = verbose,
            )
        except Exception as ex:
            report.cardiac_ok  = False
            report.cardiac_msg = str(ex)
            warnings.warn(f"[preprocess] Template removal failed: {ex}")

    elif method == "ica":
        warnings.warn(
            "[preprocess] ICA cardiac removal requires the full recording. "
            "Use preprocess_eeg_from_parquet() with cardiac_method='ica'."
        )

    elif method != "none":
        warnings.warn(
            f"[preprocess] Unknown cardiac_method '{cardiac_method}'. "
            "Choose 'none', 'template', or 'ica'."
        )

    report.elapsed_sec = time.time() - t0
    if verbose:
        print(report)

    return window, report


def preprocess_eeg_from_parquet(
    eeg_path:           str,
    channels:           List[str],
    offset_sec:         float       = 0.0,
    window_sec:         float       = 50.0,
    sfreq:              float       = 200.0,
    montage:            MontageSpec = _DEFAULT_MONTAGE,
    impute:             bool        = True,
    k_neighbors:        int         = 6,
    max_interp_gap_sec: float       = _DEFAULT_MAX_INTERP_GAP_SEC,
    rolling_median_sec: float       = _DEFAULT_ROLLING_MEDIAN_SEC,
    cardiac_method:     str         = "none",
    ekg_col:            str         = "EKG",
    epoch_ms:           float       = 300.0,
    robust_template:    bool        = True,
    n_ica_components:   int         = 15,
    max_fit_sec:        float       = 600.0,
    l_freq:             float       = 1.0,
    h_freq:             float       = 40.0,
    ica_threshold:      float       = 0.3,
    verbose:            bool        = True,
) -> Tuple[pd.DataFrame, PreprocessReport]:
    """
    Load a raw EEG parquet, run imputation and cardiac removal, return window.

    Supports all three cardiac methods including ICA (which reads the full
    recording from disk for a reliable decomposition before slicing the window).

    Parameters
    ----------
    eeg_path    : str   — path to the .parquet file
    channels    : list  — EEG channel names (not including EKG)
    offset_sec  : float — start of the 50-s label window
    window_sec  : float — length of the window to return
    sfreq       : float — sampling rate in Hz
    montage     : str or dict
        Electrode montage for spatial imputation.
        "standard_1020" for HMS 19-ch EEG.
        "biosemi64" / "biosemi128" for higher-density caps.
        Custom dict {ch: np.array([x,y,z])} for any layout.
    impute      : bool  — run NaN imputation (default True)
    k_neighbors : int   — spatial neighbours (6 for 10-20, 8+ for 64/128-ch)
    max_interp_gap_sec : float — max gap for linear interp (default 0.5 s)
    rolling_median_sec : float — rolling-median window (default 2.0 s)
    cardiac_method : str  — "none" | "template" | "ica"
    ekg_col     : str   — EKG column name (default "EKG")
    epoch_ms    : float — template half-epoch in ms (default 300)
    robust_template : bool — median template (True) or mean (False)
    n_ica_components : int  — ICA components (default 15)
    max_fit_sec : float — max seconds for ICA fitting (default 600)
    l_freq      : float — high-pass before ICA (default 1.0 Hz)
    h_freq      : float — low-pass before ICA (default 40.0 Hz)
    ica_threshold : float — min |r(IC, EKG)| to flag cardiac (default 0.3)
    verbose     : bool

    Returns
    -------
    window_clean : pd.DataFrame  — preprocessed window, shape (T, C)
    report       : PreprocessReport
    """
    import time
    t0     = time.time()
    report = PreprocessReport(
        eeg_path       = eeg_path,
        offset_sec     = offset_sec,
        window_sec     = window_sec,
        n_channels     = len(channels),
        sfreq          = sfreq,
        montage        = str(montage) if isinstance(montage, str) else "custom",
        cardiac_method = cardiac_method,
    )

    method = cardiac_method.lower().strip()

    # ── ICA: runs on full recording, returns sliced window ───────────
    if method == "ica":
        try:
            window = remove_cardiac_ica(
                eeg_path     = eeg_path,
                channels     = channels,
                offset_sec   = offset_sec,
                window_sec   = window_sec,
                sfreq        = sfreq,
                ekg_col      = ekg_col,
                n_components = n_ica_components,
                max_fit_sec  = max_fit_sec,
                l_freq       = l_freq,
                h_freq       = h_freq,
                threshold    = ica_threshold,
                verbose      = verbose,
            )
        except Exception as ex:
            report.cardiac_ok  = False
            report.cardiac_msg = str(ex)
            warnings.warn(f"[preprocess] ICA removal failed: {ex}. "
                          "Loading raw window as fallback.")
            window = _load_window(eeg_path, offset_sec, window_sec, sfreq)

    else:
        # Load window first, then optionally apply template removal
        window = _load_window(eeg_path, offset_sec, window_sec, sfreq)

    # ── Imputation (works on the sliced window) ──────────────────────
    if impute:
        window, nan_report = impute_eeg_window(
            window, channels,
            sfreq              = sfreq,
            montage            = montage,
            k_neighbors        = k_neighbors,
            max_interp_gap_sec = max_interp_gap_sec,
            rolling_median_sec = rolling_median_sec,
            verbose            = verbose,
        )
        report.imputation = nan_report

    # ── Template cardiac removal (after imputation) ──────────────────
    if method == "template":
        try:
            window = remove_cardiac_template(
                window, channels,
                sfreq    = sfreq,
                ekg_col  = ekg_col,
                epoch_ms = epoch_ms,
                robust   = robust_template,
                verbose  = verbose,
            )
        except Exception as ex:
            report.cardiac_ok  = False
            report.cardiac_msg = str(ex)
            warnings.warn(f"[preprocess] Template removal failed: {ex}")

    elif method not in ("none", "ica"):
        warnings.warn(
            f"[preprocess] Unknown cardiac_method '{cardiac_method}'. "
            "Choose 'none', 'template', or 'ica'.")

    report.elapsed_sec = time.time() - t0
    if verbose:
        print(report)

    return window, report


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _load_window(
    eeg_path:   str,
    offset_sec: float,
    window_sec: float,
    sfreq:      float,
) -> pd.DataFrame:
    """Load a fixed-length window from a parquet file."""
    df    = pd.read_parquet(eeg_path)
    start = int(offset_sec * sfreq)
    end   = start + int(window_sec * sfreq)
    return df.iloc[start:end].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience wrapper matching the old eeg_nan_sub.py API
# (backward compatibility for any code that imports from eeg_nan_sub)
# ---------------------------------------------------------------------------

def impute_eeg_window_compat(
    window:   pd.DataFrame,
    channels: List[str],
    **kwargs,
) -> Tuple[pd.DataFrame, NanReport]:
    """
    Drop-in replacement for the old eeg_nan_sub.impute_eeg_window().
    Defaults to standard_1020 montage, matching the original behaviour.
    """
    return impute_eeg_window(window, channels, **kwargs)


def remove_cardiac_artifact(
    window:   pd.DataFrame,
    channels: List[str],
    **kwargs,
) -> pd.DataFrame:
    """Drop-in replacement for eeg_nan_sub.remove_cardiac_artifact()."""
    return remove_cardiac_template(window, channels, **kwargs)


def remove_cardiac_artifact_ica(
    eeg_path: str,
    channels: List[str],
    **kwargs,
) -> pd.DataFrame:
    """Drop-in replacement for eeg_nan_sub.remove_cardiac_artifact_ica()."""
    return remove_cardiac_ica(eeg_path, channels, **kwargs)
