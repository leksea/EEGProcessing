"""
eeg_cardiac.py  —  Cardiac artifact removal for EEG
Alexandra Yakovleva, 2026
=====================================================
Two methods, same interface:

Method 1 — QRS template subtraction  (fast, no MNE required)
    Detects QRS peaks from the EKG channel, builds a per-channel average
    artifact template (±epoch_ms around each peak), and subtracts it.
    Equivalent to average artifact subtraction (AAS) — Allen et al. 1998.
    Best for: stable heart rate, consistent QRS morphology.

Method 2 — ICA  (best quality, requires MNE)
    Fits FastICA on the full recording, identifies cardiac ICs by
    correlation with the EKG channel, and reconstructs the signal
    without them.  Best for: variable HR, multiple artifact sources,
    when source decomposition is also needed downstream.

Public API
----------
    remove_cardiac_template(window, channels, sfreq, ekg_col, ...)
        -> pd.DataFrame

    remove_cardiac_ica(eeg_path, channels, offset_sec, window_sec,
                       sfreq, ekg_col, ...)
        -> pd.DataFrame
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ===========================================================================
# Shared helpers
# ===========================================================================

def _detect_qrs_peaks(
    ekg:          np.ndarray,
    sfreq:        float,
    bandpass_lo:  float = 5.0,
    bandpass_hi:  float = 20.0,
    min_rr_sec:   float = 0.35,       # ~170 bpm max
) -> np.ndarray:
    """
    Detect QRS peak sample indices from a raw EKG signal.

    Parameters
    ----------
    ekg         : 1-D array, raw EKG in any unit (µV or a.u.)
    sfreq       : sampling rate in Hz
    bandpass_lo : bandpass lower cutoff before peak detection (Hz)
    bandpass_hi : bandpass upper cutoff before peak detection (Hz)
    min_rr_sec  : minimum R-R interval in seconds (sets minimum peak distance)

    Returns
    -------
    np.ndarray of integer sample indices (QRS peaks)
    """
    from scipy.signal import butter, sosfiltfilt, find_peaks

    nyq = sfreq / 2.0
    lo  = min(bandpass_lo, nyq - 1) / nyq
    hi  = min(bandpass_hi, nyq - 1) / nyq
    if lo >= hi:
        hi = lo + 0.1

    sos       = butter(4, [lo, hi], btype="band", output="sos")
    filtered  = sosfiltfilt(sos, ekg)
    rectified = np.abs(filtered)

    threshold = float(np.percentile(rectified, 60))
    min_dist  = int(min_rr_sec * sfreq)

    peaks, _ = find_peaks(rectified, height=threshold, distance=min_dist)
    return peaks


# ===========================================================================
# Method 1 — QRS template subtraction
# ===========================================================================

def _build_artifact_template(
    sig:        np.ndarray,
    peaks:      np.ndarray,
    epoch_samp: int,
    robust:     bool = True,
) -> np.ndarray:
    """
    Build average QRS artifact template for one EEG channel.

    Parameters
    ----------
    sig        : 1-D EEG signal (same length as the EKG used for peak detection)
    peaks      : QRS peak sample indices
    epoch_samp : half-epoch length in samples  (template = 2*epoch_samp + 1)
    robust     : use median instead of mean (default True — more robust to
                 outlier epochs from movement artifact)

    Returns
    -------
    1-D template array of length 2*epoch_samp + 1
    """
    epochs = []
    n      = len(sig)
    for p in peaks:
        s = p - epoch_samp
        e = p + epoch_samp + 1
        if s < 0 or e > n:
            continue    # skip epochs that extend past the signal boundary
        epochs.append(sig[s:e])

    if not epochs:
        return np.zeros(2 * epoch_samp + 1)

    stack = np.stack(epochs, axis=0)    # (n_epochs, template_len)
    return np.median(stack, axis=0) if robust else np.mean(stack, axis=0)


def remove_cardiac_template(
    window:    pd.DataFrame,
    channels:  List[str],
    sfreq:     float = 200.0,
    ekg_col:   str   = "EKG",
    epoch_ms:  float = 300.0,    # half-epoch: ±300 ms around each QRS
    robust:    bool  = True,
    verbose:   bool  = True,
) -> pd.DataFrame:
    """
    Remove cardiac (QRS) artifact using average artifact subtraction (AAS).

    Parameters
    ----------
    window   : pd.DataFrame  — EEG window (T × channels), must include EKG
    channels : list of str   — EEG channel names to clean (NOT including EKG)
    sfreq    : float         — sampling rate in Hz (default 200)
    ekg_col  : str           — EKG column name in window (default "EKG")
    epoch_ms : float         — half-epoch length in ms (default 300)
    robust   : bool          — use median template (True) or mean (False)
    verbose  : bool          — print detection summary

    Returns
    -------
    pd.DataFrame — copy of window with cardiac artifact subtracted from
                   each channel in `channels`
    """
    window = window.copy()

    if ekg_col not in window.columns:
        warnings.warn(
            f"[cardiac/template] EKG column '{ekg_col}' not found — "
            "skipping cardiac removal.")
        return window

    ekg   = np.nan_to_num(window[ekg_col].values.astype(float))
    peaks = _detect_qrs_peaks(ekg, sfreq)

    if len(peaks) < 3:
        warnings.warn(
            f"[cardiac/template] Only {len(peaks)} QRS peaks detected — "
            "template subtraction skipped. Check EKG signal quality.")
        return window

    epoch_samp = int(epoch_ms / 1000.0 * sfreq)
    n          = len(window)

    if verbose:
        rr       = np.diff(peaks) / sfreq * 1000    # ms
        mean_hr  = 60.0 / (np.mean(rr) / 1000.0)
        print(
            f"[cardiac/template]  {len(peaks)} QRS peaks  "
            f"|  HR = {mean_hr:.0f} bpm  "
            f"|  RR = {np.mean(rr):.0f} ± {np.std(rr):.0f} ms  "
            f"|  half-epoch = {epoch_ms:.0f} ms"
        )

    for ch in channels:
        if ch not in window.columns:
            continue
        sig      = window[ch].values.astype(float)
        template = _build_artifact_template(sig, peaks, epoch_samp, robust)
        cleaned  = sig.copy()
        for p in peaks:
            s = p - epoch_samp
            e = p + epoch_samp + 1
            if s < 0 or e > n:
                continue
            cleaned[s:e] -= template
        window[ch] = cleaned

    if verbose:
        print(
            f"[cardiac/template]  Template subtraction applied "
            f"to {len(channels)} channels.")

    return window


# ===========================================================================
# Method 2 — ICA
# ===========================================================================

def remove_cardiac_ica(
    eeg_path:     str,
    channels:     List[str],
    offset_sec:   float = 0.0,
    window_sec:   float = 50.0,
    sfreq:        float = 200.0,
    ekg_col:      str   = "EKG",
    n_components: int   = 15,
    max_fit_sec:  float = 600.0,    # use up to 10 min for fitting
    l_freq:       float = 1.0,
    h_freq:       float = 40.0,
    threshold:    float = 0.3,      # min |r| with EKG to flag cardiac IC
    verbose:      bool  = True,
) -> pd.DataFrame:
    """
    Remove cardiac artifact using ICA fitted on the full recording.

    ICA is superior to template subtraction when heart rate or QRS morphology
    varies across the recording, or when multiple artifact sources need to be
    separated simultaneously.

    Algorithm
    ---------
    1. Load the full parquet file → MNE RawArray
    2. Bandpass filter l_freq–h_freq Hz (removes drift + muscle artifact
       that can dominate ICA components)
    3. Fit FastICA on up to max_fit_sec seconds
    4. Correlate each IC activation with the EKG channel
    5. Flag ICs where |r| ≥ threshold as cardiac
    6. Reconstruct signal excluding cardiac ICs
    7. Slice the requested window from the cleaned signal

    Parameters
    ----------
    eeg_path     : str   — path to the .parquet EEG file
    channels     : list  — EEG channels to clean (not including EKG)
    offset_sec   : float — start of the label window within the recording
    window_sec   : float — length of the window to return (default 50)
    sfreq        : float — sampling rate in Hz (default 200)
    ekg_col      : str   — EKG column name (default "EKG")
    n_components : int   — ICA components to fit (default 15; ≤ n_channels)
    max_fit_sec  : float — max seconds used for fitting (default 600)
    l_freq       : float — high-pass before ICA (default 1.0 Hz)
    h_freq       : float — low-pass before ICA (default 40.0 Hz)
    threshold    : float — min |r(IC, EKG)| to flag cardiac (default 0.3)
    verbose      : bool

    Returns
    -------
    pd.DataFrame — cleaned 50-second window, same columns as input parquet
    """
    try:
        import mne
        mne.set_log_level("WARNING")
    except ImportError:
        raise ImportError(
            "MNE is required for ICA cardiac removal.\n"
            "Install with:  pip install mne"
        )

    # ── 1. Load full recording ────────────────────────────────────────
    if verbose:
        print(f"[cardiac/ica] Loading: {eeg_path}")
    full  = pd.read_parquet(eeg_path)
    n_fit = min(len(full), int(max_fit_sec * sfreq))

    if verbose:
        print(
            f"[cardiac/ica]  {len(full)/sfreq:.0f} s total  |  "
            f"fitting on first {n_fit/sfreq:.0f} s  |  "
            f"{n_components} components"
        )

    fit_data = full.iloc[:n_fit]
    present  = [ch for ch in channels if ch in fit_data.columns]
    missing  = [ch for ch in channels if ch not in fit_data.columns]
    if missing and verbose:
        print(f"[cardiac/ica]  Channels not in parquet: {missing}")

    # ── 2. Build MNE RawArray (µV → V) ───────────────────────────────
    eeg_matrix = (
        np.nan_to_num(fit_data[present].values.T.astype(float)) * 1e-6
    )
    info = mne.create_info(
        ch_names = present, sfreq = sfreq, ch_types = "eeg")
    raw  = mne.io.RawArray(eeg_matrix, info, verbose=False)

    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False,
                        on_missing="ignore", verbose=False)
    except Exception:
        pass

    # ── 3. Filter before ICA ─────────────────────────────────────────
    if verbose:
        print(f"[cardiac/ica]  Filtering {l_freq}–{h_freq} Hz ...")
    raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir", verbose=False)

    # ── 4. Fit ICA ────────────────────────────────────────────────────
    n_comp = min(n_components, len(present))
    ica    = mne.preprocessing.ICA(
        n_components = n_comp,
        method       = "fastica",
        random_state = 42,
        max_iter     = 1000,
    )
    if verbose:
        print(f"[cardiac/ica]  Fitting ICA ({n_comp} components) ...")
    ica.fit(raw, verbose=False)

    # ── 5. Identify cardiac ICs via EKG correlation ──────────────────
    has_ekg        = ekg_col in full.columns
    cardiac_indices: List[int] = []

    if has_ekg:
        activations = ica.get_sources(raw).get_data()    # (n_comp, T_fit)

        from scipy.signal import butter, sosfiltfilt
        ekg_raw  = np.nan_to_num(
            fit_data[ekg_col].values[:n_fit].astype(float))
        nyq      = sfreq / 2.0
        sos      = butter(4,
                          [5.0 / nyq, min(20.0, nyq - 1) / nyq],
                          btype="band", output="sos")
        ekg_filt = sosfiltfilt(sos, ekg_raw)

        min_len   = min(activations.shape[1], len(ekg_filt))
        ekg_filt  = ekg_filt[:min_len]
        acts      = activations[:, :min_len]

        ekg_z     = (ekg_filt - ekg_filt.mean()) / (ekg_filt.std() + 1e-12)
        corrs     = np.array([
            float(np.mean(ekg_z *
                          ((ic - ic.mean()) / (ic.std() + 1e-12))))
            for ic in acts
        ])
        cardiac_indices = list(np.where(np.abs(corrs) >= threshold)[0])

        if verbose:
            top5 = np.argsort(np.abs(corrs))[::-1][:5]
            print("[cardiac/ica]  IC correlations with EKG (top 5):")
            for idx in top5:
                flag = "  ← CARDIAC" if idx in cardiac_indices else ""
                print(f"    IC{idx:02d}  r = {corrs[idx]:+.3f}{flag}")

        # Fallback: if nothing crosses threshold, take highest |r|
        if not cardiac_indices:
            best = int(np.argmax(np.abs(corrs)))
            cardiac_indices = [best]
            if verbose:
                print(
                    f"[cardiac/ica]  No IC ≥ {threshold} — "
                    f"fallback to highest: IC{best} (r={corrs[best]:+.3f})"
                )
    else:
        if verbose:
            print("[cardiac/ica]  No EKG column — using MNE topography heuristic.")
        try:
            cardiac_indices, _ = ica.find_bads_ecg(
                raw, method="ctps", verbose=False)
        except Exception as ex:
            if verbose:
                print(f"[cardiac/ica]  Heuristic failed: {ex}")
            cardiac_indices = []

    if verbose:
        print(
            f"[cardiac/ica]  Excluding {len(cardiac_indices)} "
            f"cardiac component(s): {cardiac_indices}"
        )

    # ── 6. Reconstruct without cardiac ICs ───────────────────────────
    raw_unfiltered = mne.io.RawArray(
        np.nan_to_num(fit_data[present].values.T.astype(float)) * 1e-6,
        info.copy(), verbose=False,
    )
    ica.exclude   = cardiac_indices
    raw_clean     = ica.apply(raw_unfiltered.copy(), verbose=False)
    cleaned_mat   = raw_clean.get_data() * 1e6      # back to µV  (n_ch, T_fit)

    # ── 7. Slice the requested window ────────────────────────────────
    win_start = int(offset_sec * sfreq)
    win_end   = win_start + int(window_sec * sfreq)

    # If window extends past the fit data, apply to the full recording
    if win_end > n_fit:
        if verbose:
            print("[cardiac/ica]  Window extends past fit range — "
                  "applying ICA to full recording ...")
        full_mat = (
            np.nan_to_num(full[present].values.T.astype(float)) * 1e-6
        )
        raw_full       = mne.io.RawArray(full_mat, info.copy(), verbose=False)
        raw_full_clean = ica.apply(raw_full.copy(), verbose=False)
        cleaned_mat    = raw_full_clean.get_data() * 1e6

    win_start = min(win_start, cleaned_mat.shape[1])
    win_end   = min(win_end,   cleaned_mat.shape[1])

    window_clean = cleaned_mat[:, win_start:win_end].T   # (T_win, n_ch)

    # ── 8. Rebuild DataFrame ─────────────────────────────────────────
    result = full.iloc[win_start:win_end].copy().reset_index(drop=True)
    for i, ch in enumerate(present):
        result[ch] = window_clean[:, i].astype(np.float32)

    if verbose:
        n_sec = (win_end - win_start) / sfreq
        print(
            f"[cardiac/ica]  Done.  Cleaned window: "
            f"{win_start/sfreq:.1f}–{win_end/sfreq:.1f} s  "
            f"({n_sec:.1f} s, {len(result)} samples)"
        )

    return result
