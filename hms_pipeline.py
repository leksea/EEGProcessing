"""
HMS-HBAC EEG Preprocessing Pipeline
Alexandra Yakovleva, 2026
======================================
A configurable, end-to-end preprocessing pipeline that takes raw HMS parquet
files and produces clean, model-ready bipolar EEG arrays.

Stages (in order)
-----------------
  1. Load          — read parquet, slice the 50-s label window
  2. NaN impute    — sample-level interpolation + spatial neighbour fill
                     (uses eeg_imputation.py — montage-agnostic)
  3. Cardiac       — QRS artifact removal (template | ICA | none)
                     (uses eeg_cardiac.py)
  4. Bandpass      — zero-phase Butterworth filter (default 0.5–40 Hz)
  5. Notch         — optional 60 Hz power-line notch
  6. Bipolar ref   — compute 16 double-banana bipolar differences
  7. Clip          — hard clip at ±clip_uv microvolts (default 1024)
  8. Normalise     — per-channel z-score or robust scale (median/IQR)
  9. Resample      — optional downsample to target_sfreq Hz
 10. Export        — numpy array (T x 16) + metadata dict

Each stage can be individually enabled / disabled via PipelineConfig.
The pipeline is designed to be:
  - Importable as a library (process_window, process_recording)
  - Runnable as a CLI over an entire dataset (batch mode)
  - Fully reproducible: config is serialised alongside every output

Dependencies (optional — only needed for the relevant stages)
-------------------------------------------------------------
  eeg_imputation.py   — Stage 2 NaN imputation (montage-agnostic)
  eeg_cardiac.py      — Stage 3 cardiac artifact removal
  MNE                 — required by eeg_imputation (spatial imputation)
                        and eeg_cardiac ICA mode

Usage — single window
---------------------
    from hms_pipeline import PipelineConfig, EEGPipeline

    cfg = PipelineConfig(
        cardiac_method = "ica",
        lowpass        = 40.0,
        normalise      = "zscore",
        montage        = "standard_1020",   # or "biosemi64" for 64-ch
    )
    pipe   = EEGPipeline(cfg)
    result = pipe.process(
        eeg_path   = "/data/train_eegs/1628180742.parquet",
        offset_sec = 213,
    )
    print(result.array.shape)   # (10000, 16)
    print(result.report)

Usage — batch over train.csv
-----------------------------
    python hms_pipeline.py \\
        --train_csv  /data/train.csv \\
        --eeg_dir    /data/train_eegs/ \\
        --out_dir    /data/processed/ \\
        --cardiac    ica \\
        --normalise  zscore \\
        --montage    standard_1020 \\
        --workers    4

Output layout (/data/processed/)
---------------------------------
    {eeg_id}_{label_offset}.npy      — float32 array  (T x 16)
    {eeg_id}_{label_offset}_meta.json — metadata + pipeline config
    pipeline_config.json              — top-level config snapshot
    pipeline_report.csv               — per-window processing summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, lfilter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SFREQ       = 200
WINDOW_SEC  = 50
EEG_CHANNELS = [
    "Fp1","F3","C3","P3","O1",
    "F7","T3","T5",
    "Fz","Cz","Pz",
    "Fp2","F4","C4","P4","O2",
    "F8","T4","T6",
]
BIPOLAR_CHAINS = {
    "LL": [("Fp1","F7"), ("F7","T3"), ("T3","T5"), ("T5","O1")],
    "LP": [("Fp1","F3"), ("F3","C3"), ("C3","P3"), ("P3","O1")],
    "RP": [("Fp2","F4"), ("F4","C4"), ("C4","P4"), ("P4","O2")],
    "RL": [("Fp2","F8"), ("F8","T4"), ("T4","T6"), ("T6","O2")],
}
BIPOLAR_NAMES = [
    f"{a}-{c}"
    for pairs in BIPOLAR_CHAINS.values()
    for a, c in pairs
]   # 16 channel names in chain order


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    All pipeline parameters in one place.  Serialisable to/from JSON.

    Parameters
    ----------
    sfreq           : int   — original sampling rate (200 Hz for HMS)
    window_sec      : int   — EEG window length in seconds (50 for HMS)

    --- Stage 2: NaN imputation ---
    impute          : bool  — enable NaN imputation
    montage         : str   — electrode montage for spatial imputation.
                              Named MNE montage string or "custom".
                              Default "standard_1020" (19-ch clinical EEG).
                              Other options: "biosemi64", "biosemi128", etc.
    k_neighbors     : int   — spatial neighbours for dead-channel fill.
                              Recommended: 6 for 10-20, 8 for 64-ch,
                              12 for 128-ch.
    max_interp_gap  : float — max gap (s) filled by linear interpolation
    rolling_median_sec : float — window (s) for rolling-median fallback

    --- Stage 3: Cardiac artifact ---
    cardiac_method  : str   — "none" | "template" | "ica"
    ica_n_components: int   — ICA components (ica mode only)
    ica_max_fit_sec : float — max seconds used for ICA fitting
    ica_l_freq      : float — ICA pre-filter high-pass Hz
    ica_h_freq      : float — ICA pre-filter low-pass Hz
    ica_threshold   : float — min |r| to flag cardiac IC

    --- Stage 4: Bandpass filter ---
    bandpass        : bool  — enable bandpass
    highpass        : float — high-pass cutoff Hz  (default 0.5)
    lowpass         : float — low-pass  cutoff Hz  (default 40.0)
    filter_order    : int   — Butterworth filter order

    --- Stage 5: Notch filter ---
    notch           : bool  — enable 60 Hz notch
    notch_freq      : float — notch centre Hz (default 60.0)
    notch_q         : float — notch quality factor

    --- Stage 6: Bipolar re-referencing ---
    bipolar         : bool  — compute double-banana bipolar channels
                              (False = keep 19 referential channels)

    --- Stage 7: Clipping ---
    clip            : bool  — enable hard amplitude clipping
    clip_uv         : float — clip threshold in microvolts (default 1024)

    --- Stage 8: Normalisation ---
    normalise       : str   — "none" | "zscore" | "robust"
                              zscore : subtract mean, divide std
                              robust : subtract median, divide IQR

    --- Stage 9: Resampling ---
    resample        : bool  — enable downsampling
    target_sfreq    : int   — target sampling rate Hz (default 100)
    """
    # Core
    sfreq              : int   = SFREQ
    window_sec         : int   = WINDOW_SEC

    # Stage 2
    impute             : bool  = True
    montage            : str   = "standard_1020"   # MNE montage name
    k_neighbors        : int   = 6
    max_interp_gap     : float = 0.5
    rolling_median_sec : float = 2.0

    # Stage 3
    cardiac_method     : str   = "none"    # "none" | "template" | "ica"
    ica_n_components   : int   = 15
    ica_max_fit_sec    : float = 600.0
    ica_l_freq         : float = 1.0
    ica_h_freq         : float = 40.0
    ica_threshold      : float = 0.3

    # Stage 4
    bandpass           : bool  = True
    highpass           : float = 0.5
    lowpass            : float = 40.0
    filter_order       : int   = 4

    # Stage 5
    notch              : bool  = False
    notch_freq         : float = 60.0
    notch_q            : float = 30.0

    # Stage 6
    bipolar            : bool  = True

    # Stage 7
    clip               : bool  = True
    clip_uv            : float = 1024.0

    # Stage 8
    normalise          : str   = "zscore"  # "none" | "zscore" | "robust"

    # Stage 9
    resample           : bool  = False
    target_sfreq       : int   = 100

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "PipelineConfig":
        return cls(**json.loads(s))

    @classmethod
    def from_file(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            return cls.from_json(f.read())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Output of a single pipeline run."""
    array          : np.ndarray          # float32  (T x C)
    channel_names  : List[str]           # 16 bipolar or 19 referential
    sfreq          : float               # effective sfreq after resampling
    eeg_id         : int
    offset_sec     : float
    report         : Dict                # per-stage metadata
    config         : PipelineConfig


# ---------------------------------------------------------------------------
# Internal signal-processing helpers
# ---------------------------------------------------------------------------

def _bandpass(sig: np.ndarray, lo: float, hi: float,
              fs: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    lo  = max(lo, 0.01)
    hi  = min(hi, nyq - 0.5)
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, sig)


def _notch(sig: np.ndarray, freq: float, q: float, fs: float) -> np.ndarray:
    b, a = iirnotch(freq / (fs / 2.0), q)
    return lfilter(b, a, sig)


def _bipolar_matrix(ref_matrix: np.ndarray,
                    ch_names: List[str]) -> np.ndarray:
    """
    Convert referential (T x 19) matrix to bipolar (T x 16).
    ch_names must be in the same order as ref_matrix columns.
    """
    idx = {ch: i for i, ch in enumerate(ch_names)}
    cols = []
    for pairs in BIPOLAR_CHAINS.values():
        for a, c in pairs:
            ai = idx.get(a)
            ci = idx.get(c)
            if ai is None or ci is None:
                cols.append(np.zeros(ref_matrix.shape[0]))
            else:
                cols.append(ref_matrix[:, ai] - ref_matrix[:, ci])
    return np.stack(cols, axis=1)   # (T, 16)


def _zscore(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu  = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0,  keepdims=True) + eps
    return (arr - mu) / std


def _robust_scale(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    med = np.median(arr, axis=0, keepdims=True)
    q75, q25 = np.percentile(arr, [75, 25], axis=0, keepdims=True)
    iqr = (q75 - q25) + eps
    return (arr - med) / iqr


def _resample_array(arr: np.ndarray, orig_fs: float,
                    target_fs: float) -> np.ndarray:
    """Simple decimation resample — assumes target_fs divides orig_fs."""
    factor = int(round(orig_fs / target_fs))
    if factor <= 1:
        return arr
    # Anti-alias low-pass before decimation
    nyq   = orig_fs / 2.0
    cutoff = target_fs / 2.0 * 0.9
    sos   = butter(4, cutoff / nyq, btype="low", output="sos")
    arr_f = sosfiltfilt(sos, arr, axis=0)
    return arr_f[::factor, :]


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class EEGPipeline:
    """
    Stateless (after init) EEG preprocessing pipeline.

    Instantiate once with a PipelineConfig, then call process() for each
    recording window.  Thread-safe for ProcessPoolExecutor batch use.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self._impute_available  = self._check_module("eeg_imputation")
        self._cardiac_available = self._check_module("eeg_cardiac")

    def _check_module(self, name: str) -> bool:
        try:
            __import__(name)
            return True
        except ImportError:
            warnings.warn(
                f"{name}.py not found on sys.path — "
                f"{'NaN imputation' if name == 'eeg_imputation' else 'cardiac removal'} "
                "disabled."
            )
            return False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(
        self,
        eeg_path:   str,
        offset_sec: float,
        eeg_id:     int   = -1,
    ) -> PipelineResult:
        """
        Run the full pipeline on one 50-second window.

        Parameters
        ----------
        eeg_path   : str   — path to the raw .parquet EEG file
        offset_sec : float — eeg_label_offset_seconds from train.csv
        eeg_id     : int   — recording ID (for metadata only)

        Returns
        -------
        PipelineResult
        """
        cfg    = self.cfg
        report = {"eeg_id": eeg_id, "offset_sec": offset_sec, "stages": {}}
        t0     = time.time()

        # ── Stage 1: Load ────────────────────────────────────────────
        window, full_path = self._stage_load(eeg_path, offset_sec)
        report["stages"]["load"] = {
            "shape": list(window.shape),
            "nan_count": int(window[EEG_CHANNELS].isna().sum().sum()),
        }

        # ── Stage 2: NaN imputation ──────────────────────────────────
        if cfg.impute and self._impute_available:
            window, nan_report = self._stage_impute(window)
            report["stages"]["impute"] = {
                "channels_with_nans"  : nan_report.channels_with_nans,
                "channels_all_nan"    : nan_report.channels_all_nan,
                "channels_imputed"    : nan_report.channel_nans_filled,
                "channels_unfillable" : nan_report.channels_unfillable,
            }
        else:
            # Safety: zero-fill any remaining NaNs
            for ch in EEG_CHANNELS:
                if ch in window.columns:
                    window[ch] = window[ch].fillna(0.0)
            report["stages"]["impute"] = {"skipped": True}

        # ── Stage 3: Cardiac artifact removal ────────────────────────
        cardiac = cfg.cardiac_method.lower()
        if cardiac != "none" and self._cardiac_available:
            window, cardiac_meta = self._stage_cardiac(window, full_path,
                                                        offset_sec)
            report["stages"]["cardiac"] = cardiac_meta
        else:
            report["stages"]["cardiac"] = {"method": "none"}

        # ── Extract to numpy matrix (T x 19) ─────────────────────────
        present = [ch for ch in EEG_CHANNELS if ch in window.columns]
        matrix  = np.nan_to_num(
            window[present].values.astype(np.float32))   # (T, ≤19)

        # ── Stage 4: Bandpass ─────────────────────────────────────────
        if cfg.bandpass:
            matrix = self._stage_bandpass(matrix, present)
            report["stages"]["bandpass"] = {
                "lo": cfg.highpass, "hi": cfg.lowpass,
                "order": cfg.filter_order,
            }

        # ── Stage 5: Notch ────────────────────────────────────────────
        if cfg.notch:
            matrix = self._stage_notch(matrix)
            report["stages"]["notch"] = {
                "freq": cfg.notch_freq, "q": cfg.notch_q}

        # ── Stage 6: Bipolar re-reference ─────────────────────────────
        if cfg.bipolar:
            matrix = _bipolar_matrix(matrix, present)
            ch_out = BIPOLAR_NAMES
            report["stages"]["bipolar"] = {"n_channels": 16}
        else:
            ch_out = present
            report["stages"]["bipolar"] = {"skipped": True}

        # ── Stage 7: Clip ─────────────────────────────────────────────
        if cfg.clip:
            n_clipped = int(np.sum(np.abs(matrix) > cfg.clip_uv))
            matrix    = np.clip(matrix, -cfg.clip_uv, cfg.clip_uv)
            report["stages"]["clip"] = {
                "threshold_uv" : cfg.clip_uv,
                "samples_clipped": n_clipped,
                "pct_clipped"  : round(100 * n_clipped / matrix.size, 4),
            }

        # ── Stage 8: Normalise ────────────────────────────────────────
        norm = cfg.normalise.lower()
        if norm == "zscore":
            matrix = _zscore(matrix)
            report["stages"]["normalise"] = {"method": "zscore"}
        elif norm == "robust":
            matrix = _robust_scale(matrix)
            report["stages"]["normalise"] = {"method": "robust"}
        else:
            report["stages"]["normalise"] = {"method": "none"}

        # ── Stage 9: Resample ─────────────────────────────────────────
        out_sfreq = float(cfg.sfreq)
        if cfg.resample and cfg.target_sfreq < cfg.sfreq:
            matrix    = _resample_array(matrix, cfg.sfreq, cfg.target_sfreq)
            out_sfreq = float(cfg.target_sfreq)
            report["stages"]["resample"] = {
                "from_hz": cfg.sfreq, "to_hz": cfg.target_sfreq,
                "new_shape": list(matrix.shape),
            }

        report["elapsed_sec"] = round(time.time() - t0, 3)
        report["output_shape"] = list(matrix.shape)

        return PipelineResult(
            array         = matrix.astype(np.float32),
            channel_names = ch_out,
            sfreq         = out_sfreq,
            eeg_id        = eeg_id,
            offset_sec    = offset_sec,
            report        = report,
            config        = cfg,
        )

    # ------------------------------------------------------------------
    # Per-stage private methods
    # ------------------------------------------------------------------

    def _stage_load(self, eeg_path: str,
                    offset_sec: float) -> Tuple[pd.DataFrame, str]:
        full   = pd.read_parquet(eeg_path)
        start  = int(offset_sec * self.cfg.sfreq)
        end    = start + self.cfg.window_sec * self.cfg.sfreq
        end    = min(end, len(full))
        window = full.iloc[start:end].reset_index(drop=True)
        # Add missing EEG columns as NaN
        for ch in EEG_CHANNELS:
            if ch not in window.columns:
                window[ch] = np.nan
        return window, eeg_path

    def _stage_impute(
        self, window: pd.DataFrame
    ) -> Tuple[pd.DataFrame, object]:
        from eeg_imputation import impute_eeg_window
        cfg = self.cfg
        return impute_eeg_window(
            window,
            EEG_CHANNELS,
            sfreq              = cfg.sfreq,
            montage            = cfg.montage,
            k_neighbors        = cfg.k_neighbors,
            max_interp_gap_sec = cfg.max_interp_gap,
            rolling_median_sec = cfg.rolling_median_sec,
            verbose            = False,
        )

    def _stage_cardiac(
        self, window: pd.DataFrame, eeg_path: str, offset_sec: float
    ) -> Tuple[pd.DataFrame, Dict]:
        cfg    = self.cfg
        method = cfg.cardiac_method.lower()

        if method == "template":
            from eeg_cardiac import remove_cardiac_template
            window = remove_cardiac_template(
                window, EEG_CHANNELS,
                sfreq   = cfg.sfreq,
                verbose = False,
            )
            return window, {"method": "template"}

        elif method == "ica":
            from eeg_cardiac import remove_cardiac_ica
            window = remove_cardiac_ica(
                eeg_path     = eeg_path,
                channels     = EEG_CHANNELS,
                offset_sec   = offset_sec,
                window_sec   = cfg.window_sec,
                sfreq        = cfg.sfreq,
                n_components = cfg.ica_n_components,
                max_fit_sec  = cfg.ica_max_fit_sec,
                l_freq       = cfg.ica_l_freq,
                h_freq       = cfg.ica_h_freq,
                threshold    = cfg.ica_threshold,
                verbose      = False,
            )
            return window, {"method": "ica",
                            "n_components": cfg.ica_n_components}

        return window, {"method": "none"}

    def _stage_bandpass(self, matrix: np.ndarray,
                        channels: List[str]) -> np.ndarray:
        cfg = self.cfg
        out = np.empty_like(matrix)
        for i in range(matrix.shape[1]):
            out[:, i] = _bandpass(
                matrix[:, i], cfg.highpass, cfg.lowpass,
                cfg.sfreq, cfg.filter_order)
        return out

    def _stage_notch(self, matrix: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        out = np.empty_like(matrix)
        for i in range(matrix.shape[1]):
            out[:, i] = _notch(matrix[:, i], cfg.notch_freq,
                               cfg.notch_q, cfg.sfreq)
        return out


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def _process_one(args: tuple) -> dict:
    """Worker function for ProcessPoolExecutor."""
    eeg_path, offset_sec, eeg_id, out_dir, cfg_json = args
    try:
        cfg    = PipelineConfig.from_json(cfg_json)
        pipe   = EEGPipeline(cfg)
        result = pipe.process(eeg_path, offset_sec, eeg_id)

        stem   = f"{eeg_id}_{int(offset_sec)}"
        npy_p  = os.path.join(out_dir, f"{stem}.npy")
        meta_p = os.path.join(out_dir, f"{stem}_meta.json")

        np.save(npy_p, result.array)

        meta = {
            "eeg_id"        : eeg_id,
            "offset_sec"    : offset_sec,
            "output_shape"  : list(result.array.shape),
            "channel_names" : result.channel_names,
            "sfreq"         : result.sfreq,
            "report"        : result.report,
        }
        with open(meta_p, "w") as f:
            json.dump(meta, f, indent=2)

        return {"eeg_id": eeg_id, "offset_sec": offset_sec,
                "status": "ok", "shape": list(result.array.shape),
                "elapsed_sec": result.report["elapsed_sec"]}
    except Exception as ex:
        return {"eeg_id": eeg_id, "offset_sec": offset_sec,
                "status": "error", "error": str(ex)}


def run_batch(
    train_csv : str,
    eeg_dir   : str,
    out_dir   : str,
    config    : PipelineConfig,
    workers   : int   = 1,
    limit     : Optional[int] = None,
    skip_existing : bool = True,
) -> pd.DataFrame:
    """
    Process every (eeg_id, offset) row in train.csv.

    Parameters
    ----------
    train_csv     : path to train.csv
    eeg_dir       : directory with .parquet files
    out_dir       : directory to write .npy + _meta.json files
    config        : PipelineConfig
    workers       : parallel worker processes (default 1)
    limit         : process only the first N rows (for testing)
    skip_existing : skip rows whose .npy already exists

    Returns
    -------
    pd.DataFrame  — one row per window, columns: eeg_id, offset_sec,
                    status, shape, elapsed_sec, error
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save config snapshot
    cfg_path = os.path.join(out_dir, "pipeline_config.json")
    with open(cfg_path, "w") as f:
        f.write(config.to_json())
    print(f"Config saved: {cfg_path}")

    df = pd.read_csv(train_csv)
    # Deduplicate by (eeg_id, eeg_label_offset_seconds)
    df = df.drop_duplicates(subset=["eeg_id", "eeg_label_offset_seconds"])
    if limit:
        df = df.head(limit)

    cfg_json = config.to_json()
    tasks = []
    for _, row in df.iterrows():
        eid    = int(row["eeg_id"])
        offset = float(row["eeg_label_offset_seconds"])
        stem   = f"{eid}_{int(offset)}"
        npy_p  = os.path.join(out_dir, f"{stem}.npy")
        if skip_existing and os.path.exists(npy_p):
            continue
        eeg_path = os.path.join(eeg_dir, f"{eid}.parquet")
        if not os.path.exists(eeg_path):
            continue
        tasks.append((eeg_path, offset, eid, out_dir, cfg_json))

    print(f"Processing {len(tasks)} windows with {workers} worker(s) ...")
    results = []

    if workers == 1:
        for i, task in enumerate(tasks):
            r = _process_one(task)
            results.append(r)
            status = "✓" if r["status"] == "ok" else "✗"
            print(f"  [{i+1}/{len(tasks)}] {status} eeg {r['eeg_id']}  "
                  f"offset {r['offset_sec']}s  "
                  + (f"{r.get('elapsed_sec','?')}s" if r['status']=='ok'
                     else r.get('error','')))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process_one, t): t for t in tasks}
            done = 0
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                done += 1
                status = "✓" if r["status"] == "ok" else "✗"
                print(f"  [{done}/{len(tasks)}] {status} "
                      f"eeg {r['eeg_id']}  offset {r['offset_sec']}s")

    report_df  = pd.DataFrame(results)
    report_path = os.path.join(out_dir, "pipeline_report.csv")
    report_df.to_csv(report_path, index=False)

    ok  = (report_df["status"] == "ok").sum()
    err = (report_df["status"] != "ok").sum()
    print(f"\nDone.  {ok} OK  |  {err} errors  |  report: {report_path}")
    return report_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="HMS-HBAC EEG preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--train_csv",  required=True)
    p.add_argument("--eeg_dir",    required=True)
    p.add_argument("--out_dir",    required=True,
                   help="Output directory for .npy and metadata files")

    # Stage flags
    p.add_argument("--no_impute",   action="store_true",
                   help="Disable NaN imputation")
    p.add_argument("--montage",     default="standard_1020",
                   help="MNE montage for spatial imputation "
                        "(default: standard_1020). "
                        "Other options: biosemi64, biosemi128, etc.")
    p.add_argument("--cardiac",     default="none",
                   choices=["none","template","ica"],
                   help="Cardiac artifact removal method (default: none)")
    p.add_argument("--highpass",    type=float, default=0.5,
                   help="Bandpass high-pass Hz (default 0.5)")
    p.add_argument("--lowpass",     type=float, default=40.0,
                   help="Bandpass low-pass  Hz (default 40)")
    p.add_argument("--notch",       action="store_true",
                   help="Enable 60 Hz notch filter")
    p.add_argument("--no_bipolar",  action="store_true",
                   help="Keep referential channels instead of bipolar")
    p.add_argument("--no_clip",     action="store_true",
                   help="Disable amplitude clipping")
    p.add_argument("--clip_uv",     type=float, default=1024.0,
                   help="Clip threshold in uV (default 1024)")
    p.add_argument("--normalise",   default="zscore",
                   choices=["none","zscore","robust"],
                   help="Normalisation method (default: zscore)")
    p.add_argument("--resample",    action="store_true",
                   help="Downsample to --target_sfreq")
    p.add_argument("--target_sfreq", type=int, default=100,
                   help="Target Hz if --resample (default 100)")

    # Batch control
    p.add_argument("--workers",     type=int, default=1,
                   help="Parallel worker processes (default 1)")
    p.add_argument("--limit",       type=int, default=None,
                   help="Process only first N rows (testing)")
    p.add_argument("--no_skip",     action="store_true",
                   help="Reprocess even if output already exists")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = PipelineConfig(
        impute          = not args.no_impute,
        montage         = args.montage,
        cardiac_method  = args.cardiac,
        highpass        = args.highpass,
        lowpass         = args.lowpass,
        notch           = args.notch,
        bipolar         = not args.no_bipolar,
        clip            = not args.no_clip,
        clip_uv         = args.clip_uv,
        normalise       = args.normalise,
        resample        = args.resample,
        target_sfreq    = args.target_sfreq,
    )

    run_batch(
        train_csv     = args.train_csv,
        eeg_dir       = args.eeg_dir,
        out_dir       = args.out_dir,
        config        = cfg,
        workers       = args.workers,
        limit         = args.limit,
        skip_existing = not args.no_skip,
    )