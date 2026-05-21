"""
hms_submission.py  —  Kaggle submission for HMS-HBAC
Alexandra Yakovleva, 2026
=====================================================
Generates submission.csv in the format required by the competition:
    eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

Each row is a predicted probability distribution (sums to 1.0).

Usage
-----
    # In notebook — after training
    exec(open(os.path.join(TMP, "hms_submission.py")).read())

    # Standalone
    python hms_submission.py \
        --test_csv      /data/test.csv \
        --eeg_dir       /data/test_eegs/ \
        --spec_dir      /data/test_spectrograms/ \
        --processed_dir /data/hms_processed_test/ \
        --ckpt          /data/rca_output/SimpleHMSModel_best.pt \
        --out           submission.csv

Pipeline
--------
1.  Load test.csv  (eeg_id only — no labels)
2.  Preprocess each EEG window through hms_pipeline (bandpass, bipolar,
    clip, z-score, resample)
3.  Load corresponding spectrogram from test_spectrograms/
4.  Run inference with the best checkpoint
5.  Write submission.csv
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Submission format
# ---------------------------------------------------------------------------

LABEL_COLS = [
    "seizure_vote", "lpd_vote", "gpd_vote",
    "lrda_vote",    "grda_vote", "other_vote",
]

# ---------------------------------------------------------------------------
# Test Dataset
# ---------------------------------------------------------------------------

class HMSTestDataset(Dataset):
    """
    Loads test EEG + spectrograms for inference.
    No labels — returns (eeg, spec, eeg_id).

    Expects preprocessed .npy files in processed_dir.
    Falls back to zeros if a file is missing (safe for partial runs).
    """

    EEG_CHANNELS = 16
    EEG_LENGTH   = 5000
    SPEC_CHAINS  = 6
    SPEC_FREQ    = 100
    SPEC_TIME    = 300

    def __init__(
        self,
        test_df:       pd.DataFrame,
        processed_dir: str,
        spec_dir:      str,
    ):
        self.df            = test_df.reset_index(drop=True)
        self.processed_dir = processed_dir
        self.spec_dir      = spec_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        eid    = int(row["eeg_id"])
        offset = int(row.get("eeg_label_offset_seconds", 0))
        stem   = f"{eid}_{offset}"

        # ── EEG ──────────────────────────────────────────────
        npy = os.path.join(self.processed_dir, f"{stem}.npy")
        if os.path.exists(npy):
            eeg = np.load(npy).T.astype(np.float32)   # (16, T)
        else:
            warnings.warn(f"Missing EEG: {npy}")
            eeg = np.zeros(
                (self.EEG_CHANNELS, self.EEG_LENGTH),
                dtype=np.float32)

        # Pad / trim to exact length
        T = eeg.shape[1]
        if T < self.EEG_LENGTH:
            eeg = np.pad(eeg, ((0, 0), (0, self.EEG_LENGTH - T)))
        else:
            eeg = eeg[:, :self.EEG_LENGTH]

        # Append zero EKG channel → (17, T)
        ekg = np.zeros((1, self.EEG_LENGTH), dtype=np.float32)
        eeg = np.concatenate([eeg, ekg], axis=0)

        # ── Spectrogram ───────────────────────────────────────
        sp = os.path.join(self.spec_dir, f"{stem}_spec.npy")
        if not os.path.exists(sp):
            # Try spectrogram_id if available
            if "spectrogram_id" in row:
                sp = os.path.join(
                    self.spec_dir,
                    f"{int(row['spectrogram_id'])}.npy")
        if os.path.exists(sp):
            spec = np.load(sp).astype(np.float32)
            if spec.ndim == 3:
                spec = (spec.transpose(2, 0, 1)
                        if spec.shape[2] == self.SPEC_CHAINS
                        else spec)
        else:
            warnings.warn(f"Missing spec: {sp}")
            spec = np.zeros(
                (self.SPEC_CHAINS, self.SPEC_FREQ, self.SPEC_TIME),
                dtype=np.float32)

        # Pad / trim spec
        spec = self._pad_or_trim(spec)

        return (
            torch.from_numpy(eeg),
            torch.from_numpy(spec),
            eid,
        )

    def _pad_or_trim(self, spec):
        C, F, T = spec.shape
        if F < self.SPEC_FREQ:
            spec = np.pad(spec, ((0,0),(0,self.SPEC_FREQ-F),(0,0)))
        else:
            spec = spec[:, :self.SPEC_FREQ, :]
        if T < self.SPEC_TIME:
            spec = np.pad(spec, ((0,0),(0,0),(0,self.SPEC_TIME-T)))
        else:
            spec = spec[:, :, :self.SPEC_TIME]
        return spec


# ---------------------------------------------------------------------------
# Test-time augmentation (TTA)
# ---------------------------------------------------------------------------

def _tta_augment(eeg: torch.Tensor, spec: torch.Tensor):
    """
    Yield (eeg, spec) for each TTA variant.
    Variants: original, channel sign-flip.
    """
    yield eeg, spec
    yield -eeg, spec


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model,
    loader:    DataLoader,
    device:    str,
    use_tta:   bool = True,
    n_workers: int  = 0,
) -> pd.DataFrame:
    """
    Run model inference on the test DataLoader.

    Returns DataFrame with columns: eeg_id + LABEL_COLS
    """
    model.eval()
    model.to(device)

    all_eeg_ids = []
    all_preds   = []

    for batch_eeg, batch_spec, eeg_ids in loader:
        batch_eeg  = batch_eeg.to(device)
        batch_spec = batch_spec.to(device)

        if use_tta:
            tta_preds = []
            for aug_eeg, aug_spec in _tta_augment(batch_eeg, batch_spec):
                pred = model(aug_eeg, aug_spec).cpu().numpy()
                tta_preds.append(pred)
            pred = np.mean(tta_preds, axis=0)   # average over TTA variants
        else:
            pred = model(batch_eeg, batch_spec).cpu().numpy()

        all_preds.append(pred)
        all_eeg_ids.extend(eeg_ids.numpy()
                           if hasattr(eeg_ids, "numpy")
                           else eeg_ids)

    preds = np.concatenate(all_preds, axis=0)   # (N, 6)

    # Normalise rows to sum to 1 (safety — softmax should already do this)
    preds = preds / preds.sum(axis=1, keepdims=True)

    result = pd.DataFrame(preds, columns=LABEL_COLS)
    result.insert(0, "eeg_id", all_eeg_ids)
    return result


# ---------------------------------------------------------------------------
# Ensemble multiple checkpoints
# ---------------------------------------------------------------------------

def ensemble_predict(
    model_class,
    ckpt_paths:    list,
    loader:        DataLoader,
    device:        str,
    use_tta:       bool = True,
) -> pd.DataFrame:
    """
    Average predictions from multiple checkpoints.
    All models must have the same architecture.
    """
    all_results = []
    for ckpt in ckpt_paths:
        print(f"  Loading: {ckpt}")
        model = model_class()
        model.load_state_dict(
            torch.load(ckpt, map_location=device))
        df = run_inference(model, loader, device, use_tta)
        all_results.append(df[LABEL_COLS].values)

    ensemble = np.mean(all_results, axis=0)
    ensemble = ensemble / ensemble.sum(axis=1, keepdims=True)

    result = df.copy()
    result[LABEL_COLS] = ensemble
    return result


# ---------------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------------

def preprocess_test_eegs(
    test_csv:      str,
    eeg_dir:       str,
    processed_dir: str,
    cardiac_method: str = "none",
    workers:       int  = 1,
):
    """
    Run hms_pipeline on test EEGs and write .npy files to processed_dir.
    Mirrors Cell 2 from the training notebook.
    """
    try:
        from hms_pipeline import PipelineConfig, run_batch
    except ImportError:
        raise ImportError(
            "hms_pipeline.py not found on sys.path. "
            "Copy it to TMP or SCRIPTS_DIR first.")

    os.makedirs(processed_dir, exist_ok=True)
    cfg = PipelineConfig(
        cardiac_method = cardiac_method,
        normalise      = "zscore",
        bandpass       = True,
        highpass       = 0.5,
        lowpass        = 40.0,
        bipolar        = True,
        clip           = True,
        clip_uv        = 1024.0,
        resample       = True,
        target_sfreq   = 100,
    )
    run_batch(
        train_csv     = test_csv,
        eeg_dir       = eeg_dir,
        out_dir       = processed_dir,
        config        = cfg,
        workers       = workers,
        skip_existing = True,
    )
    n = len([f for f in os.listdir(processed_dir)
             if f.endswith(".npy")])
    print(f"Preprocessed {n} test windows → {processed_dir}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def make_submission(
    test_csv:       str,
    eeg_dir:        str,
    spec_dir:       str,
    processed_dir:  str,
    ckpt:           str | list,
    output:         str  = "submission.csv",
    model_class           = None,
    device:         str  = "cpu",
    batch_size:     int  = 16,
    use_tta:        bool = True,
    preprocess:     bool = True,
    cardiac_method: str  = "none",
) -> pd.DataFrame:
    """
    End-to-end: preprocess → infer → write submission.csv

    Parameters
    ----------
    test_csv       : path to test.csv (Kaggle)
    eeg_dir        : path to test_eegs/
    spec_dir       : path to test_spectrograms/
    processed_dir  : where to write / find preprocessed .npy files
    ckpt           : checkpoint path or list of paths (ensemble)
    output         : submission CSV path
    model_class    : model constructor (default: SimpleHMSModel)
    device         : "cpu" | "cuda" | "mps"
    batch_size     : inference batch size
    use_tta        : apply test-time augmentation
    preprocess     : run pipeline on test EEGs before inference
    cardiac_method : "none" | "template" | "ica"
    """
    if model_class is None:
        from hms_model_simple import SimpleHMSModel
        model_class = SimpleHMSModel

    # ── 1. Preprocess ─────────────────────────────────────────
    if preprocess:
        print("Preprocessing test EEGs ...")
        preprocess_test_eegs(
            test_csv, eeg_dir, processed_dir, cardiac_method)

    # ── 2. Build DataLoader ───────────────────────────────────
    test_df = pd.read_csv(test_csv)
    print(f"Test rows: {len(test_df):,}")

    dataset = HMSTestDataset(test_df, processed_dir, spec_dir)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0)
    print(f"Test batches: {len(loader):,}")

    # ── 3. Infer ──────────────────────────────────────────────
    ckpt_list = [ckpt] if isinstance(ckpt, str) else ckpt
    print(f"Running inference  "
          f"({'ensemble ' + str(len(ckpt_list)) + ' ckpts' if len(ckpt_list) > 1 else 'single ckpt'}"
          f"{', TTA' if use_tta else ''}) ...")

    if len(ckpt_list) == 1:
        model = model_class()
        model.load_state_dict(
            torch.load(ckpt_list[0], map_location=device))
        result = run_inference(model, loader, device, use_tta)
    else:
        result = ensemble_predict(
            model_class, ckpt_list, loader, device, use_tta)

    # ── 4. Validate ───────────────────────────────────────────
    row_sums = result[LABEL_COLS].sum(axis=1)
    assert (row_sums - 1.0).abs().max() < 1e-4, \
        "Row probabilities do not sum to 1!"
    print(f"  Predictions: {len(result):,} rows  "
          f"✓ all rows sum to 1.0")

    # ── 5. Write CSV ──────────────────────────────────────────
    result[["eeg_id"] + LABEL_COLS].to_csv(output, index=False)
    print(f"\nSubmission saved: {output}")
    print(result[["eeg_id"] + LABEL_COLS].head(5).to_string(index=False))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate HMS-HBAC Kaggle submission CSV")
    p.add_argument("--test_csv",       required=True)
    p.add_argument("--eeg_dir",        required=True)
    p.add_argument("--spec_dir",       required=True)
    p.add_argument("--processed_dir",  required=True)
    p.add_argument("--ckpt",           required=True, nargs="+",
                   help="One or more checkpoint paths (ensemble if >1)")
    p.add_argument("--out",            default="submission.csv")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--batch_size",     type=int, default=16)
    p.add_argument("--no_tta",         action="store_true")
    p.add_argument("--no_preprocess",  action="store_true")
    p.add_argument("--cardiac_method", default="none",
                   choices=["none","template","ica"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    make_submission(
        test_csv       = args.test_csv,
        eeg_dir        = args.eeg_dir,
        spec_dir       = args.spec_dir,
        processed_dir  = args.processed_dir,
        ckpt           = args.ckpt,
        output         = args.out,
        device         = args.device,
        batch_size     = args.batch_size,
        use_tta        = not args.no_tta,
        preprocess     = not args.no_preprocess,
        cardiac_method = args.cardiac_method,
    )
