"""
hms_visualize.py  —  HMS-HBAC unified visualizer
Alexandra Yakovleva, 2026
==================================================
Three plot types in one file, sharing constants and helpers from
hms_plot_common.py.

Plot types
----------
  eeg          — longitudinal bipolar (double banana) EEG trace
  spectrogram  — 10-minute 4-chain spectrogram (quad / stack / composite)
  topomap      — 2-D scalp topographic map

Public API
----------
    plot_hms_eeg(train_csv, eeg_dir, eeg_id, patient_id,
                 t_start, t_end, uv_per_cm, lowpass, highpass,
                 apply_notch, impute, cardiac_method, mark_event,
                 output) -> str

    visualize_spectrogram(train_csv, spec_dir, eeg_id, spectrogram_id,
                          patient_id, mode, colormap, log_power,
                          mark_event, t_start, t_end, freq_max,
                          output) -> str

    plot_hms_topomap(train_csv, eeg_dir, eeg_id, patient_id,
                     t_start, t_end, metric, bipolar,
                     show_title, output) -> str

CLI usage
---------
    python hms_visualize.py eeg \\
        --train_csv /data/train.csv --eeg_dir /data/train_eegs/ \\
        --eeg_id 1628180742 --t_start 0 --t_end 10

    python hms_visualize.py spectrogram \\
        --train_csv /data/train.csv --spec_dir /data/train_spectrograms/ \\
        --eeg_id 1628180742 --mode quad --log_power

    python hms_visualize.py topomap \\
        --train_csv /data/train.csv --eeg_dir /data/train_eegs/ \\
        --eeg_id 1628180742 --t_start 20 --t_end 30 --metric rms
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from hms_plot_common import (
    # Constants
    SFREQ, WINDOW_SEC, EPOCH_START, EPOCH_END,
    EEG_CHANNELS, BIPOLAR_CHAINS, CHAIN_COLORS, CHAIN_FULL, CHAIN_LABELS,
    CHAINS, SPEC_TIME_RES, SPEC_DURATION, N_TIME_ROWS, N_FREQ_BINS, BANDS,
    VOTE_COLORS,
    # I/O
    resolve_eeg_ids, resolve_spec_ids,
    load_eeg_window, load_spectrogram, parse_spectrogram,
    # Signal processing
    bandpass, notch_filter, compute_bipolar,
    # Helpers
    get_channel_metric, build_mne_info, build_label_votes,
)


# ===========================================================================
#  EEG TRACE PLOTTER
# ===========================================================================

def _draw_vote_bar(fig, votes: Dict[str, int]) -> None:
    """Draw annotator vote proportions as a horizontal stacked bar."""
    total  = sum(votes.values()) or 1
    bar_ax = fig.add_axes([0.13, 0.005, 0.75, 0.022])
    bar_ax.set_facecolor("#111111")
    x = 0
    for lbl, count in votes.items():
        w = count / total
        bar_ax.barh(0, w, left=x,
                    color=VOTE_COLORS.get(lbl, "#888"), height=0.8)
        if w > 0.05:
            bar_ax.text(x + w / 2, 0, f"{lbl}\n{count}",
                        ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        x += w
    bar_ax.set_xlim(0, 1)
    bar_ax.set_ylim(-0.5, 0.5)
    bar_ax.axis("off")
    bar_ax.set_title("Annotator votes", color="lightgray",
                     fontsize=7.5, pad=2, loc="left")


def _plot_eeg_core(
    window,
    t_start:     float,
    t_end:       float,
    uv_per_cm:   float   = 10.0,
    lowpass:     float   = 30.0,
    highpass:    float   = 0.5,
    apply_notch: bool    = False,
    mark_event:  bool    = True,
    title:       str     = "",
    filter_info: str     = "",
    votes:       Optional[Dict[str, int]] = None,
    output:      str     = "eeg_trace.png",
) -> str:
    """Core EEG rendering — call after window is loaded and preprocessed."""

    # ── 1. Slice to display window ────────────────────────────────────
    s           = int(t_start * SFREQ)
    e           = int(t_end   * SFREQ)
    window_sl   = window.iloc[s:e].reset_index(drop=True)
    time_axis   = np.linspace(t_start, t_end, len(window_sl))

    # ── 2. Compute bipolar EEG channels ──────────────────────────────
    display_channels = []   # list of (label, chain, signal)
    for chain, pairs in BIPOLAR_CHAINS.items():
        sigs = compute_bipolar(window_sl, pairs, highpass, lowpass, apply_notch)
        for (anode, cathode), sig in zip(pairs, sigs):
            display_channels.append((f"{anode} - {cathode}", chain, sig))

    N_CH = len(display_channels)

    # ── 3. EKG ────────────────────────────────────────────────────────
    has_ekg = "EKG" in window_sl.columns
    if has_ekg:
        ekg_raw = np.nan_to_num(window_sl["EKG"].values.astype(float))
        try:
            ekg_raw = bandpass(ekg_raw, 0.5, min(40.0, SFREQ / 2 - 1), SFREQ)
        except Exception:
            pass
        if apply_notch:
            ekg_raw = notch_filter(ekg_raw)
    else:
        ekg_raw = None

    # ── 4. Figure layout ─────────────────────────────────────────────
    ROW_SPACING = uv_per_cm * 2.5
    CLIP_AMP    = uv_per_cm * 1.2
    DT          = t_end - t_start

    eeg_h = max(5.0, N_CH * 0.52 + 1.5)
    ekg_h = 1.8 if has_ekg else 0.0
    fig   = plt.figure(figsize=(14, eeg_h + ekg_h + 1.5),
                       facecolor="#0e0e0e")

    if has_ekg:
        gs     = GridSpec(2, 1, figure=fig,
                          height_ratios=[eeg_h, ekg_h],
                          hspace=0.10, top=0.95, bottom=0.09,
                          left=0.01, right=0.99)
        ax     = fig.add_subplot(gs[0])
        ax_ekg = fig.add_subplot(gs[1], sharex=ax)
    else:
        gs     = GridSpec(1, 1, figure=fig,
                          top=0.95, bottom=0.07, left=0.01, right=0.99)
        ax     = fig.add_subplot(gs[0])
        ax_ekg = None

    ax.set_facecolor("#0e0e0e")
    if ax_ekg is not None:
        ax_ekg.set_facecolor("#0e0e0e")

    # ── 5. EEG traces ────────────────────────────────────────────────
    y_positions  = [-(i * ROW_SPACING) for i in range(N_CH)]
    chain_rows: Dict[str, List[int]] = {}
    for i, (_, chain, _) in enumerate(display_channels):
        chain_rows.setdefault(chain, []).append(i)

    for i, (lbl, chain, sig) in enumerate(display_channels):
        yc          = y_positions[i]
        color       = CHAIN_COLORS.get(chain, "#aaaaaa")
        sig_clipped = np.clip(sig, -CLIP_AMP, CLIP_AMP)
        ax.axhline(yc, color="#2a2a2a", lw=0.5, zorder=1)
        ax.plot(time_axis, yc + sig_clipped,
                color=color, lw=0.75, alpha=0.92, zorder=3)
        ax.text(t_end + DT * 0.015, yc, lbl,
                va="center", ha="left",
                fontsize=7.5, color=color, fontweight="bold")

    # ── 6. Chain brackets (left) ─────────────────────────────────────
    bracket_x = t_start - DT * 0.07
    serif_w   = DT * 0.008
    for chain, rows in chain_rows.items():
        color = CHAIN_COLORS[chain]
        y_top = y_positions[rows[0]]  + ROW_SPACING * 0.45
        y_bot = y_positions[rows[-1]] - ROW_SPACING * 0.45
        y_mid = (y_top + y_bot) / 2.0
        ax.plot([bracket_x, bracket_x], [y_bot, y_top],
                color=color, lw=2.0, clip_on=False, zorder=5)
        for yt in (y_top, y_bot):
            ax.plot([bracket_x, bracket_x + serif_w], [yt, yt],
                    color=color, lw=2.0, clip_on=False, zorder=5)
        ax.text(bracket_x - DT * 0.012, y_mid,
                f"{chain}\n{CHAIN_FULL[chain]}",
                va="center", ha="right",
                fontsize=8.5, color=color, fontweight="bold",
                linespacing=1.3, clip_on=False)

    # ── 7. Labeled epoch shading ──────────────────────────────────────
    if mark_event:
        ep_lo = max(t_start, EPOCH_START)
        ep_hi = min(t_end,   EPOCH_END)
        if ep_lo < ep_hi:
            for a in ([ax] + ([ax_ekg] if ax_ekg else [])):
                a.axvspan(ep_lo, ep_hi, color="#e84a4a", alpha=0.07, zorder=0)
                a.axvline(ep_lo, color="#e84a4a", lw=1.0, ls="--",
                           alpha=0.6, zorder=2)
                a.axvline(ep_hi, color="#e84a4a", lw=1.0, ls="--",
                           alpha=0.6, zorder=2)
            ax.text((ep_lo + ep_hi) / 2,
                    y_positions[0] + ROW_SPACING * 0.65,
                    "labeled epoch",
                    ha="center", va="bottom",
                    fontsize=7, color="#e84a4a", alpha=0.85)

    # ── 8. Second-tick vertical grid ─────────────────────────────────
    for t_sec in np.arange(np.ceil(t_start), np.floor(t_end) + 1, 1.0):
        ax.axvline(t_sec, color="#222222", lw=0.6, zorder=0)

    # ── 9. EEG amplitude scale bar ────────────────────────────────────
    sb_x = t_end - DT * 0.04
    sb_y = y_positions[-1] - ROW_SPACING * 0.75
    ax.plot([sb_x, sb_x], [sb_y, sb_y + uv_per_cm], color="white", lw=1.5)
    for yt in (sb_y, sb_y + uv_per_cm):
        ax.plot([sb_x - DT*0.005, sb_x + DT*0.005], [yt, yt],
                color="white", lw=1.5)
    ax.text(sb_x + DT*0.012, sb_y + uv_per_cm / 2,
            f"{uv_per_cm:.0f} µV",
            va="center", ha="left", fontsize=8, color="white")

    # ── 10. EEG axes cleanup ──────────────────────────────────────────
    x_lo = bracket_x - DT * 0.18
    x_hi = t_end     + DT * 0.22
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_positions[-1] - ROW_SPACING * 1.4,
                y_positions[0]  + ROW_SPACING * 0.9)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x",
                   labelbottom=(ax_ekg is None),
                   colors="lightgray", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if ax_ekg is None:
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_edgecolor("#444")
        ax.set_xlabel("Time (s)", color="lightgray", fontsize=9)

    # ── 11. EKG panel ────────────────────────────────────────────────
    if ax_ekg is not None and ekg_raw is not None:
        EKG_COLOR = "#ff6b6b"
        EKG_CLIP  = float(np.percentile(np.abs(ekg_raw), 99.5)) * 1.05
        ekg_plot  = np.clip(ekg_raw, -EKG_CLIP, EKG_CLIP)

        ax_ekg.axhline(0, color="#2a2a2a", lw=0.5, zorder=1)
        ax_ekg.plot(time_axis, ekg_plot,
                    color=EKG_COLOR, lw=0.85, alpha=0.95, zorder=3)
        for t_sec in np.arange(np.ceil(t_start), np.floor(t_end) + 1, 1.0):
            ax_ekg.axvline(t_sec, color="#222222", lw=0.6, zorder=0)
        ax_ekg.text(t_end + DT*0.015, 0, "EKG",
                    va="center", ha="left",
                    fontsize=8, color=EKG_COLOR, fontweight="bold")
        ax_ekg.text(bracket_x - DT*0.012, 0, "EKG\nCardiac",
                    va="center", ha="right",
                    fontsize=8.5, color=EKG_COLOR, fontweight="bold",
                    linespacing=1.3, clip_on=False)

        ekg_sb_amp = round(EKG_CLIP * 0.5 / 50) * 50 or 100
        ekg_sb_x   = t_end - DT * 0.04
        ekg_sb_y0  = -EKG_CLIP * 0.55
        ax_ekg.plot([ekg_sb_x, ekg_sb_x],
                    [ekg_sb_y0, ekg_sb_y0 + ekg_sb_amp],
                    color="white", lw=1.5)
        for yt in (ekg_sb_y0, ekg_sb_y0 + ekg_sb_amp):
            ax_ekg.plot([ekg_sb_x - DT*0.005, ekg_sb_x + DT*0.005],
                        [yt, yt], color="white", lw=1.5)
        ax_ekg.text(ekg_sb_x + DT*0.012, ekg_sb_y0 + ekg_sb_amp / 2,
                    f"{ekg_sb_amp:.0f} µV",
                    va="center", ha="left", fontsize=7.5, color="white")
        ax_ekg.axhline(EKG_CLIP * 1.05, color="#444444", lw=1.0,
                       xmin=0, xmax=1, zorder=4)
        ax_ekg.set_xlim(x_lo, x_hi)
        ax_ekg.set_ylim(-EKG_CLIP * 1.2, EKG_CLIP * 1.4)
        ax_ekg.tick_params(axis="y", left=False, labelleft=False)
        ax_ekg.tick_params(axis="x", colors="lightgray", labelsize=8)
        ax_ekg.set_xlabel("Time (s)", color="lightgray", fontsize=9)
        for spine in ax_ekg.spines.values():
            spine.set_visible(False)
        ax_ekg.spines["bottom"].set_visible(True)
        ax_ekg.spines["bottom"].set_edgecolor("#444")

    # ── 12. Vote bar ─────────────────────────────────────────────────
    if votes:
        _draw_vote_bar(fig, votes)

    # ── 13. Title + filter info ───────────────────────────────────────
    fig.text(0.5, 0.992, title,
             ha="center", va="top", color="white",
             fontsize=11, fontweight="bold")
    fig.text(0.5, 0.974, filter_info,
             ha="center", va="top", color="#aaaaaa", fontsize=8)

    # ── 14. Legend ───────────────────────────────────────────────────
    leg_items = [
        mpatches.Patch(color=CHAIN_COLORS[c],
                       label=f"{c}  {CHAIN_FULL[c]}")
        for c in BIPOLAR_CHAINS
    ]
    if has_ekg:
        leg_items.append(mpatches.Patch(color="#ff6b6b", label="EKG  Cardiac"))
    ax.legend(handles=leg_items, loc="upper right",
              framealpha=0.15, edgecolor="#555",
              labelcolor="white", fontsize=8, ncol=2, handlelength=1.0)

    # ── 15. Save ──────────────────────────────────────────────────────
    fig.savefig(output, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {output}")
    return output


def plot_hms_eeg(
    train_csv:      str,
    eeg_dir:        str,
    eeg_id:         Optional[int]   = None,
    patient_id:     Optional[int]   = None,
    t_start:        float           = 0.0,
    t_end:          float           = 10.0,
    uv_per_cm:      float           = 10.0,
    lowpass:        float           = 30.0,
    highpass:       float           = 0.5,
    apply_notch:    bool            = False,
    impute:         bool            = True,
    cardiac_method: str             = "none",
    mark_event:     bool            = True,
    output:         str             = "eeg_trace.png",
) -> str:
    """
    Plot an EEG trace from the HMS dataset.

    Parameters
    ----------
    train_csv      : path to train.csv
    eeg_dir        : directory with .parquet EEG files
    eeg_id         : eeg_id to plot
    patient_id     : alternatively, patient_id (auto-selects first eeg_id)
    t_start        : display start (s within 50-s window, default 0)
    t_end          : display end   (s within 50-s window, default 10)
    uv_per_cm      : sensitivity µV/cm (10 = standard ICU)
    lowpass        : low-pass cutoff Hz  (default 30)
    highpass       : high-pass cutoff Hz (default 0.5)
    apply_notch    : 60 Hz notch filter
    impute         : run NaN imputation (default True)
    cardiac_method : "none" | "template" | "ica"
    mark_event     : shade the central labeled 10-s epoch
    output         : output PNG path
    """
    eeg_id, offset_sec, pid, row = resolve_eeg_ids(
        train_csv, eeg_id, patient_id)
    window = load_eeg_window(eeg_dir, eeg_id, offset_sec)

    # NaN imputation
    if impute:
        try:
            from eeg_preprocessing import preprocess_eeg_window
            window, _ = preprocess_eeg_window(
                window, EEG_CHANNELS, sfreq=SFREQ,
                impute=True, cardiac_method="none", verbose=False)
        except ImportError:
            try:
                from eeg_imputation import impute_eeg_window
                window, _ = impute_eeg_window(
                    window, EEG_CHANNELS, sfreq=SFREQ, verbose=False)
            except ImportError:
                warnings.warn("eeg_imputation.py not found — skipping imputation.")

    # Cardiac removal
    method = cardiac_method.lower().strip()
    if method == "template":
        try:
            from eeg_cardiac import remove_cardiac_template
            window = remove_cardiac_template(
                window, EEG_CHANNELS, sfreq=SFREQ, verbose=False)
        except ImportError:
            warnings.warn("eeg_cardiac.py not found — skipping cardiac removal.")
    elif method == "ica":
        try:
            from eeg_cardiac import remove_cardiac_ica
            eeg_path = os.path.join(eeg_dir, f"{eeg_id}.parquet")
            window   = remove_cardiac_ica(
                eeg_path   = eeg_path,
                channels   = EEG_CHANNELS,
                offset_sec = offset_sec,
                window_sec = WINDOW_SEC,
                sfreq      = SFREQ,
                verbose    = False,
            )
        except ImportError:
            warnings.warn("MNE / eeg_cardiac.py not found — skipping ICA.")

    if t_start < 0 or t_end > WINDOW_SEC or t_start >= t_end:
        raise ValueError(
            f"t_start/t_end must be in [0, {WINDOW_SEC}] with t_start < t_end. "
            f"Got [{t_start}, {t_end}]")

    title = (f"HMS EEG  ·  Longitudinal Bipolar (Double Banana)  ·  "
             f"patient {pid}  |  eeg_id {eeg_id}")
    filter_info = (
        f"Bandpass {highpass}–{lowpass} Hz"
        + ("  ·  60 Hz notch" if apply_notch else "")
        + f"  ·  Sensitivity {uv_per_cm:.0f} µV/cm"
        + (f"  ·  cardiac: {method}" if method != "none" else "")
        + f"  ·  t = {t_start:.1f}–{t_end:.1f} s"
          f"  (within 50-s window, label offset = {offset_sec} s)"
    )

    return _plot_eeg_core(
        window      = window,
        t_start     = t_start,
        t_end       = t_end,
        uv_per_cm   = uv_per_cm,
        lowpass     = lowpass,
        highpass    = highpass,
        apply_notch = apply_notch,
        mark_event  = mark_event,
        title       = title,
        filter_info = filter_info,
        votes       = build_label_votes(row),
        output      = output,
    )


# ===========================================================================
#  SPECTROGRAM VISUALIZER
# ===========================================================================

def _add_band_lines(ax, freq_axis, orientation="horizontal"):
    """Draw dashed lines at EEG band boundaries."""
    for _, (flo, fhi) in BANDS.items():
        for f in (flo, fhi):
            if freq_axis[0] <= f <= freq_axis[-1]:
                if orientation == "horizontal":
                    ax.axhline(f, color="white", lw=0.5, ls="--", alpha=0.4)
                else:
                    ax.axvline(f, color="white", lw=0.5, ls="--", alpha=0.4)


def _add_band_labels(ax, freq_axis):
    """Annotate band names next to the frequency axis."""
    for band, (flo, fhi) in BANDS.items():
        mid = (flo + fhi) / 2
        if freq_axis[0] <= mid <= freq_axis[-1]:
            ax.text(1.01, mid, band,
                    transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=7, color="gray")


def _imshow_spec(ax, data, time_axis, freq_axis, cmap, vmin, vmax):
    """Render a single chain spectrogram as an image."""
    return ax.imshow(
        data.T,
        aspect="auto", origin="lower",
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
    )


def _global_clim(chains_data: Dict) -> Tuple[float, float]:
    """Shared colour limits from the 1st–99th percentile across all chains."""
    all_vals = np.concatenate([v.ravel() for v in chains_data.values()])
    return float(np.nanpercentile(all_vals, 1)), float(np.nanpercentile(all_vals, 99))


def _preprocess_spec(
    chains_data: Dict,
    log_power:   bool,
    t_start:     float,
    t_end:       float,
    freq_max:    float,
    freqs:       np.ndarray,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Log-transform, crop time and frequency."""
    t0        = max(0, int(t_start / SPEC_TIME_RES))
    t1        = min(N_TIME_ROWS, int(np.ceil(t_end / SPEC_TIME_RES)))
    time_axis = np.arange(t0, t1) * SPEC_TIME_RES
    f_mask    = freqs <= freq_max
    freq_axis = freqs[f_mask]
    out = {}
    for chain, arr in chains_data.items():
        cropped = arr[t0:t1, :][:, f_mask]
        out[chain] = np.log1p(np.clip(cropped, 0, None)) if log_power else cropped
    return out, time_axis, freq_axis


def _plot_quad(chains_data, time_axis, freq_axis, cmap, event_offset,
               title, log_power) -> plt.Figure:
    """2×2 grid — one panel per chain."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#0e0e0e",
                             sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.08, wspace=0.06)
    vmin, vmax    = _global_clim(chains_data)
    chain_order   = [["LL", "RL"], ["LP", "RP"]]
    for row_i, row_chains in enumerate(chain_order):
        for col_i, chain in enumerate(row_chains):
            ax = axes[row_i][col_i]
            ax.set_facecolor("#0e0e0e")
            if chain not in chains_data:
                ax.text(0.5, 0.5, f"{chain}\n(no data)",
                        ha="center", va="center", color="gray",
                        transform=ax.transAxes)
                continue
            _imshow_spec(ax, chains_data[chain],
                         time_axis, freq_axis, cmap, vmin, vmax)
            _add_band_lines(ax, freq_axis)
            if event_offset is not None and (
                    time_axis[0] <= event_offset <= time_axis[-1]):
                ax.axvline(event_offset, color="red", lw=1.2,
                           ls="--", alpha=0.8)
            ax.set_title(CHAIN_LABELS[chain],
                         color=CHAIN_COLORS[chain], fontsize=10, pad=4)
            if row_i == 1:
                ax.set_xlabel("Time (s)", color="lightgray", fontsize=9)
            if col_i == 0:
                ax.set_ylabel("Frequency (Hz)", color="lightgray", fontsize=9)
            ax.tick_params(colors="lightgray", labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor("#444")
            _add_band_labels(ax, freq_axis)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("log(1+power)" if log_power else "Power (µV²/Hz)",
                 color="lightgray", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="lightgray", labelcolor="lightgray")
    fig.suptitle(title, color="white", fontsize=11, y=0.98)
    return fig


def _plot_stack(chains_data, time_axis, freq_axis, cmap, event_offset,
                title, log_power) -> plt.Figure:
    """4 chains stacked vertically, shared time axis."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    n    = len([c for c in CHAINS if c in chains_data])
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n),
                             facecolor="#0e0e0e", sharex=True)
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.04)
    vmin, vmax = _global_clim(chains_data)
    ax_iter    = iter(axes)
    for chain in CHAINS:
        if chain not in chains_data:
            continue
        ax = next(ax_iter)
        ax.set_facecolor("#0e0e0e")
        im = _imshow_spec(ax, chains_data[chain],
                          time_axis, freq_axis, cmap, vmin, vmax)
        _add_band_lines(ax, freq_axis)
        if event_offset is not None and (
                time_axis[0] <= event_offset <= time_axis[-1]):
            ax.axvline(event_offset, color="red", lw=1.2, ls="--", alpha=0.8)
        ax.set_ylabel(chain, color=CHAIN_COLORS[chain],
                      fontsize=11, fontweight="bold", rotation=0,
                      labelpad=30, va="center")
        ax.set_yticks([freq_axis[0], 4, 8, 12, freq_axis[-1]])
        ax.tick_params(colors="lightgray", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        _add_band_labels(ax, freq_axis)
        divider = make_axes_locatable(ax)
        cax     = divider.append_axes("right", size="1.5%", pad=0.05)
        cb      = fig.colorbar(im, cax=cax)
        cb.ax.yaxis.set_tick_params(color="lightgray", labelcolor="lightgray",
                                    labelsize=7)
    axes[-1].set_xlabel("Time (s)", color="lightgray", fontsize=9)
    fig.suptitle(title, color="white", fontsize=11, y=1.01)
    return fig


def _plot_composite(chains_data, time_axis, freq_axis, cmap, event_offset,
                    title, log_power) -> plt.Figure:
    """All chains tiled side-by-side — the CNN input format."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    available = [c for c in CHAINS if c in chains_data]
    tile      = np.concatenate([chains_data[c] for c in available], axis=0)
    vmin      = float(np.nanpercentile(tile, 1))
    vmax      = float(np.nanpercentile(tile, 99))
    dt        = time_axis[-1] - time_axis[0]
    total_t   = len(available) * dt

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0e0e0e")
    ax.set_facecolor("#0e0e0e")
    im = ax.imshow(tile.T, aspect="auto", origin="lower",
                   extent=[0, total_t, freq_axis[0], freq_axis[-1]],
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    for i, chain in enumerate(available):
        x_bnd = i * dt
        x_mid = x_bnd + dt / 2
        if i > 0:
            ax.axvline(x_bnd, color="white", lw=1.0, alpha=0.6)
        ax.text(x_mid, freq_axis[-1] * 0.95, chain,
                color=CHAIN_COLORS[chain], fontsize=12,
                fontweight="bold", ha="center", va="top")
        if event_offset is not None:
            ex = x_bnd + event_offset - time_axis[0]
            if 0 <= ex - x_bnd <= dt:
                ax.axvline(ex, color="red", lw=1.0, ls="--", alpha=0.7)

    _add_band_lines(ax, freq_axis, "horizontal")
    _add_band_labels(ax, freq_axis)
    n_tiles    = len(available)
    xtick_vals = np.linspace(0, n_tiles * dt, 7)
    xtick_labs = [f"{v % (dt + 1e-9):.0f}" for v in xtick_vals]
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(xtick_labs, color="lightgray", fontsize=8)
    ax.set_xlabel("Time within chain window (s)", color="lightgray", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", color="lightgray", fontsize=9)
    ax.tick_params(colors="lightgray")

    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="1%", pad=0.05)
    cb      = fig.colorbar(im, cax=cax)
    cb.set_label("log(1+power)" if log_power else "Power (µV²/Hz)",
                 color="lightgray", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="lightgray", labelcolor="lightgray")
    fig.suptitle(title + "\n[composite — LL | LP | RP | RL]",
                 color="white", fontsize=10)
    fig.tight_layout()
    return fig


_SPEC_MODES = {"quad": _plot_quad, "stack": _plot_stack,
               "composite": _plot_composite}


def visualize_spectrogram(
    train_csv:      str,
    spec_dir:       str,
    eeg_id:         Optional[int]   = None,
    spectrogram_id: Optional[int]   = None,
    patient_id:     Optional[int]   = None,
    mode:           str             = "quad",
    colormap:       str             = "viridis",
    log_power:      bool            = True,
    mark_event:     bool            = True,
    t_start:        float           = 0.0,
    t_end:          float           = 600.0,
    freq_max:       float           = 20.0,
    output:         str             = "spectrogram.png",
) -> str:
    """
    Visualize a 10-minute 4-chain spectrogram.

    Parameters
    ----------
    train_csv      : path to train.csv
    spec_dir       : directory with spectrogram .parquet files
    eeg_id         : look up by eeg_id
    spectrogram_id : alternatively, spectrogram_id directly
    patient_id     : alternatively, patient_id (auto-selects first)
    mode           : "quad" | "stack" | "composite"  (default: quad)
    colormap       : any matplotlib colormap  (default: viridis)
    log_power      : apply log(1+x) transform  (default: True)
    mark_event     : draw red dashed line at labeled event offset
    t_start        : crop to start time in seconds (default: 0)
    t_end          : crop to end   time in seconds (default: 600)
    freq_max       : upper frequency limit in Hz   (default: 20)
    output         : output PNG path
    """
    if mode not in _SPEC_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose: quad | stack | composite")

    eeg_id, spectrogram_id, event_offset, pid = resolve_spec_ids(
        train_csv, eeg_id, spectrogram_id, patient_id)
    print(f"patient={pid}  eeg={eeg_id}  spec={spectrogram_id}  "
          f"event@{event_offset:.0f}s")

    df             = load_spectrogram(spec_dir, spectrogram_id)
    chains_data, freqs = parse_spectrogram(df)

    t_end       = min(t_end, SPEC_DURATION)
    chains_data, time_axis, freq_axis = _preprocess_spec(
        chains_data, log_power, t_start, t_end, freq_max, freqs)

    title = (
        f"HMS Spectrogram  |  patient {pid}  |  eeg {eeg_id}  |  spec {spectrogram_id}\n"
        f"t = {t_start:.0f}–{t_end:.0f} s  |  "
        f"{'log(1+power)' if log_power else 'raw power'}  |  mode: {mode}"
    )

    fig = _SPEC_MODES[mode](
        chains_data  = chains_data,
        time_axis    = time_axis,
        freq_axis    = freq_axis,
        cmap         = colormap,
        event_offset = event_offset if mark_event else None,
        title        = title,
        log_power    = log_power,
    )
    fig.savefig(output, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {output}")
    return output


# ===========================================================================
#  TOPOMAP
# ===========================================================================

def _compute_referential_values(
    window:      pd.DataFrame,
    t_start:     float,
    t_end:       float,
    metric:      str,
    impute:      bool = True,
) -> Dict[str, float]:
    """Return {channel: scalar} for referential (raw) electrodes."""
    if impute:
        try:
            from eeg_imputation import impute_eeg_window
            window, _ = impute_eeg_window(
                window, EEG_CHANNELS, sfreq=SFREQ, verbose=False)
        except ImportError:
            warnings.warn("eeg_imputation.py not found — skipping imputation.")
    s = int(t_start * SFREQ)
    e = int(t_end   * SFREQ)
    return {
        ch: get_channel_metric(
            np.nan_to_num(window[ch].values[s:e].astype(float)), metric)
        for ch in EEG_CHANNELS
    }


def _compute_bipolar_values(
    window:  pd.DataFrame,
    t_start: float,
    t_end:   float,
    metric:  str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (positions, values) for the 16 bipolar channels."""
    try:
        import mne
    except ImportError:
        raise ImportError("MNE required for bipolar topomap. pip install mne")
    s   = int(t_start * SFREQ)
    e   = int(t_end   * SFREQ)
    montage  = mne.channels.make_standard_montage("standard_1020")
    pos_dict = {n.lower(): d["r"][:2]
                for n, d in zip(montage.ch_names, montage.dig[3:])}
    bip_pos, bip_val = [], []
    for _, pairs in BIPOLAR_CHAINS.items():
        for anode, cathode in pairs:
            if anode not in window.columns or cathode not in window.columns:
                continue
            sig = np.nan_to_num(
                (window[anode].values[s:e] - window[cathode].values[s:e]
                 ).astype(float))
            a_pos = pos_dict.get(anode.lower())
            c_pos = pos_dict.get(cathode.lower())
            if a_pos is None or c_pos is None:
                continue
            bip_pos.append((np.array(a_pos) + np.array(c_pos)) / 2.0)
            bip_val.append(get_channel_metric(sig, metric))
    return np.array(bip_pos), np.array(bip_val)


def plot_hms_topomap(
    train_csv:  str,
    eeg_dir:    str,
    eeg_id:     Optional[int]  = None,
    patient_id: Optional[int]  = None,
    t_start:    float          = 20.0,
    t_end:      float          = 30.0,
    metric:     str            = "rms",
    bipolar:    bool           = False,
    show_title: bool           = True,
    output:     str            = "topomap.png",
) -> str:
    """
    Generate a 2-D scalp topographic map.

    Parameters
    ----------
    train_csv  : path to train.csv
    eeg_dir    : directory with .parquet EEG files
    eeg_id     : eeg_id to plot
    patient_id : alternatively, patient_id
    t_start    : window start in seconds (default 20)
    t_end      : window end   in seconds (default 30)
    metric     : "rms" | "mean" | "std" | "peak"  (default: rms)
    bipolar    : plot bipolar double-banana channels (default: False)
    show_title : include auto-generated title
    output     : output PNG path
    """
    eeg_id, offset_sec, pid, _ = resolve_eeg_ids(
        train_csv, eeg_id, patient_id)

    eeg_path = os.path.join(eeg_dir, f"{eeg_id}.parquet")
    if not os.path.exists(eeg_path):
        raise FileNotFoundError(f"EEG file not found: {eeg_path}")

    window = load_eeg_window(eeg_dir, eeg_id, offset_sec)

    if t_start < 0 or t_end > WINDOW_SEC or t_start >= t_end:
        raise ValueError(
            f"t_start/t_end must be in [0, {WINDOW_SEC}] with t_start < t_end. "
            f"Got [{t_start}, {t_end}]")

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    cmap    = plt.cm.RdBu_r

    if bipolar:
        positions, values = _compute_bipolar_values(window, t_start, t_end, metric)
        med  = np.median(values)
        half = max(abs(values.max() - med), abs(med - values.min())) * 1.05
        vlim = (med - half, med + half)
        from scipy.interpolate import griddata
        grid_x, grid_y = np.mgrid[-0.12:0.12:300j, -0.12:0.12:300j]
        grid_z = griddata(positions, values, (grid_x, grid_y), method="cubic")
        im = ax.imshow(grid_z.T,
                       extent=[-0.12, 0.12, -0.12, 0.12],
                       origin="lower", cmap=cmap,
                       vmin=vlim[0], vmax=vlim[1], aspect="equal")
        ax.add_patch(plt.Circle((0, 0), 0.105,
                                fill=False, color="black", linewidth=1.5))
        ax.annotate("", xy=(0, 0.115), xytext=(0, 0.105),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
        ax.scatter(positions[:, 0], positions[:, 1],
                   c="black", s=20, zorder=5)
        ax.set_xlim(-0.13, 0.13); ax.set_ylim(-0.13, 0.13); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04,
                     label=f"{metric.upper()} amplitude (µV)")
    else:
        try:
            from mne.viz import plot_topomap
        except ImportError:
            raise ImportError("MNE required for topomap. pip install mne")
        info     = build_mne_info()
        val_dict = _compute_referential_values(window, t_start, t_end, metric)
        values   = np.array([val_dict[ch] for ch in EEG_CHANNELS])
        med      = np.median(values)
        half     = max(abs(values.max() - med), abs(med - values.min())) * 1.05
        vlim     = (med - half, med + half)
        im, _    = plot_topomap(values, info, axes=ax, cmap=cmap, vlim=vlim,
                                show=False, contours=6, sensors=True,
                                names=EEG_CHANNELS)
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04,
                     label=f"{metric.upper()} amplitude (µV)")
        idx_max = int(np.argmax(values))
        idx_min = int(np.argmin(values))
        ax.set_xlabel(
            f"Peak: {EEG_CHANNELS[idx_max]} ({values[idx_max]:.1f} µV)  |  "
            f"Min:  {EEG_CHANNELS[idx_min]} ({values[idx_min]:.1f} µV)",
            fontsize=8)

    if show_title:
        mode = "Bipolar (double banana)" if bipolar else "Referential (10-20)"
        ax.set_title(
            f"EEG Topomap  —  {mode}\n"
            f"Patient {pid}  |  EEG {eeg_id}\n"
            f"t = {t_start}–{t_end} s  |  metric: {metric}",
            fontsize=10, pad=12)

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")
    return output


# ===========================================================================
#  UNIFIED CLI
# ===========================================================================

def _cli_eeg(args):
    plot_hms_eeg(
        train_csv      = args.train_csv,
        eeg_dir        = args.eeg_dir,
        eeg_id         = args.eeg_id,
        patient_id     = args.patient_id,
        t_start        = args.t_start,
        t_end          = args.t_end,
        uv_per_cm      = args.uv_per_cm,
        lowpass        = args.lowpass,
        highpass       = args.highpass,
        apply_notch    = args.notch,
        impute         = not args.no_impute,
        cardiac_method = args.cardiac_method,
        mark_event     = not args.no_mark_event,
        output         = args.output,
    )


def _cli_spectrogram(args):
    visualize_spectrogram(
        train_csv      = args.train_csv,
        spec_dir       = args.spec_dir,
        eeg_id         = args.eeg_id,
        spectrogram_id = args.spectrogram_id,
        patient_id     = args.patient_id,
        mode           = args.mode,
        colormap       = args.colormap,
        log_power      = args.log_power,
        mark_event     = args.mark_event,
        t_start        = args.t_start,
        t_end          = args.t_end,
        freq_max       = args.freq_max,
        output         = args.output,
    )


def _cli_topomap(args):
    plot_hms_topomap(
        train_csv  = args.train_csv,
        eeg_dir    = args.eeg_dir,
        eeg_id     = args.eeg_id,
        patient_id = args.patient_id,
        t_start    = args.t_start,
        t_end      = args.t_end,
        metric     = args.metric,
        bipolar    = args.bipolar,
        show_title = not args.no_title,
        output     = args.output,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hms_visualize",
        description="HMS-HBAC visualization toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── shared args helper ────────────────────────────────────────────
    def add_common(sp, need_eeg=True, need_spec=False):
        sp.add_argument("--train_csv",   required=True)
        if need_eeg:
            sp.add_argument("--eeg_dir", required=True)
        if need_spec:
            sp.add_argument("--spec_dir", required=True)
        sp.add_argument("--eeg_id",      type=int, default=None)
        sp.add_argument("--patient_id",  type=int, default=None)
        sp.add_argument("--output",      default=None)

    # ── eeg ──────────────────────────────────────────────────────────
    sp_eeg = sub.add_parser("eeg", help="EEG trace (double banana)")
    add_common(sp_eeg)
    sp_eeg.add_argument("--t_start",       type=float, default=0.0)
    sp_eeg.add_argument("--t_end",         type=float, default=10.0)
    sp_eeg.add_argument("--uv_per_cm",     type=float, default=10.0)
    sp_eeg.add_argument("--lowpass",       type=float, default=30.0)
    sp_eeg.add_argument("--highpass",      type=float, default=0.5)
    sp_eeg.add_argument("--notch",         action="store_true")
    sp_eeg.add_argument("--no_impute",     action="store_true")
    sp_eeg.add_argument("--cardiac_method", default="none",
                        choices=["none","template","ica"])
    sp_eeg.add_argument("--no_mark_event", action="store_true")
    sp_eeg.set_defaults(output="eeg_trace.png")
    sp_eeg.set_defaults(func=_cli_eeg)

    # ── spectrogram ───────────────────────────────────────────────────
    sp_spec = sub.add_parser("spectrogram",
                             help="10-min 4-chain spectrogram")
    add_common(sp_spec, need_eeg=False, need_spec=True)
    sp_spec.add_argument("--spectrogram_id", type=int, default=None)
    sp_spec.add_argument("--mode",     default="quad",
                         choices=["quad","stack","composite"])
    sp_spec.add_argument("--colormap", default="viridis")
    sp_spec.add_argument("--log_power",    action="store_true", default=True)
    sp_spec.add_argument("--no_log_power", action="store_false", dest="log_power")
    sp_spec.add_argument("--mark_event",    action="store_true", default=True)
    sp_spec.add_argument("--no_mark_event", action="store_false", dest="mark_event")
    sp_spec.add_argument("--t_start",  type=float, default=0.0)
    sp_spec.add_argument("--t_end",    type=float, default=600.0)
    sp_spec.add_argument("--freq_max", type=float, default=20.0)
    sp_spec.set_defaults(output="spectrogram.png")
    sp_spec.set_defaults(func=_cli_spectrogram)

    # ── topomap ───────────────────────────────────────────────────────
    sp_topo = sub.add_parser("topomap", help="2-D scalp topomap")
    add_common(sp_topo)
    sp_topo.add_argument("--t_start",  type=float, default=20.0)
    sp_topo.add_argument("--t_end",    type=float, default=30.0)
    sp_topo.add_argument("--metric",   default="rms",
                         choices=["rms","mean","std","peak"])
    sp_topo.add_argument("--bipolar",  action="store_true")
    sp_topo.add_argument("--no_title", action="store_true")
    sp_topo.set_defaults(output="topomap.png")
    sp_topo.set_defaults(func=_cli_topomap)

    return p


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()
    args.func(args)
