"""
hms_check_modules.py  —  Check and install required HMS project modules
Alexandra Yakovleva, 2026
=======================================================================
Run this cell first in any HMS notebook session.

Usage
-----
    # In notebook
    exec(open(os.path.join(TMP, "hms_check_modules.py")).read())

    # Standalone
    python hms_check_modules.py
"""

import importlib
import subprocess
import sys
import os

# ---------------------------------------------------------------------------
# Required packages
# { import_name : pip_install_name }
# ---------------------------------------------------------------------------

REQUIRED = {
    # Core data
    "numpy"          : "numpy",
    "pandas"         : "pandas",
    "scipy"          : "scipy",

    # Deep learning
    "torch"          : "torch",
    "torchvision"    : "torchvision",

    # Visualisation
    "matplotlib"     : "matplotlib",
    "IPython"        : "ipython",

    # EEG / signal processing
    "mne"            : "mne",

    # ML utilities
    "sklearn"        : "scikit-learn",

    # Dimensionality reduction
    "umap"           : "umap-learn",

    # Architecture visualisation (optional)
    "torchview"      : "torchview",
    "graphviz"       : "graphviz",
}

# Packages where the import name differs from the pip name
# and where we want a specific version pin
VERSION_PINS = {
    "torch"      : "torch>=2.0",
    "torchvision": "torchvision>=0.15",
    "mne"        : "mne>=1.5",
    "umap"       : "umap-learn>=0.5",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pip_install(pkg: str) -> bool:
    """Run pip install and return True on success."""
    cmd = [sys.executable, "-m", "pip", "install", pkg, "-q"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=120)
        return result.returncode == 0
    except Exception:
        return False


def _try_import(import_name: str) -> bool:
    """Return True if the module can be imported."""
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def _get_version(import_name: str) -> str:
    """Return __version__ string or '?' if unavailable."""
    try:
        mod = importlib.import_module(import_name)
        return getattr(mod, "__version__", "?")
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

def check_and_install(verbose: bool = True) -> dict:
    """
    Check all required packages.  Attempt to install any that are missing.

    Returns
    -------
    dict with keys:
        ok       : list of (import_name, version) — already present
        installed: list of import_name — newly installed
        failed   : list of import_name — could not install
    """
    ok, installed, failed = [], [], []

    if verbose:
        print("HMS project — module check")
        print("=" * 50)

    for import_name, pip_name in REQUIRED.items():
        present = _try_import(import_name)

        if present:
            ver = _get_version(import_name)
            ok.append((import_name, ver))
            if verbose:
                print(f"  ✓  {import_name:<20}  {ver}")
        else:
            if verbose:
                print(f"  ✗  {import_name:<20}  missing — installing {pip_name} ...",
                      end="", flush=True)
            pkg_spec = VERSION_PINS.get(import_name, pip_name)
            success  = _pip_install(pkg_spec)

            if success and _try_import(import_name):
                ver = _get_version(import_name)
                installed.append(import_name)
                if verbose:
                    print(f"  installed ({ver})")
            else:
                failed.append(import_name)
                if verbose:
                    print("  FAILED")

    if verbose:
        print("=" * 50)
        print(f"  Present   : {len(ok)}")
        print(f"  Installed : {len(installed)}")
        if failed:
            print(f"  Failed    : {len(failed)}  → {failed}")
            print()
            print("  For failed packages, try manually:")
            for f in failed:
                pip_name = REQUIRED[f]
                print(f"    pip install {pip_name}")
        else:
            print(f"  Failed    : 0")
        print()

    return {"ok": ok, "installed": installed, "failed": failed}


# ---------------------------------------------------------------------------
# GPU / device check
# ---------------------------------------------------------------------------

def check_device(verbose: bool = True) -> str:
    """Report available compute device and return device string."""
    try:
        import torch
        if torch.cuda.is_available():
            name   = torch.cuda.get_device_name(0)
            mem    = torch.cuda.get_device_properties(0).total_memory
            device = "cuda"
            if verbose:
                print(f"  GPU  ✓  {name}  "
                      f"({mem / 1024**3:.1f} GB VRAM)")
        elif torch.backends.mps.is_available():
            device = "mps"
            if verbose:
                print("  GPU  ✓  Apple MPS (Metal)")
        else:
            device = "cpu"
            if verbose:
                print("  GPU  —  not available, using CPU")
    except ImportError:
        device = "cpu"
        if verbose:
            print("  GPU  —  torch not installed, using CPU")
    return device


# ---------------------------------------------------------------------------
# MPS pool divisibility check
# ---------------------------------------------------------------------------

def check_mps_pool(verbose: bool = True):
    """
    Verify that EEGBranch pool size is MPS-compatible.
    After 2 stride-2 blocks: 5000/2/2 = 1250.  1250 / 5 = 250 ✓
    """
    T = 5000
    after_blocks = T // 2 // 2
    pool_out     = 5
    ok           = (after_blocks % pool_out == 0)
    if verbose:
        status = "✓" if ok else "✗ — will crash on MPS"
        print(f"  MPS pool check  {status}"
              f"  ({after_blocks} / {pool_out} = "
              f"{after_blocks // pool_out})")
    return ok


# ---------------------------------------------------------------------------
# Run on import / exec
# ---------------------------------------------------------------------------

if __name__ == "__main__" or "__file__" not in dir():
    results = check_and_install(verbose=True)

    print("Device check")
    print("=" * 50)
    device = check_device(verbose=True)

    print()
    print("Architecture checks")
    print("=" * 50)
    check_mps_pool(verbose=True)

    print()
    if results["failed"]:
        print("⚠  Some packages could not be installed automatically.")
        print("   Install graphviz via Homebrew (macOS) or apt (Linux):")
        print("     brew install graphviz")
        print("     sudo apt install graphviz")
    else:
        print("✓  All packages ready.")

    # Export device for use in the notebook
    # (when exec'd, this sets `device` in the calling namespace)
