"""Small helpers shared by the scripts."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def pick_device(pref: str = "auto") -> str:
    """Device string for Ultralytics: '0' (first GPU), 'mps' or 'cpu'."""
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def torch_device(pref: str = "auto") -> str:
    """Device string for plain PyTorch: 'cuda', 'mps' or 'cpu'."""
    d = pick_device(pref)
    return "cuda" if d.isdigit() else d


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link to save disk space; fall back to a copy (e.g. across drives)."""
    import shutil

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except (OSError, AttributeError, NotImplementedError):
        shutil.copy2(src, dst)
