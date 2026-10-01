#!/usr/bin/env python
"""Step 9: train the stage-3 temporal classifier on the tracks from step 8.

    python scripts/09_train_temporal.py

- Slides a window (pipeline.yaml window_s, default 6 s) over each track.
- Splits train/val by VIDEO, so windows of one clip never sit on both sides.
- Also scores the rule-based motion heuristic on the same val windows, so you
  can see whether the learned model actually beats it.
- Saves models/temporal.pt and models/temporal_metrics.json.
Set temporal_checkpoint: models/temporal.pt in configs/pipeline.yaml to use it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from smokedet.motion import MotionParams, motion_score  # noqa: E402
from smokedet.utils import load_yaml, torch_device  # noqa: E402


def load_tracks(folder: Path):
    tracks = []
    for f in sorted(folder.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        tracks.append({
            "feats": z["feats"].astype(np.float32), "label": int(z["label"]), "video": str(z["video"]),
            "fps": float(z["fps"]), "names": [str(n) for n in z["feature_names"]],
        })
    return tracks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", default=ROOT / "data/tracks", type=Path)
    ap.add_argument("--config", default=ROOT / "configs/pipeline.yaml", type=Path)
    ap.add_argument("--out", default=ROOT / "models", type=Path)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--hop-s", type=float, default=1.0, help="window stride for training data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from torch import nn

    from smokedet.temporal import TemporalCNN, make_windows

    torch.manual_seed(args.seed)
    cfg = load_yaml(args.config)
    tracks = load_tracks(args.tracks)
    if len(tracks) < 4:
        sys.exit(f"Only {len(tracks)} tracks in {args.tracks}. Label more clips and run step 8 first.")
    names = tracks[0]["names"]
    fps = float(np.median([t["fps"] for t in tracks]))
    window = int(round(cfg.get("window_s", 6) * fps))
    hop = max(1, int(round(args.hop_s * fps)))
    d_idx = names.index("d_min")

    videos = np.array([t["video"] for t in tracks])
    labels = np.array([t["label"] for t in tracks])
    if len(set(labels)) < 2:
        sys.exit("Need both smoking and not_smoking clips.")
    # stratified by label, grouped by video: both classes land in val, and no video is split
    n_splits = max(2, int(round(1 / args.val_frac)))
    n_splits = min(n_splits, len(set(videos[labels == 0])), len(set(videos[labels == 1])))
    if n_splits < 2:
        sys.exit("Need clips from at least 2 different videos per class.")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    tr_idx, va_idx = next(sgkf.split(np.zeros(len(tracks)), labels, groups=videos))

    def windows_for(idx):
        X, y, raw_d = [], [], []
        for i in idx:
            w = make_windows(tracks[i]["feats"], window, hop)
            X.append(w)
            y.append(np.full(len(w), tracks[i]["label"], dtype=np.float32))
            raw_d.append(w[:, :, d_idx])
        return np.concatenate(X), np.concatenate(y), np.concatenate(raw_d)

    Xtr, ytr, _ = windows_for(tr_idx)
    Xva, yva, dva = windows_for(va_idx)
    if len(set(yva)) < 2:
        print("WARNING: the validation split has only one class; AUROC will be undefined. Add more clips.")

    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    # floor the std: a feature that never varies in training (e.g. no cigars in the clips)
    # would otherwise be divided by ~0 and explode the first time it does vary
    std = np.maximum(Xtr.reshape(-1, Xtr.shape[-1]).std(0), 0.05)
    norm = lambda X: np.clip((X - mean) / std, -10, 10).astype(np.float32)  # noqa: E731

    device = torch_device(args.device)
    model = TemporalCNN(len(names)).to(device)
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Xtr_t, ytr_t = torch.from_numpy(norm(Xtr)), torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(norm(Xva)).to(device)

    def evaluate() -> tuple[float, float, np.ndarray]:
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xva_t)).cpu().numpy()
        auc = roc_auc_score(yva, p) if len(set(yva)) > 1 else float("nan")
        return auc, f1_score(yva, p >= 0.5, zero_division=0), p

    best_auc, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        total = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s : s + args.batch]
            xb = Xtr_t[b].to(device)
            xb = xb + 0.02 * torch.randn_like(xb)  # light noise augmentation
            loss = loss_fn(model(xb), ytr_t[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
        sched.step()
        auc, f1, _ = evaluate()
        score = auc if np.isfinite(auc) else f1
        if score > best_auc:
            best_auc, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  loss {total / len(perm):.4f}  val AUROC {auc:.3f}  F1 {f1:.3f}")

    model.load_state_dict(best_state)
    auc, f1, p = evaluate()

    mp = MotionParams(**cfg.get("motion", {}))
    heur = np.array([motion_score(d, fps, window, mp) for d in dva])
    heur_auc = roc_auc_score(yva, heur) if len(set(yva)) > 1 else float("nan")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": best_state, "feature_names": names, "window": window, "fps": fps,
         "mean": mean, "std": std, "model_kwargs": {}},
        args.out / "temporal.pt",
    )
    metrics = {
        "val_windows": int(len(yva)), "train_windows": int(len(ytr)),
        "val_tracks": int(len(va_idx)), "train_tracks": int(len(tr_idx)),
        "temporal_auroc": auc, "temporal_f1@0.5": f1,
        "heuristic_auroc": heur_auc,
        "note": "Window labels are clip labels, so some 'smoking' windows contain no puff. "
                "Judge the final system with event-level metrics on whole videos (step 10).",
    }
    (args.out / "temporal_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nLearned model: val AUROC {auc:.3f}, F1 {f1:.3f}   |   motion heuristic: AUROC {heur_auc:.3f}")
    print(f"Saved {args.out / 'temporal.pt'}")


if __name__ == "__main__":
    main()
