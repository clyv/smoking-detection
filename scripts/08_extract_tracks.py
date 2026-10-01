#!/usr/bin/env python
"""Step 8: turn labelled video clips into per-person feature sequences.

    python scripts/08_extract_tracks.py --clips data/clips/labels.csv

labels.csv columns (see data/clips/labels_template.csv):
  video_path   path to the clip (absolute, or relative to the repo)
  label        smoking / not_smoking   (or 1 / 0)
  start_s      optional start second
  end_s        optional end second
  notes        free text, ignored

One label per clip. Record clips with ONE main person where you can; by
default only that person (the longest, largest track) is kept. Good
not_smoking clips are drinking, eating, phone calls, touching the face.

For every kept track, writes data/tracks/<clip>_t<id>.npz with the (T, F)
feature matrix at ~target_fps, the label and the feature names.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smokedet.features import feature_names, sequence_features  # noqa: E402
from smokedet.taxonomy import load_taxonomy  # noqa: E402
from smokedet.utils import load_yaml, pick_device  # noqa: E402

POSITIVE = {"1", "smoking", "smoke", "yes", "true", "positive"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default=ROOT / "data/clips/labels.csv", type=Path)
    ap.add_argument("--config", default=ROOT / "configs/pipeline.yaml", type=Path)
    ap.add_argument("--taxonomy", default=ROOT / "configs/taxonomy.yaml", type=Path)
    ap.add_argument("--out", default=ROOT / "data/tracks", type=Path)
    ap.add_argument("--all-tracks", action="store_true", help="keep every person, not just the main one")
    ap.add_argument("--min-seconds", type=float, default=2.0, help="ignore tracks shorter than this")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import cv2

    from smokedet.frame_processor import FrameProcessor

    cfg = load_yaml(args.config)
    tax = load_taxonomy(args.taxonomy)
    names = feature_names(tax.classes)
    proc = FrameProcessor(cfg, tax, device=pick_device(args.device))
    args.out.mkdir(parents=True, exist_ok=True)
    clips = pd.read_csv(args.clips)

    summary = []
    for ci, row in clips.iterrows():
        video = Path(str(row["video_path"]))
        video = video if video.is_absolute() else ROOT / video
        label = int(str(row["label"]).strip().lower() in POSITIVE)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"[skip] cannot open {video}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, int(round(fps / cfg.get("target_fps", 10))))
        eff_fps = fps / stride
        start = float(row["start_s"]) if "start_s" in row and pd.notna(row["start_s"]) else 0.0
        end = float(row["end_s"]) if "end_s" in row and pd.notna(row["end_s"]) else None
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        proc.reset_tracker()

        tracks: dict[int, dict] = {}  # id -> {step: features}, areas
        frame_no, step = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = start + frame_no / fps
            if end is not None and t > end:
                break
            if frame_no % stride == 0:
                people, _ = proc.process(frame)
                for p in people:
                    tr = tracks.setdefault(p.track_id, {"steps": {}, "area": []})
                    tr["steps"][step] = p.features
                    tr["area"].append(float((p.box[2] - p.box[0]) * (p.box[3] - p.box[1])))
                step += 1
            frame_no += 1
        cap.release()

        min_steps = int(args.min_seconds * eff_fps)
        candidates = {tid: tr for tid, tr in tracks.items() if len(tr["steps"]) >= min_steps}
        if not candidates:
            print(f"[{video.name}] no track longer than {args.min_seconds}s")
            continue
        if not args.all_tracks:
            main_id = max(candidates, key=lambda k: len(candidates[k]["steps"]) * np.mean(candidates[k]["area"]))
            candidates = {main_id: candidates[main_id]}

        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", video.stem)
        for tid, tr in candidates.items():
            s0, s1 = min(tr["steps"]), max(tr["steps"])
            frames = [tr["steps"].get(s) for s in range(s0, s1 + 1)]
            feats = sequence_features(frames, len(names))
            out = args.out / f"{ci:04d}_{stem}_t{tid}.npz"
            np.savez_compressed(
                out, feats=feats, label=label, fps=eff_fps, video=str(video), track_id=tid,
                feature_names=np.array(names), clip_index=ci,
            )
            summary.append({"file": out.name, "video": video.name, "label": label, "seconds": len(feats) / eff_fps})
        print(f"[{video.name}] label={label}  kept {len(candidates)} track(s)")

    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(args.out / "summary.csv", index=False)
        print(f"\n{len(df)} tracks; {int(df['label'].sum())} smoking, {int((1 - df['label']).sum())} not smoking; "
              f"{df['seconds'].sum() / 60:.1f} minutes in total.")


if __name__ == "__main__":
    main()
