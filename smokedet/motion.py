"""Rule-based hand-to-mouth detector.

Works on day one, before any video is labelled, and later serves as the
baseline the learned temporal model has to beat.

A puff looks like: hand away from the mouth -> hand at the mouth for roughly
0.3-5 s -> hand away again. Phone calls keep the hand up much longer; eating
and drinking repeat the motion much more often. Both are down-weighted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MotionParams:
    near_thr: float = 0.6
    far_thr: float = 1.0
    min_dwell_s: float = 0.3
    max_dwell_s: float = 5.0
    lookback_s: float = 3.0
    busy_window_s: float = 20.0
    busy_max_approaches: int = 4
    busy_penalty: float = 0.5
    smooth: int = 3


def _smooth(d: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(d) < k:
        return d.astype(float)
    pad = np.pad(d.astype(float), (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def find_episodes(d: np.ndarray, fps: float, p: MotionParams) -> list[tuple[int, int | None]]:
    """Hand-at-mouth episodes as (start, end) frame indices; end is None if still ongoing.

    An episode starts when the wrist comes within near_thr of the mouth, having
    been beyond far_thr within the last lookback_s, and ends when it moves
    beyond far_thr again.
    """
    d = _smooth(np.asarray(d, dtype=float), p.smooth)
    lookback = max(1, int(round(p.lookback_s * fps)))
    episodes: list[tuple[int, int | None]] = []
    last_far = -10**9
    start = None
    for i, v in enumerate(d):
        if start is None:
            if v > p.far_thr:
                last_far = i
            elif v < p.near_thr and i - last_far <= lookback:
                start = i
        elif v > p.far_thr:
            episodes.append((start, i))
            start = None
            last_far = i
    if start is not None:
        episodes.append((start, None))
    return episodes


def motion_score(d_hist: np.ndarray, fps: float, window_frames: int, p: MotionParams) -> float:
    """Score in [0, 1] that the last `window_frames` contain a puff-like motion.

    d_hist is the d_min series for this person over the recent buffer (it can
    be longer than the window; the extra history feeds the "busy hands" check).
    """
    d_hist = np.asarray(d_hist, dtype=float)
    n = len(d_hist)
    if n == 0:
        return 0.0
    w_start = max(0, n - window_frames)
    min_dwell = p.min_dwell_s * fps
    max_dwell = p.max_dwell_s * fps

    score = 0.0
    episodes = find_episodes(d_hist, fps, p)
    for s, e in episodes:
        end = n if e is None else e
        if end <= w_start:
            continue  # finished before this window
        dwell = end - s
        if e is not None:
            s_ = 1.0 if min_dwell <= dwell <= max_dwell else 0.2
        else:
            s_ = 0.7 if min_dwell <= dwell <= max_dwell else (0.2 if dwell > max_dwell else 0.3)
        score = max(score, s_)

    busy_frames = int(round(p.busy_window_s * fps))
    recent = [s for s, _ in episodes if s >= n - busy_frames]
    if len(recent) > p.busy_max_approaches:
        score *= p.busy_penalty
    return float(score)
