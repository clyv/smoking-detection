"""Tests for the parts that run without a GPU, Ultralytics or PyTorch.

    pytest -q
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smokedet.features import BASE_FEATURES, FeatureParams, feature_names, frame_features, sequence_features  # noqa: E402
from smokedet.fusion import EventAggregator, FusionParams, fuse, window_object_score  # noqa: E402
from smokedet.hashing import UnionFind, near_duplicate_pairs, phash_pair, roboflow_stem  # noqa: E402
from smokedet.metrics import average_precision, evaluate_detections, nms_per_class  # noqa: E402
from smokedet.motion import MotionParams, find_episodes, motion_score  # noqa: E402
from smokedet.splits import group_split  # noqa: E402
from smokedet.taxonomy import Taxonomy, build_id_map, norm  # noqa: E402
from smokedet.yolo_io import label_path_for, read_labels, write_labels  # noqa: E402

TAX = Taxonomy(classes=["cigarette", "vape", "cigar", "pipe", "smoke"], smoking_objects=["cigarette", "vape", "cigar", "pipe"],
               prompts={"vape": ["vape", "e-cigarette"]})


# --- taxonomy / io ----------------------------------------------------------------

def test_build_id_map_and_unmapped():
    cmap = {norm("Cigarette"): "cigarette", norm("Person"): None}
    mapping, unmapped = build_id_map(["Cigarette", "Person", "weird"], cmap, TAX)
    assert mapping == {0: 0, 1: None, 2: None}
    assert unmapped == ["weird"]
    with pytest.raises(ValueError):
        build_id_map(["x"], {"x": "lighter"}, TAX)


def test_flat_prompts_synonyms():
    prompts, to_class = TAX.flat_prompts()
    assert prompts[:3] == ["cigarette", "vape", "e-cigarette"]
    assert to_class[1] == to_class[2] == 1


def test_read_labels_polygon_and_roundtrip(tmp_path):
    p = tmp_path / "labels" / "a.txt"
    p.parent.mkdir()
    p.write_text("0 0.5 0.5 0.2 0.1\n1 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n")
    boxes = read_labels(p)
    assert boxes[0] == (0, 0.5, 0.5, 0.2, 0.1)
    c, cx, cy, w, h = boxes[1]
    assert c == 1 and cx == pytest.approx(0.2) and w == pytest.approx(0.2) and h == pytest.approx(0.3)
    write_labels(p, boxes)
    assert len(read_labels(p)) == 2


def test_label_path_for():
    assert label_path_for(Path("/d/merged/images/x.jpg")) == Path("/d/merged/labels/x.txt")


# --- hashing / dedupe ----------------------------------------------------------------

def _img(path: Path, seed: int, flip: bool = False) -> Path:
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (200, 160), tuple(int(v) for v in rng.integers(0, 255, 3)))
    d = ImageDraw.Draw(im)
    for _ in range(12):
        x, y = rng.integers(0, 180, 2)
        d.rectangle([int(x), int(y), int(x) + 30, int(y) + 20], fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im.save(path)
    return path


def test_phash_catches_mirrored_copy(tmp_path):
    a = phash_pair(_img(tmp_path / "a.jpg", 1))
    b = phash_pair(_img(tmp_path / "b.jpg", 1, flip=True))
    c = phash_pair(_img(tmp_path / "c.jpg", 99))
    h = np.array([a[0], b[0], c[0]], dtype=np.uint64)
    m = np.array([a[1], b[1], c[1]], dtype=np.uint64)
    pairs = near_duplicate_pairs(h, m, threshold=6)
    assert (0, 1) in pairs
    assert all(2 not in p for p in pairs)


def test_near_duplicate_pairs_chunking():
    far = 0xFFFF0000FFFF0000  # 32 bits away from 0
    h = np.array([0, 1, 3, far, far + 7], dtype=np.uint64)  # 0~1~3 within 2 bits; far pair within 3
    pairs = near_duplicate_pairs(h, h, threshold=3, chunk=2)
    assert set(pairs) == {(0, 1), (0, 2), (1, 2), (3, 4)}


def test_union_find_and_stem():
    uf = UnionFind(4)
    uf.union(0, 2)
    uf.union(2, 3)
    assert uf.find(3) == uf.find(0) != uf.find(1)
    assert roboflow_stem("IMG_42_jpg.rf.abc123.jpg") == roboflow_stem("IMG_42_jpg.rf.zzz999.jpg") == "IMG_42_jpg"
    assert roboflow_stem("plain.png") == "plain"


def test_group_split_keeps_groups_together():
    groups = [f"g{i // 3}" for i in range(300)]
    split = group_split(groups, 0.1, seed=1)
    for g in set(groups):
        assert len({s for s, gg in zip(split, groups) if gg == g}) == 1
    assert 0.08 <= split.count("val") / len(split) <= 0.15


# --- metrics ----------------------------------------------------------------------

def test_average_precision_known_values():
    assert average_precision(np.array([1, 1]), np.array([0.9, 0.8]), 2) == pytest.approx(1.0)
    assert average_precision(np.array([1, 0]), np.array([0.9, 0.8]), 2) == pytest.approx(0.5)
    assert average_precision(np.array([0, 1]), np.array([0.9, 0.8]), 1) == pytest.approx(0.5)
    assert np.isnan(average_precision(np.array([]), np.array([]), 0))


def test_evaluate_detections_image_level_and_size():
    gts = {
        "a": {"w": 100, "h": 100, "boxes": np.array([[0, 10, 10, 20, 20]])},       # small cigarette
        "b": {"w": 100, "h": 100, "boxes": np.zeros((0, 5))},                       # negative
        "c": {"w": 400, "h": 400, "boxes": np.array([[1, 0, 0, 200, 200]])},        # large vape
    }
    preds = {
        "a": np.array([[0, 0.9, 10, 10, 20, 20]]),
        "b": np.array([[0, 0.8, 50, 50, 60, 60]]),                                  # false alarm
        "c": np.zeros((0, 6)),                                                     # missed
    }
    r = evaluate_detections(gts, preds, TAX.classes, TAX.smoking_ids, conf_thr=0.25)
    assert r["AP50"]["cigarette"] == pytest.approx(1.0)   # the false alarm ranks below the hit
    assert r["AP50"]["vape"] == 0.0
    il = r["image_level"]
    assert (il["tp"], il["fp"], il["fn"], il["tn"]) == (1, 1, 1, 0)
    assert r["recall_by_size"]["small"] == 1.0 and r["recall_by_size"]["large"] == 0.0


def test_nms_merges_synonyms():
    p = np.array([[1, 0.9, 0, 0, 10, 10], [1, 0.7, 0, 0, 10, 11], [0, 0.6, 0, 0, 10, 10]])
    out = nms_per_class(p)
    assert len(out) == 2


# --- features ---------------------------------------------------------------------

def _pose(wrist_xy):
    kp = np.zeros((17, 2))
    kp[0] = [100, 50]                 # nose
    kp[1], kp[2] = [95, 45], [105, 45]  # eyes
    kp[5], kp[6] = [80, 90], [120, 90]  # shoulders: width 40
    kp[7], kp[8] = [75, 120], [125, 120]
    kp[9] = wrist_xy                  # left wrist
    kp[10] = [125, 150]               # right wrist low
    return kp, np.ones(17)


def test_frame_features_hand_at_mouth_with_cigarette():
    kp, conf = _pose([100, 56])
    objs = [("cigarette", 0.8, np.array([98, 54, 106, 58])), ("vape", 0.9, np.array([0, 150, 10, 160]))]
    f = frame_features(kp, conf, np.array([60, 30, 140, 250]), objs, TAX.classes, FeatureParams())
    names = feature_names(TAX.classes)
    assert f[names.index("d_left")] < 0.3
    assert f[names.index("obj_cigarette")] == pytest.approx(0.8)
    assert f[names.index("obj_vape")] == 0.0          # far from face and hands


def test_frame_features_hand_down():
    kp, conf = _pose([75, 160])
    f = frame_features(kp, conf, np.array([60, 30, 140, 250]), [], TAX.classes, FeatureParams())
    assert f[feature_names(TAX.classes).index("d_min")] > 1.5


def test_sequence_features_fills_gaps():
    n = len(feature_names(TAX.classes))
    kp, conf = _pose([100, 56])
    f = frame_features(kp, conf, np.array([60, 30, 140, 250]), [], TAX.classes, FeatureParams())
    seq = sequence_features([None, f, None, f], n)
    present = BASE_FEATURES.index("present")
    assert seq.shape == (4, n)
    assert list(seq[:, present]) == [0, 1, 0, 1]
    assert np.isfinite(seq).all()


# --- motion heuristic ---------------------------------------------------------------

FPS = 10.0


def _series(segments):
    return np.concatenate([np.full(int(s * FPS), v, dtype=float) for v, s in segments])


def test_single_puff_scores_high():
    d = _series([(2.0, 3), (0.3, 1.5), (2.0, 1.5)])     # away, 1.5 s at mouth, away
    assert len(find_episodes(d, FPS, MotionParams())) == 1
    assert motion_score(d, FPS, int(6 * FPS), MotionParams()) == 1.0


def test_phone_call_scores_low():
    d = _series([(2.0, 2), (0.4, 15), (2.0, 1)])        # 15 s at the face
    assert motion_score(d, FPS, int(6 * FPS), MotionParams()) <= 0.2


def test_no_approach_scores_zero():
    d = _series([(2.0, 10)])
    assert motion_score(d, FPS, int(6 * FPS), MotionParams()) == 0.0


def test_eating_rhythm_is_penalised():
    bite = [(2.0, 1.5), (0.3, 0.6)]
    d = _series(bite * 10 + [(2.0, 1)])                  # 10 quick approaches in ~21 s
    assert motion_score(d, FPS, int(6 * FPS), MotionParams()) <= 0.5


# --- fusion -------------------------------------------------------------------------

def test_fusion_rules():
    p = FusionParams()
    assert fuse(0.5, 0.6, p)[0] and fuse(0.5, 0.6, p)[2] == "object+motion"
    assert fuse(0.8, 0.0, p)[2] == "object"
    assert not fuse(0.5, 0.0, p)[0]
    assert not fuse(0.0, 0.9, p)[0]                      # motion alone needs 0.95
    assert fuse(0.0, 0.97, p)[2] == "motion"


def test_window_object_score_ignores_single_spike():
    m = np.zeros((60, 4))
    m[10, 0] = 0.95
    assert window_object_score(m, 80) == 0.0
    m[:30, 0] = 0.6
    assert window_object_score(m, 80) == pytest.approx(0.6)


def test_event_aggregator_hysteresis():
    agg = EventAggregator(n_on=4, k_on=2, m_off=3)
    seq = [False, True, True, True, False, False, False, False]
    states = [agg.update(1, float(t), d, 0.7 if d else 0.1, 0.6, 0.6, "object+motion" if d else "") for t, d in enumerate(seq)]
    assert states == [False, False, True, True, True, True, False, False]
    events = agg.finish(10.0)
    assert len(events) == 1 and events[0]["start_s"] == 2.0 and events[0]["end_s"] == 6.0


# --- steps 2-4 end to end on a fake Roboflow download -----------------------------------

def _fake_source(raw: Path, name: str, names: list[str], items: list[tuple[str, list[str], int]]):
    """items: (file name, label lines, image seed)."""
    d = raw / name
    for split in ("train", "valid"):
        (d / split / "images").mkdir(parents=True, exist_ok=True)
        (d / split / "labels").mkdir(parents=True, exist_ok=True)
    (d / "data.yaml").write_text(yaml.safe_dump({"names": names}))
    for i, (fname, lines, seed) in enumerate(items):
        split = "train" if i % 3 else "valid"
        _img(d / split / "images" / fname, seed)
        (d / split / "labels" / (Path(fname).stem + ".txt")).write_text("\n".join(lines) + "\n")


def test_pipeline_steps_2_to_4(tmp_path):
    raw = tmp_path / "raw"
    # source A: 12 photos, each with 2 Roboflow augmentations
    items_a = []
    for k in range(12):
        for aug in range(2):
            items_a.append((f"p{k}_jpg.rf.{aug}x.jpg", ["0 0.5 0.5 0.1 0.05", "1 0.5 0.5 0.6 0.9"], 100 + k))
    items_a.append(("onlyperson_jpg.rf.q.jpg", ["1 0.5 0.5 0.6 0.9"], 500))       # emptied by mapping -> dropped
    _fake_source(raw, "src_a", ["Cigarette", "person"], items_a)
    # source B (train): 6 photos, one is a copy of A's photo p0 -> cross-source duplicate
    items_b = [(f"b{k}.jpg", ["0 0.4 0.4 0.1 0.1"], 200 + k) for k in range(5)] + [("copy_of_p0.jpg", ["0 0.5 0.5 0.1 0.05"], 100)]
    _fake_source(raw, "src_b", ["vape"], items_b)
    # source T (test): includes a copy of A's p1 -> train copies must be dropped
    items_t = [(f"t{k}.jpg", ["0 0.5 0.5 0.1 0.1"], 300 + k) for k in range(4)] + [("copy_of_p1.jpg", ["0 0.5 0.5 0.1 0.05"], 101)]
    _fake_source(raw, "src_t", ["cigarette"], items_t)

    sources = {
        "roboflow_format": "yolov8",
        "sources": {
            "src_a": {"workspace": "w", "project": "a", "role": "train", "priority": 1, "class_map": {"cigarette": "cigarette", "person": None}},
            "src_b": {"workspace": "w", "project": "b", "role": "train", "priority": 2, "class_map": {"vape": "vape"}},
            "src_t": {"workspace": "w", "project": "t", "role": "test", "priority": 9, "class_map": {"cigarette": "cigarette"}},
        },
    }
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump(sources))
    tax = {"classes": TAX.classes, "smoking_objects": TAX.smoking_objects, "prompts": {}}
    (tmp_path / "taxonomy.yaml").write_text(yaml.safe_dump(tax))
    merged, data = tmp_path / "merged", tmp_path / "data"

    def run(script, *a):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, a)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    run("02_merge_unify.py", "--sources", tmp_path / "sources.yaml", "--taxonomy", tmp_path / "taxonomy.yaml",
        "--raw", raw, "--out", merged)
    import pandas as pd

    m = pd.read_csv(merged / "manifest.csv")
    assert len(m) == 24 + 6 + 5                       # the person-only image was dropped
    assert not m["image"].str.contains("onlyperson").any()
    lbl = (merged / "labels" / "src_a__p0_jpg.rf.0x.txt").read_text().split()
    assert lbl[0] == "0" and len(lbl) == 5            # person box removed

    run("03_dedupe.py", "--merged", merged, "--workers", 2)
    dd = pd.read_csv(merged / "manifest_dedup.csv")
    dropped = dd[~dd["keep"]]
    assert set(dropped["drop_reason"]) == {"test_leak", "cross_source_dup"}
    assert dropped[dropped["drop_reason"] == "test_leak"]["orig_stem"].str.startswith("p1_jpg").all()
    assert (dropped[dropped["drop_reason"] == "cross_source_dup"]["source"] == "src_b").all()

    out = run("04_split.py", "--taxonomy", tmp_path / "taxonomy.yaml", "--merged", merged,
              "--negatives", tmp_path / "none", "--data-dir", data, "--val-frac", 0.25)
    split = {s: (data / "splits" / f"{s}.txt").read_text().split() for s in ("train", "val", "test")}
    assert len(split["test"]) == 5
    assert not set(split["train"]) & set(split["val"])
    # augmented copies of one photo never straddle train and val
    stem_of = dict(zip(dd["image"], dd["orig_stem"]))
    assert not {stem_of[p] for p in split["train"]} & {stem_of[p] for p in split["val"]}
    data_yaml = yaml.safe_load((data / "smoking.yaml").read_text())
    assert data_yaml["names"][0] == "cigarette"
    assert "Boxes per class" in out
