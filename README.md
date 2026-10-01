# Smoking Detection

Detect people smoking cigarettes, vapes, cigars and pipes in images and video (hookah excluded).

The system has three stages, so it does not rely on spotting a 15-pixel cigarette alone:

```
frame ─┬─> [1] object detector ──> cigarette / vape / cigar / pipe / smoke boxes ──┐
       │                                                                          ├─> per-person features ─> [3] temporal decision ─> smoking events
       └─> [2] pose + tracking ──> each person's keypoints over time ──────────────┘     (object near mouth/hand,     (object and motion agree,
                                                                                          wrist-to-mouth distance)       or one is very confident)
```

Everything runs today: before you train anything, stage 1 uses YOLO-World zero-shot and stage 3 uses a hand-to-mouth heuristic. Each trained piece then replaces its stand-in.

---

## Setup

Tested with Python 3.11 and Ultralytics 8.4. Training is easiest in WSL/Ubuntu.

**1. PyTorch with GPU support (install this first).** RTX 50-series cards need a PyTorch build for CUDA 12.8 or newer.

- **WSL / Linux:** `pip install torch torchvision` (the default Linux wheels on PyPI are CUDA builds).
- **Windows:** the PyPI wheels are CPU-only. Use the command from the [PyTorch "Get Started" selector](https://pytorch.org/get-started/locally/) for CUDA 12.8 or newer.

Check it worked:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**2. Everything else:**

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install fiftyone pytest                           # optional: step 5 and the tests
pytest -q                                             # 22 tests, CPU only, ~2 s
```

**3. Roboflow API key** (free account): Roboflow → Settings → API Keys.

```bash
export ROBOFLOW_API_KEY=your_key                      # PowerShell: $env:ROBOFLOW_API_KEY="your_key"
```

---

## Workflow

| Step | Script | What it does | Output |
| --- | --- | --- | --- |
| 1 | `01_download_sources.py` | Downloads the Roboflow datasets and **audits their class names** | `data/raw/`, `audit/report.md`, sample crops |
| — | *you* | Read the audit, fix `configs/sources.yaml` | |
| 2 | `02_merge_unify.py` | Remaps every source to one label set | `data/merged/` |
| 3 | `03_dedupe.py` | Finds duplicate photos within and across sources | `manifest_dedup.csv` |
| 4 | `04_split.py` | Train/val split by photo group; whole sources held out as test | `data/splits/`, `data/smoking.yaml` |
| 5 | `05_hard_negatives.py` | People drinking, eating, on the phone (no smoking) | `data/negatives/` |
| 6 | `06_yoloworld_baseline.py` | Zero-shot baseline; false-positive mining; pre-labelling | `runs/baselines/` |
| 7 | `07_train_detector.py` | Trains YOLO11 / YOLO26 / YOLO-World and scores it on the test set | `runs/detect/<name>/` |
| 8 | `08_extract_tracks.py` | Labelled video clips → per-person feature sequences | `data/tracks/` |
| 9 | `09_train_temporal.py` | Trains the temporal classifier; compares it with the heuristic | `models/temporal.pt` |
| 10 | `10_run_pipeline.py` | Full pipeline on a video or webcam | annotated video + events JSON |

Re-run step 4 after step 5 so the negatives are included.

### Step 1 in detail: the audit is the important part

```bash
python scripts/01_download_sources.py
```

The class names in `configs/sources.yaml` came from the Roboflow web pages and are **not verified**. Open `audit/report.md` and the crops under `audit/<source>/<class>/`, then for every class decide what it maps to:

- `cigarette`, `vape`, `cigar`, `pipe` or `smoke`, or
- `null` to drop it.

Lines marked `REVIEW` are the uncertain ones. For example, a `smoking` class might box the hand and cigarette together (map it to `cigarette`) or the whole person (drop it). If a source only has whole-person boxes, set `role: skip`.

Step 2 refuses to run while any class is unmapped, so nothing slips through silently.

### First commands, start to finish

```bash
python scripts/01_download_sources.py
# ...edit configs/sources.yaml from audit/report.md...
python scripts/02_merge_unify.py --overwrite
python scripts/03_dedupe.py
python scripts/05_hard_negatives.py --max 3000
python scripts/04_split.py
python scripts/06_yoloworld_baseline.py --mode eval --split test        # the number to beat
python scripts/07_train_detector.py --model yolo11s.pt --name y11s_960
python scripts/07_train_detector.py --model yolo26s.pt --name y26s_960
python scripts/10_run_pipeline.py --video some_clip.mp4                  # works with YOLO-World already
```

When a detector beats the baseline, point `configs/pipeline.yaml` at it:

```yaml
detector:
  type: yolo
  weights: runs/detect/y11s_960/weights/best.pt
```

---

## Design decisions

**"person" is not a detector class.** People come from the pose model in stage 2. Several sources never label people; training a person class on them would teach the model that unlabelled people are background.

**Test sources are held out whole.** `smoker_yolo` and `livestream` (role: test) never touch training. A random split of merged Roboflow data leaks near-copies across splits, which is a big reason published numbers look inflated.

**Duplicates.** Roboflow augmentation produces several copies of each photo, and many datasets copy each other. Step 3 groups copies by the Roboflow file-name stem and by perceptual hash (mirrored copies included):

| Group spans | Action |
| --- | --- |
| a test source and a train source | drop the train copies |
| several train sources | keep the highest-priority source's copies only |
| one source | keep all; same `group_id`, so never split across train and val |

Spot-check `data/merged/duplicates_sample.csv` and tune `--threshold` (default 6 of 64 bits).

**Images that lose all their boxes are dropped.** If an image only had `person` / `smoker` boxes, it probably shows a cigarette nobody labelled, so it would be a poisoned negative. Override with `--keep-emptied`.

**Partially labelled classes.** Step 2 prints which unified classes each source labels. If, say, only some sources label `smoke`, unlabelled smoke elsewhere counts as background. Fix it by pre-labelling with step 6 (`--mode prelabel --classes smoke`) and reviewing the boxes.

**Hard negatives are screened.** Every COCO candidate is run through YOLO-World first; anything that looks like a smoking object goes to `data/negatives/review/` for a manual look instead of being used. About 20% of the negatives go to the test set, so you get a false-alarm rate.

---

## YOLO-World: what it can and cannot do here

YOLO-World is an open-vocabulary detector: it has no fixed label list; you give it text prompts (`configs/taxonomy.yaml → prompts`). It was pretrained on Objects365, which has a single combined **"Cigar/Cigarette"** class, so cigarettes are a concept it has actually seen. There is no vape or pipe class in Objects365, so those rely on text generalisation and will be weaker.

Expect it to struggle with small, distant cigarettes and to mix up cigars and cigarettes. In this repo it is used as:

- **the baseline** to beat (step 6 `--mode eval`, scored exactly like the trained detectors),
- **the stage-1 stand-in** so the pipeline runs before your detector exists,
- **a screen** for hard negatives (step 5),
- **a miner** of false positives on non-smoking images (step 6 `--mode mine`),
- **a pre-labeller** for missing classes (step 6 `--mode prelabel`),
- **a training candidate**: `07_train_detector.py --model yolov8s-worldv2.pt` fine-tunes it.

The first YOLO-World run downloads its CLIP text encoder, so it needs internet access once. [YOLOE](https://docs.ultralytics.com/models/yoloe) uses the same `set_classes()` API and is worth trying as an alternative.

---

## Video data for stages 2 and 3

Copy `data/clips/labels_template.csv` to `data/clips/labels.csv`, one row per clip:

```
video_path,label,start_s,end_s,notes
data/clips/videos/smoker_01.mp4,smoking,0,,cigarette outdoors
data/clips/videos/coffee_01.mp4,not_smoking,,,drinking coffee
```

- Prefer clips with **one main person**; by default only the longest, largest track is kept.
- Aim for at least **50–100 clips per class** from different people and places as a starting point.
- `not_smoking` clips should be the hard cases: drinking, eating, phone calls, touching the face, inhalers.
- Labels are per clip, so a smoking clip's windows between puffs are still labelled "smoking". The object cue (cigarette in hand) carries those windows.

Then `08_extract_tracks.py` → `09_train_temporal.py` → set `temporal_checkpoint: models/temporal.pt` in `configs/pipeline.yaml`.

---

## What to measure

Accuracy is not the target. Watch these instead:

- **Image-level false-positive rate** on the hard negatives in the test set (steps 6 and 7 print it).
- **Recall by object size**: small objects are where detectors fail.
- **Per-class AP@0.5**, especially vape vs cigarette.
- **Event-level precision and recall** on held-out videos, and **false alarms per camera-hour**. Step 10 writes `events.json`; compare it with hand-marked smoking intervals.

---

## Known gaps and next steps

- Class names in `configs/sources.yaml` are unverified until you run the audit.
- Cigar and pipe data is thin (deferred for now); expect those classes to be weak.
- The motion heuristic's thresholds (`configs/pipeline.yaml → motion`) are educated guesses; tune them on your first clips.
- Motion alone can raise an alarm above `fusion.mot_strong` (0.95). Set it to `1.01` to require object evidence for every alarm.
- No event-level evaluation script yet; that is the natural next addition once you have marked-up videos.
- Check each dataset's license before any commercial use.

## What was tested

- All 22 unit and end-to-end tests pass (`pytest -q`): label remapping, polygon conversion, hashing including mirrored copies, duplicate resolution, group-safe splitting, AP and image-level metrics, features, the motion heuristic, fusion and event smoothing, and steps 2–4 on a synthetic Roboflow-style dataset.
- Smoke-tested with Ultralytics 8.4 on CPU: the step 1 audit, steps 2–4, one-epoch training plus test scoring (step 7), track extraction on real people (step 8), temporal training (step 9), and the full pipeline with both the learned model and the heuristic (step 10).
- Not run end to end here: real Roboflow downloads (needs your API key), FiftyOne COCO download (step 5), and YOLO-World prompts (its text-encoder download was blocked in the build sandbox). Those paths follow the current Ultralytics docs.

## Layout

```
configs/      taxonomy.yaml (labels, prompts) · sources.yaml (datasets) · pipeline.yaml (stages 2-3)
smokedet/     shared code: io, taxonomy, hashing, splits, metrics, features, motion, fusion, temporal, frame processor
scripts/      01-10, run in order
tests/        pytest suite (no GPU needed)
data/         created by the scripts (raw, merged, negatives, splits, clips, tracks)
```
