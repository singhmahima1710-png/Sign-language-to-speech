# Sign Language → Speech

Real-time static sign (e.g. ASL fingerspelling) recognition:
**webcam → hand landmarks + classical CV features → classifier → text → speech.**

Built for an Image & Video Processing course: it combines a modern landmark
detector (MediaPipe) with an explicit classical image-processing pipeline
(HSV skin segmentation, morphology, contours, convex hull, Hu moments).

## Quick start

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py                    # menu launcher for everything below
```

**Python 3.9 – 3.12 required** (the pinned MediaPipe version has no 3.13 build).
A webcam is needed for options 1, 2 and 5.

## Recommended workflow

| Step | Menu option | Script | What it does |
|------|-------------|--------|--------------|
| 0 | 1 | `classical_demo.py` | Live view of the classical CV stages; calibrate skin color for your lighting |
| 1 | 2 or 3 | `data_collection.py` / `build_dataset_from_images.py` | Create `data/landmarks.csv` from your webcam, or from an image-folder dataset |
| 2 | 4 | `train_model.py` | Train Random Forest, print accuracy, save `model/` |
| 3 | 5 | `realtime_recognition.py` | Live sign → text → speech |
| 4 | 6 | `evaluate_ablation.py` | Landmarks-only vs classical-only vs hybrid comparison (for your report) |

### Step 1 — Get training data

**Option A: record yourself** (`data_collection.py`)
- Press a letter key (a–z) to choose the sign, then **SPACE** to start/stop recording.
- **ENTER** lets you type a custom label (e.g. a word). **ESC** quits.
- Aim for 150–300 samples per sign. Move/rotate your hand slightly between samples.
- Record in the same lighting you'll use for the final demo.

**Option B: use an existing dataset** (`build_dataset_from_images.py`)
```bash
python build_dataset_from_images.py path/to/asl_alphabet_train --max-per-class 300
```
Folder layout: one sub-folder per label (`A/`, `B/`, …). Images where no hand is found
are skipped. Works with the Kaggle "ASL Alphabet" dataset.

### Step 3 — Real-time controls
- Hold a sign steady (~15 frames) and it's added to the current word.
- **SPACE** commit word · **b** backspace · **c** clear · **s** speak sentence · **q** quit

## How it works

```
Webcam frame
   │
   ├───────────────────────────────┐
   ▼                                ▼
MediaPipe Hands                 Classical CV (preprocessing.py)
21 landmarks (x,y,z)            HSV skin mask → morphology (open/close)
   │                            → largest contour (inside hand bbox)
   ▼                            → hull, solidity, extent, aspect ratio,
Normalize: subtract wrist,        convexity defects, Hu moments
divide by hand size                │
   │                                │
   └────────────┬───────────────────┘
                ▼
   76-d hybrid feature vector (63 landmark + 13 classical)
                ▼
   Random Forest classifier → letter/sign + confidence
                ▼
   Stability filter (N consecutive frames, min confidence, cooldown)
                ▼
   Text buffer (word → sentence) → pyttsx3 text-to-speech
```

### The 13 classical features
`solidity, extent, aspect ratio, finger-gap count (convexity defects),
7 log-scaled Hu moments, equivalent-diameter ratio, circularity.`
Finger-gap depth is measured relative to hand size, so it works at different distances.

## Suggested report experiment
`python evaluate_ablation.py` trains the same classifier on three feature sets with
5-fold CV and a held-out split, and saves `results/ablation.csv`. Discuss when
classical features help (silhouette-distinct signs) vs when landmarks dominate.

## Known limitations (worth stating in your report)
- **Static signs only.** Letters/handshapes work; signs involving motion (J, Z, most
  real words) need a temporal model (LSTM/Transformer on frame sequences, e.g. WLASL).
- **Skin segmentation depends on lighting and skin tone.** Use the calibration step in
  `classical_demo.py`; note `data_collection.py` / `realtime_recognition.py` use the
  default HSV range in `preprocessing.py` (edit `DEFAULT_LOWER_HSV/UPPER_HSV` there with
  your calibrated values printed by the demo). Skin-colored backgrounds hurt it.
- **Keep your hand reasonably close** (roughly ≥150 px wide in frame): the fixed
  morphology kernels erase finger gaps on very small hands.
- When the classical mask fails, its 13 features fall back to zeros and the classifier
  relies on landmarks alone.
- Similar-looking letters (e.g. M/N/T, U/V) are the most commonly confused.
- Single hand only (`max_num_hands=1`). For two-handed signs, set it to 2 in
  `utils.get_hands_detector` and concatenate both hands' features.
- pyttsx3 uses your OS's offline voices (needs `espeak` on Linux).

## Files
```
main.py                      menu launcher
utils.py                     MediaPipe wrapper + landmark normalization + hybrid features
preprocessing.py             classical CV pipeline
classical_demo.py            live classical-CV visualization (no MediaPipe)
data_collection.py           record dataset from webcam
build_dataset_from_images.py build dataset from labeled image folders
train_model.py               train + evaluate + save classifier
realtime_recognition.py      live sign → text → speech
evaluate_ablation.py         feature-set comparison experiment
requirements.txt
```
