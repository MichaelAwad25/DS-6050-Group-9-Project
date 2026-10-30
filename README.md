# ECGBeats — DS 6050 Group 09

ECGBeats is a University of Virginia DS 6050 group project exploring heartbeat classification using the MIT-BIH Arrhythmia Database. This repository contains ECG preprocessing and a teammate-developed baseline CNN implementation.

## Dataset

Source: [MIT-BIH Arrhythmia Database (PhysioNet, v1.0.0)](https://physionet.org/content/mitdb/1.0.0/). The database contains 48 annotated, approximately 30-minute, two-channel ECG recordings sampled at 360 Hz. The `atr` annotations identify beat locations and types.

The dataset is downloaded locally; raw ECG files and generated NumPy arrays should not be committed to Git.

## Repository layout

```text
data/
  raw/                   # Downloaded PhysioNet recordings (ignored by Git)
  processed/             # Generated NumPy arrays (ignored by Git)
models/
  BaselineModel.py       # Baseline CNN (developed separately)
notebooks/
  preprocessing.ipynb    # ECG exploration and visualization
scripts/
  preprocess.py          # Filtering, beat extraction, label mapping
  split_data.py          # Patient-separated train/validation/test split
  normalize_data.py      # Training-only z-score normalization
requirements.txt
README.md
```

## Environment setup

Use Python 3.11 with the dependencies in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

If the requirements file is incomplete, preprocessing needs at least `numpy`, `scipy`, `wfdb`, and `scikit-learn`. The notebook also uses `pandas`, `matplotlib`, `jupyter`, and `ipykernel`; model training uses PyTorch.

## Download the ECG recordings

Run from the repository root:

```bash
mkdir -p data/raw data/processed
python -c "import wfdb; wfdb.dl_database('mitdb', dl_dir='data/raw', records='all', annotators=['atr'])"
```

Confirm that `data/raw/` contains 48 files each with the `.hea`, `.dat`, and `.atr` extensions.

## Reproduce preprocessing

Run these commands in order from the repository root:

```bash
python scripts/preprocess.py
python scripts/split_data.py
python scripts/normalize_data.py
```

### 1. Filtering and heartbeat extraction

`scripts/preprocess.py` reads the first signal channel of each record (not necessarily MLII for every recording), applies a fourth-order zero-phase Butterworth bandpass filter from 0.5 to 40 Hz, and extracts a 252-sample window centered on each eligible expert-annotated beat: 90 samples before and 162 after the annotation (approximately 0.25 seconds before and 0.45 seconds after at 360 Hz). Unknown annotation symbols, out-of-bounds windows, and non-finite windows are excluded.

The project uses the following five-class, AAMI-style mapping:

| Class | ID | Original annotation symbols |
|---|---:|---|
| N — Normal | 0 | `N`, `L`, `R`, `e`, `j` |
| S — Supraventricular | 1 | `A`, `a`, `J`, `S` |
| V — Ventricular | 2 | `V`, `E` |
| F — Fusion | 3 | `F` |
| Q — Unknown/Paced | 4 | `/`, `f`, `Q` |

This is a project-specific mapping; particularly, the Q class includes paced beats. It should not be assumed to be identical to every published AAMI evaluation protocol.

The script produces `X_all.npy` (unnormalized windows), `y_all.npy` (integer class labels), and `record_groups.npy` (source recording identifiers).

### 2. Patient-separated splitting

`scripts/split_data.py` uses `GroupShuffleSplit` to keep each recording in only one split. Recordings 201 and 202 are explicitly treated as one patient group. The script searches 1,000 random seeds, requires at least 30 beats from each class in each split, and chooses the candidate closest to the full-dataset class proportions and approximate 70/15/15 split sizes. It saves `train_idx.npy`, `val_idx.npy`, `test_idx.npy`, and `split_metadata.npz`.

Important evaluation limitation: this is a class-distribution-optimized exploratory split, not a published MIT-BIH benchmark split. The split selection uses labels from all recordings, including the eventual test set. Do not tune or reselect the test split based on model results; for comparison with published research, consider a fixed inter-patient benchmark protocol. Class proportions remain unequal across splits, especially for rare classes.

### 3. Normalization

`scripts/normalize_data.py` computes one global mean and standard deviation from training windows only and applies `(x - mean) / std` to training, validation, and test windows. It saves six arrays (`X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`) and `normalization_stats.npz`. The feature arrays remain two-dimensional `(number_of_beats, 252)`; the baseline PyTorch dataset class adds the channel dimension.

## Verified preprocessing output

The following outputs were observed on the initial run:

| Dataset | Beats | Shape of features |
|---|---:|---|
| All | 109,452 | `(109452, 252)` |
| Train | 74,431 | `(74431, 252)` |
| Validation | 17,227 | `(17227, 252)` |
| Test | 17,794 | `(17794, 252)` |

The selected split seed was 77. Training-only normalization statistics were mean 0.010421719908351524 and standard deviation 0.39447492408220847. The normalized training mean was approximately zero and its standard deviation was 1.0.

Class counts by split:

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| N | 63,354 | 14,096 | 13,144 |
| S | 2,612 | 101 | 68 |
| V | 3,922 | 911 | 2,402 |
| F | 390 | 36 | 376 |
| Q | 4,153 | 2,083 | 1,804 |

These counts are observations for the current pipeline, not guarantees if preprocessing settings or splitting logic are changed.

