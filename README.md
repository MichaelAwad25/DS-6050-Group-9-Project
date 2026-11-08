# ECGBeats — DS 6050 Group 09

ECGBeats is a University of Virginia DS 6050 group project comparing a compact 1-D CNN with a matched CNN–Transformer for heartbeat classification on the MIT-BIH Arrhythmia Database.

## Dataset and approved protocol

Source: [MIT-BIH Arrhythmia Database (PhysioNet, v1.0.0)](https://physionet.org/content/mitdb/1.0.0/). The database contains 48 annotated, approximately 30-minute, two-channel ECG recordings sampled at 360 Hz. The `atr` annotations identify beat locations and types.

The pipeline implements the approved four-class protocol:

- **Records 102, 104, 107 and 217 are excluded.** The raw download may still contain all 48 records; preprocessing skips the four exclusions.
- **The MLII lead is selected by name** from `record.sig_name`. This matters for record 114, where MLII is channel index 1. A record without MLII is skipped and logged in the preprocessing metadata.
- **One-second windows** are extracted: 144 samples before and 216 after each annotation, for 360 samples at 360 Hz.
- **Paced, unclassifiable and non-beat annotations are excluded** (`/`, `f`, `Q`, and every symbol outside the mapping below).

The dataset is downloaded locally; raw ECG files and generated NumPy arrays should not be committed to Git.

## Repository layout

```text
data/
  raw/                     # Downloaded PhysioNet recordings (ignored by Git)
  processed/               # Generated arrays and metadata (ignored by Git)
models/
  BaselineModel.py         # Matched CNN and CNN-Transformer training/evaluation
notebooks/
  preprocessing.ipynb      # ECG exploration and visualization
scripts/
  pipeline_utils.py        # Shared constants and validation helpers
  preprocess.py            # MLII filtering and four-class beat extraction
  split_data.py            # Frozen patient-separated 60/20/20 split
  normalize_data.py        # Training-only z-score normalization
paper/
  main.tex                 # Milestone I report source
  arxiv.sty                # Report style file
  references.bib           # Bibliography
  DS6050_F26_G09_ECGBeats.pdf
docs/
  G09MileI_feedback.pdf    # Milestone I instructor feedback
results/
  figures/                 # Training curves
  runs/                    # Per-run configs, checkpoints and metrics (ignored by Git)
requirements.txt
README.md
```

## Environment setup

Use Python 3.12 with the pinned dependencies in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The project is also configured as a uv project via `pyproject.toml` and `.python-version`:

```bash
uv sync
```

If PyTorch is installed separately, use the command appropriate for the local CPU or CUDA version. `tqdm` is required by the training script.

## Download the ECG recordings

Run from the repository root:

```bash
mkdir -p data/raw data/processed
python -c "import wfdb; wfdb.dl_database('mitdb', dl_dir='data/raw', records='all', annotators=['atr'])"
```

Confirm that `data/raw/` contains 48 records, each with `.hea`, `.dat` and `.atr` files.

## Reproduce the pipeline

Run these commands in order from the repository root:

```bash
python scripts/preprocess.py
python scripts/split_data.py
python scripts/normalize_data.py
```

Split and normalization accept `--data-dir`; preprocessing accepts `--raw-dir` and `--processed-dir` for alternate locations.

### 1. Filtering and heartbeat extraction

`scripts/preprocess.py` selects MLII by name, applies a fourth-order zero-phase Butterworth bandpass filter from 0.5 to 40 Hz, and extracts a 360-sample window for each eligible expert-annotated beat: 144 samples before and 216 after the annotation.

The four-class mapping is:

| Class | ID | Original annotation symbols |
|---|---:|---|
| N — Normal | 0 | `N`, `L`, `R`, `e`, `j` |
| S — Supraventricular | 1 | `A`, `a`, `J`, `S` |
| V — Ventricular | 2 | `V`, `E` |
| F — Fusion | 3 | `F` |

The script produces:

- `X_all.npy` — unnormalized windows, shape `(beats, 360)`;
- `y_all.npy` — integer labels in `0..3`;
- `record_groups.npy` — source recording identifiers;
- `preprocessing_metadata.json` — protocol settings, included/skipped records, per-record beat counts, class counts and SHA-256 hashes of the arrays.

Out-of-bounds windows and non-finite windows are excluded. Records missing MLII are skipped and reported.

### 2. Patient-separated splitting

`scripts/split_data.py` uses `GroupShuffleSplit` to keep every patient in exactly one split. Recordings 201 and 202 are explicitly treated as one patient group. The default target is approximately **60% training, 20% validation and 20% test**.

The default split uses a **predeclared fixed seed of 42**. It does not search seeds using labels. The split must represent all four classes in every partition; classes with fewer than 30 beats in a partition are reported as low-support warnings. If a fixed seed leaves a class absent, the script stops rather than silently producing a different design.

For research comparisons only, `python scripts/split_data.py --search-seeds` performs a **label-informed** seed search with a declared minimum support of 30 beats per class per split. This mode is recorded in the manifest as label-informed and must be disclosed before model comparisons. Do not reselect the split because test performance looks better.

The script writes `train_idx.npy`, `val_idx.npy`, `test_idx.npy`, `split_metadata.npz` and `split_manifest.json`. The manifest records the seed and selection method, target ratios, actual record IDs and patient groups, class support, support warnings, input hashes and index hashes.

### 3. Normalization

`scripts/normalize_data.py` verifies the split manifest and array generations, checks that the indices are unique, disjoint and complete, and confirms the inputs are finite. It computes one global mean and standard deviation from **training windows only**, applies `(x - mean) / std` to all splits, and saves:

- `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`;
- `normalization_stats.npz`;
- `normalization_metadata.json` — statistics, split hash, source/index hashes and output array hashes.

The arrays remain two-dimensional `(beats, 360)`; the PyTorch dataset adds the channel dimension.

## Model training and evaluation

Train the matched CNN control and CNN–Transformer without touching the test split:

```bash
python models/BaselineModel.py --model both
```

When configurations are frozen, run the explicit final evaluation:

```bash
python models/BaselineModel.py --model both --run-test
```

To evaluate an existing best checkpoint without retraining:

```bash
python models/BaselineModel.py \
  --test-checkpoint results/runs/<run>/best_checkpoint.pt \
  --run-test
```

Both models use the same data, CNN backbone, linear classification head, optimizer, learning-rate schedule, seed 42, batch size 256, ten epochs and checkpoint rule. The hybrid adds a learned positional embedding and a single Transformer encoder layer between the shared backbone and head.

Key evaluation behavior:

- **Checkpoint rule:** the best validation macro-F1 checkpoint is saved and restored before evaluation; ties keep the earlier checkpoint.
- **Checkpoint-only evaluation:** `--test-checkpoint ... --run-test` evaluates a saved checkpoint without retraining and verifies that it belongs to the current split and normalization generation.
- **Macro-F1:** computed across the fixed four classes, with an absent class contributing zero and being flagged in the metrics.
- **Loss:** epoch loss is example-weighted (`sum(batch_loss × batch_size) / total_examples`), so a short final batch does not receive the same weight as a full batch.
- **Validation reporting:** per-class sensitivity, precision, F1 and support are printed and saved for every epoch.
- **Test access:** the test split is not loaded or evaluated unless `--run-test` is passed. `test_access.json` records whether it was accessed.
- **Latency:** batch-one float32 latency is measured with warm-up, repetitions, recorded device and accelerator synchronization.

Each run is saved under `results/runs/<timestamp>_<model>_seed42/` with:

```text
config.json
best_checkpoint.pt
history.csv
history.json
model_summary.json
latency.json
test_access.json
test_metrics.json       # only with --run-test
predictions.npz         # only with --run-test
```

Training curves are written to `results/figures/`.

## Reporting notes

- Re-run preprocessing, splitting and normalization after changing any protocol setting. The previous five-class, 252-sample results and split counts are obsolete and must not be reported as four-class results.
- Do not reuse the old training-curve image as evidence for the revised protocol. Retrain and report the new curves, checkpoint epoch, per-class validation metrics and confusion matrix.
- The fixed 60/20/20 split is an internal exploratory split, not a published MIT-BIH benchmark. Report the exact records, class support and any label-informed split selection.
- Test macros are not to be used for model selection. Re-run final evaluation only after configurations are frozen, and disclose any earlier test access.

## Suggested execution order

1. `python scripts/preprocess.py`
2. `python scripts/split_data.py` (or `--search-seeds` with disclosure)
3. `python scripts/normalize_data.py`
4. `python models/BaselineModel.py --model both`
5. Freeze configurations and inspect validation results.
6. `python models/BaselineModel.py --model both --run-test`
