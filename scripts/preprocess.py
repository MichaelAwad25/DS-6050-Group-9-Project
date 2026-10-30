"""Extract filtered, labeled heartbeat windows from MIT-BIH recordings."""


from pathlib import Path
from collections import Counter

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LOWCUT = 0.5
HIGHCUT = 40.0
FILTER_ORDER = 4
SAMPLES_BEFORE = 90
SAMPLES_AFTER = 162
WINDOW_SIZE = SAMPLES_BEFORE + SAMPLES_AFTER

BEAT_CLASSES = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "A": 1, "a": 1, "J": 1, "S": 1,
    "V": 2, "E": 2,
    "F": 3,
    "/": 4, "f": 4, "Q": 4
}

CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def preprocess_record(record_id):
    """Filter one ECG recording and extract labeled heartbeat windows."""

    record_path = str(RAW_DIR / record_id)

    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    fs = record.fs

    signal = record.p_signal[:, 0]

    sos = butter(
        FILTER_ORDER,
        [LOWCUT, HIGHCUT],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    filtered_signal = sosfiltfilt(sos, signal)

    windows = []
    labels = []

    for sample, symbol in zip(annotation.sample, annotation.symbol):

        if symbol not in BEAT_CLASSES:
            continue

        start = sample - SAMPLES_BEFORE
        end = sample + SAMPLES_AFTER

        if start < 0 or end > len(filtered_signal):
            continue

        beat = filtered_signal[start:end]

        if len(beat) != WINDOW_SIZE:
            continue

        if not np.all(np.isfinite(beat)):
            continue

        windows.append(beat)
        labels.append(BEAT_CLASSES[symbol])

    X = np.asarray(windows, dtype=np.float32).reshape(-1, WINDOW_SIZE)
    y = np.asarray(labels, dtype=np.int64)

    return X, y


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    record_ids = sorted(path.stem for path in RAW_DIR.glob("*.hea"))

    if len(record_ids) != 48:
        raise ValueError(
            f"Expected 48 recordings, found {len(record_ids)}"
        )

    all_windows = []
    all_labels = []
    all_record_ids = []

    for record_id in record_ids:
        print(f"Processing record {record_id}...")

        X, y = preprocess_record(record_id)

        all_windows.append(X)
        all_labels.append(y)

        all_record_ids.extend([record_id] * len(y))

        print(f"  Extracted {len(y)} beats")

    X_all = np.concatenate(all_windows, axis=0)
    y_all = np.concatenate(all_labels, axis=0)
    record_groups = np.asarray(all_record_ids)

    np.save(PROCESSED_DIR / "X_all.npy", X_all)
    np.save(PROCESSED_DIR / "y_all.npy", y_all)
    np.save(PROCESSED_DIR / "record_groups.npy", record_groups)

    print("\nPreprocessing complete!")
    print("X_all shape:", X_all.shape)
    print("y_all shape:", y_all.shape)
    print("Record groups shape:", record_groups.shape)

    counts = Counter(y_all.tolist())

    print("\nFinal class distribution:")
    for class_id, name in enumerate(CLASS_NAMES):
        print(f"{name}: {counts.get(class_id, 0):,}")

    print(f"\nSaved files to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
