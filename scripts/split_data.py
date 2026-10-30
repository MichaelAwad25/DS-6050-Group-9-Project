
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def class_counts(y):
    return np.bincount(y, minlength=5)


def print_distribution(name, labels, records):
    counts = class_counts(labels)

    print(f"\n{name}")
    print("-" * 45)
    print(f"Total beats: {len(labels):,}")
    print(f"Recordings: {len(np.unique(records))}")

    for i, class_name in enumerate(CLASS_NAMES):
        percentage = 100 * counts[i] / len(labels)
        print(f"{class_name}: {counts[i]:,} ({percentage:.2f}%)")


def main():
    y = np.load(DATA_DIR / "y_all.npy")
    records = np.load(DATA_DIR / "record_groups.npy")
    if len(y) != len(records):
        raise ValueError("Labels and record groups must have matching lengths.")

    patient_groups = np.where(records == "202", "201", records)

    best_score = float("inf")
    best_split = None
    best_seed = None

    overall_proportions = class_counts(y) / len(y)

    for seed in range(1000):

        splitter1 = GroupShuffleSplit(
            n_splits=1,
            test_size=0.30,
            random_state=seed
        )

        train_idx, temp_idx = next(
            splitter1.split(y, y, groups=patient_groups)
        )

        splitter2 = GroupShuffleSplit(
            n_splits=1,
            test_size=0.50,
            random_state=seed
        )

        val_rel, test_rel = next(
            splitter2.split(
                y[temp_idx],
                y[temp_idx],
                groups=patient_groups[temp_idx]
            )
        )

        val_idx = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]

        splits = [train_idx, val_idx, test_idx]

        valid = True

        for idx in splits:
            counts = class_counts(y[idx])

            if np.any(counts < 30):
                valid = False
                break

        if not valid:
            continue

        score = 0.0

        for idx in splits:
            proportions = class_counts(y[idx]) / len(idx)
            score += np.sum(
                np.abs(proportions - overall_proportions)
            )

        target_sizes = [0.70, 0.15, 0.15]

        for idx, target in zip(splits, target_sizes):
            score += abs(len(idx) / len(y) - target)

        if score < best_score:
            best_score = score
            best_split = splits
            best_seed = seed

    if best_split is None:
        raise RuntimeError(
            "No suitable split found. "
            "Try more seeds or revise the constraints."
        )

    train_idx, val_idx, test_idx = best_split

    train_groups = set(patient_groups[train_idx])
    val_groups = set(patient_groups[val_idx])
    test_groups = set(patient_groups[test_idx])

    if not (train_groups.isdisjoint(val_groups)
            and train_groups.isdisjoint(test_groups)
            and val_groups.isdisjoint(test_groups)):
        raise ValueError("Patient groups overlap between splits.")

    print("Patient-group overlap check passed!")
    print("Selected random seed:", best_seed)
    print("Split score:", round(best_score, 4))

    print_distribution("TRAIN", y[train_idx], records[train_idx])
    print_distribution("VALIDATION", y[val_idx], records[val_idx])
    print_distribution("TEST", y[test_idx], records[test_idx])

    np.save(DATA_DIR / "train_idx.npy", train_idx)
    np.save(DATA_DIR / "val_idx.npy", val_idx)
    np.save(DATA_DIR / "test_idx.npy", test_idx)
    np.savez(DATA_DIR / "split_metadata.npz", seed=best_seed, score=best_score)

    print("\nImproved split indices saved successfully.")


if __name__ == "__main__":
    main()
