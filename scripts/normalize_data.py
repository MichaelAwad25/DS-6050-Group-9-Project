from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def main():


    X = np.load(DATA_DIR / "X_all.npy", mmap_mode="r")
    y = np.load(DATA_DIR / "y_all.npy")

    train_idx = np.load(DATA_DIR / "train_idx.npy")
    val_idx = np.load(DATA_DIR / "val_idx.npy")
    test_idx = np.load(DATA_DIR / "test_idx.npy")


    X_train = np.array(X[train_idx], dtype=np.float32)
    X_val = np.array(X[val_idx], dtype=np.float32)
    X_test = np.array(X[test_idx], dtype=np.float32)

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    if any(len(features) != len(labels) for features, labels in [
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    ]):
        raise ValueError("Feature and label counts do not match.")


    train_mean = np.mean(X_train, dtype=np.float64)
    train_std = np.std(X_train, dtype=np.float64)

    print("Training mean:", train_mean)
    print("Training standard deviation:", train_std)

    if not np.isfinite(train_std) or train_std < 1e-8:
        raise ValueError("Invalid training standard deviation.")


    X_train = ((X_train - train_mean) / train_std).astype(np.float32)
    X_val = ((X_val - train_mean) / train_std).astype(np.float32)
    X_test = ((X_test - train_mean) / train_std).astype(np.float32)


    print("\nNormalized dataset shapes:")

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    for name, array in [
        ("X_train", X_train),
        ("X_val", X_val),
        ("X_test", X_test)
    ]:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinite values.")

    print("\nNormalized training mean:", np.mean(X_train))
    print("Normalized training std:", np.std(X_train))


    np.save(DATA_DIR / "X_train.npy", X_train)
    np.save(DATA_DIR / "y_train.npy", y_train)

    np.save(DATA_DIR / "X_val.npy", X_val)
    np.save(DATA_DIR / "y_val.npy", y_val)

    np.save(DATA_DIR / "X_test.npy", X_test)
    np.save(DATA_DIR / "y_test.npy", y_test)

    np.savez(
        DATA_DIR / "normalization_stats.npz",
        mean=train_mean,
        std=train_std
    )

    print("\nAll normalized datasets saved successfully!")


if __name__ == "__main__":
    main()
