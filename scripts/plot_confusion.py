"""Plot best-epoch validation confusion matrices (row-normalized, counts annotated)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["N", "S", "V", "F"]

p = argparse.ArgumentParser()
p.add_argument(
    "--run",
    action="append",
    required=True,
    metavar="TITLE=RUN_DIR",
    help="Repeatable. Example: --run 'CNN=results/runs/prelim_cnn_seed42'",
)
p.add_argument("--out", type=Path, default=Path("results/figures/confusion_matrices.png"))
p.add_argument("--cols", type=int, default=2)
a = p.parse_args()

runs = []
for spec in a.run:
    title, path = spec.split("=", 1)
    with open(Path(path) / "val_metrics_best.json", encoding="utf-8") as handle:
        runs.append((title, np.asarray(json.load(handle)["confusion_matrix"])))

rows = int(np.ceil(len(runs) / a.cols))
fig, axes = plt.subplots(rows, a.cols, figsize=(4.2 * a.cols, 3.9 * rows), squeeze=False)
for ax in axes.ravel():
    ax.axis("off")

for ax, (title, cm) in zip(axes.ravel(), runs):
    ax.axis("on")
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(
                j, i, f"{cm[i, j]:,}\n({100 * norm[i, j]:.0f}%)",
                ha="center", va="center", fontsize=8,
                color="white" if norm[i, j] > 0.5 else "black",
            )
    ax.set_xticks(range(4), CLASSES)
    ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)

plt.tight_layout()
a.out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(a.out, dpi=200)
print(f"Saved {a.out}")
