"""Overlay CNN and hybrid training curves from two run directories."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--cnn", type=Path, required=True)
p.add_argument("--hybrid", type=Path, required=True)
p.add_argument("--out", type=Path, default=Path("results/figures/comparison.png"))
a = p.parse_args()

runs = {}
for name, path in (("CNN", a.cnn), ("CNN-Transformer", a.hybrid)):
    with open(path / "history.json", encoding="utf-8") as handle:
        runs[name] = json.load(handle)

panels = [
    ("train_loss", "Train loss"),
    ("val_loss", "Validation loss"),
    ("val_macro_f1", "Validation macro-F1"),
]
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (key, title) in zip(axes, panels):
    for name, h in runs.items():
        ax.plot(h["epoch"], h[key], marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.grid(True)
    ax.legend()
plt.tight_layout()
a.out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(a.out, dpi=150)
print(f"Saved {a.out}")
