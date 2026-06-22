"""Money-shot figure for the EEG-JEPA project: seizure-detection AUROC by method.

Values are the held-out (patient-disjoint) TUSZ results from the runs logged in
AVANCEMENT_EEG_JEPA.md (with-projector probe protocol; energy = H1 detector).
Run:  python -m examples.eeg.make_figure
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, AUROC, kind) — kind drives the color
RESULTS = [
    ("VICReg\n(healthy)", 0.815, "repr"),
    ("SIGReg", 0.793, "repr"),
    ("Collapse\n(std=0)", 0.739, "repr"),
    ("Random\nencoder", 0.565, "floor"),
    ("Prediction\nenergy (H1)", 0.520, "energy"),
]
COLORS = {"repr": "#2a7fb8", "floor": "#9e9e9e", "energy": "#c0392b"}

labels = [r[0] for r in RESULTS]
vals = [r[1] for r in RESULTS]
colors = [COLORS[r[2]] for r in RESULTS]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, vals, color=colors, width=0.65)
ax.axhline(0.5, ls="--", c="k", lw=1)
ax.text(len(labels) - 0.4, 0.51, "chance (0.5)", fontsize=9, va="bottom", ha="right")

for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=10)

ax.set_ylabel("Seizure-detection AUROC (held-out patients)")
ax.set_ylim(0.45, 0.88)
ax.set_title("TUSZ seizure detection — EEG-JEPA\n"
             "Representation separates seizures; prediction energy does not (H1 negative)",
             fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "auroc_summary.png")
fig.savefig(out, dpi=150)
print(f"saved -> {out}")
