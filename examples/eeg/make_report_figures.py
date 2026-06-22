"""Generate the synthesis figures for the EEG-JEPA report.

All AUROC values are the held-out (patient-disjoint) results logged in
AVANCEMENT_EEG_JEPA.md. Run:  python -m examples.eeg.make_report_figures
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
BLUE, GREY, RED, GREEN = "#2a7fb8", "#9e9e9e", "#c0392b", "#27ae60"


def bars(fname, labels, vals, colors, title, ylabel="Seizure AUROC (held-out)",
         ylim=(0.45, 0.86), chance=0.5, note=None):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    b = ax.bar(labels, vals, color=colors, width=0.62)
    ax.axhline(chance, ls="--", c="k", lw=1); ax.text(len(labels)-0.45, chance+0.005, "chance", fontsize=8, ha="right")
    for r, v in zip(b, vals):
        ax.text(r.get_x()+r.get_width()/2, v+0.006, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel(ylabel); ax.set_ylim(*ylim); ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(fontsize=9)
    if note:
        fig.text(0.5, -0.02, note, ha="center", fontsize=7.5, style="italic")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", fname)


# Fig 1 — THE ENERGY ARC: same world model, read four ways (+ chance)
bars("fig1_energy_arc.png",
     ["Euclidean\nenergy ‖r‖²", "Mahalanobis\nenergy (0-label)", "Residual\nvector r", "Representation\n(two-view)"],
     [0.57, 0.67, 0.72, 0.81], [RED, "#e67e22", BLUE, GREEN],
     "The energy is the wrong reader: same JEPA, four ways to read it",
     note="Euclidean & Mahalanobis: same frames (0.568/0.670). Residual-vector probe 0.723. Representation = two-view 0.81. Seizure, TUSZ held-out.")

# Fig 2 — DETECTION: representation route, regularizer & collapse (H2/H3)
bars("fig2_detection_regularizers.png",
     ["VICReg", "SIGReg", "Collapse\n(proj)", "Collapse\n(no proj)", "Random\nencoder"],
     [0.815, 0.793, 0.739, 0.687, 0.565], [GREEN, BLUE, "#e67e22", RED, GREY],
     "Detection (seiz vs bckg): representation works; anti-collapse matters",
     note="Two-view SSL, frozen encoder + linear probe, patient-disjoint, same protocol (300 rec/split, out_dim 256).")

# Fig 3 — DETECTION vs PREDICTION
bars("fig3_detection_vs_prediction.png",
     ["Detection\n(during seizure)", "Prediction\n(60s before)", "Random"],
     [0.81, 0.539, 0.50], [GREEN, RED, GREY],
     "A seizure is recognizable, not predictable (short horizon)",
     ylim=(0.45, 0.86),
     note="Detection = seiz-vs-bckg (two-view repr). Prediction = pre-ictal[onset-60s,onset) vs interictal. TUSZ recordings are minutes (no hours-ahead possible).")

# Fig 4 — ENERGY GRADIENT: unpredictable (artifacts) vs predictable (seizure)
bars("fig4_energy_predictability.png",
     ["Muscle\nartifact", "Eye\nartifact", "Seizure"],
     [0.545, 0.539, 0.520], [BLUE, "#5dade2", RED],
     "Prediction energy ∝ unpredictability (weak but ordered)",
     ylabel="Energy AUROC (held-out)", ylim=(0.45, 0.60), chance=0.5,
     note="Same energy world model. Unpredictable (muscle/eye) ≥ predictable (rhythmic seizure). Effect small (whole-brain energy dilutes localized events).")

# Fig 5 — per-channel energy heatmap on a seizure (honest: energy does NOT spike)
d = np.load("/lustre/work/vivatech-dreamingmachines/lguerin/checkpoints/eeg/chan_energy/chan_heatmap.npz", allow_pickle=True)
key = max(d.keys(), key=lambda k: int(d[k][-1].sum()))   # recording with most seizure frames
arr = d[key]; E, y = arr[:-1], arr[-1]
fig, ax = plt.subplots(figsize=(9, 4))
t = np.arange(E.shape[1]) * 2.0 / 60.0                    # frames of 2 s -> minutes
im = ax.imshow(E, aspect="auto", cmap="viridis", extent=[t[0], t[-1], 19, 0])
ax.set_xlabel("time (min)"); ax.set_ylabel("EEG channel"); ax.set_title("Per-channel prediction energy — seizure shaded (energy does NOT spike on it)", fontsize=10)
for j in np.where(y > 0)[0]:
    ax.axvspan(t[j]-1/60, t[j]+1/60, color="red", alpha=0.25)
fig.colorbar(im, ax=ax, label="prediction energy")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_energy_heatmap.png"), dpi=150); plt.close(fig)
print("saved fig5_energy_heatmap.png  (recording:", key, ")")
