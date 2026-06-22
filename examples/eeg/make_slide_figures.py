"""Slide-ready training-curve figures, parsed from the real SLURM logs.
Run from repo root:  python -m examples.eeg.make_slide_figures
"""
import ast
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.2})
OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
RE = re.compile(r"epoch (\d+) loss=([\d.eE+-]+) (\{[^}]*\})")
GREEN, BLUE, RED, ORANGE = "#27ae60", "#2a7fb8", "#c0392b", "#e67e22"


def load(jobid):
    f = glob.glob(f"eeg_*_{jobid}.out")[0]
    ep, loss, d = [], [], []
    for line in open(f):
        m = RE.search(line)
        if m:
            ep.append(int(m.group(1))); loss.append(float(m.group(2))); d.append(ast.literal_eval(m.group(3)))
    return ep, loss, d


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved", name)


# ---- 1. H2 collapse money-shot: variance dies when std_coeff=0 ----
ev, _, dv = load("75213")   # VICReg healthy
ec, _, dc = load("75222")   # collapse std=0
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
a1.plot(ev, [x["var_loss"] for x in dv], color=GREEN, marker="o", ms=3, label="VICReg (healthy)")
a1.plot(ec, [x["var_loss"] for x in dc], color=RED, marker="o", ms=3, label="Collapse (std_coeff=0)")
a1.set(xlabel="epoch", ylabel="variance loss  (hinge std)", title="Removing the variance term collapses the encoder")
a1.legend(frameon=False)
a2.plot(ev, [x["invariance_loss"] for x in dv], color=GREEN, marker="o", ms=3, label="VICReg (healthy)")
a2.plot(ec, [x["invariance_loss"] for x in dc], color=RED, marker="o", ms=3, label="Collapse (std_coeff=0)")
a2.set(xlabel="epoch", ylabel="invariance loss  (MSE of views)", title="Collapse satisfies invariance trivially (→ 0)", yscale="log")
a2.legend(frameon=False)
save(fig, "slide_collapse.png")

# ---- 2. VICReg vs SIGReg: both converge cleanly (two-view, no collapse) ----
ev2, lv, _ = load("75213"); es2, ls, _ = load("75214")
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(ev2, lv, color=GREEN, marker="o", ms=3, label="VICReg")
ax.plot(es2, ls, color=BLUE, marker="o", ms=3, label="SIGReg (BCS)")
ax.set(xlabel="epoch", ylabel="total SSL loss", title="Two-view SSL converges — VICReg vs SIGReg")
ax.legend(frameon=False)
save(fig, "slide_vicreg_vs_sigreg.png")

# ---- 3. Energy world model: prediction energy ↓ on normal EEG ----
ee, le, de = load("75176")
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(ee, le, color="#9e9e9e", marker="o", ms=3, label="total loss")
ax.plot(ee, [x["energy"] for x in de], color=RED, marker="o", ms=3, label="prediction energy  ‖ẑ−z‖²")
ax.set(xlabel="epoch", ylabel="loss", title="Energy world model trains fine on normal EEG")
ax.legend(frameon=False)
save(fig, "slide_energy_training.png")

# ---- 4. Long 3h run: smooth convergence (out_dim 512, 1328 epochs) ----
e3, l3, _ = load("75485")
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(e3, l3, color=GREEN, lw=1.4)
ax.set(xlabel="epoch", ylabel="total SSL loss", title="VICReg, 3 h run (1328 epochs) — smooth, no collapse", yscale="log")
save(fig, "slide_vicreg_3h.png")
