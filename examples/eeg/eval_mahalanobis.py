"""Mahalanobis prediction energy — the 'right' zero-label energy (experiment A).

Euclidean energy ||r||^2 fails on seizures (~0.52) because a seizure shifts the
prediction residual r=z_hat-z in DIRECTIONS, not in total magnitude. A supervised
probe on the residual vector recovers it (0.72). Can we recover it WITHOUT labels?

Mahalanobis energy: E_M(r) = (r-mu)^T Sigma^-1 (r-mu), with mu, Sigma estimated on
NORMAL (bckg) residuals only (zero seizure labels). It upweights directions where the
model normally predicts well -> a deviation there is anomalous. This keeps the
zero-label anomaly-detection story while using the residual's structure.

Run:  python -m examples.eeg.eval_mahalanobis --ckpt <.../energy_tusz_seqnorm/latest.pth.tar>
"""
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from examples.eeg.main_energy import build_energy_jepa
from examples.eeg.eval_energy_probe import extract


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(st["cfg"])
    frame = int(cfg.data.frame_sec * cfg.data.sfreq); norm = cfg.data.get("norm", "frame")
    model = build_energy_jepa(cfg.model).to(device); model.load_state_dict(st["model"]); model.eval()

    print(f"=== Mahalanobis vs Euclidean prediction energy | norm={norm} ===", flush=True)
    Ztr, Rtr, ytr = extract(model, "train", frame, device, norm, 300, 40, seed=0)
    Zev, Rev, yev = extract(model, "eval", frame, device, norm, 300, 40, seed=1)
    print(f"frames: train {len(ytr)} ({int(ytr.sum())} seiz) | eval {len(yev)} ({int(yev.sum())} seiz)", flush=True)

    # Fit normal residual statistics on TRAIN bckg only (zero-label, shrinkage covariance).
    cov = LedoitWolf().fit(Rtr[ytr == 0])

    maha = cov.mahalanobis(Rev)              # squared Mahalanobis distance (higher = anomalous)
    eucl = (Rev ** 2).sum(1)                 # Euclidean energy ||r||^2 (== the scalar energy)
    print(f"[Euclidean energy ||r||^2  ] seizure AUROC={roc_auc_score(yev, eucl):.4f}", flush=True)
    print(f"[Mahalanobis energy (zero-label)] seizure AUROC={roc_auc_score(yev, maha):.4f}", flush=True)

    # Same idea on the encoder representation z_t (OOD-style detector), for context.
    covz = LedoitWolf().fit(Ztr[ytr == 0])
    mahaz = covz.mahalanobis(Zev)
    print(f"[Mahalanobis on z_t (encoder)    ] seizure AUROC={roc_auc_score(yev, mahaz):.4f}", flush=True)
    print("\n>>> ref: scalar energy ~0.52 | supervised residual-probe 0.72 | two-view 0.81", flush=True)


if __name__ == "__main__":
    main()
