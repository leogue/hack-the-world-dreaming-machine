"""Probe the world model's INTERNAL STATE for seizures (user Q2).

The scalar prediction energy ||ẑ_{t+1}-z_{t+1}||² fails on seizures (~0.52). But does
the world model's internal state encode the seizure even if its energy does not? We
linear-probe two internal signals of the frozen energy model (TUSZ-trained), at the
2 s frame level, patient-disjoint:
  * Z  = the ENERGY-trained encoder's representation z_t  (does its perception encode it?)
  * R  = the per-dimension residual r_t = ẑ_{t+1} - z_{t+1}  (the energy is just ||R||²;
         a linear probe can weight R's directions -> more than the scalar)
Compare to: scalar energy ~0.52, and the two-view encoder probe ~0.81.

Run:  python -m examples.eeg.eval_energy_probe --ckpt <.../energy_tusz_seqnorm/latest.pth.tar>
"""
import glob
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import pyedflib

from examples.eeg.eval_tusz import parse_seiz
from examples.eeg.main_energy import build_energy_jepa

TUSZ = "/lustre/work/pdl17890/udl806719/datasets/Neuro/TUSZ_PREPROCESSED/edf"
CH, SFREQ, MAX_SEC = 19, 200, 300


@torch.no_grad()
def recording_feats(model, edf, frame, device, norm, bckg_keep, rng):
    """-> (Z[n,D], R[n,D], y[n]) per-frame internal features + seiz label."""
    intervals = parse_seiz(edf[:-4] + ".csv_bi")
    try:
        f = pyedflib.EdfReader(edf)
    except Exception:
        return None
    try:
        if f.signals_in_file < CH:
            return None
        nsamp = min(int(min(f.getNSamples()[:CH])), MAX_SEC * SFREQ)
        nfr = nsamp // frame
        if nfr < 4:
            return None
        sig = np.empty((CH, nfr * frame), dtype=np.float32)
        for c in range(CH):
            sig[c] = f.readSignal(c, 0, nfr * frame)
    finally:
        f._close()

    if norm == "seq":
        mu = sig.mean(1, keepdims=True); sd = sig.std(1, keepdims=True) + 1e-6
        x = ((sig - mu) / sd).reshape(CH, nfr, frame).astype(np.float32)
    else:
        x = sig.reshape(CH, nfr, frame)
        mu = x.mean(2, keepdims=True); sd = x.std(2, keepdims=True) + 1e-6
        x = ((x - mu) / sd).astype(np.float32)
    seq = torch.from_numpy(x).unsqueeze(0).to(device)

    z = model.encoder.frames(seq)                         # [1, D, nfr, 1, 1]
    D, T = z.shape[1], z.shape[2]
    src = z[:, :, :-1].reshape(D, T - 1).T.reshape(T - 1, D, 1, 1, 1)
    a = torch.zeros(T - 1, model.action_dim, 1, device=device)
    pred = model.predictor(src, a).reshape(T - 1, D).cpu().numpy()
    zt = z.reshape(D, T).T.cpu().numpy()                   # [T, D]
    resid = pred - zt[1:]                                  # [T-1, D], aligned to frames 1..T-1

    # frame f = 1..T-1 : Z=zt[f], R=resid[f-1], label = seiz(center of frame f)
    Z, R, y = [], [], []
    s_idx, b_idx = [], []
    for f_ in range(1, T):
        c = (f_ * frame + frame / 2) / SFREQ
        lab = int(any(s <= c < e for s, e in intervals))
        (s_idx if lab else b_idx).append(f_)
    if bckg_keep and len(b_idx) > bckg_keep:
        b_idx = list(rng.choice(b_idx, bckg_keep, replace=False))
    for f_ in s_idx + b_idx:
        Z.append(zt[f_]); R.append(resid[f_ - 1])
        y.append(int(f_ in s_idx))
    if not y:
        return None
    return np.stack(Z), np.stack(R), np.array(y)


def extract(model, split, frame, device, norm, max_rec, bckg_keep, seed):
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(os.path.join(TUSZ, split, "**", "*.edf"), recursive=True))
    rng.shuffle(files)
    if max_rec:
        files = files[:max_rec]
    Z, R, Y = [], [], []
    for edf in files:
        out = recording_feats(model, edf, frame, device, norm, bckg_keep, rng)
        if out:
            Z.append(out[0]); R.append(out[1]); Y.append(out[2])
    return np.concatenate(Z), np.concatenate(R), np.concatenate(Y)


def probe(Xtr, ytr, Xev, yev, tag):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    auroc = roc_auc_score(yev, clf.predict_proba(sc.transform(Xev))[:, 1])
    print(f"[{tag}] seizure AUROC={auroc:.4f}", flush=True)


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(st["cfg"])
    frame = int(cfg.data.frame_sec * cfg.data.sfreq); norm = cfg.data.get("norm", "frame")
    model = build_energy_jepa(cfg.model).to(device); model.load_state_dict(st["model"]); model.eval()

    print(f"=== probe world-model internal state | norm={norm}, frame={frame} ===", flush=True)
    Ztr, Rtr, ytr = extract(model, "train", frame, device, norm, 300, 20, seed=0)
    Zev, Rev, yev = extract(model, "eval", frame, device, norm, 300, 20, seed=1)
    print(f"frames: train {len(ytr)} ({int(ytr.sum())} seiz) | eval {len(yev)} ({int(yev.sum())} seiz)", flush=True)
    probe(Ztr, ytr, Zev, yev, "Z  energy-encoder repr")
    probe(Rtr, ytr, Rev, yev, "R  residual vector (energy=||R||^2)")
    probe(np.c_[Ztr, Rtr], ytr, np.c_[Zev, Rev], yev, "Z+R concat")
    print("\n>>> compare: scalar energy ~0.52 | two-view encoder ~0.81", flush=True)


if __name__ == "__main__":
    main()
