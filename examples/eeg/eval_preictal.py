"""Seizure PREDICTION (not detection): pre-ictal vs interictal — short horizon.

Detection asks "is this window a seizure?" (we get ~0.81). Prediction asks "is a
seizure COMING?" -- classify the PRE-ICTAL lead-up (the LEAD seconds before onset, the
seizure itself EXCLUDED) vs INTERICTAL baseline (far from any seizure). Frozen two-view
encoder features + linear probe, patient-disjoint. Any AUROC > chance = we can forecast
seizures short-horizon. (TUSZ recordings are minutes, so horizon is ~tens of seconds,
not hours.)

Anti-confound: interictal windows are drawn from the SAME seizure recordings (far from
the seizure) AND from seizure-free recordings, so the probe can't just learn
"seizure-patient vs not".

Run:  python -m examples.eeg.eval_preictal --ckpt <.../solid_sigreg/latest.pth.tar>
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
from examples.eeg.main import build_encoder

TUSZ = "/lustre/work/pdl17890/udl806719/datasets/Neuro/TUSZ_PREPROCESSED/edf"
CH, SFREQ, WIN = 19, 200, 10 * 200      # 10 s windows
MAX_SEC = 900
LEAD = 60.0     # pre-ictal horizon: window within [onset-LEAD, onset) -> positive
FAR = 180.0     # interictal: window > FAR from any seizure -> negative


def _zscore(x):
    mu = x.mean(1, keepdims=True); sd = x.std(1, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


@torch.no_grad()
def recording_windows(encoder, edf, device, rng, interictal_keep):
    """-> (feats[n,D], y[n]) with y=1 pre-ictal, y=0 interictal (ictal/buffer skipped)."""
    iv = parse_seiz(edf[:-4] + ".csv_bi")
    onsets = [s for s, e in iv]
    try:
        f = pyedflib.EdfReader(edf)
    except Exception:
        return None
    try:
        if f.signals_in_file < CH:
            return None
        nsamp = min(int(min(f.getNSamples()[:CH])), MAX_SEC * SFREQ)
        nw = nsamp // WIN
        if nw < 2:
            return None
        sig = np.empty((CH, nw * WIN), dtype=np.float32)
        for c in range(CH):
            sig[c] = f.readSignal(c, 0, nw * WIN)
    finally:
        f._close()

    def dist_to_seiz(s0, s1):
        if not iv:
            return 1e9
        ds = []
        for a, b in iv:
            if s1 <= a:
                ds.append(a - s1)        # window before this seizure
            elif s0 >= b:
                ds.append(s0 - b)        # window after this seizure
            else:
                ds.append(0.0)           # overlap (shouldn't reach here)
        return min(ds)

    wins, labs = [], []
    for j in range(nw):
        s0, s1 = j * WIN / SFREQ, (j + 1) * WIN / SFREQ        # window [s0, s1) in seconds
        if any(a < s1 and s0 < b for a, b in iv):              # overlaps a seizure -> ictal, skip
            continue
        d = dist_to_seiz(s0, s1)                               # distance to nearest seizure (s)
        lead = min(((o - s1) for o in onsets if o >= s1), default=1e9)  # lead-in to a future onset
        if 0 <= lead <= LEAD:
            wins.append(_zscore(sig[:, j * WIN:(j + 1) * WIN])); labs.append(1)   # pre-ictal
        elif d > FAR:
            wins.append(_zscore(sig[:, j * WIN:(j + 1) * WIN])); labs.append(0)   # interictal
    if not wins:
        return None
    # cap interictal per recording (pre-ictal is rare -> keep all)
    idx_pos = [i for i, l in enumerate(labs) if l == 1]
    idx_neg = [i for i, l in enumerate(labs) if l == 0]
    if interictal_keep and len(idx_neg) > interictal_keep:
        idx_neg = list(rng.choice(idx_neg, interictal_keep, replace=False))
    keep = idx_pos + idx_neg
    X = np.stack([wins[i] for i in keep])
    z = encoder.represent(torch.from_numpy(X).to(device)).cpu().numpy()
    return z, np.array([labs[i] for i in keep])


def extract(encoder, split, device, max_rec, seed, interictal_keep):
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(os.path.join(TUSZ, split, "**", "*.edf"), recursive=True))
    rng.shuffle(files)
    if max_rec:
        files = files[:max_rec]
    X, Y = [], []
    for edf in files:
        out = recording_windows(encoder, edf, device, rng, interictal_keep)
        if out:
            X.append(out[0]); Y.append(out[1])
    return np.concatenate(X), np.concatenate(Y)


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(st["cfg"])
    enc = build_encoder(cfg.model).to(device); enc.load_state_dict(st["encoder"]); enc.eval()
    rnd = build_encoder(cfg.model).to(device); rnd.eval()

    print(f"=== seizure PREDICTION: pre-ictal[onset-{LEAD:.0f}s,onset) vs interictal(>{FAR:.0f}s) ===", flush=True)
    Xtr, ytr = extract(enc, "train", device, 500, 0, interictal_keep=8)
    Xev, yev = extract(enc, "eval", device, 500, 1, interictal_keep=8)
    print(f"train: {len(ytr)} ({int(ytr.sum())} pre-ictal) | eval: {len(yev)} ({int(yev.sum())} pre-ictal)", flush=True)

    def probe(Xtr, Xev, tag):
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        print(f"[{tag}] pre-ictal AUROC={roc_auc_score(yev, clf.predict_proba(sc.transform(Xev))[:,1]):.4f}", flush=True)

    probe(Xtr, Xev, f"SSL encoder ({os.path.basename(os.path.dirname(ckpt))})")
    Xtr_r, _ = extract(rnd, "train", device, 500, 0, 8)
    Xev_r, _ = extract(rnd, "eval", device, 500, 1, 8)
    probe(Xtr_r, Xev_r, "RANDOM encoder (floor)")
    print(">>> ref: seizure DETECTION (during seizure) ~0.81 | chance 0.5", flush=True)


if __name__ == "__main__":
    main()
