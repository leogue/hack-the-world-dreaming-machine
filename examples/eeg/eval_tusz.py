"""EEG — TUSZ seizure probe (Étape 1: get a real number out of the SSL encoder).

Sanity-probe analogue of our energy detector: does the FROZEN encoder linearly
separate *seizure* windows from *background* windows, on HELD-OUT patients?

  * window label: seiz (1) if the window center falls inside a `.csv_bi` `seiz`
    interval, else bckg (0).
  * patient-disjoint by construction: fit the probe on TUSZ `train` patients,
    score on `eval` patients (the official split shares no patients).
  * metric: AUROC (threshold-free, robust to the ~13% seizure imbalance) +
    balanced accuracy. We also probe a RANDOM (untrained) encoder of the same
    architecture as a FLOOR — the SSL number only means something above it.

This is NOT the energy detector yet (that predicts t -> t+1); it answers the
prerequisite question "is there seizure-relevant info in the representation at
all?" before we invest in the predictive/energy route.

Run:  python -m examples.eeg.eval_tusz --ckpt <.../latest.pth.tar>
"""
import glob
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import pyedflib

from examples.eeg.main import build_encoder
from eb_jepa.datasets.eeg.dataset import preprocess_montage

TUSZ = "/lustre/work/pdl17890/udl806719/datasets/Neuro/TUSZ_PREPROCESSED/edf"
CH, SFREQ = 19, 200
WIN = 10 * SFREQ          # 2000 samples = 10 s window (matches training)
MAX_SEC = 300             # read at most first 5 min/recording (bounds cold EDF IO)
# Preprocessing must match the checkpoint's training (set from cfg in main()).
MONTAGE, BAND_LOW, BAND_HIGH = "referential", 0.0, 0.0


def parse_seiz(csv_bi):
    """Return list of (start_sec, stop_sec) seizure intervals from a .csv_bi."""
    intervals = []
    try:
        with open(csv_bi) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("channel"):
                    continue
                p = [c.strip() for c in line.split(",")]
                if len(p) >= 4 and p[3] == "seiz":
                    intervals.append((float(p[1]), float(p[2])))
    except FileNotFoundError:
        pass
    return intervals


def _zscore(x):  # [C, T] per-channel z-score, as in training
    mu = x.mean(1, keepdims=True)
    sd = x.std(1, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def recording_windows(edf, bckg_keep, rng):
    """[(window[C,T], label)] for one recording: all seiz windows + <=bckg_keep bckg."""
    intervals = parse_seiz(edf[:-4] + ".csv_bi")
    try:
        f = pyedflib.EdfReader(edf)
    except Exception:
        return [], []
    try:
        if f.signals_in_file < CH:
            return [], []
        nsamp = int(min(f.getNSamples()[:CH]))
        nsamp = min(nsamp, MAX_SEC * SFREQ)          # cap read length (cold-cache EDF IO)
        if nsamp <= WIN + 1:
            return [], []
        sig = np.empty((CH, nsamp), dtype=np.float32)
        for c in range(CH):
            sig[c] = f.readSignal(c, 0, nsamp)
    finally:
        f._close()
    sig = preprocess_montage(sig, SFREQ, MONTAGE, BAND_LOW, BAND_HIGH)   # referential|bipolar + bandpass

    seiz_w, bckg_w = [], []
    for j in range(nsamp // WIN):                 # non-overlapping 10 s windows
        s = j * WIN
        center = (s + WIN / 2) / SFREQ
        w = _zscore(sig[:, s:s + WIN])
        if any(a <= center < b for a, b in intervals):
            seiz_w.append(w)
        else:
            bckg_w.append(w)
    if bckg_keep is not None and len(bckg_w) > bckg_keep:
        idx = rng.choice(len(bckg_w), bckg_keep, replace=False)
        bckg_w = [bckg_w[k] for k in idx]
    wins = seiz_w + bckg_w
    labs = [1] * len(seiz_w) + [0] * len(bckg_w)
    return wins, labs


@torch.no_grad()
def extract(encoder, split, device, max_rec, bckg_keep, seed=0):
    """Frozen encoder -> (X[N, D], y[N]) window-level features + seiz/bckg labels."""
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(os.path.join(TUSZ, split, "**", "*.edf"), recursive=True))
    rng.shuffle(files)
    if max_rec:
        files = files[:max_rec]
    X, y = [], []
    t0, n_seiz = time.time(), 0
    for i, edf in enumerate(files):
        wins, labs = recording_windows(edf, bckg_keep, rng)
        if not wins:
            continue
        batch = torch.from_numpy(np.stack(wins)).to(device)   # [n, C, T]
        z = encoder.represent(batch).cpu().numpy()            # [n, D]
        X.append(z); y.extend(labs); n_seiz += sum(labs)
        if (i + 1) % 200 == 0:
            print(f"  [{split}] {i+1}/{len(files)} rec, {sum(len(a) for a in X)} win "
                  f"({n_seiz} seiz) {time.time()-t0:.0f}s", flush=True)
    X = np.concatenate(X, 0)
    y = np.array(y)
    print(f"  [{split}] DONE {len(y)} windows, {int(y.sum())} seiz / {int((1-y).sum())} bckg",
          flush=True)
    return X, y


def probe(Xtr, ytr, Xev, yev, tag):
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(Xtr), ytr)
    p = clf.predict_proba(scaler.transform(Xev))[:, 1]
    auroc = roc_auc_score(yev, p)
    bacc = balanced_accuracy_score(yev, (p >= 0.5).astype(int))
    print(f"[{tag}] AUROC={auroc:.4f}  balanced_acc={bacc:.4f}", flush=True)
    return {"tag": tag, "auroc": float(auroc), "balanced_acc": float(bacc)}


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    max_tr = int(sys.argv[sys.argv.index("--max-train") + 1]) if "--max-train" in sys.argv else 400
    max_ev = int(sys.argv[sys.argv.index("--max-eval") + 1]) if "--max-eval" in sys.argv else 400
    bckg_keep = int(sys.argv[sys.argv.index("--bckg-keep") + 1]) if "--bckg-keep" in sys.argv else 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])

    # Preprocessing must match training (montage / bandpass stored in the ckpt cfg).
    global MONTAGE, BAND_LOW, BAND_HIGH
    MONTAGE = cfg.data.get("montage", "referential")
    BAND_LOW = float(cfg.data.get("band_low", 0.0))
    BAND_HIGH = float(cfg.data.get("band_high", 0.0))
    print(f"=== preprocessing: montage={MONTAGE} band=({BAND_LOW},{BAND_HIGH}) ===", flush=True)

    # Trained SSL encoder
    enc = build_encoder(cfg.model).to(device)
    enc.load_state_dict(state["encoder"]); enc.eval()
    print("=== extract features (frozen encoder) ===", flush=True)
    Xtr, ytr = extract(enc, "train", device, max_tr, bckg_keep, seed=0)
    Xev, yev = extract(enc, "eval", device, max_ev, bckg_keep, seed=1)

    print("=== probe (patient-disjoint: fit train, score eval) ===", flush=True)
    probe(Xtr, ytr, Xev, yev, f"SSL encoder ({os.path.basename(os.path.dirname(ckpt))})")
    if "--no-floor" not in sys.argv:
        rnd = build_encoder(cfg.model).to(device); rnd.eval()   # random-init floor
        Xtr_r, _ = extract(rnd, "train", device, max_tr, bckg_keep, seed=0)
        Xev_r, _ = extract(rnd, "eval", device, max_ev, bckg_keep, seed=1)
        probe(Xtr_r, ytr, Xev_r, yev, "RANDOM encoder (floor)")


if __name__ == "__main__":
    main()
