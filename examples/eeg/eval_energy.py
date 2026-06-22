"""EEG energy detector eval — Étape 2 (H1): is prediction energy a seizure detector?

For each held-out (eval) recording, cut it into consecutive 2 s frames, encode the
latent trajectory, and read the per-frame prediction energy ||ẑ_{t+1}-z_{t+1}||².
That energy is the ZERO-LABEL anomaly score. We then compare it to the seizure
labels (.csv_bi) at FRAME level and report AUROC, against:
  * a RANDOM-init energy model (floor),
  * (for context) the Étape-1 two-view probe: AUROC 0.756 / random 0.716.

Also dumps the energy curve + seizure mask for a few seizure recordings (money-shot).

Run:  python -m examples.eeg.eval_energy --ckpt <.../energy_tusz/latest.pth.tar>
"""
import glob
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

import pyedflib

from examples.eeg.eval_tusz import parse_seiz
from examples.eeg.main_energy import build_energy_jepa

TUSZ = "/lustre/work/pdl17890/udl806719/datasets/Neuro/TUSZ_PREPROCESSED/edf"
CH = 19


MAX_SEC = 600   # read at most the first 10 min of each recording (bounds EDF IO)


@torch.no_grad()
def recording_energy(model, edf, frame, device, norm="frame"):
    """-> (energy[n-1], seiz_label[n-1]) at frame level, or (None, None)."""
    intervals = parse_seiz(edf[:-4] + ".csv_bi")
    try:
        f = pyedflib.EdfReader(edf)
    except Exception:
        return None, None
    try:
        if f.signals_in_file < CH:
            return None, None
        nsamp = int(min(f.getNSamples()[:CH]))
        nsamp = min(nsamp, MAX_SEC * 200)            # cap duration
        nfr = nsamp // frame
        if nfr < 3:
            return None, None
        sig = np.empty((CH, nfr * frame), dtype=np.float32)
        for c in range(CH):
            sig[c] = f.readSignal(c, 0, nfr * frame)
    finally:
        f._close()

    if norm == "seq":      # per-channel over the whole read -> amplitude survives
        mu = sig.mean(1, keepdims=True); sd = sig.std(1, keepdims=True) + 1e-6
        x = ((sig - mu) / sd).reshape(CH, nfr, frame).astype(np.float32)
    else:                  # "frame": z-score each frame independently
        x = sig.reshape(CH, nfr, frame)
        mu = x.mean(2, keepdims=True); sd = x.std(2, keepdims=True) + 1e-6
        x = ((x - mu) / sd).astype(np.float32)
    seq = torch.from_numpy(x).unsqueeze(0).to(device)     # [1, C, nfr, L]

    z = model.encoder.frames(seq)                          # [1, D, nfr, 1, 1]
    D, T = z.shape[1], z.shape[2]
    src = z[:, :, :-1].reshape(D, T - 1).T.reshape(T - 1, D, 1, 1, 1)  # each frame as own sample
    a = torch.zeros(T - 1, model.action_dim, 1, device=device)
    pred = model.predictor(src, a)                         # [T-1, D, 1, 1, 1]
    tgt = z[:, :, 1:].reshape(D, T - 1).T.reshape(T - 1, D, 1, 1, 1)
    energy = ((pred - tgt) ** 2).flatten(1).mean(1).cpu().numpy()      # [T-1]

    # frame j+1 label = its center inside a seiz interval
    centers = (np.arange(1, T) * frame + frame / 2) / 200.0
    lab = np.array([any(s <= c < e for s, e in intervals) for c in centers], dtype=int)
    return energy, lab


def _eval_files(n_bckg=150, seed=0):
    """All seizure recordings (positives) + a sample of seizure-free ones."""
    files = sorted(glob.glob(os.path.join(TUSZ, "eval", "**", "*.edf"), recursive=True))
    seiz = [f for f in files if parse_seiz(f[:-4] + ".csv_bi")]
    bckg = [f for f in files if f not in set(seiz)]
    rng = np.random.default_rng(seed)
    rng.shuffle(bckg)
    return seiz + bckg[:n_bckg]


def collect(model, frame, device, files, norm="frame"):
    E, Y, money = [], [], []
    for edf in files:
        e, y = recording_energy(model, edf, frame, device, norm)
        if e is None:
            continue
        E.append(e); Y.append(y)
        if y.sum() > 0 and len(money) < 4:                 # keep a few seizure recs for plotting
            money.append((os.path.basename(edf), e, y))
    return np.concatenate(E), np.concatenate(Y), money


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    frame = int(cfg.data.frame_sec * cfg.data.sfreq)
    norm = cfg.data.get("norm", "frame")

    model = build_energy_jepa(cfg.model).to(device)
    model.load_state_dict(state["model"]); model.eval()
    rnd = build_energy_jepa(cfg.model).to(device); rnd.eval()   # floor

    files = _eval_files()
    print(f"=== {len(files)} eval recordings (all seizure + 150 bckg sample) | norm={norm} ===", flush=True)

    E, Y, money = collect(model, frame, device, files, norm)
    print(f"frames: {len(Y)}  seiz={int(Y.sum())}  bckg={int((1-Y).sum())}", flush=True)
    print(f"[ENERGY trained]  AUROC={roc_auc_score(Y, E):.4f}", flush=True)

    Er, Yr, _ = collect(rnd, frame, device, files, norm)
    print(f"[ENERGY random ]  AUROC={roc_auc_score(Yr, Er):.4f}", flush=True)

    out = os.path.join(os.path.dirname(ckpt), "moneyshot.npz")
    np.savez(out, **{f"rec{i}_{n}": np.stack([e, y]) for i, (n, e, y) in enumerate(money)})
    print(f"[money-shot] saved {len(money)} seizure recordings -> {out}", flush=True)


if __name__ == "__main__":
    main()
