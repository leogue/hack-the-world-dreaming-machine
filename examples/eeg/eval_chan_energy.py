"""Per-channel energy detector eval (exp B) — does localizing fix the dilution?

The channel-wise world model gives prediction energy per (channel, time). We aggregate
over channels (max / mean / q90) into a per-frame seizure score and compute AUROC on
held-out patients. If max-over-channels >> whole-brain energy (0.52), the whole-brain
dilution was the culprit. Also dumps a channel x time energy heatmap for a seizure
recording (money-shot). Optionally a zero-label Mahalanobis per-channel score.

Run:  python -m examples.eeg.eval_chan_energy --ckpt <.../chan_energy/latest.pth.tar>
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
from examples.eeg.main_chan_energy import build_chan_energy_jepa

TUSZ = "/lustre/work/pdl17890/udl806719/datasets/Neuro/TUSZ_PREPROCESSED/edf"
CH, SFREQ, MAX_SEC = 19, 200, 300


@torch.no_grad()
def recording_chan_energy(model, edf, frame, device, norm):
    """-> (E[C, nfr-1] per-channel energy, y[nfr-1] seiz label) or None."""
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
    seq = torch.from_numpy(x).unsqueeze(0).to(device)         # [1, C, nfr, L]
    z = model.frames(seq)                                     # [1, C, nfr, D]
    E = model.energy(z)[0].cpu().numpy()                      # [C, nfr-1]
    y = np.array([int(any(s <= (j * frame + frame / 2) / SFREQ < e for s, e in intervals))
                  for j in range(1, nfr)])
    return E, y


def collect(model, frame, device, norm, files, keep_heat=0):
    aggs = {"max": [], "mean": [], "q90": []}
    Y, heat = [], []
    for edf in files:
        out = recording_chan_energy(model, edf, frame, device, norm)
        if out is None:
            continue
        E, y = out
        aggs["max"].append(E.max(0)); aggs["mean"].append(E.mean(0))
        aggs["q90"].append(np.quantile(E, 0.9, axis=0))
        Y.append(y)
        if keep_heat and y.sum() > 0 and len(heat) < keep_heat:
            heat.append((os.path.basename(edf), E, y))
    Y = np.concatenate(Y)
    return {k: np.concatenate(v) for k, v in aggs.items()}, Y, heat


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(st["cfg"])
    frame = int(cfg.data.frame_sec * cfg.data.sfreq); norm = cfg.data.get("norm", "frame")
    model = build_chan_energy_jepa(cfg.model).to(device); model.load_state_dict(st["model"]); model.eval()
    rnd = build_chan_energy_jepa(cfg.model).to(device); rnd.eval()

    files = sorted(glob.glob(os.path.join(TUSZ, "eval", "**", "*.edf"), recursive=True))
    np.random.default_rng(1).shuffle(files); files = files[:300]
    print(f"=== per-channel energy detector | {len(files)} eval rec | norm={norm} ===", flush=True)

    aggs, Y, heat = collect(model, frame, device, norm, files, keep_heat=4)
    print(f"frames: {len(Y)}  seiz={int(Y.sum())}", flush=True)
    for k in ("max", "mean", "q90"):
        print(f"[trained] channel-energy {k}-over-channels: seizure AUROC={roc_auc_score(Y, aggs[k]):.4f}", flush=True)
    aggs_r, Yr, _ = collect(rnd, frame, device, norm, files)
    print(f"[random ] channel-energy max-over-channels: seizure AUROC={roc_auc_score(Yr, aggs_r['max']):.4f}", flush=True)

    out = os.path.join(os.path.dirname(ckpt), "chan_heatmap.npz")
    np.savez(out, **{f"rec{i}_{n}": np.vstack([E, y[None]]) for i, (n, E, y) in enumerate(heat)})
    print(f"[heatmap] {len(heat)} seizure recordings -> {out}", flush=True)
    print(">>> ref: whole-brain energy 0.52 | Mahalanobis 0.67 | two-view 0.81", flush=True)


if __name__ == "__main__":
    main()
