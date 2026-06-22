"""Artifact energy eval — the world-model boundary test (original contribution).

Thesis: a JEPA world model's prediction energy detects UNPREDICTABLE anomalies but
not PREDICTABLE ones. We already showed it FAILS on seizures (rhythmic -> predictable,
AUROC ~0.52). Here we test the complementary case: muscle/eye artifacts (erratic ->
unpredictable). Prediction: the SAME frozen energy model gives a HIGH artifact AUROC.

Same world model (trained on TUSZ normal EEG, no labels), evaluated on TUAR (the
dedicated artifact corpus). Per-frame prediction energy = zero-label anomaly score.
- artifact frame: its center falls in a `musc`/`eyem` event (any channel).
- clean frame: its center is NOT covered by ANY annotation (TUAR annotates artifacts;
  unannotated = clean EEG).
TUAR EDF has 23 channels; the first 19 match the model's montage exactly -> take [:19].

Run:  python -m examples.eeg.eval_tuar_artifact --ckpt <.../energy_tusz_seqnorm/latest.pth.tar>
"""
import glob
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

import pyedflib

from examples.eeg.main_energy import build_energy_jepa

TUAR = "/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAR-TUEP/TUAR_PREPROCESSED/edf"
CH, SFREQ, MAX_SEC = 19, 200, 600


def parse_intervals(csv):
    """-> (musc_intervals, eyem_intervals, any_annotation_intervals) in seconds."""
    musc, eyem, anyev = [], [], []
    try:
        with open(csv) as f:
            for line in f:
                p = [c.strip() for c in line.strip().split(",")]
                if len(p) < 4 or p[0] == "channel":
                    continue
                s, e, lab = float(p[1]), float(p[2]), p[3]
                anyev.append((s, e))
                if "musc" in lab:      # muscle: high-frequency, most unpredictable
                    musc.append((s, e))
                if "eyem" in lab:      # eye movement: slower
                    eyem.append((s, e))
    except FileNotFoundError:
        pass
    return musc, eyem, anyev


def _covered(c, intervals):
    return any(s <= c < e for s, e in intervals)


@torch.no_grad()
def recording_energy(model, edf, frame, device, norm):
    musc, eyem, anyev = parse_intervals(edf[:-4] + ".csv")
    try:
        f = pyedflib.EdfReader(edf)
    except Exception:
        return None, None
    try:
        if f.signals_in_file < CH:
            return None, None
        nsamp = min(int(min(f.getNSamples()[:CH])), MAX_SEC * SFREQ)
        nfr = nsamp // frame
        if nfr < 3:
            return None, None
        sig = np.empty((CH, nfr * frame), dtype=np.float32)
        for c in range(CH):
            sig[c] = f.readSignal(c, 0, nfr * frame)     # first 19 channels = model montage
    finally:
        f._close()

    if norm == "seq":
        mu = sig.mean(1, keepdims=True); sd = sig.std(1, keepdims=True) + 1e-6
        x = ((sig - mu) / sd).reshape(CH, nfr, frame).astype(np.float32)
    else:
        x = sig.reshape(CH, nfr, frame)
        mu = x.mean(2, keepdims=True); sd = x.std(2, keepdims=True) + 1e-6
        x = ((x - mu) / sd).astype(np.float32)
    seq = torch.from_numpy(x).unsqueeze(0).to(device)        # [1, C, nfr, L]

    z = model.encoder.frames(seq)                            # [1, D, nfr, 1, 1]
    D, T = z.shape[1], z.shape[2]
    src = z[:, :, :-1].reshape(D, T - 1).T.reshape(T - 1, D, 1, 1, 1)
    a = torch.zeros(T - 1, model.action_dim, 1, device=device)
    pred = model.predictor(src, a)
    tgt = z[:, :, 1:].reshape(D, T - 1).T.reshape(T - 1, D, 1, 1, 1)
    energy = ((pred - tgt) ** 2).flatten(1).mean(1).cpu().numpy()   # [T-1]

    E, L = [], []   # label: 2=musc, 1=eyem(only), 0=clean (else skip)
    for j in range(1, T):                                    # frame j+1 = energy[j-1]
        c = (j * frame + frame / 2) / SFREQ
        if _covered(c, musc):
            E.append(energy[j - 1]); L.append(2)
        elif _covered(c, eyem):
            E.append(energy[j - 1]); L.append(1)
        elif not _covered(c, anyev):                         # truly clean (unannotated)
            E.append(energy[j - 1]); L.append(0)
    return E, L


def collect(model, frame, device, norm, files):
    E, L = [], []
    for edf in files:
        e, l = recording_energy(model, edf, frame, device, norm)
        if e:
            E += e; L += l
    return np.array(E), np.array(L)


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    frame = int(cfg.data.frame_sec * cfg.data.sfreq)
    norm = cfg.data.get("norm", "frame")

    model = build_energy_jepa(cfg.model).to(device)
    model.load_state_dict(state["model"]); model.eval()
    rnd = build_energy_jepa(cfg.model).to(device); rnd.eval()

    files = sorted(f for f in glob.glob(os.path.join(TUAR, "*.edf")) if "_seiz" not in f)
    print(f"=== TUAR {len(files)} recordings | norm={norm} | predictability gradient ===", flush=True)

    def report(tag, E, L):
        clean = E[L == 0]
        for code, name in [(2, "musc (muscle, high-freq)"), (1, "eyem (eye, slower)")]:
            pos = E[L == code]
            if len(pos) and len(clean):
                y = np.r_[np.ones(len(pos)), np.zeros(len(clean))]
                s = np.r_[pos, clean]
                print(f"  [{tag}] {name:26s} vs clean: AUROC={roc_auc_score(y, s):.4f} "
                      f"(n_pos={len(pos)})", flush=True)
        both = E[L >= 1]
        y = np.r_[np.ones(len(both)), np.zeros(len(clean))]
        print(f"  [{tag}] {'musc+eyem combined':26s} vs clean: AUROC={roc_auc_score(y, np.r_[both, clean]):.4f}", flush=True)

    E, L = collect(model, frame, device, norm, files)
    print(f"frames: musc={int((L==2).sum())} eyem={int((L==1).sum())} clean={int((L==0).sum())}", flush=True)
    print("--- trained world model ---"); report("trained", E, L)
    Er, Lr = collect(rnd, frame, device, norm, files)
    print("--- random floor ---"); report("random", Er, Lr)
    print(f"\n>>> SEIZURE energy AUROC (TUSZ, same model) ~0.52 — predictable anomaly.\n"
          f">>> thesis = gradient: muscle > eye > seizure (more unpredictable -> higher energy)",
          flush=True)


if __name__ == "__main__":
    main()
