"""TUSZ sequence dataset for the predictive (energy) JEPA — Étape 2.

Each item is a SEQUENCE of `seq_len` consecutive short frames read from one
recording: shape [C, seq_len, frame_len], per-frame z-scored. The encoder turns
each frame into a latent vector -> a latent trajectory the predictor rolls over.

`bckg_only=True` keeps only recordings with NO seizure interval (the ~87% pure-
normal recordings) -> we pretrain the world model on normal brain dynamics only,
which is what makes the energy a zero-label anomaly signal at eval time.
"""
import glob
import os
from dataclasses import dataclass

import numpy as np
import torch

import pyedflib

from examples.eeg.eval_tusz import parse_seiz


@dataclass
class SeqConfig:
    data_root: str = "/lustre/work/pdl17890/udl806719/datasets/Neuro/TUSZ_PREPROCESSED/edf"
    split: str = "train"
    n_channels: int = 19
    sfreq: int = 200
    frame_sec: float = 2.0       # one latent frame = 2 s
    seq_len: int = 16            # frames per sequence (16 * 2 s = 32 s context)
    epoch_size: int = 4000       # random sequences per epoch
    batch_size: int = 128
    num_workers: int = 16
    bckg_only: bool = True       # keep only seizure-free recordings (pretrain on normal)
    norm: str = "frame"          # "frame" = z-score each 2s frame (kills amplitude);
                                 # "seq"   = z-score per channel over the whole sequence
                                 #            (inter-frame amplitude survives)


def _list_bckg_only(root, split, bckg_only):
    files = sorted(glob.glob(os.path.join(root, split, "**", "*.edf"), recursive=True))
    if not bckg_only:
        return files
    keep = [f for f in files if not parse_seiz(f[:-4] + ".csv_bi")]  # no seiz interval
    if not keep:
        raise RuntimeError(f"no seizure-free recordings under {root}/{split}")
    return keep


class TUSZSeqDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: SeqConfig):
        self.cfg = cfg
        self.frame = int(cfg.frame_sec * cfg.sfreq)        # 400 samples
        self.total = self.frame * cfg.seq_len              # samples per sequence
        self.files = _list_bckg_only(cfg.data_root, cfg.split, cfg.bckg_only)
        self._rng = np.random.default_rng()

    def __len__(self):
        return self.cfg.epoch_size

    def _read_seq(self):
        cfg = self.cfg
        for _ in range(8):                                 # retry on short/bad files
            path = self.files[self._rng.integers(len(self.files))]
            try:
                f = pyedflib.EdfReader(path)
            except Exception:
                continue
            try:
                if f.signals_in_file < cfg.n_channels:
                    continue
                nsamp = int(min(f.getNSamples()[:cfg.n_channels]))
                if nsamp <= self.total + 1:
                    continue
                start = int(self._rng.integers(0, nsamp - self.total))
                x = np.empty((cfg.n_channels, self.total), dtype=np.float32)
                for c in range(cfg.n_channels):
                    x[c] = f.readSignal(c, start, self.total)
            finally:
                f._close()
            if cfg.norm == "seq":   # per-channel over the whole 32s -> amplitude survives
                mu = x.mean(1, keepdims=True); sd = x.std(1, keepdims=True) + 1e-6
                x = (x - mu) / sd
                x = x.reshape(cfg.n_channels, cfg.seq_len, self.frame)
            else:                   # "frame": z-score each 2s frame independently
                x = x.reshape(cfg.n_channels, cfg.seq_len, self.frame)
                mu = x.mean(2, keepdims=True); sd = x.std(2, keepdims=True) + 1e-6
                x = (x - mu) / sd
            return x.astype(np.float32)
        return None

    def __getitem__(self, i):
        self._rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
        x = self._read_seq()
        if x is None:
            x = np.zeros((self.cfg.n_channels, self.cfg.seq_len, self.frame), dtype=np.float32)
        return torch.from_numpy(x)                          # [C, seq_len, frame]


def make_seq_loader(cfg: SeqConfig):
    ds = TUSZSeqDataset(cfg)
    return torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=cfg.num_workers > 0)
