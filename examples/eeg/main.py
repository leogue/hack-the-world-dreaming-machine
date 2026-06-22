"""EEG — SSL pretraining entrypoint (self-supervised representation learning).

Research question: can two-view invariance learning on unlabeled EEG learn
features that linearly separate *normal vs abnormal* recordings, generalizing
to held-out (patient-disjoint) subjects?

The DATA + TRAINING LOOP are provided. The two modelling pieces you implement
are marked `# TODO` below — that is the whole point of the track:
  1. the 1D encoder over [B, C=19, T]
  2. the SSL objective (two-view VICReg  *or*  predictive JEPA)
The downstream probe + metric is the third `# TODO`, in eval.py.

Run:  python -m examples.eeg.main --fname examples/eeg/cfgs/train.yaml
"""
import math
import os
import sys
import time

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, make_loader

# Reuse the eb_jepa core — DO NOT reimplement these:
#   eb_jepa.architectures: Projector (MLP), RNNPredictor (GRU)
#   eb_jepa.losses:        VICRegLoss (inv+var+cov), BCS (SIGReg), VCLoss
from eb_jepa.architectures import Projector
from eb_jepa.losses import BCS, VICRegLoss


# --------------------------------------------------------------------------- #
# 1) ENCODER  — 1D strided-conv stack over [B, C=19, T]
# --------------------------------------------------------------------------- #
class Conv1dEncoder(nn.Module):
    """Strided Conv1d stack: [B, C, T] -> [B, D]. Each block halves time
    (kernel 7, stride 2, BatchNorm + GELU); a final global average pool over
    time gives one [B, D] representation per window."""

    def __init__(self, in_channels=19, out_dim=256, hidden=64, depth=4):
        super().__init__()
        chans = [in_channels] + [min(hidden * 2**i, out_dim) for i in range(depth - 1)] + [out_dim]
        blocks = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            blocks += [
                nn.Conv1d(cin, cout, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(cout),
                nn.GELU(),
            ]
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = out_dim

    def represent(self, x):
        """[B, C, T] -> [B, D] (time-pooled global representation)."""
        h = self.net(x)              # [B, D, T']
        return self.pool(h).squeeze(-1)  # [B, D]

    def forward(self, x):
        return self.represent(x)


def build_encoder(cfg):
    """1D encoder over [B, C=n_channels, T] -> [B, D]; exposes .represent / .out_dim."""
    return Conv1dEncoder(
        in_channels=cfg.in_channels,
        out_dim=cfg.out_dim,
        hidden=cfg.get("hidden", 64),
        depth=cfg.get("depth", 4),
    )


# --------------------------------------------------------------------------- #
# 2) SSL OBJECTIVE  — two-view invariance (VICReg or SIGReg), config-switched
# --------------------------------------------------------------------------- #
class TwoViewSSL(nn.Module):
    """Encode two augmented views -> Projector -> anti-collapse regularizer.

    `regularizer: vicreg` -> VICRegLoss (std_coeff, cov_coeff)   [H3 arm A]
    `regularizer: sigreg` -> BCS / SIGReg (lmbd, num_slices)     [H3 arm B]
    Set std_coeff (vicreg) or lmbd (sigreg) to 0 to *force collapse* — that is
    the H2 ablation (a collapsed encoder is a blind detector)."""

    def __init__(self, encoder, cfg):
        super().__init__()
        self.encoder = encoder
        spec = cfg.get("projector", f"{encoder.out_dim}-1024-1024")
        # `projector: none` -> regularize the ENCODER output directly (so std_coeff=0
        # actually collapses the encoder, not just the projector — clean H2 ablation).
        self.projector = nn.Identity() if str(spec).lower() in ("none", "") else Projector(spec)
        self.kind = cfg.get("regularizer", "vicreg")
        if self.kind == "vicreg":
            self.reg = VICRegLoss(
                std_coeff=cfg.get("std_coeff", 1.0),
                cov_coeff=cfg.get("cov_coeff", 1.0),
            )
        elif self.kind == "sigreg":
            self.reg = BCS(num_slices=cfg.get("num_slices", 256), lmbd=cfg.get("lmbd", 10.0))
        else:
            raise ValueError(f"unknown regularizer '{self.kind}' (vicreg|sigreg)")

    def compute_loss(self, batch):
        v1, v2 = batch
        z1 = self.projector(self.encoder.represent(v1))
        z2 = self.projector(self.encoder.represent(v2))
        out = self.reg(z1, z2)
        logs = {k: float(v.detach()) for k, v in out.items() if k != "loss"}
        return out["loss"], logs


def build_ssl(encoder, cfg):
    """Two-view invariance objective; keeps the anti-collapse term (the whole point)."""
    return TwoViewSSL(encoder, cfg)


# --------------------------------------------------------------------------- #
# TRAINING LOOP
# --------------------------------------------------------------------------- #
def _lr_factor(elapsed, warmup, total):
    """Linear warmup then cosine decay, driven by elapsed wall-clock time."""
    if elapsed < warmup:
        return elapsed / max(1.0, warmup)
    p = (elapsed - warmup) / max(1.0, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


def run(fname="examples/eeg/cfgs/train.yaml", cfg=None, folder=None, **overrides):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    dcfg = EEGConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dcfg.mode = "ssl"
    loader = make_loader(dcfg)

    encoder = build_encoder(cfg.model).to(device)
    ssl = build_ssl(encoder, cfg.model).to(device)
    base_lr = cfg.optim.lr
    opt = torch.optim.AdamW(ssl.parameters(), lr=base_lr, weight_decay=cfg.optim.weight_decay)

    # Time-budgeted cosine schedule: train for `train_seconds` (robust to throughput),
    # with warmup. If train_seconds is unset, fall back to fixed epochs + constant LR.
    budget = cfg.optim.get("train_seconds", None)
    warmup = cfg.optim.get("warmup_seconds", 300)
    max_epochs = cfg.optim.get("epochs", 10 ** 9) if budget else cfg.optim.epochs

    ckpt_dir = folder or cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    from examples.eeg import wb
    wb.init(cfg, group="two-view")

    def save(epoch):
        torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, "latest.pth.tar"))

    t_start, epoch, stop = time.time(), 0, False
    while not stop and epoch < max_epochs:
        ssl.train()
        t0, nb = time.time(), 0
        for batch in loader:
            if budget:
                lr = base_lr * _lr_factor(time.time() - t_start, warmup, budget)
                for g in opt.param_groups:
                    g["lr"] = lr
            batch = batch.to(device) if torch.is_tensor(batch) else [b.to(device) for b in batch]
            opt.zero_grad(set_to_none=True)
            loss, logs = ssl.compute_loss(batch)
            loss.backward(); opt.step()
            nb += 1
            if budget and time.time() - t_start >= budget:
                stop = True; break
        dt = time.time() - t0
        print(f"[eeg] epoch {epoch} loss={loss.item():.4f} {logs} | {dt:.1f}s "
              f"({nb * cfg.data.batch_size / dt:.0f} win/s) lr={opt.param_groups[0]['lr']:.2e} "
              f"elapsed={(time.time() - t_start) / 60:.1f}min", flush=True)
        wb.log({"epoch": epoch, "loss": float(loss.detach()),
                "lr": opt.param_groups[0]["lr"], **{k: float(v) for k, v in logs.items()}})
        save(epoch); epoch += 1
    save(epoch)
    wb.finish()
    print(f"[eeg] done ({epoch} epochs, {(time.time() - t_start) / 60:.1f} min) -> {ckpt_dir}/latest.pth.tar", flush=True)


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/eeg/cfgs/train.yaml"
    run(fname=fname)
