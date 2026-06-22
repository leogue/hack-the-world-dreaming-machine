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
import os
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.architectures import Projector
from eb_jepa.losses import VICRegLoss

from eb_jepa.datasets.eeg.dataset import EEGConfig, make_loader

# Reuse the eb_jepa core — DO NOT reimplement these:
#   eb_jepa.architectures: Projector (MLP), RNNPredictor (GRU)
#   eb_jepa.losses:        VICRegLoss (inv+var+cov), VCLoss (variance+covariance)


# --------------------------------------------------------------------------- #
# 1) ENCODER  — # TODO
# --------------------------------------------------------------------------- #
def _cfg_get(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default

def build_encoder(cfg):
    """TODO: return a 1D encoder mapping an EEG window [B, C=n_channels, T] to a
    representation. Expose `.represent(x) -> [B, D]` (global pooled over time)
    and an `.out_dim` attribute. If you go for the predictive objective, also
    expose `.frames(x) -> [B, F, D]` (a short latent sequence) and `.n_frames`.

    Hints: a strided Conv1d stack (kernel 7, stride 2, BatchNorm + GELU) that
    downsamples time, followed by global average pooling, is a strong baseline
    for [B, 19, 2000]. eb_jepa.architectures has 2D image/video encoders to take
    inspiration from, not a 1D one — so this lives here."""
    class EEGConvEncoder(nn.Module):
        def __init__(
            self,
            in_channels=19,
            out_dim=256,
            hidden=64,
            depth=4,
            kernel_size=7,
        ):
            super().__init__()
            self.out_dim = out_dim

            layers = []
            c_in = in_channels
            c = hidden

            for i in range(depth):
                layers += [
                    nn.Conv1d(
                        c_in,
                        c,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(c),
                    nn.GELU(),
                ]
                c_in = c
                c = min(c * 2, out_dim)

            self.net = nn.Sequential(*layers)
            self.proj = nn.Conv1d(c_in, out_dim, kernel_size=1)
            self.pool = nn.AdaptiveAvgPool1d(1)

        def frames(self, x):
            # x: [B, C, T]
            h = self.net(x)
            h = self.proj(h)          # [B, D, F]
            h = h.transpose(1, 2)     # [B, F, D]
            return h

        def represent(self, x):
            h = self.net(x)
            h = self.proj(h)          # [B, D, F]
            h = self.pool(h).squeeze(-1)
            return h                  # [B, D]

        def forward(self, x):
            return self.represent(x)

    return EEGConvEncoder(
        in_channels=_cfg_get(cfg, "in_channels", 19),
        out_dim=_cfg_get(cfg, "out_dim", 256),
        hidden=_cfg_get(cfg, "hidden", 64),
        depth=_cfg_get(cfg, "depth", 4),
        kernel_size=_cfg_get(cfg, "kernel_size", 7),
    )


# --------------------------------------------------------------------------- #
# 2) SSL OBJECTIVE  — # TODO
# --------------------------------------------------------------------------- #
def build_ssl(encoder, cfg):
    """TODO: return an nn.Module exposing `compute_loss(batch) -> (loss, logs)`.
    Pick one:
      * two-view VICReg (natural choice): the dataset already returns (v1, v2);
        encoder.represent each view -> eb_jepa Projector -> VICRegLoss
        (invariance + variance + covariance). batch = (v1, v2).
      * predictive JEPA (optional): encode frames, roll an eb_jepa RNNPredictor
        from a context frame to predict future frame latents vs an EMA target;
        add VCLoss (anti-collapse) on the online latents.
    Keep the variance/covariance (anti-collapse) term — it is what stops the
    encoder from mapping every window to the same point."""
    class EEGVICRegSSL(nn.Module):
        def __init__(self, encoder, cfg):
            super().__init__()
            self.encoder = encoder

            proj_hidden = _cfg_get(cfg, "proj_hidden", 1024)
            proj_out = _cfg_get(cfg, "proj_out", 1024)
            projector_spec = _cfg_get(
                cfg,
                "projector",
                f"{encoder.out_dim}-{proj_hidden}-{proj_out}",
            )

            self.projector = Projector(projector_spec)
            self.loss_fn = VICRegLoss(
                std_coeff=_cfg_get(cfg, "std_coeff", 1.0),
                cov_coeff=_cfg_get(cfg, "cov_coeff", 1.0),
            )

        def compute_loss(self, batch):
            v1, v2 = batch

            h1 = self.encoder.represent(v1)
            h2 = self.encoder.represent(v2)

            z1 = self.projector(h1)
            z2 = self.projector(h2)

            out = self.loss_fn(z1, z2)
            loss = out["loss"]

            logs = {
                "inv": float(out["invariance_loss"].detach().cpu()),
                "var": float(out["var_loss"].detach().cpu()),
                "cov": float(out["cov_loss"].detach().cpu()),
            }
            return loss, logs

    return EEGVICRegSSL(encoder, cfg)

# --------------------------------------------------------------------------- #
# TRAINING LOOP  — provided
# --------------------------------------------------------------------------- #
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
    print(next(iter(loader))[0].shape)

    encoder = build_encoder(cfg.model).to(device)
    ssl = build_ssl(encoder, cfg.model).to(device)
    opt = torch.optim.AdamW(ssl.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    ckpt_dir = folder or cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    for epoch in range(cfg.optim.epochs):
        ssl.train()
        
        for batch in loader:
            batch = batch.to(device) if torch.is_tensor(batch) else [b.to(device) for b in batch]
            opt.zero_grad(set_to_none=True)
            loss, logs = ssl.compute_loss(batch)
            loss.backward(); opt.step()

        """
        batch = next(iter(loader))
        batch = [b.to(device) for b in batch]

        loss, logs = ssl.compute_loss(batch)

        print(loss)
        print(logs)
        """

        print(f"[eeg] epoch {epoch} loss={loss.item():.4f} {logs}", flush=True)
        torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, "latest.pth.tar"))
    print(f"[eeg] done -> {ckpt_dir}/latest.pth.tar")


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/eeg/cfgs/train.yaml"
    run(fname=fname)
