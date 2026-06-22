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
    import torch
    import torch.nn as nn

    class EEGOneSecEncoder(nn.Module):
        def __init__(
            self,
            in_channels=19,
            out_dim=256,
            hidden=64,
            depth=3,
            kernel_size=7,
            sample_rate=200,
        ):
            super().__init__()
            self.out_dim = out_dim
            self.sample_rate = sample_rate
            self.patch_size = sample_rate  # 1 second = 200 samples

            layers = []
            c_in = in_channels
            c = hidden

            for _ in range(depth):
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
            self.proj = nn.Linear(c_in, out_dim)

        def encode_patch(self, x):
            # x: [B, C, 200]
            h = self.net(x)           # [B, H, T']
            h = h.mean(dim=-1)        # [B, H]
            h = self.proj(h)          # [B, D]
            return h

        def frames(self, x):
            # x: [B, C, T], e.g. [B, 19, 2000]
            B, C, T = x.shape

            n_frames = T // self.patch_size
            T_used = n_frames * self.patch_size

            x = x[:, :, :T_used]

            # [B, C, n_frames, 200]
            x = x.view(B, C, n_frames, self.patch_size)

            # [B, n_frames, C, 200]
            x = x.permute(0, 2, 1, 3)

            # [B*n_frames, C, 200]
            x = x.reshape(B * n_frames, C, self.patch_size)

            h = self.encode_patch(x)  # [B*n_frames, D]

            # [B, n_frames, D]
            h = h.view(B, n_frames, self.out_dim)

            return h

        def represent(self, x):
            # recording/window-level embedding for linear probe
            h = self.frames(x)        # [B, F, D]
            return h.mean(dim=1)      # [B, D]

        def forward(self, x):
            return self.represent(x)

    return EEGOneSecEncoder(
        in_channels=getattr(cfg, "in_channels", 19),
        out_dim=getattr(cfg, "out_dim", 256),
        hidden=getattr(cfg, "hidden", 64),
        depth=getattr(cfg, "depth", 3),
        kernel_size=getattr(cfg, "kernel_size", 7),
        sample_rate=getattr(cfg, "sample_rate", 200),
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
    import copy
    import torch.nn as nn
    import torch.nn.functional as F

    from eb_jepa.architectures import Projector
    from eb_jepa.losses import VCLoss

    class EEGPredictiveVICRegJEPA(nn.Module):
        def __init__(self, encoder, cfg):
            super().__init__()
            self.encoder = encoder

            self.target_encoder = copy.deepcopy(encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False

            self.pred_steps = getattr(cfg, "pred_steps", 4)
            self.ema = getattr(cfg, "ema", 0.99)

            self.predictor = nn.GRU(
                input_size=encoder.out_dim,
                hidden_size=encoder.out_dim,
                num_layers=getattr(cfg, "pred_layers", 1),
                batch_first=True,
            )

            self.head = nn.Linear(encoder.out_dim, encoder.out_dim)

            projector_spec = getattr(
                cfg,
                "projector",
                f"{encoder.out_dim}-1024-1024",
            )
            self.projector = Projector(projector_spec)

            self.regularizer = VCLoss(
                std_coeff=getattr(cfg, "std_coeff", 10.0),
                cov_coeff=getattr(cfg, "cov_coeff", 100.0),
            )

            self.reg_weight = getattr(cfg, "reg_weight", 1.0)

        @torch.no_grad()
        def update_target_encoder(self):
            for p_online, p_target in zip(
                self.encoder.parameters(),
                self.target_encoder.parameters(),
            ):
                p_target.data.mul_(self.ema).add_(
                    p_online.data,
                    alpha=1.0 - self.ema,
                )

        def _parse_reg_loss(self, out, device):
            if isinstance(out, dict):
                loss = out["loss"]
                logs = {k: float(v.detach().cpu()) for k, v in out.items() if torch.is_tensor(v)}
                return loss, logs

            if isinstance(out, tuple):
                loss = out[0]
                logs = {
                    "reg": float(out[0].detach().cpu()),
                }
                if len(out) > 1 and torch.is_tensor(out[1]):
                    logs["reg_unweighted"] = float(out[1].detach().cpu())
                if len(out) > 2 and isinstance(out[2], dict):
                    for k, v in out[2].items():
                        if torch.is_tensor(v):
                            logs[k] = float(v.detach().cpu())
                return loss, logs

            return out, {"reg": float(out.detach().cpu())}

        def compute_loss(self, batch):
            if isinstance(batch, (list, tuple)):
                x_context, x_target = batch
            else:
                x_context = batch
                x_target = batch

            online_frames = self.encoder.frames(x_context)  # [B, F, D]

            with torch.no_grad():
                target_frames = self.target_encoder.frames(x_target)

            # next-frame prediction:
            # frame 0 predicts frame 1
            # frame 1 predicts frame 2
            # ...
            context = online_frames[:, :-1]                 # [B, F-1, D]
            target = target_frames[:, 1:]                   # [B, F-1, D]

            pred, _ = self.predictor(context)               # [B, F-1, D]
            pred = self.head(pred)                          # [B, F-1, D]

            pred_loss = F.mse_loss(pred, target.detach())

            z = online_frames.reshape(-1, online_frames.shape[-1])
            z = self.projector(z)

            reg_out = self.regularizer(z)
            reg_loss, reg_logs = self._parse_reg_loss(
                reg_out,
                x_context.device,
            )

            loss = pred_loss + self.reg_weight * reg_loss

            self.update_target_encoder()

            logs = {
                "pred_energy": float(pred_loss.detach().cpu()),
                "reg_loss": float(reg_loss.detach().cpu()),
                **reg_logs,
            }

            return loss, logs

        @torch.no_grad()
        def compute_energy(self, batch):
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            online_frames = self.encoder.frames(x)

            with torch.no_grad():
                target_frames = self.target_encoder.frames(x)

            context = online_frames[:, :-1]
            target = target_frames[:, 1:]

            pred, _ = self.predictor(context)
            pred = self.head(pred)

            energy_per_frame = ((pred - target) ** 2).mean(dim=-1)
            energy_per_window = energy_per_frame.mean(dim=1)

            return energy_per_window

    return EEGPredictiveVICRegJEPA(encoder, cfg)

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
