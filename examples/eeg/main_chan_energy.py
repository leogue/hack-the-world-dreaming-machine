"""Channel-wise predictive world model (experiment B) — prediction, per channel.

Same predictive paradigm as main_energy.py, but each EEG channel is encoded and
predicted INDEPENDENTLY: a shared 1-channel encoder maps each channel's frame to a
latent, and a shared GRU predicts each channel's next-frame latent from its past.
The prediction energy is then PER CHANNEL x time -> it localizes (a seizure/artifact
on a few channels is no longer diluted into a whole-brain scalar), and gives a
channel x time energy heatmap (money-shot). Trained on TUSZ normal (bckg) only.

Run:  python -m examples.eeg.main_chan_energy --fname examples/eeg/cfgs/chan_energy.yaml
"""
import os
import sys
import time

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.architectures import RNNPredictor
from eb_jepa.losses import CovarianceLoss, HingeStdLoss
from examples.eeg.dataset_seq import SeqConfig, make_seq_loader
from examples.eeg.main import Conv1dEncoder, _lr_factor


class ChannelEnergyJEPA(nn.Module):
    """Per-channel encoder + per-channel GRU prediction + VC anti-collapse."""

    def __init__(self, cfg):
        super().__init__()
        D = cfg.out_dim
        self.enc = Conv1dEncoder(in_channels=1, out_dim=D, hidden=cfg.get("hidden", 32),
                                 depth=cfg.get("depth", 5))   # encodes ONE channel
        self.action_dim = cfg.get("action_dim", 1)
        self.predictor = RNNPredictor(hidden_size=D, action_dim=self.action_dim,
                                      num_layers=cfg.get("rnn_layers", 1), final_ln=nn.LayerNorm(D))
        self.std_fn = HingeStdLoss(std_margin=1.0)
        self.cov_fn = CovarianceLoss()
        self.std_coeff = cfg.get("std_coeff", 1.0)
        self.cov_coeff = cfg.get("cov_coeff", 1.0)

    def frames(self, obs):
        """obs [B, C, T, L] -> per-channel latents [B, C, T, D]."""
        B, C, T, L = obs.shape
        z = self.enc.represent(obs.reshape(B * C * T, 1, L))   # [B*C*T, D]
        return z.reshape(B, C, T, -1)

    def energy(self, z):
        """[B, C, T, D] -> per-(B,C,t) prediction energy [B, C, T-1] (teacher-forced)."""
        B, C, T, D = z.shape
        src = z[:, :, :-1].reshape(B * C * (T - 1), D, 1, 1, 1)
        a = torch.zeros(B * C * (T - 1), self.action_dim, 1, device=z.device)
        pred = self.predictor(src, a).reshape(B, C, T - 1, D)
        return ((pred - z[:, :, 1:]) ** 2).mean(-1)           # [B, C, T-1]

    def compute_loss(self, seq):
        z = self.frames(seq)                                  # [B, C, T, D]
        e = self.energy(z).mean()
        flat = z.reshape(-1, z.shape[-1])                     # [N, D] all channel-frames
        std, cov = self.std_fn(flat), self.cov_fn(flat)
        loss = e + self.std_coeff * std + self.cov_coeff * cov
        return loss, {"energy": float(e.detach()), "std_loss": float(std.detach()),
                      "cov_loss": float(cov.detach())}


def build_chan_energy_jepa(cfg):
    return ChannelEnergyJEPA(cfg)


def run(fname, cfg=None, **ov):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if ov:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in ov.items()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)
    loader = make_seq_loader(SeqConfig(**OmegaConf.to_container(cfg.data, resolve=True)))
    model = build_chan_energy_jepa(cfg.model).to(device)
    base_lr = cfg.optim.lr
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=cfg.optim.weight_decay)
    budget = cfg.optim.get("train_seconds", None)
    warmup = cfg.optim.get("warmup_seconds", 300)
    max_epochs = cfg.optim.get("epochs", 10 ** 9) if budget else cfg.optim.epochs
    ckpt = cfg.meta.ckpt_dir
    os.makedirs(ckpt, exist_ok=True)
    print(f"[chan] {len(loader.dataset.files)} recordings (bckg_only={cfg.data.bckg_only})", flush=True)
    t0, epoch, stop = time.time(), 0, False
    while not stop and epoch < max_epochs:
        model.train(); te, nb = time.time(), 0
        for seq in loader:
            if budget:
                lr = base_lr * _lr_factor(time.time() - t0, warmup, budget)
                for g in opt.param_groups:
                    g["lr"] = lr
            seq = seq.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss, logs = model.compute_loss(seq)
            loss.backward(); opt.step(); nb += 1
            if budget and time.time() - t0 >= budget:
                stop = True; break
        print(f"[chan] epoch {epoch} loss={loss.item():.4f} {logs} | {time.time()-te:.1f}s "
              f"lr={opt.param_groups[0]['lr']:.2e} elapsed={(time.time()-t0)/60:.1f}min", flush=True)
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)}, os.path.join(ckpt, "latest.pth.tar"))
        epoch += 1
    print(f"[chan] done ({epoch} ep) -> {ckpt}/latest.pth.tar", flush=True)


if __name__ == "__main__":
    fn = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv else "examples/eeg/cfgs/chan_energy.yaml"
    run(fname=fn)
