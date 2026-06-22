"""EEG predictive (energy) JEPA — Étape 2, the world-model route (H1).

Encode a sequence of consecutive frames into a latent trajectory z_1..z_T, then
let an RNN predict z_{t+1} from z_t (teacher-forced, dummy zero action = autonomous
dynamics). Train on NORMAL EEG only -> the prediction energy ||ẑ_{t+1}-z_{t+1}||²
is low on normal and (hypothesis) spikes on seizures unseen in training.

Reuses eb_jepa components: RNNPredictor (GRU), VCLoss (anti-collapse), SquareLossSeq
(the MSE energy). The Conv1d frame encoder is the Étape-1 one, applied per frame.

Run:  python -m examples.eeg.main_energy --fname examples/eeg/cfgs/energy_tusz.yaml
"""
import os
import sys
import time

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.architectures import RNNPredictor
from eb_jepa.losses import SquareLossSeq, VCLoss
from examples.eeg.dataset_seq import SeqConfig, make_seq_loader
from examples.eeg.main import Conv1dEncoder


class FramesEncoder(nn.Module):
    """Per-frame Conv1d encoder. obs [B, C, Tf, L] -> trajectory [B, D, Tf, 1, 1]."""

    def __init__(self, in_channels, out_dim, hidden, depth):
        super().__init__()
        self.enc = Conv1dEncoder(in_channels, out_dim, hidden, depth)
        self.out_dim = out_dim

    def frames(self, obs):
        B, C, Tf, L = obs.shape
        x = obs.permute(0, 2, 1, 3).reshape(B * Tf, C, L)   # [B*Tf, C, L]
        z = self.enc.represent(x)                           # [B*Tf, D]
        z = z.reshape(B, Tf, -1).permute(0, 2, 1)           # [B, D, Tf]
        return z.unsqueeze(-1).unsqueeze(-1)                # [B, D, Tf, 1, 1]

    def represent(self, x):
        return self.enc.represent(x)


class EnergyJEPA(nn.Module):
    """Predictive JEPA: frame encoder + RNN predictor + VCLoss anti-collapse."""

    def __init__(self, cfg):
        super().__init__()
        D = cfg.out_dim
        self.encoder = FramesEncoder(cfg.in_channels, D, cfg.get("hidden", 64), cfg.get("depth", 5))
        self.action_dim = cfg.get("action_dim", 1)          # dummy action -> autonomous dynamics
        self.predictor = RNNPredictor(
            hidden_size=D, action_dim=self.action_dim,
            num_layers=cfg.get("rnn_layers", 1), final_ln=nn.LayerNorm(D))
        self.reg = VCLoss(std_coeff=cfg.get("std_coeff", 1.0), cov_coeff=cfg.get("cov_coeff", 1.0))
        self.predcost = SquareLossSeq()

    def _energy_per_step(self, z):
        """z [B, D, T, 1, 1] -> per-(sample,step) energy [B, T-1] (teacher-forced)."""
        B, D, T = z.shape[0], z.shape[1], z.shape[2]
        a = torch.zeros(B, self.action_dim, 1, device=z.device)
        es = []
        for t in range(T - 1):
            zt = z[:, :, t:t + 1].reshape(B, D, 1, 1, 1)     # true frame t (teacher forcing)
            pred = self.predictor(zt, a)                     # ẑ_{t+1}  [B, D, 1, 1, 1]
            tgt = z[:, :, t + 1:t + 2]                       # z_{t+1}
            es.append(((pred - tgt) ** 2).flatten(1).mean(1))   # [B]
        return torch.stack(es, dim=1)                        # [B, T-1]

    def compute_loss(self, seq):
        z = self.encoder.frames(seq)                         # [B, D, T, 1, 1]
        energy = self._energy_per_step(z).mean()             # scalar prediction energy
        rloss, _, rdict = self.reg(z)                        # anti-collapse on trajectory
        loss = energy + rloss
        return loss, {"energy": float(energy.detach()), **rdict}

    @torch.no_grad()
    def per_frame_energy(self, seq):
        """Eval: unreduced energy per frame -> [B, T-1] (frame t+1's anomaly score)."""
        return self._energy_per_step(self.encoder.frames(seq))


def build_energy_jepa(cfg):
    return EnergyJEPA(cfg)


def run(fname, cfg=None, **overrides):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    loader = make_seq_loader(SeqConfig(**OmegaConf.to_container(cfg.data, resolve=True)))
    model = build_energy_jepa(cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    ckpt_dir = cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    from examples.eeg import wb
    wb.init(cfg, group="energy")
    print(f"[energy] {len(loader.dataset.files)} recordings (bckg_only={cfg.data.bckg_only})", flush=True)
    for epoch in range(cfg.optim.epochs):
        model.train()
        t0, nb = time.time(), 0
        for seq in loader:
            seq = seq.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss, logs = model.compute_loss(seq)
            loss.backward(); opt.step(); nb += 1
        dt = time.time() - t0
        print(f"[energy] epoch {epoch} loss={loss.item():.4f} {logs} | {dt:.1f}s "
              f"({nb * cfg.data.batch_size / dt:.0f} seq/s)", flush=True)
        wb.log({"epoch": epoch, "loss": float(loss.detach()), **{k: float(v) for k, v in logs.items()}})
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, "latest.pth.tar"))
    wb.finish()
    print(f"[energy] done -> {ckpt_dir}/latest.pth.tar", flush=True)


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/eeg/cfgs/energy_tusz.yaml"
    run(fname=fname)
