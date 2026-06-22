"""Tiny optional Weights & Biases logger (enabled by cfg.meta.wandb)."""
import os

from omegaconf import OmegaConf

_run = None


def init(cfg, group):
    global _run
    if not bool(cfg.meta.get("wandb", False)):
        return None
    import wandb
    _run = wandb.init(
        project=str(cfg.meta.get("wandb_project", "eeg-jepa")),
        name=str(cfg.meta.get("wandb_name", os.path.basename(str(cfg.meta.ckpt_dir)))),
        group=group,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return _run


def log(d):
    if _run is not None:
        import wandb
        wandb.log(d)


def finish():
    global _run
    if _run is not None:
        import wandb
        wandb.finish()
        _run = None
