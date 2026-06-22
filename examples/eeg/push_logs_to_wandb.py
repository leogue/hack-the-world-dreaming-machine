"""Push our real training curves (parsed from SLURM logs) to W&B as online runs.

We already trained these models; their per-epoch metrics are in the .out logs. Rather
than burn contended GPUs to reproduce identical curves, we replay the logged metrics
into W&B from the login node (which has internet). Run from the repo root:
    python -m examples.eeg.push_logs_to_wandb
"""
import ast
import glob
import re

import wandb

RE = re.compile(r"epoch (\d+) loss=([\d.eE+-]+) (\{[^}]*\})")
PROJECT = "eeg-jepa"

# (jobid, run name, group)  — the models we present
RUNS = [
    ("75213", "vicreg",            "detection"),
    ("75214", "sigreg",            "detection"),
    ("75222", "collapse_std0",     "detection"),
    ("75485", "vicreg_3h",         "detection-solid"),
    ("75486", "sigreg_3h",         "detection-solid"),
    ("75176", "energy_worldmodel", "energy"),
    ("76854", "chan_energy",       "energy"),
]


def find(jobid):
    g = glob.glob(f"eeg_*_{jobid}.out")
    return g[0] if g else None


def parse(path):
    rows = []
    for line in open(path):
        m = RE.search(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), ast.literal_eval(m.group(3))))
    return rows


def main():
    for jobid, name, group in RUNS:
        path = find(jobid)
        if not path:
            print(f"skip {name}: no log for {jobid}"); continue
        rows = parse(path)
        if not rows:
            print(f"skip {name}: no epochs in {path}"); continue
        run = wandb.init(project=PROJECT, name=name, group=group,
                         config={"jobid": jobid, "source_log": path}, reinit=True)
        for ep, loss, d in rows:
            run.log({"epoch": ep, "loss": loss, **{k: float(v) for k, v in d.items()}})
        run.finish()
        print(f"pushed {name}: {len(rows)} epochs from {path}")


if __name__ == "__main__":
    main()
