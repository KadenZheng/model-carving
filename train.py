"""Train a full-depth or carved Transformer.

    python train.py configs/standard_140m.yaml             # full-depth baseline
    python train.py configs/standard_140m.yaml --carving   # Model Carving
"""

import argparse
import os

import jax
import yaml

from carving.training.loop import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--carving", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:  # one process per GPU under SLURM
        jax.distributed.initialize()
    run(yaml.safe_load(open(args.config)), args.carving, args.out, args.save_every, args.resume)


if __name__ == "__main__":
    main()
