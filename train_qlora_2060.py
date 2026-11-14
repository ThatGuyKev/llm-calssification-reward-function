"""Launcher for train_qlora.py with defaults that fit an RTX 2060 (6 GB, Turing).

Same training code, different defaults:
  - Falcon3-1B-Base, 4-bit NF4, fp16 + GradScaler (train_qlora.py auto-selects fp16 on pre-Ampere GPUs)
  - gradient checkpointing on, max length 512, micro-batch 4 x accum 8 (effective 32)
  - 20k training rows (~2-2.5 h); pass --n-train 0 for all rows (~6-7 h)
  - small prediction batch, separate output dir

Any flag you pass overrides these defaults, e.g.:
  python train_qlora_2060.py --smoke
  python train_qlora_2060.py --n-train 10000
  python train_qlora_2060.py --bs 2 --accum 16      # if you hit CUDA out-of-memory
"""
import runpy, sys
from pathlib import Path

DEFAULTS = {
    "--model": "tiiuae/Falcon3-1B-Base",
    "--max-len": "512",
    "--bs": "4",
    "--accum": "8",
    "--n-train": "20000",
    "--pred-bs": "8",
    "--out": "outputs/falcon3-1b-qlora-2060",
}

for flag, value in DEFAULTS.items():
    if flag not in sys.argv:
        sys.argv += [flag, value]
if "--grad-ckpt" not in sys.argv:
    sys.argv.append("--grad-ckpt")

runpy.run_path(str(Path(__file__).with_name("train_qlora.py")), run_name="__main__")
