"""Create a tar.gz archive of training NPZ packs for HPC transfer.

Usage:
    python -m scripts.package_hpc --packs data/*_torax_training.npz --out torax_packs.tar.gz
"""
from __future__ import annotations
import argparse
import tarfile
import os


def main():
    ap = argparse.ArgumentParser(description="Package NPZ training packs into a tar.gz")
    ap.add_argument("--packs", nargs="+", required=True, help="List of NPZ pack paths (glob expansion handled by shell)")
    ap.add_argument("--out", default="torax_packs.tar.gz", help="Output tar.gz filename")
    args = ap.parse_args()

    with tarfile.open(args.out, "w:gz") as tf:
        for p in args.packs:
            if not os.path.isfile(p):
                print(f"[skip] {p} (not found)")
                continue
            tf.add(p, arcname=os.path.basename(p))
            print(f"[add] {p}")
    print(f"[package] Wrote {args.out}")

if __name__ == "__main__":
    main()
