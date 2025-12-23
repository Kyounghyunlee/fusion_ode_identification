#!/usr/bin/env python
"""Time padding strictness smoke test.

Goal: ensure pad_time_to_max_strict() remains strictly increasing even if
values are later downcast to float32 (e.g. x64 disabled).

Run via:
  ./scripts/run_training_gpu.sh --python scripts/smoke_time_padding_strict.py
"""

import os
import sys
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fusion_ode_identification.data import pad_time_to_max_strict


def assert_strict(x: np.ndarray, name: str):
    d = np.diff(x)
    if not np.all(d > 0):
        bad = np.where(d <= 0)[0][:10]
        raise SystemExit(f"FAIL: {name} not strictly increasing; idx={bad} d[idx]={d[bad]}")


def main():
    # This branch expects x64 enabled. We still check that strictness survives a
    # float32 downcast in a realistic time range.
    base = 0.0
    dt = 1e-3

    t1 = base + dt * np.arange(50, dtype=np.float64)
    t2 = base + dt * np.arange(80, dtype=np.float64)
    t3 = base + dt * np.arange(10, dtype=np.float64)

    times_list = [t1, t2, t3]
    lens_list = [len(t1), len(t2), len(t3)]

    padded = pad_time_to_max_strict(times_list, lens_list)
    padded64 = np.array(padded, dtype=np.float64)
    padded32 = padded64.astype(np.float32)

    for i in range(padded64.shape[0]):
        assert_strict(padded64[i], f"padded64[{i}]")
        assert_strict(padded32[i], f"padded32[{i}]")

    print("PASS")


if __name__ == "__main__":
    main()
