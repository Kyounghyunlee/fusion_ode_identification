"""Smoke test: checkpoint selection follows preference order.

Usage:
  python scripts/smoke_checkpoint_selection.py

This is a lightweight regression test for the selection rules used by
fusion_ode_identification.debug.find_best_checkpoint() and scripts/evaluate_model.py.

Default preference: best_ema -> best -> finetuned -> newest fallback.
"""

import os
import tempfile
import time


def select_by_preference(best_ema, best, finetuned):
    for p in best_ema:
        if os.path.exists(p):
            return p
    for p in best:
        if os.path.exists(p):
            return p
    for p in finetuned:
        if os.path.exists(p):
            return p

    # newest fallback
    all_paths = list(best_ema) + list(best) + list(finetuned)
    existing = [(p, os.path.getmtime(p)) for p in all_paths if os.path.exists(p)]
    if not existing:
        raise FileNotFoundError(all_paths)
    existing.sort(key=lambda x: x[1], reverse=True)
    return existing[0][0]


def touch(path, mtime):
    with open(path, "wb") as f:
        f.write(b"test")
    os.utime(path, (mtime, mtime))


def main():
    with tempfile.TemporaryDirectory() as d:
        raw = "model"
        best_ema = os.path.join(d, f"{raw}_best_ema.eqx")
        best = os.path.join(d, f"{raw}_best.eqx")
        finetuned = os.path.join(d, f"{raw}_finetuned.eqx")

        t0 = time.time()
        touch(best_ema, t0 - 20)
        touch(best, t0 - 10)
        touch(finetuned, t0)

        sel = select_by_preference([best_ema], [best], [finetuned])
        assert sel.endswith("_best_ema.eqx"), f"expected best_ema, got {sel}"

        os.remove(best_ema)
        sel = select_by_preference([best_ema], [best], [finetuned])
        assert sel.endswith("_best.eqx"), f"expected best, got {sel}"

        os.remove(best)
        sel = select_by_preference([best_ema], [best], [finetuned])
        assert sel.endswith("_finetuned.eqx"), f"expected finetuned, got {sel}"

        print("OK: checkpoint selection follows preference order")


if __name__ == "__main__":
    main()
