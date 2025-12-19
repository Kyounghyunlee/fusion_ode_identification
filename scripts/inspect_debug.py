# scripts/inspect_debug.py

import numpy as np
import os
import diffrax

def inspect_data(shot):
    path = f"data/{shot}_torax_training.npz"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    print(f"--- Inspecting {path} ---")
    d = np.load(path)
    print("Keys:", list(d.keys()))
    
    for k in ['Te', 'ne', 'rho', 't_ts', 't']:
        if k in d:
            v = d[k]
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")
            print(f"    min={np.nanmin(v)}, max={np.nanmax(v)}")
            print(f"    nans={np.isnan(v).sum()}, infs={np.isinf(v).sum()}")
            if k == 'rho':
                print(f"    values={v}")

def inspect_failure():
    # Find a failure file
    fail_dir = "logs/sanity_run_01/failures"
    if not os.path.exists(fail_dir):
        print("No failure dir")
        return
    
    files = os.listdir(fail_dir)
    if not files:
        print("No failure files")
        return
    
    # Pick one
    f = files[0]
    path = os.path.join(fail_dir, f)
    print(f"--- Inspecting Failure: {path} ---")
    try:
        d = np.load(path)
        print("Keys:", list(d.keys()))
        for k in d.keys():
            if k.startswith("extra_"):
                print(f"{k}: {d[k]}")
    except Exception as e:
        print(f"Failed to load failure file: {e}")

if __name__ == "__main__":
    inspect_data(27567)
    inspect_failure()
    print(f"diffrax.RESULTS.successful: {diffrax.RESULTS.successful}")
