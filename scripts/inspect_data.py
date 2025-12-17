import numpy as np
import glob
import os

files = glob.glob("data/*_torax_training.npz")
print(f"Found {len(files)} files")

for f in files:
    d = np.load(f)
    Te = d["Te"]
    print(f"{os.path.basename(f)}: Te shape={Te.shape}, min={np.nanmin(Te):.2f}, max={np.nanmax(Te):.2f}, mean={np.nanmean(Te):.2f}")
    if "Vprime" in d:
        Vp = d["Vprime"]
        print(f"  Vprime: min={Vp.min():.2e}, max={Vp.max():.2e}, mean={Vp.mean():.2e}")
