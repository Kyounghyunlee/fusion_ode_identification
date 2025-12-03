import numpy as np
import os
import glob

def inspect_file(filepath):
    print(f"\n{'='*80}")
    print(f"Inspecting: {os.path.basename(filepath)}")
    print(f"{'='*80}")
    
    try:
        d = np.load(filepath)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    keys_to_check = ['t', 'rho', 'Te', 'ne', 'Vprime', 'P_nbi', 'Ip', 'nebar']
    
    # Check available keys
    print(f"Keys available: {list(d.keys())}")
    
    # Check dimensions and NaNs
    for k in keys_to_check:
        if k not in d:
            print(f"MISSING: {k}")
            continue
            
        arr = d[k]
        print(f"\nField: {k}")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        
        if np.issubdtype(arr.dtype, np.number):
            n_nan = np.isnan(arr).sum()
            n_inf = np.isinf(arr).sum()
            print(f"  NaNs: {n_nan} ({n_nan/arr.size*100:.2f}%)")
            print(f"  Infs: {n_inf} ({n_inf/arr.size*100:.2f}%)")
            
            if arr.size > 0:
                # Compute min/max ignoring NaNs
                with np.errstate(invalid='ignore'):
                    if np.all(np.isnan(arr)):
                         print("  Min: NaN")
                         print("  Max: NaN")
                    else:
                         print(f"  Min: {np.nanmin(arr):.4e}")
                         print(f"  Max: {np.nanmax(arr):.4e}")
                         print(f"  Mean: {np.nanmean(arr):.4e}")
        
        # Specific checks
        if k == 'Te':
            # Check for valid profiles (no NaNs in the entire profile)
            valid_profiles = np.sum(~np.isnan(arr).any(axis=1))
            print(f"  Valid Profiles (rows without NaNs): {valid_profiles} / {arr.shape[0]}")
            
            # Check boundary values
            if arr.ndim > 1 and arr.shape[1] > 0:
                print(f"  Center (rho=0) NaNs: {np.isnan(arr[:, 0]).sum()}")
                print(f"  Edge (rho=1) NaNs: {np.isnan(arr[:, -1]).sum()}")

        if k == 'rho':
            print(f"  Values: {arr}")

def main():
    data_dir = "/home/kyoung/Code/Torax_project/data"
    files = glob.glob(os.path.join(data_dir, "*_training*.npz"))
    files.sort()
    
    for f in files:
        inspect_file(f)

if __name__ == "__main__":
    main()
