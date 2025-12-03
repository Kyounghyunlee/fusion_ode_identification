import numpy as np

data = np.load('simulate_outputs.npz')
print("Keys:", list(data.keys()))

if 'Te' in data:
    te = data['Te']
    print(f"Te shape: {te.shape}")
    print(f"Te min: {np.nanmin(te)}")
    print(f"Te max: {np.nanmax(te)}")
    
    neg_mask = te <= 0
    print(f"Count <= 0: {neg_mask.sum()}")
    print(f"Unique values <= 0: {np.unique(te[neg_mask])}")
    
    # Check if entire rows are bad
    bad_rows = np.all(te <= 0, axis=1)
    print(f"Rows with all <= 0: {bad_rows.sum()}")
    
    # Check if it's just the boundary or something
    print(f"First row sample: {te[0]}")
    
    valid_te = te[te > 0]
    if len(valid_te) > 0:
        print(f"Valid Te min: {np.min(valid_te)}")
