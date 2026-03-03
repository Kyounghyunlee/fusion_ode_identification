#!/usr/bin/env python3
"""Compare two NPZ files to check if they're identical."""

import numpy as np
import sys


def compare_npz_files(file1: str, file2: str) -> bool:
    """Compare two NPZ files.
    
    Returns True if identical, False otherwise.
    """
    try:
        d1 = np.load(file1)
        d2 = np.load(file2)
        
        # Check if same keys
        keys1 = set(d1.files)
        keys2 = set(d2.files)
        
        if keys1 != keys2:
            print(f"❌ Different keys!")
            print(f"  Only in {file1}: {keys1 - keys2}")
            print(f"  Only in {file2}: {keys2 - keys1}")
            return False
        
        # Compare each array
        differences = []
        for key in sorted(keys1):
            arr1 = d1[key]
            arr2 = d2[key]
            
            # Check shape
            if arr1.shape != arr2.shape:
                differences.append(f"  {key}: different shapes {arr1.shape} vs {arr2.shape}")
                continue
            
            # Check dtype
            if arr1.dtype != arr2.dtype:
                differences.append(f"  {key}: different dtypes {arr1.dtype} vs {arr2.dtype}")
                continue
            
            # For string/object/bytes arrays, compare as strings
            if arr1.dtype.kind in ('U', 'S', 'O') or arr2.dtype.kind in ('U', 'S', 'O'):
                str1 = str(arr1) if arr1.ndim == 0 else arr1
                str2 = str(arr2) if arr2.ndim == 0 else arr2
                if not np.array_equal(str1, str2):
                    differences.append(f"  {key}: different values (string/object type)")
                continue
            
            # For numeric arrays, check with tolerance
            try:
                if not np.allclose(arr1, arr2, rtol=1e-10, atol=1e-12, equal_nan=True):
                    # Check if exact match first
                    if not np.array_equal(arr1, arr2, equal_nan=True):
                        max_diff = np.nanmax(np.abs(arr1.astype(float) - arr2.astype(float))) if arr1.size > 0 else 0
                        differences.append(f"  {key}: different values (max diff: {max_diff:.3e})")
            except (TypeError, ValueError):
                if not np.array_equal(arr1, arr2):
                    differences.append(f"  {key}: different values")
        
        if differences:
            print(f"❌ Found {len(differences)} difference(s):")
            for diff in differences[:10]:  # Show first 10
                print(diff)
            if len(differences) > 10:
                print(f"  ... and {len(differences) - 10} more")
            return False
        
        print("✅ Files are identical!")
        return True
        
    except Exception as e:
        print(f"❌ Error comparing files: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 compare_npz.py <file1.npz> <file2.npz>")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    print(f"Comparing:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    print()
    
    identical = compare_npz_files(file1, file2)
    sys.exit(0 if identical else 1)
