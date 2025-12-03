"""
Download selected MAST Level-2 data products from the STFC S3 object store
and save them locally as NetCDF files, plus a quick summary plot.

Workflow overview:
- For each shot in SHOTS, open the remote Zarr store via fsspec/s3.
- For each group name in GROUPS, open that Zarr group as an xarray Dataset.
- Sanitize attributes so the Dataset is NetCDF-writable, then save to disk.
- If a "summary" group exists, generate a small figure (Ip and <ne>). 

The defaults focus on the 24209–24211 shots and only download the summary/thomson/equilibrium/gas_injection groups needed by the training pipeline.

Notes:
- Access is anonymous (anon=True) against endpoint_url.
- A local simplecache (".cache") is used to speed up repeated access.
"""

import os
import time
from numbers import Number

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/CI environments
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr

# S3-compatible endpoint hosting the MAST Level-2 Zarr stores
endpoint_url = "https://s3.echo.stfc.ac.uk"

# Which discharges (shots) to fetch by default
SHOTS = [24209, 24210, 24211]

# Core groups required for the data-driven training path
ESSENTIAL_GROUPS = [
    "summary",
    "thomson_scattering",
    "equilibrium",
    "gas_injection",
    "spectrometer_visible",
]

# Root directory for all outputs (NetCDFs and plots)
OUT_ROOT = "data"


def _with_retries(label: str, func, max_retries: int = 3, retry_delay: float = 1.0):
    """Retry helper with exponential backoff."""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            if attempt == max_retries:
                break
            sleep_for = retry_delay * attempt
            print(f"   ⚠️  {label} failed ({attempt}/{max_retries}); retrying in {sleep_for:.1f}s: {exc}")
            time.sleep(sleep_for)
    print(f"❌ {label} exceeded retry budget ({max_retries}). Last error: {last_error}")
    raise last_error

# ---------- helpers ----------

def open_store(shot: int, cache_dir: str, max_retries: int, retry_delay: float):
    """Create a fsspec-backed Zarr store for a given shot.

    Uses a simple local cache in .cache to reduce repeated network access.
    Returns a zarr Store compatible with xr.open_zarr.
    """
    url = f"s3://mast/level2/shots/{shot}.zarr"
    print(f"\n🔗 Connecting to {url}")
    def _connect():
        return zarr.storage.FsspecStore.from_url(
            url,
            storage_options=dict(
                protocol="simplecache",
                target_protocol="s3",
                cache_storage=cache_dir,
                target_options=dict(
                    anon=True,               # public, no credentials required
                    endpoint_url=endpoint_url,  # S3-compatible endpoint
                    asynchronous=True,       # enable async capable fsspec client
                ),
            ),
        )

    try:
        return _with_retries(f"open store for shot {shot}", _connect, max_retries, retry_delay)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"❌ Failed to open store for shot {shot}: {exc}")
        return None


def open_group_as_dataset(store, group: str, max_retries: int, retry_delay: float):
    """Open a named Zarr group from the store as an xarray Dataset.

    Returns None if the group does not exist or cannot be read.
    """
    def _open():
        return xr.open_zarr(store, group=group)

    try:
        return _with_retries(f"open group '{group}'", _open, max_retries, retry_delay)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"   ⚠️  Could not open group '{group}': {exc}")
        return None


def make_netcdf_safe(ds: xr.Dataset) -> xr.Dataset:
    """Convert all non-serialisable attrs (dicts, custom objects) to strings.

    NetCDF writers can fail if attrs contain complex objects; this routine
    normalizes them to simple types and leaves arrays/lists as-is when safe.
    """
    ds = ds.copy()

    def _clean_attr_dict(attr_dict):
        clean = {}
        for k, v in attr_dict.items():
            if isinstance(v, (str, bytes, Number, np.generic, np.ndarray, list, tuple)):
                clean[k] = v
            elif isinstance(v, dict) and k == "license":
                # make something human-readable
                name = v.get("name", "")
                url = v.get("url", "")
                clean[k] = f"{name} ({url})"
            else:
                clean[k] = str(v)
        return clean

    # Clean global attributes
    ds.attrs = _clean_attr_dict(ds.attrs)

    # Clean per-variable attributes
    for var in ds.variables:
        ds[var].attrs = _clean_attr_dict(ds[var].attrs)

    return ds


def save_dataset(ds: xr.Dataset, shot: int, group: str, overwrite: bool) -> str:
    """Write a Dataset to NetCDF; optionally skip if file already exists."""
    out_dir = os.path.join(OUT_ROOT, str(shot))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{group}.nc")

    if not overwrite and os.path.exists(out_path):
        print(f"   ⏭️  {out_path} exists, skipping (use --overwrite to refresh)")
        return out_path

    print(f"   💾 Saving {group} -> {out_path}")
    ds_clean = make_netcdf_safe(ds)
    ds_clean.to_netcdf(out_path)
    return out_path


def quick_summary_plot(ds_summary: xr.Dataset, shot: int):
    """Produce a small figure for Ip and line-averaged density if present."""
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    if "ip" in ds_summary:
        ds_summary["ip"].plot(ax=ax[0])
        ax[0].set_ylabel("Ip [A]")
    else:
        ax[0].text(0.5, 0.5, "no 'ip' in summary",
                   transform=ax[0].transAxes, ha="center", va="center")

    if "line_average_n_e" in ds_summary:
        ds_summary["line_average_n_e"].plot(ax=ax[1])
        ax[1].set_ylabel(r"<ne> [m$^{-3}$]")
    else:
        ax[1].text(0.5, 0.5, "no 'line_average_n_e'",
                   transform=ax[1].transAxes, ha="center", va="center")

    ax[1].set_xlabel("time [s]")
    plt.suptitle(f"Shot {shot} summary")

    # Save plots to a dedicated subfolder
    plots_dir = os.path.join(OUT_ROOT, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f"summary_{shot}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   📈 Saved summary plot -> {out_path}")


# ---------- main ----------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download MAST Level-2 groups to NetCDF")
    ap.add_argument("--shots", type=int, nargs="+", help="Shot numbers to fetch (default: built-in SHOTS)")
    ap.add_argument(
        "--groups",
        type=str,
        nargs="+",
        choices=ESSENTIAL_GROUPS,
        help="Groups to download (default: essential groups)",
    )
    ap.add_argument("--out-root", type=str, default=OUT_ROOT, help="Output root directory (default: data)")
    ap.add_argument("--cache-dir", type=str, default=".cache", help="Local cache directory for fsspec simplecache")
    ap.add_argument("--max-retries", type=int, default=3, help="Max retries for store/group access")
    ap.add_argument("--retry-delay", type=float, default=1.0, help="Base delay (s) for retry backoff")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing NetCDF files")
    ap.add_argument("--no-plots", action="store_true", help="Disable summary plot generation")
    args = ap.parse_args()

    shots = args.shots if args.shots is not None else SHOTS
    groups = args.groups if args.groups is not None else ESSENTIAL_GROUPS
    OUT_ROOT = args.out_root
    cache_dir = args.cache_dir
    max_retries = max(1, args.max_retries)
    retry_delay = max(0.1, args.retry_delay)
    overwrite = bool(args.overwrite)
    make_plots = not args.no_plots

    # Ensure output root exists
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Iterate through each requested shot and try to fetch groups
    for shot in shots:
        print("\n===============================")
        print(f"   Processing shot {shot}")
        print("===============================")

        store = open_store(shot, cache_dir, max_retries, retry_delay)
        if store is None:
            continue

        ds_summary = None

        # Attempt to open and save each group for this shot
        for group in groups:
            print(f" - Trying group: {group}")
            existing_path = os.path.join(OUT_ROOT, str(shot), f"{group}.nc")
            if not overwrite and os.path.exists(existing_path):
                print(f"   ⏭️  {existing_path} exists, skipping download")
                if make_plots and group == "summary":
                    try:
                        ds_summary = xr.open_dataset(existing_path)
                    except Exception as exc:  # pylint: disable=broad-except
                        print(f"   ⚠️  Failed to reopen cached summary for plotting: {exc}")
                continue

            ds = open_group_as_dataset(store, group, max_retries, retry_delay)
            if ds is None:
                continue

            save_dataset(ds, shot, group, overwrite=overwrite)

            if group == "summary":
                ds_summary = ds

        # Optionally create a quick-look plot if summary is available
        if make_plots and ds_summary is not None:
            quick_summary_plot(ds_summary, shot)
            close_fn = getattr(ds_summary, "close", None)
            if callable(close_fn):
                close_fn()
        elif ds_summary is None:
            print("   ⚠️  No 'summary' group found for this shot, skipping plot.")
