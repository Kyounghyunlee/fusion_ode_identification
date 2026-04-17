"""
Download selected MAST Level-2 data products from the STFC S3 object store
and save them locally as NetCDF files, plus a quick summary plot.

Workflow overview:
- For each shot (from --shots, config.data.shots, or detected *_torax_training.npz), open the remote Zarr store via fsspec/s3.
- For each requested target, open the relevant Zarr group as an xarray Dataset.
- Sanitize attributes so the Dataset is NetCDF-writable, then save to disk.
- For D-alpha, extract a compact `d_alpha.nc` sidecar from the visible spectrometer group.
- If a "summary" group exists, generate a small figure (Ip and <ne>). 

Notes:
- Access is anonymous (anon=True) against endpoint_url.
- A local simplecache (".cache") is used to speed up repeated access.
"""

# preprocessing/download_data.py

import os
import shutil
import time
from numbers import Number
from typing import Optional, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/CI environments
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
import requests
import fsspec

# S3-compatible endpoint hosting the MAST Level-2 Zarr stores
endpoint_url = "https://s3.echo.stfc.ac.uk"

# REST API root for locating EFIT/IDA3 files
API_ROOT = "https://mastapp.site/json"

# Remote Zarr groups available in the Level-2 store.
REMOTE_GROUPS = [
    "summary",
    "thomson_scattering",
    "equilibrium",
    "gas_injection",
    "spectrometer_visible",
]

# Download targets exposed to the CLI.
DOWNLOAD_TARGETS = REMOTE_GROUPS + ["d_alpha"]

# Default targets for the training pipeline.
ESSENTIAL_GROUPS = [
    "summary",
    "thomson_scattering",
    "equilibrium",
    "gas_injection",
    "d_alpha",
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


def _clean_json_items(resp_json):
    if not resp_json:
        return []
    items = resp_json.get("items") if isinstance(resp_json, dict) else None
    return items if isinstance(items, list) else []


def query_efit_file(shot: int, max_items: int = 3):
    params = {"filters": f"shot_id$eq:{shot},class$eq:efm", "size": max_items}
    resp = requests.get(f"{API_ROOT}/files", params=params, timeout=15)
    if resp.status_code == 404:
        # Endpoint not available on this API; gracefully skip
        print("   ℹ️  EFIT endpoint not found on mastapp; skipping EFIT fetch for this shot")
        return []
    if resp.status_code != 200:
        raise RuntimeError(f"EFIT API {resp.status_code}: {resp.text[:200]}")
    return _clean_json_items(resp.json())


def download_efit_file(item: dict, dest_dir: str) -> str:
    url = item.get("url") or item.get("s3_path") or ""
    ep = item.get("endpoint_url") or endpoint_url
    fname = item.get("filename") or os.path.basename(url) or "efit_ida3"
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, fname)
    if not url:
        raise ValueError("EFIT metadata missing url")

    if url.startswith("s3://"):
        fs = fsspec.filesystem("s3", anon=True, client_kwargs={"endpoint_url": ep})
        with fs.open(url, "rb") as fsrc, open(out_path, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
        return out_path

    if url.startswith("http"):
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, "wb") as fdst:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fdst.write(chunk)
        return out_path

    raise ValueError(f"Unrecognized EFIT url scheme: {url}")


def _try_open_dataset(path: str):
    engines = ["netcdf4", "h5netcdf", None]
    groups = [None, "equilibrium"]
    for grp in groups:
        for eng in engines:
            try:
                if grp is None:
                    return xr.open_dataset(path, engine=eng)
                return xr.open_dataset(path, engine=eng, group=grp)
            except Exception:
                continue
    return None


def convert_efit_to_equilibrium_nc(src_path: str, out_path: str) -> bool:
    ds = _try_open_dataset(src_path)
    if ds is None:
        return False
    if "psi" not in ds and "psirz" in ds:
        ds = ds.rename({"psirz": "psi"})
    ds_clean = make_netcdf_safe(ds)
    ds_clean.to_netcdf(out_path)
    try:
        ds.close()
    except Exception:
        pass
    return True

# ---------- helpers ----------

def open_store(shot: int, cache_dir: str, max_retries: int, retry_delay: float):
    """Create a fsspec-backed Zarr store for a given shot.

    Uses a simple local cache in .cache to reduce repeated network access.
    Returns a zarr Store compatible with xr.open_zarr.
    """
    url = f"s3://mast/level2/shots/{shot}.zarr"
    print(f"\n🔗 Connecting to {url}")
    async_flag = bool(os.getenv("MAST_S3_ASYNC", "0") == "1")

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
                    asynchronous=async_flag,
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
        try:
            return xr.open_zarr(store, group=group, consolidated=True)
        except Exception:
            return xr.open_zarr(store, group=group, consolidated=False)

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


def extract_dalpha_dataset(ds: xr.Dataset) -> Optional[xr.Dataset]:
    """Extract a compact D-alpha dataset from the visible spectrometer group."""
    source_name = None
    for candidate in ("D_alpha", "filter_spectrometer_dalpha_voltage", "d_alpha"):
        if candidate in ds:
            source_name = candidate
            break
    if source_name is None:
        return None

    dalpha = ds[source_name]
    time_dim = next((dim for dim in dalpha.dims if "time" in dim.lower()), None)
    if time_dim is None:
        time_dim = dalpha.dims[-1]

    other_dims = [dim for dim in dalpha.dims if dim != time_dim]
    if len(other_dims) == 0:
        time_coord = ds.coords[time_dim].values if time_dim in ds.coords else np.arange(dalpha.size, dtype=float)
        channel_dim = "dalpha_channel"
        channel_names = np.array([str(dalpha.attrs.get("uda_name", source_name))], dtype="U64")
        dalpha_values = np.asarray(dalpha.transpose(time_dim).values, dtype=np.float64)[None, :]
    else:
        if len(other_dims) == 1:
            channel_dim = other_dims[0]
            dalpha = dalpha.transpose(channel_dim, time_dim)
            dalpha_values = np.asarray(dalpha.values, dtype=np.float64)
            if channel_dim in ds.coords:
                channel_names = np.asarray(ds.coords[channel_dim].values).astype("U64")
            else:
                channel_names = np.array([f"{source_name}_{i:02d}" for i in range(dalpha_values.shape[0])], dtype="U64")
        else:
            channel_dim = "dalpha_channel"
            dalpha = dalpha.stack({channel_dim: other_dims}).transpose(channel_dim, time_dim)
            dalpha_values = np.asarray(dalpha.values, dtype=np.float64)
            channel_names = np.asarray([str(v) for v in dalpha.coords[channel_dim].values], dtype="U64")
        time_coord = ds.coords[time_dim].values if time_dim in ds.coords else np.arange(dalpha_values.shape[1], dtype=float)

    out = xr.Dataset(
        data_vars={
            "D_alpha": ((channel_dim, "time"), dalpha_values),
            "D_alpha_sum": (("time",), np.nansum(dalpha_values, axis=0)),
        },
        coords={
            channel_dim: channel_names,
            "time": np.asarray(time_coord, dtype=float),
        },
        attrs={
            "description": "Extracted D-alpha channels from the visible spectrometer diagnostic",
            "source_group": str(ds.attrs.get("name", "spectrometer_visible")),
            "source_variable": source_name,
        },
    )
    out["D_alpha"].attrs = {
        "description": str(dalpha.attrs.get("description", "D-alpha voltage channels")),
        "units": str(dalpha.attrs.get("units", ds.attrs.get("units", ""))),
        "label": str(dalpha.attrs.get("label", ds.attrs.get("label", ""))),
    }
    out["D_alpha_sum"].attrs = {
        "description": "Sum of all available D-alpha channels",
        "units": str(dalpha.attrs.get("units", ds.attrs.get("units", ""))),
    }
    return out


def save_dalpha_dataset(ds: xr.Dataset, shot: int, overwrite: bool, out_root: str) -> Optional[str]:
    dalpha_ds = extract_dalpha_dataset(ds)
    if dalpha_ds is None:
        print("   ⚠️  No D-alpha variable found in visible spectrometer group")
        return None
    return save_dataset(dalpha_ds, shot, "d_alpha", overwrite=overwrite, out_root=out_root)


def save_dataset(ds: xr.Dataset, shot: int, group: str, overwrite: bool, out_root: str) -> str:
    """Write a Dataset to NetCDF; optionally skip if file already exists."""
    out_dir = os.path.join(out_root, str(shot))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{group}.nc")

    if not overwrite and os.path.exists(out_path):
        print(f"   ⏭️  {out_path} exists, skipping (use --overwrite to refresh)")
        return out_path

    print(f"   💾 Saving {group} -> {out_path}")
    ds_clean = make_netcdf_safe(ds)
    ds_clean.to_netcdf(out_path)
    return out_path


def quick_summary_plot(ds_summary: xr.Dataset, shot: int, out_root: str):
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
    plots_dir = os.path.join(out_root, "plots")
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
    ap.add_argument("--shots", type=int, nargs="+", required=True, help="Shot numbers to fetch (required)")
    ap.add_argument(
        "--groups",
        type=str,
        nargs="+",
        choices=DOWNLOAD_TARGETS,
        help="Groups to download (default: essential groups)",
    )
    ap.add_argument("--out-root", type=str, default=OUT_ROOT, help="Output root directory (default: data or config.data_dir)")
    ap.add_argument("--cache-dir", type=str, default=".cache", help="Local cache directory for fsspec simplecache")
    ap.add_argument("--clear-cache", action="store_true", help="Delete cache directory at start")
    ap.add_argument("--max-retries", type=int, default=3, help="Max retries for store/group access")
    ap.add_argument("--retry-delay", type=float, default=1.0, help="Base delay (s) for retry backoff")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing NetCDF files")
    ap.add_argument("--no-plots", action="store_true", help="Disable summary plot generation")
    ap.add_argument("--fetch-efit", action="store_true", help="Fetch EFIT/IDA3 equilibrium via REST and replace equilibrium.nc if successful")
    args = ap.parse_args()
    out_root = args.out_root
    shots = list(args.shots)

    groups = args.groups if args.groups is not None else ESSENTIAL_GROUPS
    cache_dir = args.cache_dir
    max_retries = max(1, args.max_retries)
    retry_delay = max(0.1, args.retry_delay)
    overwrite = bool(args.overwrite)
    make_plots = not args.no_plots
    fetch_efit = bool(args.fetch_efit)

    # Ensure output root exists
    os.makedirs(out_root, exist_ok=True)
    if args.clear_cache and os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

    if len(shots) == 0:
        print("No shots specified (--shots is required). Exiting.")
        raise SystemExit(1)

    # Iterate through each requested shot and try to fetch groups
    for shot in shots:
        print("\n===============================")
        print(f"   Processing shot {shot}")
        print("===============================")

        store = open_store(shot, cache_dir, max_retries, retry_delay)
        if store is None:
            continue

        ds_summary = None
        ds_cache = {}

        def get_dataset(group_name: str):
            if group_name in ds_cache:
                return ds_cache[group_name]
            ds_opened = open_group_as_dataset(store, group_name, max_retries, retry_delay)
            if ds_opened is not None:
                ds_cache[group_name] = ds_opened
            return ds_opened

        # Attempt to open and save each group for this shot
        for group in groups:
            print(f" - Trying group: {group}")
            existing_path = os.path.join(out_root, str(shot), f"{group}.nc")

            if group == "d_alpha":
                if not overwrite and os.path.exists(existing_path):
                    print(f"   ⏭️  {existing_path} exists, skipping download")
                    continue

                local_visible = os.path.join(out_root, str(shot), "spectrometer_visible.nc")
                if not overwrite and os.path.exists(local_visible):
                    try:
                        with xr.open_dataset(local_visible) as local_ds:
                            save_dalpha_dataset(local_ds, shot, overwrite=True, out_root=out_root)
                        continue
                    except Exception as exc:  # pylint: disable=broad-except
                        print(f"   ⚠️  Failed to extract D-alpha from cached spectrometer file: {exc}")

                ds_visible = get_dataset("spectrometer_visible")
                if ds_visible is None:
                    continue
                save_dalpha_dataset(ds_visible, shot, overwrite=True, out_root=out_root)
                continue

            if not overwrite and os.path.exists(existing_path):
                print(f"   ⏭️  {existing_path} exists, skipping download")
                if make_plots and group == "summary":
                    try:
                        ds_summary = xr.open_dataset(existing_path)
                    except Exception as exc:  # pylint: disable=broad-except
                        print(f"   ⚠️  Failed to reopen cached summary for plotting: {exc}")
                continue

            ds = get_dataset(group)
            if ds is None:
                continue
            if len(ds.data_vars) == 0:
                print(f"   ⚠️  Group '{group}' empty; skipping")
                continue

            save_dataset(ds, shot, group, overwrite=overwrite, out_root=out_root)
            if group == "summary":
                ds_summary = ds

        # Optionally create a quick-look plot if summary is available
        if make_plots and ds_summary is not None:
            quick_summary_plot(ds_summary, shot, out_root=out_root)
            close_fn = getattr(ds_summary, "close", None)
            if callable(close_fn):
                close_fn()
        elif ds_summary is None:
            print("   ⚠️  No 'summary' group found for this shot, skipping plot.")

        # Fetch EFIT/IDA3 via REST if requested
        if fetch_efit:
            try:
                items = query_efit_file(shot)
                if not items:
                    print("   ⚠️  No EFIT entries from REST API for this shot")
                else:
                    item = items[0]
                    raw_path = download_efit_file(item, os.path.join(out_root, str(shot)))
                    out_eq = os.path.join(out_root, str(shot), "equilibrium.nc")
                    if convert_efit_to_equilibrium_nc(raw_path, out_eq):
                        print(f"   ✅ EFIT equilibrium saved -> {out_eq}")
                    else:
                        print("   ⚠️  EFIT file downloaded but could not be parsed into equilibrium.nc")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"   ⚠️  EFIT REST download failed: {exc}")

        for ds_cached in ds_cache.values():
            try:
                ds_cached.close()
            except Exception:
                pass
