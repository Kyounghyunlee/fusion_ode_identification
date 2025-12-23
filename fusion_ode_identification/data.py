"""Data loading and preprocessing utilities."""
# fusion_ode_identification/data.py

import glob
import logging
import os
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np

from .model import CONTROL_NAMES
from .types import ShotBundle


def pad_to_max(arrays, mode: str = "constant", constant_values: float = 0.0):
    max_len = max(x.shape[0] for x in arrays)
    out = []
    for x in arrays:
        pad_len = max_len - x.shape[0]
        if pad_len > 0:
            if mode == "edge":
                padding = jnp.repeat(x[-1:], pad_len, axis=0)
                out.append(jnp.concatenate([x, padding], axis=0))
            else:
                padding = jnp.full((pad_len,) + x.shape[1:], constant_values, dtype=x.dtype)
                out.append(jnp.concatenate([x, padding], axis=0))
        else:
            out.append(x)
    return jnp.stack(out)


def pad_time_to_max_strict(times_list, lens_list):
    """Pad time arrays while keeping them strictly increasing."""
    max_len = int(max(lens_list))
    padded = []
    for t, L in zip(times_list, lens_list):
        L = int(L)
        t_np = np.asarray(t, dtype=float)
        t_np = t_np[:L]

        pad_len = max_len - L
        if L < 2:
            eps = 1e-3
        else:
            dt = np.diff(t_np)
            dt = dt[dt > 0]
            eps = 1e-3 if dt.size == 0 else float(np.min(dt) * 0.1)
            t_scale = max(1.0, float(np.abs(t_np[-1])) + 1.0)
            eps = max(eps, 1e-9 * t_scale)
        if pad_len > 0:
            t_last = float(t_np[-1])
            # Ensure the increment survives potential float64->float32 truncation when x64 is disabled.
            ulp64 = np.finfo(np.float64).eps * max(1.0, abs(t_last))
            ulp32 = np.finfo(np.float32).eps * max(1.0, abs(t_last))
            step = max(float(eps), float(ulp64), float(ulp32))
            tail = t_last + step * np.arange(1, pad_len + 1, dtype=np.float64)
            t_np = np.concatenate([t_np, tail], axis=0)
        padded.append(t_np)
    return jnp.array(np.stack(padded), dtype=jnp.float64)


def assert_strictly_increasing(ts, name):
    d = np.diff(ts)
    if not np.all(d > 0):
        bad = np.where(d <= 0)[0][:10]
        raise ValueError(f"{name} not strictly increasing; first bad idx: {bad}, d[idx]={d[bad]}")


def load_data(config) -> Tuple[ShotBundle, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = config["data"]["data_dir"]
    shot_list = config["data"]["shots"]
    if shot_list == "all":
        files = glob.glob(os.path.join(data_dir, "*_torax_training.npz"))
    else:
        files = [os.path.join(data_dir, f"{s}_torax_training.npz") for s in shot_list]

    if len(files) == 0:
        raise FileNotFoundError(f"No training packs found in {data_dir}")

    print(f"Loading {len(files)} shots...")

    ref_data = np.load(files[0])
    rho_ref_np = np.array(ref_data["rho"], dtype=float)
    if not np.all(np.diff(rho_ref_np) > 0):
        order_ref = np.argsort(rho_ref_np)
        rho_ref_np = rho_ref_np[order_ref]

    def interp_profile_to_grid(values_t_s, mask_t_s, rho_src, rho_dst):
        Nt = values_t_s.shape[0]
        out_val = np.zeros((Nt, rho_dst.size), dtype=float)
        out_mask = np.zeros_like(out_val)
        for i in range(Nt):
            row_val = values_t_s[i]
            row_mask = mask_t_s[i].astype(bool) & np.isfinite(row_val) & np.isfinite(rho_src)
            if np.count_nonzero(row_mask) == 0:
                continue
            rs = rho_src[row_mask]
            vs = row_val[row_mask]
            ms = mask_t_s[i][row_mask]
            idx = np.argsort(rs)
            rs, vs, ms = rs[idx], vs[idx], ms[idx]
            rs, uniq_idx = np.unique(rs, return_index=True)
            vs = vs[uniq_idx]
            ms = ms[uniq_idx]
            out_val[i] = np.interp(rho_dst, rs, vs, left=0.0, right=0.0)
            out_mask[i] = np.interp(rho_dst, rs, ms, left=0.0, right=0.0)
        out_mask = (out_mask > 0.5).astype(float)
        return out_val, out_mask

    raw_shots: List[dict] = []

    for f in files:
        d = np.load(f)

        ts_t_full = np.array(d["t_ts"], dtype=float)
        ts_Te_raw = np.array(d["Te"], dtype=float)
        mask_raw = np.array(d["Te_mask"], dtype=float)
        ne_raw = np.array(d["ne"], dtype=float)
        ne_mask_raw = np.array(d.get("ne_mask", np.isfinite(ne_raw)), dtype=float)

        mask_full = np.where(np.isnan(ts_Te_raw), 0.0, mask_raw)
        ts_Te_full = np.nan_to_num(ts_Te_raw, nan=0.0, posinf=0.0, neginf=0.0)
        ne_mask_full = np.where(np.isnan(ne_raw), 0.0, ne_mask_raw)
        ne_full = np.nan_to_num(ne_raw, nan=0.0, posinf=0.0, neginf=0.0)

        ctrl_t_full = np.array(d["t"], dtype=float)
        ctrl_vals_full = np.stack([np.array(d[k], dtype=float) for k in CONTROL_NAMES], axis=-1)
        ctrl_vals_full = np.nan_to_num(ctrl_vals_full, nan=0.0, posinf=0.0, neginf=0.0)
        regime_full = np.array(d.get("regime", np.zeros_like(ctrl_t_full)), dtype=float)

        ctrl_order = np.argsort(ctrl_t_full)
        ctrl_t_full = ctrl_t_full[ctrl_order]
        ctrl_vals_full = ctrl_vals_full[ctrl_order]
        regime_full = regime_full[ctrl_order]
        keep_c = np.concatenate([[True], np.diff(ctrl_t_full) > 0])
        ctrl_t_full = ctrl_t_full[keep_c]
        ctrl_vals_full = ctrl_vals_full[keep_c]
        regime_full = regime_full[keep_c]

        rho_src = np.array(d["rho"], dtype=float)
        if (rho_src.shape[0] != rho_ref_np.size) or (not np.allclose(rho_src, rho_ref_np)):
            ts_Te_full, mask_full = interp_profile_to_grid(ts_Te_full, mask_full, rho_src, rho_ref_np)
            ne_full, ne_mask_full = interp_profile_to_grid(ne_full, ne_mask_full, rho_src, rho_ref_np)
            Vprime_src = np.array(d["Vprime"], dtype=float)
            rho_order = np.argsort(rho_src)
            rho_src_sorted = rho_src[rho_order]
            Vprime_src_sorted = Vprime_src[rho_order]
            Vprime_full = np.interp(rho_ref_np, rho_src_sorted, Vprime_src_sorted, left=Vprime_src_sorted[0], right=Vprime_src_sorted[-1])
        else:
            Vprime_full = np.array(d["Vprime"], dtype=float)

        t0 = max(ts_t_full[0], ctrl_t_full[0])
        t1 = min(ts_t_full[-1], ctrl_t_full[-1])
        ts_mask = (ts_t_full >= t0) & (ts_t_full <= t1)
        ctrl_mask = (ctrl_t_full >= t0) & (ctrl_t_full <= t1)

        ts_t = ts_t_full[ts_mask]
        ts_Te = ts_Te_full[ts_mask]
        mask = mask_full[ts_mask]
        ne_vals = ne_full[ts_mask]
        ne_mask = ne_mask_full[ts_mask]

        order = np.argsort(ts_t)
        ts_t, ts_Te, mask, ne_vals, ne_mask = (
            ts_t[order],
            ts_Te[order],
            mask[order],
            ne_vals[order],
            ne_mask[order],
        )
        keep = np.concatenate([[True], np.diff(ts_t) > 0])
        ts_t, ts_Te, mask, ne_vals, ne_mask = (
            ts_t[keep],
            ts_Te[keep],
            mask[keep],
            ne_vals[keep],
            ne_mask[keep],
        )

        regime_ts = np.interp(ts_t, ctrl_t_full, regime_full)
        regime_mask = (
            ((regime_ts > 0.5) & (regime_ts < 1.5)) | ((regime_ts > 2.5) & (regime_ts < 3.5))
        ).astype(np.float64)

        if ts_t.size < 2:
            logging.warning(f"[data] skipping {os.path.basename(f)}: insufficient overlapping ts_t length={ts_t.size}")
            continue

        ctrl_vals_ts = np.stack(
            [
                np.interp(ts_t, ctrl_t_full, ctrl_vals_full[:, i], left=ctrl_vals_full[0, i], right=ctrl_vals_full[-1, i])
                for i in range(ctrl_vals_full.shape[-1])
            ],
            axis=-1,
        )
        ctrl_means = ctrl_vals_ts.mean(axis=0)
        ctrl_stds = ctrl_vals_ts.std(axis=0)

        raw_shots.append(
            dict(
                ts_t=jnp.array(ts_t),
                ts_Te=ts_Te,
                mask=mask,
                ne_vals=ne_vals,
                ne_mask=ne_mask,
                Vprime=Vprime_full,
                regime_ts=jnp.array(regime_ts),
                regime_mask=jnp.array(regime_mask),
                ctrl_t=jnp.array(ts_t),
                ctrl_vals=jnp.array(ctrl_vals_ts),
                ctrl_means=jnp.array(ctrl_means),
                ctrl_stds=jnp.array(ctrl_stds),
                shot_id=int(os.path.basename(f).split("_")[0]),
            )
        )

    if len(raw_shots) == 0:
        raise RuntimeError("No usable shots after overlap/monotonic filtering (raw_shots empty).")

    col_cov_stack = np.stack([s["mask"].mean(axis=0) for s in raw_shots], axis=0)
    col_cov_min = col_cov_stack.min(axis=0)
    tau_cap = float(config["data"].get("intersection_rho_threshold", 0.05))
    I_cap = np.flatnonzero(col_cov_min >= tau_cap)
    if I_cap.size < 2:
        k = min(max(5, rho_ref_np.size // 4), rho_ref_np.size)
        I_cap = np.argsort(col_cov_min)[::-1][:k]
        I_cap = np.sort(I_cap)
    rho_cap = rho_ref_np[I_cap]

    n_int = int(config["data"].get("rom_n_interior", 8))
    rho_cap_min = float(rho_cap.min()) if rho_cap.size > 0 else float(rho_ref_np[1])
    strategy = str(config.get("data", {}).get("rom_interior_strategy", "chebyshev")).lower()

    if strategy == "none":
        rho_int = np.array([], dtype=float)
    elif n_int > 0:
        if strategy == "chebyshev":
            k_int = np.arange(n_int)
            x = 0.5 * (1 - np.cos(np.pi * (k_int + 1) / (n_int + 1)))
            rho_int = rho_cap_min * x
        elif strategy == "uniform":
            rho_int = np.linspace(0.0, rho_cap_min, n_int + 2)[1:-1]
        elif strategy == "edge_refine":
            # Monotone spacing in (0, rho_cap_min) that concentrates nodes near rho_cap_min
            # (coarse near core, fine near the first observed radius).
            power = float(config.get("data", {}).get("rom_edge_refine_power", 3.0))
            power = max(1.0, power)
            s = (np.arange(1, n_int + 1, dtype=float) / (n_int + 1))
            rho_int = rho_cap_min * (1.0 - (1.0 - s) ** power)
        else:
            raise ValueError(f"Unknown data.rom_interior_strategy={strategy!r}")
    else:
        rho_int = np.array([], dtype=float)
    edge_mode = str(config.get("data", {}).get("edge_bc_mode", "use_last_observed")).lower()
    include_rho1 = edge_mode == "extrapolate_to_1"

    parts = [np.array([0.0]), rho_int, rho_cap]
    if include_rho1:
        parts.append(np.array([1.0]))

    rho_rom = np.unique(np.sort(np.concatenate(parts)))
    if rho_rom.size > 2:
        rho_rom[1] = max(rho_rom[1], 1e-3)
        rho_rom = np.unique(np.sort(rho_rom))

    def find_obs_idx(rho_cap_vals, rho_rom_vals):
        idxs = []
        for r in rho_cap_vals:
            idxs.append(int(np.argmin(np.abs(rho_rom_vals - r))))
        if len(idxs) == 0:
            return np.arange(min(5, rho_rom_vals.size), dtype=np.int32)
        seen = set()
        out = []
        for j in idxs:
            if j not in seen:
                seen.add(j)
                out.append(j)
        return np.array(out, dtype=np.int32)

    obs_idx = find_obs_idx(rho_cap, rho_rom)
    if obs_idx.size == 0:
        obs_idx = np.arange(min(5, rho_rom.size), dtype=np.int32)

    # The solver always treats the last ROM node as a Dirichlet boundary value.
    # Therefore, we must ensure `obs_idx` never includes the boundary index.
    # IMPORTANT: do not use `np.clip(..., N-2)` here; clipping can introduce
    # duplicates (e.g. boundary_idx -> N-2), which silently double-weights a
    # column in the loss.
    if rho_rom.size >= 2:
        boundary_idx = int(rho_rom.size - 1)
        obs_idx = obs_idx.astype(np.int32)
        obs_idx = obs_idx[obs_idx != boundary_idx]
        # Keep stable ordering and uniqueness.
        seen = set()
        obs_idx = np.array([j for j in obs_idx.tolist() if (j not in seen and not seen.add(j))], dtype=np.int32)

    if obs_idx.size == 0:
        obs_idx = np.arange(min(5, max(rho_rom.size - 1, 1)), dtype=np.int32)

    edge_idx_ref = int(rho_rom.size - 1)
    rho_edge_ref = float(rho_rom[edge_idx_ref])

    bundles_list = []

    for shot in raw_shots:
        ts_t = shot["ts_t"]
        ts_Te_rom, mask_rom = interp_profile_to_grid(shot["ts_Te"], shot["mask"], rho_ref_np, rho_rom)
        ne_rom, _ = interp_profile_to_grid(shot["ne_vals"], shot["ne_mask"], rho_ref_np, rho_rom)

        Te0 = ts_Te_rom[0]
        if (not np.isfinite(Te0).any()) or (mask_rom[0].sum() == 0):
            Te0 = 100.0 * (1.0 - rho_rom**2) + 10.0

        # Robust edge BC: use outermost observed point per time, extrapolate to rho=1, then time-interp any gaps
        Te_edge_filled = np.empty((ts_Te_rom.shape[0],), dtype=float)

        if edge_mode == "use_last_observed":
            raw = ts_Te_rom[:, edge_idx_ref].astype(float)
            raw_m = (mask_rom[:, edge_idx_ref] > 0.5) & np.isfinite(raw)
            Te_edge_filled[:] = np.nan
            Te_edge_filled[raw_m] = raw[raw_m]

        elif edge_mode == "extrapolate_to_1":
            for i in range(ts_Te_rom.shape[0]):
                m = (mask_rom[i] > 0.5) & np.isfinite(ts_Te_rom[i])
                if not np.any(m):
                    Te_edge_filled[i] = np.nan
                    continue

                idx = np.where(m)[0]
                j1 = idx[-1]
                if j1 == rho_rom.size - 1:
                    Te_edge_filled[i] = ts_Te_rom[i, j1]
                elif idx.size >= 2:
                    j0 = idx[-2]
                    r0, r1 = rho_rom[j0], rho_rom[j1]
                    T0, T1 = ts_Te_rom[i, j0], ts_Te_rom[i, j1]
                    if np.isfinite(r0) and np.isfinite(r1) and (r1 > r0 + 1e-12):
                        slope = (T1 - T0) / (r1 - r0)
                        Te_edge_filled[i] = T1 + slope * (1.0 - r1)
                    else:
                        Te_edge_filled[i] = T1
                else:
                    Te_edge_filled[i] = ts_Te_rom[i, j1]
        else:
            raise ValueError(f"Unknown data.edge_bc_mode={edge_mode!r}")

        if np.all(~np.isfinite(Te_edge_filled)):
            Te_edge_filled = np.full_like(Te_edge_filled, 200.0)
        else:
            ok = np.isfinite(Te_edge_filled)
            if np.count_nonzero(ok) == 1:
                Te_edge_filled[:] = Te_edge_filled[ok][0]
            else:
                Te_edge_filled = np.interp(
                    np.array(ts_t, dtype=float),
                    np.array(ts_t, dtype=float)[ok],
                    Te_edge_filled[ok],
                    left=Te_edge_filled[ok][0],
                    right=Te_edge_filled[ok][-1],
                )

        Te_edge_filled = np.clip(Te_edge_filled, 5.0, 5000.0)

        Vprime_rom = np.interp(rho_rom, rho_ref_np, shot["Vprime"])
        if np.allclose(Vprime_rom, 1.0):
            Vprime_rom = 2.0 * rho_rom
        Vprime_rom = np.clip(Vprime_rom, 0.0, None)
        if Vprime_rom.size > 1:
            core_floor = 0.125 * max(Vprime_rom[1], 1e-6)
            Vprime_rom[0] = max(Vprime_rom[0], core_floor)
        Vprime_rom = np.clip(Vprime_rom, 1e-6, None)

        bundles_list.append({
            "ts_t": ts_t,
            "ts_Te": jnp.array(ts_Te_rom),
            "mask": jnp.array(mask_rom),
            "obs_idx": jnp.array(obs_idx, dtype=jnp.int32),
            "regime_ts": shot["regime_ts"],
            "regime_mask": shot["regime_mask"],
            "Te0": jnp.array(Te0),
            "z0": 0.0,
            "latent_idx": jnp.array([], dtype=jnp.int32),
            "latent_proj": jnp.zeros((0, 0), dtype=jnp.float64),
            "rho_rom": jnp.array(rho_rom),
            "Vprime_rom": jnp.array(Vprime_rom),
            "ctrl_t": shot["ctrl_t"],
            "ctrl_vals": shot["ctrl_vals"],
            "ctrl_means": shot["ctrl_means"],
            "ctrl_stds": shot["ctrl_stds"],
            "ne_vals": jnp.array(ne_rom),
            "Te_edge": jnp.array(Te_edge_filled),
            "edge_idx": jnp.array(edge_idx_ref, dtype=jnp.int32),
            "rho_edge": jnp.array(rho_edge_ref, dtype=jnp.float64),
            "shot_id": jnp.array(shot["shot_id"]),
            "t_len": len(ts_t),
        })

    if len(bundles_list) == 0:
        raise RuntimeError("No bundles constructed (bundles_list empty) after ROM processing.")

    t_len_list = [b["t_len"] for b in bundles_list]
    ts_t_stack = pad_time_to_max_strict([b["ts_t"] for b in bundles_list], t_len_list)
    ts_Te_stack = pad_to_max([b["ts_Te"] for b in bundles_list], mode="constant")
    mask_stack = pad_to_max([b["mask"] for b in bundles_list], mode="constant")
    regime_ts_stack = pad_to_max([b["regime_ts"] for b in bundles_list], mode="edge")
    regime_mask_stack = pad_to_max([b["regime_mask"] for b in bundles_list], mode="constant")
    Te0_stack = jnp.stack([b["Te0"] for b in bundles_list])
    z0_stack = jnp.array([b["z0"] for b in bundles_list])

    rho_rom_stack = jnp.stack([b["rho_rom"] for b in bundles_list])
    Vprime_rom_stack = jnp.stack([b["Vprime_rom"] for b in bundles_list])

    ctrl_t_stack = pad_time_to_max_strict([b["ctrl_t"] for b in bundles_list], t_len_list)
    ctrl_vals_stack = pad_to_max([b["ctrl_vals"] for b in bundles_list], mode="edge")
    ctrl_means_stack = jnp.stack([b["ctrl_means"] for b in bundles_list])
    ctrl_stds_stack = jnp.stack([b["ctrl_stds"] for b in bundles_list])

    ne_vals_stack = pad_to_max([b["ne_vals"] for b in bundles_list], mode="edge")
    Te_edge_stack = pad_to_max([b["Te_edge"] for b in bundles_list], mode="edge")
    edge_idx_stack = jnp.stack([b["edge_idx"] for b in bundles_list])
    rho_edge_stack = jnp.stack([b["rho_edge"] for b in bundles_list])

    shot_id_stack = jnp.stack([b["shot_id"] for b in bundles_list])
    obs_idx_stack = jnp.stack([b["obs_idx"] for b in bundles_list])
    t_len_stack = jnp.array(t_len_list, dtype=jnp.int32)

    ts_t_np = np.array(ts_t_stack)
    ctrl_t_np = np.array(ctrl_t_stack)
    for i in range(ts_t_np.shape[0]):
        assert_strictly_increasing(ts_t_np[i], "ts_t_stack")
        assert_strictly_increasing(ctrl_t_np[i], "ctrl_t_stack")

    latent_idx_stack = jnp.stack([b["latent_idx"] for b in bundles_list]) if bundles_list[0]["latent_idx"].size > 0 else jnp.zeros((len(bundles_list), 0), dtype=jnp.int32)
    latent_proj_stack = jnp.stack([b["latent_proj"] for b in bundles_list]) if bundles_list[0]["latent_proj"].size > 0 else jnp.zeros((len(bundles_list), 0, 0), dtype=jnp.float64)

    stacked_bundle = ShotBundle(
        ts_t_stack,
        ts_Te_stack,
        mask_stack,
        obs_idx_stack,
        regime_ts_stack,
        regime_mask_stack,
        Te0_stack,
        z0_stack,
        latent_idx_stack,
        latent_proj_stack,
        shot_id_stack,
        t_len_stack,
        rho_rom_stack,
        Vprime_rom_stack,
        ctrl_t_stack,
        ctrl_vals_stack,
        ctrl_means_stack,
        ctrl_stds_stack,
        ne_vals_stack,
        Te_edge_stack,
        edge_idx_stack,
        rho_edge_stack,
    )

    print(f"[data] Loaded and stacked {len(bundles_list)} shots.")
    print(f"       Max time steps: {ts_t_stack.shape[1]}")

    return stacked_bundle, rho_rom, rho_cap, obs_idx


def log_data_scale(all_bundles: ShotBundle, obs_idx: jnp.ndarray):
    ts_Te = np.array(all_bundles.ts_Te)
    mask = np.array(all_bundles.mask)
    t_len = np.array(all_bundles.t_len)
    obs = np.array(obs_idx[0] if obs_idx.ndim > 1 else obs_idx)

    time_mask = (np.arange(ts_Te.shape[1])[None, :] < t_len[:, None]).astype(float)
    obs_mask = mask[:, :, obs] * time_mask[:, :, None]
    obs_vals = ts_Te[:, :, obs]

    if np.any(obs_mask > 0):
        vals = obs_vals[obs_mask > 0]
        logging.info(
            "[data] Te(obs) scale: min=%.3f max=%.3f median=%.3f",
            float(np.min(vals)),
            float(np.max(vals)),
            float(np.median(vals)),
        )
    else:
        logging.info("[data] Te(obs) scale: no observed points for stats")

    Te_edge = np.array(all_bundles.Te_edge)
    edge_time_mask = (np.arange(Te_edge.shape[1])[None, :] < t_len[:, None]).astype(float)
    vals = Te_edge[edge_time_mask > 0]
    if vals.size > 0:
        logging.info(
            "[data] Te_edge(BC) scale: min=%.3f max=%.3f median=%.3f",
            float(np.min(vals)),
            float(np.max(vals)),
            float(np.median(vals)),
        )
    else:
        logging.info("[data] Te_edge(BC) scale: no valid times for stats")
