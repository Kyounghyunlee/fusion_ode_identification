"""Normalization utilities for TORAX training inputs.

Provides simple z-score normalization with persistent stats.
Stats can be serialized to JSON and reused across training runs
for consistent normalization of inputs (P_nbi, Ip, nebar, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from typing import Dict
import numpy as np

@dataclass
class NormStats:
    mean: float
    std: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-12)

    @staticmethod
    def from_array(x: np.ndarray) -> "NormStats":
        finite = np.isfinite(x)
        if not finite.any():
            return NormStats(0.0, 1.0)
        return NormStats(float(np.nanmean(x[finite])), float(np.nanstd(x[finite]) + 1e-12))


def compute_stats(inputs: Dict[str, np.ndarray]) -> Dict[str, NormStats]:
    """Compute normalization stats for each input key."""
    return {k: NormStats.from_array(v) for k, v in inputs.items()}


def normalize_inputs(inputs: Dict[str, np.ndarray], stats: Dict[str, NormStats]) -> Dict[str, np.ndarray]:
    """Apply provided stats to input dictionary."""
    return {k: stats[k].apply(v) if k in stats else v for k, v in inputs.items()}


def save_stats(stats: Dict[str, NormStats], path: str) -> None:
    with open(path, "w") as f:
        json.dump({k: asdict(v) for k, v in stats.items()}, f, indent=2)


def load_stats(path: str) -> Dict[str, NormStats]:
    with open(path) as f:
        raw = json.load(f)
    return {k: NormStats(**v) for k, v in raw.items()}
