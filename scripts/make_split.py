"""Create the grouped train/validation/test split.

Groups are experimental sessions, identified by contiguity of shot numbers
(a gap of more than 20 shot numbers starts a new session). Neighboring MAST
shots share machine setup and intent, so splitting by session prevents
session leakage into the held-out sets.

Role assignment is deterministic and blind to model performance: groups are
ordered by their smallest shot number and assigned by SHA-256 hash of the
group key, targeting ~20% of shots for the locked TEST set and ~20% of the
remainder for VALIDATION. The test set must be evaluated once, at the end.

Usage: python scripts/make_split.py  ->  data/split.json
"""

import glob
import hashlib
import json
import os

GAP = 20
TEST_FRAC = 0.20
VAL_FRAC = 0.20


def main():
    shots = sorted(
        int(os.path.basename(p).split("_")[0]) for p in glob.glob("data/*_torax_training.npz")
    )
    if not shots:
        raise SystemExit("no packs found")

    groups = []
    cur = [shots[0]]
    for s in shots[1:]:
        if s - cur[-1] > GAP:
            groups.append(cur)
            cur = [s]
        else:
            cur.append(s)
    groups.append(cur)

    def h(g):
        return int(hashlib.sha256(f"session-{g[0]}".encode()).hexdigest(), 16) / 2**256

    ordered = sorted(groups, key=h)
    n_total = len(shots)
    test, rest = [], []
    n_test = 0
    for g in ordered:
        if n_test < TEST_FRAC * n_total:
            test.append(g)
            n_test += len(g)
        else:
            rest.append(g)
    val, train = [], []
    n_val = 0
    n_rest = sum(len(g) for g in rest)
    for g in sorted(rest, key=h):
        if n_val < VAL_FRAC * n_rest:
            val.append(g)
            n_val += len(g)
        else:
            train.append(g)

    split = {
        "rule": f"session groups by shot-number gap > {GAP}; SHA-256 hash order; "
                f"~{TEST_FRAC:.0%} shots to locked test, ~{VAL_FRAC:.0%} of remainder to validation",
        "groups": [{"session_start": g[0], "shots": g} for g in groups],
        "train": sorted(s for g in train for s in g),
        "val": sorted(s for g in val for s in g),
        "test": sorted(s for g in test for s in g),
    }
    with open("data/split.json", "w") as f:
        json.dump(split, f, indent=1)
    print(f"{len(groups)} sessions | train {len(split['train'])} / val {len(split['val'])} / TEST(locked) {len(split['test'])}")
    print("test sessions:", [g[0] for g in test])


if __name__ == "__main__":
    main()
