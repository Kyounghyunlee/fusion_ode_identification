"""Audit weak labels against independent session-log annotations.

Cohorts:
  confirmed_transition - extractor found >=1 H segment AND the session log
                         affirms H-mode / L-H transition.
  confirmed_negative   - extractor found none AND the log does not affirm
                         H-mode (or explicitly denies it) - these support
                         the negative-control claim.
  ambiguous            - extractor and log disagree, or the log is empty.

Writes data/label_audit.json with per-shot evidence snippets.
"""

import glob
import json
import os
import re

import numpy as np

POS = re.compile(r"(l-?h\s+transition|enters?\s+h-?mode|h-?mode\s+(from|at)|into\s+h-?mode|elmy\s+h-?mode|elm\s*free\s*h?-?mode|h-?mode\s+lost|loose\s+h-?mode|good\s+h-?mode|long\s+h-?mode)", re.I)
NEG = re.compile(r"(no\s+h-?mode|not\s+.{0,20}h-?mode|close\s+to\s+h-?mode|almost\s+h-?mode|fail(ed)?\s+to\s+.{0,20}h-?mode|no\s+l-?h)", re.I)


def main():
    with open("data/session_logs.json") as f:
        logs = json.load(f)
    out = {"cohorts": {"confirmed_transition": [], "confirmed_negative": [], "ambiguous": []}, "shots": {}}
    for p in sorted(glob.glob("data/*_torax_training.npz")):
        shot = os.path.basename(p).split("_")[0]
        d = np.load(p, allow_pickle=True)
        has_h = bool(np.any(d["regime"] == 3))
        log = logs.get(shot, {})
        text = " ".join([log.get("preshot", "") or "", log.get("postshot", "") or ""])
        neg_hit = bool(NEG.search(text))
        pos_hit = bool(POS.search(text)) and not neg_hit
        heating = (log.get("heating") or "").lower()
        ohmic = "ohmic" in heating or heating == ""

        if has_h and pos_hit:
            cohort = "confirmed_transition"
        elif not has_h and not pos_hit:
            cohort = "confirmed_negative"
        else:
            cohort = "ambiguous"
        out["cohorts"][cohort].append(int(shot))
        out["shots"][shot] = {
            "extractor_H": has_h,
            "log_positive": pos_hit,
            "log_negative_phrase": neg_hit,
            "heating": log.get("heating"),
            "postshot_snippet": (log.get("postshot", "") or "")[:220],
            "cohort": cohort,
        }
    with open("data/label_audit.json", "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out["cohorts"].items():
        print(f"{k}: {len(v)}")
    print("ambiguous shots:", out["cohorts"]["ambiguous"])


if __name__ == "__main__":
    main()
