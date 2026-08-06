"""Fetch session-log annotations for every corpus shot from the MAST
catalogue API. Independent evidence for the label audit.

Writes data/session_logs.json: {shot: {"preshot":..., "postshot":..., "heating":...}}
"""

import glob
import json
import os
import sys

import requests

API = "https://mastapp.site/json/shots"


def main():
    shots = sorted(
        int(os.path.basename(p).split("_")[0]) for p in glob.glob("data/*_torax_training.npz")
    )
    out = {}
    if os.path.exists("data/session_logs.json"):
        with open("data/session_logs.json") as f:
            out = json.load(f)
    for s in shots:
        if str(s) in out:
            continue
        try:
            r = requests.get(API, params={"filters": f"shot_id$eq:{s}", "size": 1}, timeout=30)
            items = r.json().get("items", [])
            if items:
                it = items[0]
                out[str(s)] = {
                    "heating": it.get("heating"),
                    "preshot": (it.get("preshot_description") or "").strip(),
                    "postshot": (it.get("postshot_description") or "").strip(),
                    "campaign": it.get("campaign"),
                }
        except Exception as e:
            print(f"{s}: ERR {e}", file=sys.stderr)
    with open("data/session_logs.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"fetched {len(out)}/{len(shots)} session logs")


if __name__ == "__main__":
    main()
