"""Quick look at a labeled recording from `stress_cam.py run --record <name>`.

  python analyze.py recordings/session1_20261116-120000.csv

Prints the mean and std of each stress cue per label, and (if both are present) the AUC of
the stress score for telling "calm" apart from "stressed".
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

from face_features import CUE_NAMES


def to_float(value: str) -> float:
    return math.nan if value == "" else float(value)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="a recording CSV written by `stress_cam.py run --record`")
    args = parser.parse_args(argv)

    with open(args.csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["face_found"] == "1"]
    if not rows:
        raise SystemExit("No rows with a face were found in this recording.")

    labels = sorted({r["label"] for r in rows})
    print(f"{len(rows)} frames with a face, {len(labels)} labels seen: {', '.join(labels)}\n")

    header = f"{'label':<12} {'n':>6}  " + "  ".join(f"{name:>16}" for name in CUE_NAMES)
    print(header)
    for label in labels:
        sub = [r for r in rows if r["label"] == label]
        cells = []
        for name in CUE_NAMES:
            values = np.array([to_float(r[name]) for r in sub])
            cells.append(f"{np.nanmean(values):6.3f} +- {np.nanstd(values):5.3f}")
        print(f"{label:<12} {len(sub):>6}  " + "  ".join(f"{c:>16}" for c in cells))

    calm = np.array([to_float(r["stress"]) for r in rows if r["label"] == "calm"])
    stressed = np.array([to_float(r["stress"]) for r in rows if r["label"] == "stressed"])
    calm, stressed = calm[~np.isnan(calm)], stressed[~np.isnan(stressed)]
    if len(calm) == 0 or len(stressed) == 0:
        print("\nNeed both 'calm' and 'stressed' frames with a stress score to compute an AUC.")
        return

    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.zeros(len(calm)), np.ones(len(stressed))])
    scores = np.concatenate([calm, stressed])
    print(f"\nstress score AUC, calm ({len(calm)}) vs stressed ({len(stressed)}): "
          f"{roc_auc_score(y, scores):.3f} (0.5 = guessing, 1.0 = perfect)")


if __name__ == "__main__":
    main(sys.argv[1:])
