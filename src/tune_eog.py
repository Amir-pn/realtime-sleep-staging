"""Tune the EOG eye-movement detector: which filter + slope threshold best separates REM from N2/N3.

Usage:  python src/tune_eog.py --data data/processed
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from sklearn.metrics import roc_auc_score

from stream import FS, SCALE_UV

FILTERS = {
    "lp5 (current)": butter(4, 5, btype="low", fs=FS, output="sos"),
    "bp1-5": butter(4, [1, 5], btype="band", fs=FS, output="sos"),
    "bp2-8": butter(4, [2, 8], btype="band", fs=FS, output="sos"),
}
THRESHOLDS = [300, 400, 600, 800, 1000, 1500]   # µV/s


def count(eog, sos, thr):
    lp = sosfiltfilt(sos, eog, axis=-1)
    slope = np.abs(np.diff(lp, axis=-1)) * FS
    return np.array([len(find_peaks(s, height=thr, distance=int(0.3 * FS))[0]) for s in slope])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--per_stage", type=int, default=30)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    eog, lab = [], []
    for f in sorted(Path(a.data).glob("SC*.npz")):
        d = np.load(f)
        s, e = d["trim"]
        y, st = d["labels"][s:e], d["status"][s:e]
        for k in (0, 2, 3, 4):                       # W, N2, N3, REM
            idx = np.where((y == k) & (st == 0))[0]
            if len(idx):
                pick = rng.choice(idx, min(a.per_stage, len(idx)), replace=False)
                eog.append(d["signals"][pick, 2].astype(np.float64) * SCALE_UV[2, 0])
                lab.append(np.full(len(pick), k))
    eog, lab = np.concatenate(eog), np.concatenate(lab)
    print(f"{len(lab)} epochs sampled")

    rows = []
    rem_vs_nrem = np.isin(lab, [2, 3, 4])
    for fname, sos in FILTERS.items():
        for thr in THRESHOLDS:
            c = count(eog, sos, thr)
            auc = roc_auc_score(lab[rem_vs_nrem] == 4, c[rem_vs_nrem])
            med = {n: np.median(c[lab == k]) for n, k in [("W", 0), ("N2", 2), ("N3", 3), ("REM", 4)]}
            rows.append(dict(filter=fname, thr=thr, AUC_REM_vs_N2N3=round(auc, 3), **med))

    df = pd.DataFrame(rows).sort_values("AUC_REM_vs_N2N3", ascending=False)
    print(df.to_string(index=False))
    best = df.iloc[0]
    print(f"\nBest: filter={best['filter']}, threshold={best['thr']} µV/s, AUC={best['AUC_REM_vs_N2N3']}")
