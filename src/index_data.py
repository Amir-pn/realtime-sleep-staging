"""Step 2 - Index Sleep-EDF sleep-cassette (SC) recordings.

Usage:  python src/index_data.py --raw data/raw --out data/index.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def index_sc(raw_dir):
    raw_dir = Path(raw_dir)
    psgs = sorted(raw_dir.rglob("SC4*-PSG.edf"))
    # PSG 'SC4001E0' pairs with hypnogram 'SC4001EC' -> match on first 6 chars
    hyps = {h.name[:6]: h for h in raw_dir.rglob("SC4*-Hypnogram.edf")}

    rows, missing = [], []
    for p in psgs:
        key = p.name[:6]
        if key not in hyps:
            missing.append(p.name)
            continue
        rows.append(dict(key=key, subject=int(key[3:5]), night=int(key[5]),
                         psg=str(p), hyp=str(hyps[key])))

    if missing:
        print(f"[WARN] {len(missing)} PSG files without hypnogram: {missing}")
    if not rows:
        raise FileNotFoundError(f"No SC PSG/hypnogram pairs found under {raw_dir}")

    df = pd.DataFrame(rows)
    print(f"Found {len(df)} nights from {df.subject.nunique()} subjects "
          f"(expected 153 nights / 78 subjects)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/index.csv")
    a = ap.parse_args()
    df = index_sc(a.raw)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Saved {a.out}")
