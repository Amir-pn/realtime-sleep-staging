"""Run StreamProcessor over every night (as a simulated stream) and save one .npz per night.

Usage:  python src/build_dataset.py --index data/index.csv --out data/processed --jobs 4

Saved per night:
  signals   (n_trim, 3, 3000) float16  scaled epochs, trimmed range only (keeps disk ~3 GB)
  features  (n_all, 34)       float32  all epochs of the night
  qc_flags  (n_all, 3)  status (n_all,)  warmup (n_all,)
  labels    (n_all,)          int8     -1 = ignore (movement / unscored)
  trim      [start, end)      trimmed range -> signals[i] == epoch start+i
"""
import argparse
import traceback
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from labels import STAGE_NAMES, epoch_labels, trim_range
from stream import CH_NAMES, EPOCH, FS, StreamProcessor

EXPECTED_EDF20 = {"W": 8285, "N1": 2804, "N2": 17799, "N3": 5703, "REM": 7717}


def process_night(row, out_dir):
    raw = mne.io.read_raw_edf(row["psg"], include=CH_NAMES, preload=True, verbose="ERROR")
    missing = set(CH_NAMES) - set(raw.ch_names)
    if missing:
        raise ValueError(f"missing channels {missing}; available: {raw.ch_names}")
    if raw.info["sfreq"] != FS:
        raise ValueError(f"sfreq {raw.info['sfreq']} != {FS}")

    x = raw.get_data(picks=CH_NAMES, units="uV")
    n_ep = x.shape[1] // EPOCH
    y = epoch_labels(row["hyp"], n_ep, raw)
    start, end = trim_range(y)

    proc = StreamProcessor()
    sigs = np.zeros((end - start, 3, EPOCH), dtype=np.float16)
    feats, flags, status, warm = [], [], [], []
    for k in range(n_ep):
        o = proc.process(x[:, k * EPOCH:(k + 1) * EPOCH])
        feats.append(o["features"]); flags.append(o["qc_flags"])
        status.append(o["status"]); warm.append(o["warmup"])
        if start <= k < end:
            sigs[k - start] = o["signal"]

    np.savez_compressed(
        Path(out_dir) / f"{row['key']}.npz",
        signals=sigs, features=np.stack(feats), qc_flags=np.stack(flags),
        status=np.array(status, np.int8), warmup=np.array(warm, bool),
        labels=y, trim=np.array([start, end]),
        subject=row["subject"], night=row["night"],
    )
    yt = y[start:end]
    return dict(key=row["key"], subject=row["subject"], n_all=n_ep, n_trim=end - start,
                bad_epochs=int(np.sum(np.array(status)[start:end] == 2)),
                **{s: int(np.sum(yt == i)) for i, s in enumerate(STAGE_NAMES)})


def safe(row, out_dir):
    try:
        return process_night(row, out_dir), None
    except Exception as e:
        return None, (row["key"], f"{type(e).__name__}: {e}", traceback.format_exc())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(a.index).to_dict("records")
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(safe)(r, out) for r in rows)

    ok = [r for r, e in res if r is not None]
    errs = [e for r, e in res if e is not None]

    if ok:
        df = pd.DataFrame(ok).sort_values("key")
        df.to_csv(out / "summary.csv", index=False)
        print("\nClass counts (trimmed):")
        print("  All SC      :", df[STAGE_NAMES].sum().to_dict())
        print("  Subjects<20 :", df[df.subject < 20][STAGE_NAMES].sum().to_dict())
        print("  Expected EDF-20 (MMTSleepNet):", EXPECTED_EDF20)
        print("  BAD-signal epochs in trimmed range:", int(df.bad_epochs.sum()))

    if errs:
        with open(out / "errors.log", "w") as f:
            for key, msg, tb in errs:
                f.write(f"=== {key}: {msg}\n{tb}\n")
        print(f"\n[ERROR] {len(errs)} nights failed -> {out/'errors.log'}")
        for key, msg, _ in errs:
            print(f"  {key}: {msg}")
        raise SystemExit(1)
    print(f"\nDone: {len(ok)} nights processed.")
