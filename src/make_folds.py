"""Step 6 - Subject-wise folds.

Usage:
  python src/make_folds.py --data data/processed --out data/folds.json            # EDF-78: 10-fold
  python src/make_folds.py --data data/processed --out data/folds20.json --edf20  # EDF-20: LOSO (20 folds)

Each fold: test subjects, and training subjects split into train / val (~10%, for early stopping).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="data/folds.json")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--edf20", action="store_true", help="subjects 0-19, leave-one-subject-out")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    subjects = sorted({int(np.load(f)["subject"]) for f in Path(a.data).glob("SC*.npz")})
    if a.edf20:
        subjects = [s for s in subjects if s < 20]
        k = len(subjects)
    else:
        k = a.k
    if len(subjects) < k:
        raise ValueError(f"Only {len(subjects)} subjects for {k} folds")

    rng = np.random.default_rng(a.seed)
    subjects = np.array(subjects)
    folds = []
    for i, (tr, te) in enumerate(KFold(k, shuffle=True, random_state=a.seed).split(subjects)):
        tr_subj = rng.permutation(subjects[tr])
        n_val = max(1, int(round(len(tr_subj) * a.val_frac)))
        folds.append(dict(fold=i,
                          train=sorted(tr_subj[n_val:].tolist()),
                          val=sorted(tr_subj[:n_val].tolist()),
                          test=sorted(subjects[te].tolist())))

    # leakage check: no subject in two sets of the same fold; every subject tested once
    for f in folds:
        assert not (set(f["train"]) & set(f["val"]) or set(f["train"]) & set(f["test"])
                    or set(f["val"]) & set(f["test"])), f"leakage in fold {f['fold']}"
    tested = sorted(s for f in folds for s in f["test"])
    assert tested == sorted(subjects.tolist()), "not every subject tested exactly once"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(folds, open(a.out, "w"), indent=1)
    print(f"{k} folds, {len(subjects)} subjects -> {a.out}")
    print(f"fold 0: train {len(folds[0]['train'])}, val {len(folds[0]['val'])}, test {len(folds[0]['test'])} subjects")


if __name__ == "__main__":
    main()
