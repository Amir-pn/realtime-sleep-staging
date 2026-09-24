"""Step 7 - XGBoost baseline on handcrafted features (the floor the deep model must beat).

Usage:
  python src/baseline_xgb.py --folds data/folds.json --tag edf78               # per-epoch only
  python src/baseline_xgb.py --folds data/folds.json --tag edf78 --context 2   # + 2 previous epochs (causal)

Training/testing on trimmed epochs, labels >= 0, status != BAD.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from labels import STAGE_NAMES
from stream import STATUS_BAD


def load(data_dir, context):
    X, y, subj = [], [], []
    for f in sorted(Path(data_dir).glob("SC*.npz")):
        d = np.load(f)                       # lazy: signals are never decompressed here
        F = d["features"]
        if context:                          # causal lags: epoch t sees t-1 ... t-context
            lags = [F] + [np.vstack([np.zeros((k, F.shape[1]), F.dtype), F[:-k]])
                          for k in range(1, context + 1)]
            F = np.hstack(lags)
        s, e = d["trim"]
        yy, st, FF = d["labels"][s:e], d["status"][s:e], F[s:e]
        m = (yy >= 0) & (st != STATUS_BAD)
        X.append(FF[m]); y.append(yy[m]); subj.append(np.full(m.sum(), int(d["subject"])))
    return np.vstack(X), np.concatenate(y), np.concatenate(subj)


def metrics(y, p):
    return dict(ACC=100 * accuracy_score(y, p), MF1=100 * f1_score(y, p, average="macro"),
                kappa=cohen_kappa_score(y, p),
                **{f"F1_{n}": 100 * v for n, v in zip(STAGE_NAMES, f1_score(y, p, average=None))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--folds", default="data/folds.json")
    ap.add_argument("--out", default="results/baseline_xgb")
    ap.add_argument("--tag", default="edf78")
    ap.add_argument("--context", type=int, default=0)
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    a = ap.parse_args()

    folds = json.load(open(a.folds))
    X, y, subj = load(a.data, a.context)
    print(f"{len(y)} epochs, {X.shape[1]} features, context={a.context}")

    y_all, p_all, rows = [], [], []
    for f in folds:
        tr, va, te = (np.isin(subj, f[k]) for k in ("train", "val", "test"))
        if not (tr.any() and va.any() and te.any()):
            raise ValueError(f"fold {f['fold']}: empty split - check folds file vs data")
        clf = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                            device=a.device, early_stopping_rounds=50,
                            eval_metric="mlogloss")
        clf.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]),
                eval_set=[(X[va], y[va])], verbose=False)
        p = clf.predict(X[te])
        m = metrics(y[te], p)
        rows.append(dict(fold=f["fold"], **m))
        print(f"fold {f['fold']:2d}: ACC {m['ACC']:.1f}  MF1 {m['MF1']:.1f}  k {m['kappa']:.3f}")
        y_all.append(y[te]); p_all.append(p)

    y_all, p_all = np.concatenate(y_all), np.concatenate(p_all)
    overall = metrics(y_all, p_all)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = f"{a.tag}_ctx{a.context}"
    pd.DataFrame(rows).to_csv(out / f"{name}_folds.csv", index=False)
    pd.DataFrame(confusion_matrix(y_all, p_all), index=STAGE_NAMES, columns=STAGE_NAMES) \
        .to_csv(out / f"{name}_confusion.csv")
    json.dump(overall, open(out / f"{name}_overall.json", "w"), indent=1)

    print(f"\n=== Pooled over {len(folds)} folds ({name}) ===")
    print("  " + "  ".join(f"{k} {v:.3f}" if k == "kappa" else f"{k} {v:.1f}" for k, v in overall.items()))
    print(pd.DataFrame(confusion_matrix(y_all, p_all), index=STAGE_NAMES, columns=STAGE_NAMES))


if __name__ == "__main__":
    main()
