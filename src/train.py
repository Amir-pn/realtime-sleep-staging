"""Step 8 - Train the per-epoch CNN (no context).

Usage:
  python src/train.py --folds data/folds.json --fold 0          # one fold (quick check)
  python src/train.py --folds data/folds.json --fold all        # all 10 folds + pooled metrics
Options: --mmap (low RAM), --epochs, --bs, --lr, --device
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from dataset import SleepData
from labels import STAGE_NAMES
from model import SleepEpochNet

CLASS_W = [1.0, 1.8, 1.0, 1.25, 1.2]   # W, N1, N2, N3, REM (as MMTSleepNet)


def metrics(y, p):
    return dict(ACC=100 * accuracy_score(y, p), MF1=100 * f1_score(y, p, average="macro"),
                kappa=cohen_kappa_score(y, p),
                **{f"F1_{n}": 100 * v for n, v in zip(STAGE_NAMES, f1_score(y, p, average=None, labels=range(5)))})


def augment(x, m):
    B = x.shape[0]
    x = x * torch.empty(B, 3, 1, device=x.device).uniform_(0.8, 1.2)       # amplitude scaling
    x = x + 0.05 * torch.randn_like(x)                                    # noise
    x = torch.roll(x, int(torch.randint(-100, 101, (1,))), dims=-1)       # small time shift
    m = m.clone()
    m[:, 2] *= (torch.rand(B, device=x.device) > 0.2).float()             # EOG dropout (20%)
    return x, m


@torch.no_grad()
def predict(model, data, idx, dev, bs=1024):
    model.eval()
    ys, ps, pr, ii = [], [], [], []
    for b in range(0, len(idx), bs):
        x, f, m, y, i = data.batch(idx[b:b + bs], dev)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            prob = torch.softmax(model(x, f, m).float(), 1)
        ys.append(y.cpu().numpy()); pr.append(prob.cpu().numpy()); ii.append(i)
    pr = np.concatenate(pr)
    return np.concatenate(ys), pr.argmax(1), pr, np.concatenate(ii)


def run_fold(fold, data, a, dev, out):
    torch.manual_seed(a.seed + fold["fold"]); rng = np.random.default_rng(a.seed + fold["fold"])
    tr, va, te = (data.indices(fold[k]) for k in ("train", "val", "test"))
    data.fit_feat_norm(tr)

    model = SleepEpochNet(n_feat=data.feat.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_W, device=dev))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    best, best_ep, ckpt = -1, 0, out / f"{a.tag}_fold{fold['fold']}.pt"
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); perm = rng.permutation(tr); tot = 0.0
        for b in range(0, len(perm), a.bs):
            x, f, m, y, _ = data.batch(perm[b:b + a.bs], dev)
            x, m = augment(x, m)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = crit(model(x, f, m), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y)
        sched.step()
        yv, pv, _, _ = predict(model, data, va, dev)
        mf1 = 100 * f1_score(yv, pv, average="macro")
        flag = ""
        if mf1 > best:
            best, best_ep, flag = mf1, ep, " *"
            torch.save(dict(model=model.state_dict(), f_mu=data.f_mu, f_sd=data.f_sd), ckpt)
        print(f"  ep {ep:2d}  loss {tot / len(perm):.3f}  val MF1 {mf1:.1f}  "
              f"acc {100 * accuracy_score(yv, pv):.1f}  {time.time() - t0:.0f}s{flag}")
        if ep - best_ep >= a.patience:
            print(f"  early stop (best ep {best_ep})"); break

    model.load_state_dict(torch.load(ckpt, weights_only=False)["model"])
    yt, pt, prob, it = predict(model, data, te, dev)
    np.savez(out / f"{a.tag}_fold{fold['fold']}_pred.npz", idx=it, y=yt, p=pt, prob=prob)
    return yt, pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--folds", default="data/folds.json")
    ap.add_argument("--fold", default="0", help="fold number or 'all'")
    ap.add_argument("--out", default="results/cnn")
    ap.add_argument("--tag", default="cnn_edf78")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mmap", action="store_true", help="don't load signals into RAM")
    a = ap.parse_args()

    dev = torch.device(a.device)
    if dev.type != "cuda":
        print("[WARN] training on CPU - this will be very slow")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = SleepData(a.data, a.cache, in_ram=not a.mmap)
    folds = json.load(open(a.folds))
    if a.fold != "all":
        folds = [f for f in folds if f["fold"] == int(a.fold)]
    n_par = sum(p.numel() for p in SleepEpochNet(n_feat=data.feat.shape[1]).parameters())
    print(f"model params: {n_par:,}  device: {dev}")

    y_all, p_all, rows = [], [], []
    for f in folds:
        print(f"\n=== fold {f['fold']} ===")
        yt, pt = run_fold(f, data, a, dev, out)
        m = metrics(yt, pt); rows.append(dict(fold=f["fold"], **m))
        print(f"fold {f['fold']} TEST: ACC {m['ACC']:.1f}  MF1 {m['MF1']:.1f}  k {m['kappa']:.3f}")
        y_all.append(yt); p_all.append(pt)

    y_all, p_all = np.concatenate(y_all), np.concatenate(p_all)
    overall = metrics(y_all, p_all)
    name = f"{a.tag}_fold{a.fold}"
    pd.DataFrame(rows).to_csv(out / f"{name}_folds.csv", index=False)
    json.dump(overall, open(out / f"{name}_overall.json", "w"), indent=1)
    print(f"\n=== Pooled ({len(folds)} fold(s)) ===")
    print("  " + "  ".join(f"{k} {v:.3f}" if k == "kappa" else f"{k} {v:.1f}" for k, v in overall.items()))
    print(pd.DataFrame(confusion_matrix(y_all, p_all, labels=range(5)), index=STAGE_NAMES, columns=STAGE_NAMES))


if __name__ == "__main__":
    main()
