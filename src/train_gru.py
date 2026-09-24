"""Step 9a - Causal GRU on frozen CNN embeddings (per fold).

Usage:  python src/train_gru.py --folds data/folds.json
Needs:  results/cnn/cnn_edf78_fold<k>.pt for every fold (Step 8)
Output: results/gru/gru_edf78_fold<k>.pt, *_pred.npz, pooled metrics for provisional + revised outputs
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score

from dataset import SleepData
from labels import STAGE_NAMES
from model import SleepEpochNet, SleepSeqNet
from stream import STATUS_BAD
from train import CLASS_W, metrics

L = 20   # context: 20 epochs = 10 min


@torch.no_grad()
def embed_all(data, ck, dev):
    enc = SleepEpochNet(n_feat=data.feat.shape[1]).to(dev)
    enc.load_state_dict(ck["model"]); enc.eval()
    data.f_mu, data.f_sd = ck["f_mu"], ck["f_sd"]
    N = len(data.y)
    out = torch.zeros(N, enc.proj[1].out_features, device=dev)
    for b in range(0, N, 1024):
        idx = np.arange(b, min(b + 1024, N))
        x, f, m, _, _ = data.batch(idx, dev)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            out[b:b + len(idx)] = enc.embed(x, f, m).float()
    return out


class Seq:
    """Builds causal windows on the GPU from contiguous per-night cache positions."""
    def __init__(self, data, emb, dev):
        self.emb = emb
        self.night = torch.from_numpy(data.night.astype(np.int64)).to(dev)
        self.bad = torch.from_numpy(data.status == STATUS_BAD).to(dev)
        self.y = torch.from_numpy(data.y.astype(np.int64)).to(dev)
        self.offs = torch.arange(-L + 1, 1, device=dev)
        self.N = len(data.y)

    def windows(self, T):
        w = T[:, None] + self.offs
        wc = w.clamp(min=0)
        v = (w >= 0) & (self.night[wc] == self.night[T][:, None]) & ~self.bad[wc]
        return self.emb[wc], v.float()

    def targets(self, T):
        ok_t = (self.y[T] >= 0) & ~self.bad[T]
        yp = torch.where(ok_t, self.y[T], torch.full_like(T, -100))
        P = (T - 1).clamp(min=0)
        ok_p = (T > 0) & (self.night[P] == self.night[T]) & (self.y[P] >= 0) & ~self.bad[P]
        yr = torch.where(ok_p, self.y[P], torch.full_like(T, -100))
        return yp, yr


@torch.no_grad()
def predict(net, S, E, bs=4096):
    """E: evaluation epochs. provisional = window ending at e; revised = window ending at e+1."""
    net.eval()
    pp, pr = [], []
    for b in range(0, len(E), bs):
        e = E[b:b + bs]
        lp, _ = net(*S.windows(e)); pp.append(torch.softmax(lp, 1))
        n = (e + 1).clamp(max=S.N - 1)
        same = ((e + 1) < S.N) & (S.night[n] == S.night[e])
        _, lr = net(*S.windows(n))
        r = torch.softmax(lr, 1)
        r[~same] = pp[-1][~same]                       # last epoch of night: no future -> provisional
        pr.append(r)
    return torch.cat(pp).cpu().numpy(), torch.cat(pr).cpu().numpy()


def run_fold(fold, data, a, dev, out):
    torch.manual_seed(a.seed + fold["fold"])
    ck = torch.load(Path(a.cnn_dir) / f"{a.cnn_tag}_fold{fold['fold']}.pt", map_location=dev, weights_only=False)
    t0 = time.time()
    emb = embed_all(data, ck, dev)
    S = Seq(data, emb, dev)
    tr = torch.from_numpy(data.indices(fold["train"], valid_only=False)).to(dev)
    va = torch.from_numpy(data.indices(fold["val"])).to(dev)
    te = torch.from_numpy(data.indices(fold["test"])).to(dev)
    print(f"  embeddings: {time.time() - t0:.0f}s")

    net = SleepSeqNet(emb=emb.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_W, device=dev), ignore_index=-100)
    yv = data.y[va.cpu().numpy()]
    best, ckpt = -1, out / f"{a.tag}_fold{fold['fold']}.pt"
    for ep in range(a.epochs):
        net.train(); perm = tr[torch.randperm(len(tr), device=dev)]; tot = 0.0
        for b in range(0, len(perm), a.bs):
            T = perm[b:b + a.bs]
            z, v = S.windows(T)
            yp, yr = S.targets(T)
            lp, lr = net(z, v)
            loss = crit(lp, yp) + crit(lr, yr)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * len(T)
        sched.step()
        pp, pr = predict(net, S, va)
        mf1 = 100 * f1_score(yv, pp.argmax(1), average="macro")
        flag = ""
        if mf1 > best:
            best, flag = mf1, " *"
            torch.save(dict(model=net.state_dict(), L=L), ckpt)
        print(f"  ep {ep:2d}  loss {tot / len(perm):.3f}  val MF1 prov {mf1:.1f}  "
              f"rev {100 * f1_score(yv, pr.argmax(1), average='macro'):.1f}{flag}")

    net.load_state_dict(torch.load(ckpt, weights_only=False)["model"])
    pp, pr = predict(net, S, te)
    yt = data.y[te.cpu().numpy()]
    np.savez(out / f"{a.tag}_fold{fold['fold']}_pred.npz", idx=te.cpu().numpy(), y=yt,
             prov=pp.argmax(1), rev=pr.argmax(1), prob_prov=pp, prob_rev=pr)
    return yt, pp.argmax(1), pr.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--folds", default="data/folds.json")
    ap.add_argument("--cnn_dir", default="results/cnn")
    ap.add_argument("--cnn_tag", default="cnn_edf78")
    ap.add_argument("--out", default="results/gru")
    ap.add_argument("--tag", default="gru_edf78")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mmap", action="store_true")
    a = ap.parse_args()

    dev = torch.device(a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    folds = json.load(open(a.folds))
    missing = [f["fold"] for f in folds
               if not (Path(a.cnn_dir) / f"{a.cnn_tag}_fold{f['fold']}.pt").exists()]
    if missing:
        raise FileNotFoundError(f"CNN checkpoints missing for folds {missing} - finish Step 8 first")
    data = SleepData(a.data, a.cache, in_ram=not a.mmap)
    print(f"GRU params: {sum(p.numel() for p in SleepSeqNet().parameters()):,}  context {L} epochs  device {dev}")

    Y, P, R = [], [], []
    for f in folds:
        print(f"\n=== fold {f['fold']} ===")
        y, p, r = run_fold(f, data, a, dev, out)
        mp, mr = metrics(y, p), metrics(y, r)
        print(f"fold {f['fold']} TEST  provisional: ACC {mp['ACC']:.1f} MF1 {mp['MF1']:.1f} k {mp['kappa']:.3f}"
              f"  |  revised: ACC {mr['ACC']:.1f} MF1 {mr['MF1']:.1f} k {mr['kappa']:.3f}")
        Y.append(y); P.append(p); R.append(r)

    Y, P, R = np.concatenate(Y), np.concatenate(P), np.concatenate(R)
    for name, pred in [("prov", P), ("rev", R)]:
        m = metrics(Y, pred)
        json.dump(m, open(out / f"{a.tag}_{name}_overall.json", "w"), indent=1)
        print(f"\n=== Pooled {len(folds)} folds - {'provisional (live)' if name == 'prov' else 'revised (+30 s)'} ===")
        print("  " + "  ".join(f"{k} {v:.3f}" if k == "kappa" else f"{k} {v:.1f}" for k, v in m.items()))
        print(pd.DataFrame(confusion_matrix(Y, pred, labels=range(5)), index=STAGE_NAMES, columns=STAGE_NAMES))


if __name__ == "__main__":
    main()
