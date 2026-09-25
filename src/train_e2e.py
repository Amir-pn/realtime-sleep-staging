"""Step 9b - End-to-end fine-tuning: unfreeze the CNN encoder and train it together with the causal GRU.

Starts from the Step 8 CNN and Step 9a GRU checkpoints of the same fold.
Training uses contiguous chunks of C epochs (+19 epochs of past context), so every epoch passes
through the CNN about once per pass, and every target sees exactly the same 20-epoch causal window
it sees at inference.

Usage:  python src/train_e2e.py --folds data/folds.json            (add --resume to skip finished folds)
Output: results/e2e/e2e_edf78_fold<k>.pt (encoder + GRU), *_pred.npz, pooled metrics (prov / rev)
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
from stream import N_FPZ, N_PZ, STATUS_BAD
from train import CLASS_W, augment, metrics
from train_gru import L, Seq, predict


@torch.no_grad()
def embed_all(enc, data, dev, bs=1024):
    enc.eval()
    N = len(data.y)
    out = torch.zeros(N, enc.proj[1].out_features, device=dev)
    for b in range(0, N, bs):
        idx = np.arange(b, min(b + bs, N))
        x, f, m, _, _ = data.batch(idx, dev)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            out[b:b + len(idx)] = enc.embed(x, f, m).float()
    return out


class Chunks:
    """Contiguous chunks inside one night: C target epochs + (L-1) past-context epochs."""
    def __init__(self, data, subjects, C, dev):
        self.d, self.C, self.dev = data, C, dev
        self.bad = data.status == STATUS_BAD
        self.nights = []
        for n in np.unique(data.night[np.isin(data.subj, subjects)]):
            pos = np.where(data.night == n)[0]          # cache is contiguous per night
            self.nights.append((pos[0], pos[-1] + 1))

    def starts(self, rng):
        s = []
        for ns, ne in self.nights:
            st = ns + int(rng.integers(0, self.C))       # random offset -> all positions covered over epochs
            while st + self.C <= ne:
                s.append((st, ns)); st += self.C
        rng.shuffle(s)
        return s

    def load(self, batch):
        d, C = self.d, self.C
        P = np.stack([np.arange(st - (L - 1), st + C) for st, _ in batch])        # (B, C+L-1) positions
        NS = np.array([ns for _, ns in batch])[:, None]
        ctx_ok = P >= NS                                                           # context before night start = missing
        Pc = np.maximum(P, NS)
        flat = Pc.ravel()
        x = d.sig[flat].astype(np.float32)
        m = (d.qc[flat] == 0).astype(np.float32)
        f = (d.feat[flat] - d.f_mu) / d.f_sd
        f[:, :N_FPZ] *= m[:, 0:1]; f[:, N_FPZ:N_FPZ + N_PZ] *= m[:, 1:2]; f[:, N_FPZ + N_PZ:] *= m[:, 2:3]
        valid = (ctx_ok & ~self.bad[Pc]).astype(np.float32)
        y = d.y[Pc].astype(np.int64)
        ok = (y >= 0) & ~self.bad[Pc] & ctx_ok
        y = np.where(ok, y, -100)
        yp = y[:, L - 1:]                         # provisional targets: the C chunk epochs
        yr = y[:, L - 2:-1]                       # revised targets: previous epoch of each
        T = lambda arr: torch.from_numpy(np.ascontiguousarray(arr)).to(self.dev)
        return T(x), T(f.astype(np.float32)), T(m), T(valid), T(yp), T(yr)


def freeze_bn(module):
    """Keep BN statistics AND dropout as they were when the GRU was trained (deterministic embeddings)."""
    for mod in module.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.Dropout)):
            mod.eval()


def run_fold(fold, data, a, dev, out):
    k = fold["fold"]
    torch.manual_seed(a.seed + k); rng = np.random.default_rng(a.seed + k)
    ck = torch.load(Path(a.cnn_dir) / f"{a.cnn_tag}_fold{k}.pt", map_location=dev, weights_only=False)
    gk = torch.load(Path(a.gru_dir) / f"{a.gru_tag}_fold{k}.pt", map_location=dev, weights_only=False)
    data.f_mu, data.f_sd = ck["f_mu"], ck["f_sd"]

    enc = SleepEpochNet(n_feat=data.feat.shape[1]).to(dev); enc.load_state_dict(ck["model"])
    seq = SleepSeqNet(emb=enc.proj[1].out_features).to(dev); seq.load_state_dict(gk["model"])
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": a.lr_enc},
                             {"params": seq.parameters(), "lr": a.lr_gru}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_W, device=dev), ignore_index=-100)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    chunks = Chunks(data, fold["train"], a.chunk, dev)
    va = torch.from_numpy(data.indices(fold["val"])).to(dev)
    te = torch.from_numpy(data.indices(fold["test"])).to(dev)
    yv = data.y[va.cpu().numpy()]

    def validate():
        S = Seq(data, embed_all(enc, data, dev), dev)
        pp, _ = predict(seq, S, va)
        return 100 * f1_score(yv, pp.argmax(1), average="macro")

    ckpt = out / f"{a.tag}_fold{k}.pt"
    best = validate()                                   # frozen starting point: never end up worse on validation
    torch.save(dict(encoder=enc.state_dict(), gru=seq.state_dict(), L=L, f_mu=data.f_mu, f_sd=data.f_sd), ckpt)
    print(f"  start (frozen)  val MF1 prov {best:.1f} *")
    for ep in range(a.epochs):
        enc.train(); freeze_bn(enc); seq.train()
        t0, tot, n = time.time(), 0.0, 0
        st = chunks.starts(rng)
        for b in range(0, len(st), a.bs):
            x, f, m, v, yp, yr = chunks.load(st[b:b + a.bs])
            B = yp.shape[0]
            if not a.no_aug:
                x, m = augment(x, m)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                z = enc.embed(x, f, m).float().view(B, a.chunk + L - 1, -1)
                zw = z.unfold(1, L, 1).permute(0, 1, 3, 2).reshape(-1, L, z.shape[-1])   # (B*C, L, emb)
                vw = v.unfold(1, L, 1).reshape(-1, L)
                lp, lr = seq(zw, vw)
                loss = crit(lp.float(), yp.reshape(-1)) + crit(lr.float(), yr.reshape(-1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * B; n += B
        sched.step()
        mf1 = validate(); flag = ""
        if mf1 > best:
            best, flag = mf1, " *"
            torch.save(dict(encoder=enc.state_dict(), gru=seq.state_dict(), L=L, f_mu=data.f_mu, f_sd=data.f_sd), ckpt)
        print(f"  ep {ep:2d}  loss {tot / max(n, 1):.3f}  val MF1 prov {mf1:.1f}  {time.time() - t0:.0f}s{flag}")

    best_ck = torch.load(ckpt, weights_only=False)
    enc.load_state_dict(best_ck["encoder"]); seq.load_state_dict(best_ck["gru"])
    S = Seq(data, embed_all(enc, data, dev), dev)
    pp, pr = predict(seq, S, te)
    yt = data.y[te.cpu().numpy()]
    np.savez(out / f"{a.tag}_fold{k}_pred.npz", idx=te.cpu().numpy(), y=yt, prov=pp.argmax(1), rev=pr.argmax(1),
             prob_prov=pp, prob_rev=pr)
    return yt, pp.argmax(1), pr.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--folds", default="data/folds.json")
    ap.add_argument("--cnn_dir", default="results/cnn"); ap.add_argument("--cnn_tag", default="cnn_edf78")
    ap.add_argument("--gru_dir", default="results/gru"); ap.add_argument("--gru_tag", default="gru_edf78")
    ap.add_argument("--out", default="results/e2e"); ap.add_argument("--tag", default="e2e_edf78")
    ap.add_argument("--fold", default="all", help="fold number or 'all'")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=20, help="target epochs per chunk")
    ap.add_argument("--bs", type=int, default=24, help="chunks per batch (more chunks = more nights per batch)")
    ap.add_argument("--lr_enc", type=float, default=2e-5)
    ap.add_argument("--lr_gru", type=float, default=1e-4)
    ap.add_argument("--no_aug", action="store_true", help="disable signal augmentation during fine-tuning")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mmap", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip folds that already have predictions")
    a = ap.parse_args()

    dev = torch.device(a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    folds = json.load(open(a.folds))
    if a.fold != "all":
        folds = [f for f in folds if f["fold"] == int(a.fold)]
    missing = [f["fold"] for f in folds
               if not (Path(a.cnn_dir) / f"{a.cnn_tag}_fold{f['fold']}.pt").exists()
               or not (Path(a.gru_dir) / f"{a.gru_tag}_fold{f['fold']}.pt").exists()]
    if missing:
        raise FileNotFoundError(f"CNN or GRU checkpoints missing for folds {missing} - run Steps 8 and 9a first")
    data = SleepData(a.data, a.cache, in_ram=not a.mmap)
    print(f"end-to-end fine-tuning  chunk {a.chunk}+{L - 1} epochs x {a.bs}  "
          f"lr enc {a.lr_enc} / gru {a.lr_gru}  augmentation {'off' if a.no_aug else 'on'}  device {dev}")

    Y, P, R = [], [], []
    for f in folds:
        pred = out / f"{a.tag}_fold{f['fold']}_pred.npz"
        if a.resume and pred.exists():
            d = np.load(pred); y, p, r = d["y"], d["prov"], d["rev"]
            print(f"\n=== fold {f['fold']}: loaded saved predictions ===")
        else:
            print(f"\n=== fold {f['fold']} ===")
            y, p, r = run_fold(f, data, a, dev, out)
        mp, mr = metrics(y, p), metrics(y, r)
        print(f"fold {f['fold']} TEST  provisional: ACC {mp['ACC']:.1f} MF1 {mp['MF1']:.1f} k {mp['kappa']:.3f}"
              f"  |  revised: ACC {mr['ACC']:.1f} MF1 {mr['MF1']:.1f} k {mr['kappa']:.3f}")
        Y.append(y); P.append(p); R.append(r)

    Y, P, R = np.concatenate(Y), np.concatenate(P), np.concatenate(R)
    for name, pred in [("prov", P), ("rev", R)]:
        m = metrics(Y, pred)
        suffix = "" if a.fold == "all" else f"_fold{a.fold}"
        json.dump(m, open(out / f"{a.tag}{suffix}_{name}_overall.json", "w"), indent=1)
        print(f"\n=== Pooled {len(folds)} folds - {'provisional (live)' if name == 'prov' else 'revised (+30 s)'} ===")
        print("  " + "  ".join(f"{k} {v:.3f}" if k == "kappa" else f"{k} {v:.1f}" for k, v in m.items()))
        print(pd.DataFrame(confusion_matrix(Y, pred, labels=range(5)), index=STAGE_NAMES, columns=STAGE_NAMES))


if __name__ == "__main__":
    main()
