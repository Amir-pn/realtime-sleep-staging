"""External test on Sleep-EDF Sleep Telemetry (ST) - zero-shot, no retraining.

Every ST night is streamed through the SAME causal StreamProcessor, then scored by all 10
fold models (CNN encoder + causal GRU) trained on the Sleep Cassette (SC) study.
Reports the 10-model ensemble (averaged probabilities) and the mean +/- SD of the single models.

Usage:
  python src/eval_st.py --raw data/raw --out results/st --jobs 4
Needs: results/cnn/cnn_edf78_fold<k>.pt and results/gru/gru_edf78_fold<k>.pt for k = 0..9
"""
import argparse
import json
import traceback
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix

from labels import STAGE_NAMES, epoch_labels, trim_range
from model import SleepEpochNet, SleepSeqNet
from stream import CH_NAMES, EPOCH, FEATURE_NAMES, FS, N_FPZ, N_PZ, STATUS_BAD, StreamProcessor
from train import metrics


# ------------------------------------------------------------------ 1. index + stream ST nights
def index_st(raw_dir):
    raw_dir = Path(raw_dir)
    hyps = {h.name[:6]: h for h in raw_dir.rglob("ST7*-Hypnogram.edf")}
    rows, missing = [], []
    for p in sorted(raw_dir.rglob("ST7*-PSG.edf")):
        key = p.name[:6]
        if key not in hyps:
            missing.append(p.name); continue
        rows.append(dict(key=key, subject=int(key[3:5]), night=int(key[5]), psg=str(p), hyp=str(hyps[key])))
    if missing:
        print(f"[WARN] no hypnogram for: {missing}")
    if not rows:
        raise FileNotFoundError(f"No ST7* PSG/hypnogram pairs under {raw_dir}")
    df = pd.DataFrame(rows)
    print(f"ST: {len(df)} nights from {df.subject.nunique()} subjects (expected 44 / 22)")
    return df


def stream_night(row, out_dir):
    out = Path(out_dir) / f"{row['key']}.npz"
    if out.exists():
        return row["key"], None
    try:
        raw = mne.io.read_raw_edf(row["psg"], include=CH_NAMES, preload=True, verbose="ERROR")
        missing = set(CH_NAMES) - set(raw.ch_names)
        if missing:
            raise ValueError(f"missing channels {missing}; available: {raw.ch_names}")
        if raw.info["sfreq"] != FS:
            raise ValueError(f"sfreq {raw.info['sfreq']} != {FS}")
        x = raw.get_data(picks=CH_NAMES, units="uV")
        n = x.shape[1] // EPOCH
        y = epoch_labels(row["hyp"], n, raw)
        s, e = trim_range(y)
        proc = StreamProcessor()
        sig = np.zeros((n, 3, EPOCH), np.float16)
        feat = np.zeros((n, len(FEATURE_NAMES)), np.float32)
        qc = np.zeros((n, 3), np.uint8); st = np.zeros(n, np.int8)
        for k in range(n):
            o = proc.process(x[:, k * EPOCH:(k + 1) * EPOCH])
            sig[k], feat[k], qc[k], st[k] = o["signal"], o["features"], o["qc_flags"], o["status"]
        np.savez_compressed(out, signals=sig, features=feat, qc_flags=qc, status=st, labels=y,
                            trim=np.array([s, e]), subject=row["subject"], night=row["night"])
        return row["key"], None
    except Exception as ex:
        return row["key"], f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}"


# ------------------------------------------------------------------ 2. score one night with one fold model
@torch.no_grad()
def score_night(d, enc, seq, ck, L, dev):
    sig, feat, qc, st = d["signals"], d["features"], d["qc_flags"], d["status"]
    n = len(st)
    m = (qc == 0).astype(np.float32)
    f = (feat - ck["f_mu"]) / ck["f_sd"]
    f[:, :N_FPZ] *= m[:, 0:1]; f[:, N_FPZ:N_FPZ + N_PZ] *= m[:, 1:2]; f[:, N_FPZ + N_PZ:] *= m[:, 2:3]
    emb, p_cnn = [], []
    for b in range(0, n, 512):
        xs = torch.from_numpy(sig[b:b + 512].astype(np.float32)).to(dev)
        fs = torch.from_numpy(f[b:b + 512].astype(np.float32)).to(dev)
        ms = torch.from_numpy(m[b:b + 512]).to(dev)
        z = enc.embed(xs, fs, ms)
        emb.append(z); p_cnn.append(torch.softmax(enc.head(z), 1).cpu().numpy())
    emb = torch.cat(emb)
    w = np.arange(n)[:, None] + np.arange(-L + 1, 1)
    wc = np.clip(w, 0, None)
    v = torch.from_numpy(((w >= 0) & (st[wc] != STATUS_BAD)).astype(np.float32)).to(dev)
    wct = torch.from_numpy(wc).to(dev)
    pp, pr = [], []
    for b in range(0, n, 1024):
        lp, lr = seq(emb[wct[b:b + 1024]], v[b:b + 1024])
        pp.append(torch.softmax(lp, 1).cpu().numpy()); pr.append(torch.softmax(lr, 1).cpu().numpy())
    p_prov = np.concatenate(pp); rev_next = np.concatenate(pr)
    p_rev = p_prov.copy(); p_rev[:-1] = rev_next[1:]          # revised stage of t comes from window ending at t+1
    return np.concatenate(p_cnn), p_prov, p_rev


def fmt(m):
    return "  ".join(f"{k} {v:.3f}" if k == "kappa" else f"{k} {v:.1f}" for k, v in m.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--work", default="data/processed_st")
    ap.add_argument("--out", default="results/st")
    ap.add_argument("--cnn_dir", default="results/cnn"); ap.add_argument("--cnn_tag", default="cnn_edf78")
    ap.add_argument("--gru_dir", default="results/gru"); ap.add_argument("--gru_tag", default="gru_edf78")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    work, out = Path(a.work), Path(a.out)
    work.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)

    # checkpoints first, so a missing file fails before the long streaming step
    ckpts = [(Path(a.cnn_dir) / f"{a.cnn_tag}_fold{k}.pt", Path(a.gru_dir) / f"{a.gru_tag}_fold{k}.pt") for k in range(a.folds)]
    missing = [str(p) for pair in ckpts for p in pair if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))

    # 1. stream ST nights (cached in data/processed_st)
    df = index_st(a.raw)
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(stream_night)(r, work) for r in df.to_dict("records"))
    errs = [(k, e) for k, e in res if e]
    if errs:
        (out / "errors.log").write_text("\n\n".join(f"=== {k}\n{e}" for k, e in errs))
        print(f"[ERROR] {len(errs)} nights failed -> {out / 'errors.log'}")
        for k, e in errs:
            print("  ", k, e.splitlines()[0])
    nights = {p.stem: dict(np.load(p)) for p in sorted(work.glob("ST7*.npz"))}
    print(f"{len(nights)} ST nights ready")

    # 2. score with each fold model
    probs = {key: {"cnn": [], "prov": [], "rev": []} for key in nights}
    for k, (cp, gp) in enumerate(ckpts):
        ck = torch.load(cp, map_location=dev, weights_only=False)
        gk = torch.load(gp, map_location=dev, weights_only=False)
        enc = SleepEpochNet(n_feat=len(FEATURE_NAMES)).to(dev); enc.load_state_dict(ck["model"]); enc.eval()
        seq = SleepSeqNet(emb=enc.proj[1].out_features).to(dev); seq.load_state_dict(gk["model"]); seq.eval()
        for key, d in nights.items():
            pc, pp, pr = score_night(d, enc, seq, ck, gk["L"], dev)
            probs[key]["cnn"].append(pc); probs[key]["prov"].append(pp); probs[key]["rev"].append(pr)
        print(f"  scored with fold model {k}")

    # 3. metrics: trimmed window (main, same rule as SC) and whole night (secondary)
    rows, conf = [], None
    for scope in ["trimmed", "whole night"]:
        Y, P = [], {o: [[] for _ in range(a.folds)] + [[]] for o in ["cnn", "prov", "rev"]}   # last slot = ensemble
        for key, d in nights.items():
            y, st = d["labels"], d["status"]
            s, e = d["trim"] if scope == "trimmed" else (0, len(y))
            keep = np.zeros(len(y), bool); keep[s:e] = True
            keep &= (y >= 0) & (st != STATUS_BAD)
            Y.append(y[keep])
            for o in P:
                for k in range(a.folds):
                    P[o][k].append(probs[key][o][k][keep].argmax(1))
                P[o][-1].append(np.mean(probs[key][o], axis=0)[keep].argmax(1))
        Y = np.concatenate(Y)
        for o, name in [("cnn", "CNN (1 epoch)"), ("prov", "CNN+GRU provisional (live)"), ("rev", "CNN+GRU revised (+30 s)")]:
            single = [metrics(Y, np.concatenate(P[o][k])) for k in range(a.folds)]
            ens = metrics(Y, np.concatenate(P[o][-1]))
            rows.append(dict(scope=scope, model=name, kind="ensemble", **ens))
            rows.append(dict(scope=scope, model=name, kind="single mean", **{m: np.mean([s_[m] for s_ in single]) for m in ens}))
            rows.append(dict(scope=scope, model=name, kind="single SD", **{m: np.std([s_[m] for s_ in single]) for m in ens}))
            if scope == "trimmed" and o == "prov":
                conf = confusion_matrix(Y, np.concatenate(P[o][-1]), labels=range(5))
        if scope == "trimmed":
            counts = {STAGE_NAMES[i]: int(np.sum(Y == i)) for i in range(5)}

    tab = pd.DataFrame(rows)
    tab.to_csv(out / "st_results.csv", index=False)
    pd.DataFrame(conf, index=STAGE_NAMES, columns=STAGE_NAMES).to_csv(out / "st_confusion_prov_ensemble.csv")
    json.dump(counts, open(out / "st_class_counts.json", "w"), indent=1)

    print(f"\nST class counts (trimmed, scored): {counts}  total {sum(counts.values())}")
    for scope in ["trimmed", "whole night"]:
        print(f"\n=== Sleep-EDF ST, zero-shot, {scope} ===")
        t = tab[tab.scope == scope]
        for _, r in t.iterrows():
            print(f"  {r.model:28s} {r.kind:12s} " + fmt({c: r[c] for c in tab.columns if c not in ('scope', 'model', 'kind')}))
    print("\nConfusion matrix, CNN+GRU provisional ensemble (trimmed):")
    print(pd.DataFrame(conf, index=STAGE_NAMES, columns=STAGE_NAMES))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
