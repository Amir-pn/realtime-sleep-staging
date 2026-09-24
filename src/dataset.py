"""Dataset: builds a fast cache once, then serves batches (per-epoch now, sequences in Step 9).

Cache (data/cache/), trimmed epochs of all nights, contiguous per night:
  signals.npy (N, 3, 3000) float16
  meta.npz    features, labels, status, qc_flags, subject, night, pos
"""
from pathlib import Path

import numpy as np
import torch

from stream import EPOCH, N_FPZ, N_PZ, STATUS_BAD


def build_cache(data_dir, cache_dir):
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(data_dir.glob("SC*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz in {data_dir}")
    sizes = [int(np.diff(np.load(f)["trim"])[0]) for f in files]
    N = sum(sizes)
    sig = np.lib.format.open_memmap(cache_dir / "signals.npy", mode="w+",
                                    dtype=np.float16, shape=(N, 3, EPOCH))
    meta = {k: [] for k in ["features", "labels", "status", "qc_flags", "subject", "night", "pos"]}
    o = 0
    for i, (f, n) in enumerate(zip(files, sizes)):
        d = np.load(f)
        s, e = d["trim"]
        sig[o:o + n] = d["signals"]
        meta["features"].append(d["features"][s:e]); meta["labels"].append(d["labels"][s:e])
        meta["status"].append(d["status"][s:e]); meta["qc_flags"].append(d["qc_flags"][s:e])
        meta["subject"].append(np.full(n, int(d["subject"]), np.int16))
        meta["night"].append(np.full(n, i, np.int16))
        meta["pos"].append(np.arange(s, e, dtype=np.int32))
        o += n
        print(f"\rcaching {i + 1}/{len(files)}", end="")
    sig.flush(); del sig
    np.savez(cache_dir / "meta.npz", **{k: np.concatenate(v) for k, v in meta.items()},
             night_keys=np.array([f.stem for f in files]))
    print(f"\ncache: {N} epochs -> {cache_dir}")


class SleepData:
    def __init__(self, data_dir="data/processed", cache_dir="data/cache", in_ram=True):
        cache_dir = Path(cache_dir)
        if not (cache_dir / "signals.npy").exists() or not (cache_dir / "meta.npz").exists():
            build_cache(data_dir, cache_dir)
        print("loading signals" + (" into RAM (~3.5 GB)" if in_ram else " (memmap)") + "...")
        self.sig = np.load(cache_dir / "signals.npy", mmap_mode=None if in_ram else "r")
        m = np.load(cache_dir / "meta.npz")
        self.feat = m["features"].astype(np.float32)
        self.y, self.status, self.qc = m["labels"], m["status"], m["qc_flags"]
        self.subj, self.night, self.pos = m["subject"], m["night"], m["pos"]
        self.night_keys = m["night_keys"]
        self.f_mu = np.zeros(self.feat.shape[1], np.float32)
        self.f_sd = np.ones(self.feat.shape[1], np.float32)

    def indices(self, subjects, valid_only=True):
        m = np.isin(self.subj, subjects)
        if valid_only:
            m &= (self.y >= 0) & (self.status != STATUS_BAD)
        return np.where(m)[0]

    def fit_feat_norm(self, idx):
        self.f_mu = self.feat[idx].mean(0)
        self.f_sd = self.feat[idx].std(0) + 1e-6

    def batch(self, idx, device):
        idx = np.sort(idx)
        x = torch.from_numpy(self.sig[idx].astype(np.float32))
        mask = torch.from_numpy((self.qc[idx] == 0).astype(np.float32))       # (B,3) channel OK
        f = (self.feat[idx] - self.f_mu) / self.f_sd
        # keep features of failed channels at 0 after normalisation
        f[:, :N_FPZ] *= mask[:, 0:1].numpy()
        f[:, N_FPZ:N_FPZ + N_PZ] *= mask[:, 1:2].numpy()
        f[:, N_FPZ + N_PZ:] *= mask[:, 2:3].numpy()
        return (x.to(device, non_blocking=True), torch.from_numpy(f).to(device),
                mask.to(device), torch.from_numpy(self.y[idx].astype(np.int64)).to(device), idx)
