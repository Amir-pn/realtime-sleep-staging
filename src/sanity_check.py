"""Step 5 - Sanity checks on data/processed.

Usage:  python src/sanity_check.py --data data/processed --out results/sanity --night SC4001

Produces:
  1_psd_per_stage.png      mean spectrum per stage (Fpz-Cz, Pz-Oz, EOG)
  2_features_per_stage.png key handcrafted features per stage
  3_night_<key>.png        hypnogram + feature traces for one night
  qc_counts.csv            QC flag counts per channel
  feature_medians.csv      median of every feature per stage
Prints PASS/FAIL for expected physiological patterns.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from labels import STAGE_NAMES
from stream import FEATURE_NAMES, FS, QC_AMP, QC_CLIP, QC_FLAT, QC_HF, SCALE_UV

COLORS = ["#e41a1c", "#ff7f00", "#377eb8", "#4daf4a", "#984ea3"]
CH = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG"]
KEY_FEATS = ["fpz_rel_delta", "fpz_slowwave_frac", "pz_rel_alpha", "pz_theta_alpha",
             "pz_spindle_per_min", "fpz_spindle_per_min", "eog_sem_rel", "eog_movement_count"]


def load_all(data_dir, per_stage_per_night, rng):
    feats, labels, status, psd_sig, psd_lab, qc_rows = [], [], [], [], [], []
    files = sorted(Path(data_dir).glob("SC*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in {data_dir}")
    for f in files:
        d = np.load(f)
        s, e = d["trim"]
        y = d["labels"][s:e]
        st = d["status"][s:e]
        feats.append(d["features"][s:e]); labels.append(y); status.append(st)
        qc_rows.append(d["qc_flags"][s:e])
        # sample a few epochs per stage for spectra (keeps memory low)
        for k in range(5):
            idx = np.where((y == k) & (st == 0))[0]
            if len(idx):
                pick = rng.choice(idx, min(per_stage_per_night, len(idx)), replace=False)
                psd_sig.append(d["signals"][pick].astype(np.float32) * SCALE_UV[None])
                psd_lab.append(np.full(len(pick), k))
    return (np.concatenate(feats), np.concatenate(labels), np.concatenate(status),
            np.concatenate(psd_sig), np.concatenate(psd_lab), np.concatenate(qc_rows))


def plot_psd(sig, lab, out):
    fr, p = welch(sig, FS, nperseg=400, axis=-1)          # (n, 3, f)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for c in range(3):
        for k in range(5):
            m = np.log10(p[lab == k, c].mean(0) + 1e-12)
            ax[c].plot(fr, m, color=COLORS[k], label=STAGE_NAMES[k])
        for lo, hi, name in [(0.5, 2, "δ"), (8, 12, "α"), (11, 16, "σ")]:
            ax[c].axvspan(lo, hi, alpha=0.08, color="gray")
            ax[c].text((lo + hi) / 2, ax[c].get_ylim()[1], name, ha="center", va="top")
        ax[c].set(title=CH[c], xlabel="Hz", xlim=(0, 35 if c < 2 else 20))
    ax[0].set_ylabel("log10 PSD (µV²/Hz)"); ax[0].legend()
    fig.tight_layout(); fig.savefig(out / "1_psd_per_stage.png", dpi=150); plt.close(fig)


def plot_features(df, out):
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    for a, f in zip(ax.flat, KEY_FEATS):
        data = [df.loc[df.stage == k, f].values for k in range(5)]
        bp = a.boxplot(data, tick_labels=STAGE_NAMES, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], COLORS):
            patch.set_facecolor(c); patch.set_alpha(0.5)
        a.set_title(f)
    fig.tight_layout(); fig.savefig(out / "2_features_per_stage.png", dpi=150); plt.close(fig)


def plot_night(data_dir, key, out):
    d = np.load(Path(data_dir) / f"{key}.npz")
    y, F, st = d["labels"], d["features"], d["status"]
    s, e = d["trim"]
    t = np.arange(len(y)) * 30 / 3600
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    rows = [("fpz_rel_delta", "δ (Fpz)"), ("pz_rel_alpha", "α (Pz)"),
            ("pz_spindle_per_min", "spindles/min"), ("eog_movement_count", "eye movements")]
    fig, ax = plt.subplots(len(rows) + 2, 1, figsize=(16, 11), sharex=True)
    hyp = np.array([0, 2, 3, 4, 1])  # display order: W, REM, N1, N2, N3 (top→bottom)
    yy = np.where(y >= 0, hyp[np.clip(y, 0, 4)], np.nan)
    ax[0].step(t, -yy, where="post", color="k")
    ax[0].set_yticks([0, -1, -2, -3, -4], ["W", "REM", "N1", "N2", "N3"])
    ax[0].axvspan(t[s], t[e - 1], color="green", alpha=0.07, label="trimmed range")
    ax[0].legend(loc="upper right"); ax[0].set_title(f"{key} hypnogram")
    for a, (f, lbl) in zip(ax[1:-1], rows):
        a.plot(t, F[:, idx[f]], lw=0.6); a.set_ylabel(lbl)
    ax[-1].step(t, st, where="post", color="r"); ax[-1].set_ylabel("QC status\n0 ok 1 deg 2 bad")
    ax[-1].set_xlabel("hours")
    fig.tight_layout(); fig.savefig(out / f"3_night_{key}.png", dpi=130); plt.close(fig)


def check_patterns(med, mean):
    """Expected physiology -> PASS/FAIL (medians per stage)."""
    W, N1, N2, N3, R = range(5)
    checks = [
        ("N3 has highest relative delta (Fpz)", med.loc[N3, "fpz_rel_delta"] == med["fpz_rel_delta"].max()),
        ("N3 has highest slow-wave fraction", med.loc[N3, "fpz_slowwave_frac"] == med["fpz_slowwave_frac"].max()),
        ("W alpha (Pz) > N1, N2, N3", med.loc[W, "pz_rel_alpha"] > med.loc[[N1, N2, N3], "pz_rel_alpha"].max()),
        ("N2 spindles (Pz) > W, N1, REM", med.loc[N2, "pz_spindle_per_min"] > med.loc[[W, N1, R], "pz_spindle_per_min"].max()),
        ("REM eye movements > N2, N3 (mean)", mean.loc[R, "eog_movement_count"] > mean.loc[[N2, N3], "eog_movement_count"].max()),
        ("N1 theta/alpha (Pz) > W", med.loc[N1, "pz_theta_alpha"] > med.loc[W, "pz_theta_alpha"]),
    ]
    n_fail = 0
    print("\nPhysiology checks (medians, QC-OK epochs):")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        n_fail += not ok
    return n_fail


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results/sanity")
    ap.add_argument("--night", default="SC4001")
    ap.add_argument("--per_stage", type=int, default=20, help="epochs per stage per night for PSD")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    F, y, st, sig, sig_lab, qc = load_all(a.data, a.per_stage, rng)
    print(f"Loaded {len(y)} trimmed epochs, {len(sig)} sampled for spectra")

    # QC counts
    qc_df = pd.DataFrame({CH[c]: {name: int(np.sum(qc[:, c] & bit > 0))
                                  for name, bit in [("flat", QC_FLAT), ("clip", QC_CLIP),
                                                    ("amplitude", QC_AMP), ("high_freq", QC_HF)]}
                          for c in range(3)})
    qc_df.to_csv(out / "qc_counts.csv"); print("\nQC flags (trimmed range):\n", qc_df)

    # features per stage (valid labels, QC OK)
    m = (y >= 0) & (st == 0)
    df = pd.DataFrame(F[m], columns=FEATURE_NAMES); df["stage"] = y[m]
    med = df.groupby("stage").median()
    med.index = [STAGE_NAMES[i] for i in med.index]
    med.T.to_csv(out / "feature_medians.csv")
    med.index = range(5)

    plot_psd(sig, sig_lab, out)
    plot_features(df, out)
    plot_night(a.data, a.night, out)

    mean = df.groupby("stage").mean()
    n_fail = check_patterns(med, mean)
    print(f"\nPlots saved to {out}")
    if n_fail:
        print(f"[WARN] {n_fail} physiology check(s) failed - inspect plots before training.")
