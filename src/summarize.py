"""Results table (Sleep-EDF-78, 10-fold subject-wise) + CPU latency per 30 s epoch.

Usage:  python src/summarize.py
Output: results/summary_table.csv
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import SleepEpochNet, SleepSeqNet
from stream import EPOCH, FEATURE_NAMES, StreamProcessor

R = Path("results")
OURS = [  # (name, json path, params, causal)
    ("XGBoost features (1 epoch)", R / "baseline_xgb/edf78_ctx0_overall.json", "-", "yes"),
    ("XGBoost features (+2 past epochs)", R / "baseline_xgb/edf78_ctx2_overall.json", "-", "yes"),
    ("Ours: CNN (1 epoch)", R / "cnn/cnn_edf78_foldall_overall.json", "34.6K", "yes"),
    ("Ours: CNN+GRU provisional (live)", R / "gru/gru_edf78_prov_overall.json", "60.4K", "yes"),
    ("Ours: CNN+GRU revised (+30 s)", R / "gru/gru_edf78_rev_overall.json", "60.4K", "yes (30 s delay)"),
    ("Ours: CNN+GRU fine-tuned, live", R / "e2e/e2e_edf78_prov_overall.json", "60.4K", "yes"),
    ("Ours: CNN+GRU fine-tuned, revised", R / "e2e/e2e_edf78_rev_overall.json", "60.4K", "yes (30 s delay)"),
]
# Sleep-EDF-78 results reported in the original papers (protocols differ slightly)
PAPERS = [
    ("AttnSleepNet", 81.3, 75.1, 0.74, 92.0, 42.0, 85.0, 82.1, 74.2, "520K", "no"),
    ("TinySleepNet", 83.1, 78.1, 0.77, 92.8, 51.0, 85.3, 81.1, 80.3, "1.3M", "LSTM causal, offline eval"),
    ("ULW-SleepNet (ICASSP 2026)", 81.4, 74.0, 0.74, 93.0, 39.3, 83.8, 79.6, 74.4, "13.3K", "no"),
    ("MMTSleepNet (Appl Intell 2026)", 82.9, 77.7, 0.76, 93.1, 49.0, 84.9, 81.5, 80.1, "1.96M", "no"),
]
COLS = ["Model", "ACC", "MF1", "kappa", "W", "N1", "N2", "N3", "REM", "Params", "Causal"]


def table():
    rows = []
    for name, p, par, causal in OURS:
        if not p.exists():
            print(f"[missing] {p} - row skipped"); continue
        m = json.load(open(p))
        rows.append([name, m["ACC"], m["MF1"], m["kappa"], m["F1_W"], m["F1_N1"], m["F1_N2"],
                     m["F1_N3"], m["F1_REM"], par, causal])
    rows += [list(r) for r in PAPERS]
    df = pd.DataFrame(rows, columns=COLS)
    for c in COLS[1:9]:
        df[c] = df[c].astype(float).round(3 if c == "kappa" else 1)
    df.to_csv(R / "summary_table.csv", index=False)
    print(df.to_string(index=False))


def latency(n=100):
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    proc = StreamProcessor()
    chunks = [rng.normal(0, 30, (3, EPOCH)) for _ in range(n)]
    t0 = time.perf_counter()
    for c in chunks:
        proc.process(c)
    t_proc = (time.perf_counter() - t0) / n * 1000

    enc, seq = SleepEpochNet(n_feat=len(FEATURE_NAMES)).eval(), SleepSeqNet().eval()
    x, f, m = torch.randn(1, 3, EPOCH), torch.randn(1, len(FEATURE_NAMES)), torch.ones(1, 3)
    z, v = torch.randn(1, 20, 64), torch.ones(1, 20)
    with torch.no_grad():
        for _ in range(10):
            enc.embed(x, f, m); seq(z, v)
        t0 = time.perf_counter()
        for _ in range(n):
            enc.embed(x, f, m)
        t_cnn = (time.perf_counter() - t0) / n * 1000
        t0 = time.perf_counter()
        for _ in range(n):
            seq(z, v)
        t_gru = (time.perf_counter() - t0) / n * 1000
    tot = t_proc + t_cnn + t_gru
    print(f"\nCPU latency per 30 s epoch (1 thread): preprocessing+features {t_proc:.1f} ms, "
          f"CNN {t_cnn:.1f} ms, GRU {t_gru:.1f} ms  ->  total {tot:.1f} ms "
          f"(real-time factor {tot / 30000:.5f})")


if __name__ == "__main__":
    table()
    latency()
