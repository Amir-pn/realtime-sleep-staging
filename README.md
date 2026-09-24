# Real-Time, Explainable Sleep Staging (EEG + EOG)

A fully causal sleep-staging pipeline that outputs a sleep stage (W, N1, N2, N3, REM) every 30 seconds using only past data.

![Architecture](docs/model_architecture.png)

## Results (Sleep-EDF-78, 10-fold subject-wise)

| Model | Accuracy | Macro F1 | κ | Parameters |
|---|---|---|---|---|
| XGBoost, 34 features (1 epoch) | 77.3 | 72.3 | 0.693 | – |
| XGBoost, features + 2 past epochs | 81.0 | 76.5 | 0.742 | – |
| CNN (1 epoch) | 80.7 | 75.0 | 0.735 | 34.6K |
| **CNN + causal GRU, live** | **84.1** | **80.0** | **0.783** | 60.4K |
| CNN + causal GRU, revised (+30 s) | 84.2 | 79.9 | 0.784 | 60.4K |

CPU latency: about 9 ms per 30 s epoch (single thread).

## Key features

- **Causal end to end:** causal filters with carried state, fixed µV scaling, no whole-night statistics, and a one-directional GRU over the last 10 minutes.
- **Hybrid model:** a separable multi-scale CNN on the raw EEG/EOG signals plus 34 physiology features (DWT band energies, spindles, slow waves, eye movements).
- **Signal-quality gate:** bad epochs are flagged ("poor signal") and masked in the context model.
- **Two outputs:** a provisional stage immediately and a revised stage 30 s later.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux/macOS: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU build; omit the index URL for CPU
pip install -r requirements.txt
```

## Data

Download the Sleep-EDF Expanded database (Sleep Cassette files) from PhysioNet: https://physionet.org/content/sleep-edfx/1.0.0/ and place it in `data/raw/`. The data is not included in this repository.

## Pipeline

```bash
python src/index_data.py --raw data/raw --out data/index.csv
python src/build_dataset.py --index data/index.csv --out data/processed --jobs 4
python src/sanity_check.py --data data/processed --out results/sanity
python src/make_folds.py --data data/processed --out data/folds.json
python src/baseline_xgb.py --folds data/folds.json --context 0
python src/baseline_xgb.py --folds data/folds.json --context 2
python src/train.py --folds data/folds.json --fold all --epochs 20 --patience 20
python src/train_gru.py --folds data/folds.json
python src/summarize.py
```

| Script | Purpose |
|---|---|
| `index_data.py` | Pair PSG and hypnogram files |
| `labels.py` | R&K → AASM labels, wake trimming |
| `stream.py` | Causal StreamProcessor: filters, QC gate, 34 features |
| `build_dataset.py` | Stream every night through the processor |
| `sanity_check.py` / `tune_eog.py` | Physiology checks, eye-movement detector tuning |
| `make_folds.py` | Subject-wise folds (no person in two sets) |
| `baseline_xgb.py` | XGBoost baseline |
| `dataset.py`, `model.py`, `train.py` | Per-epoch CNN |
| `train_gru.py` | Causal GRU with provisional and revised heads |
| `summarize.py` | Results table and CPU latency |

## Limitations

- Single dataset: healthy adults, recorded 1987–91, R&K scoring.
- Published results are quoted from their papers, not re-run on these folds.
- Real-time use is simulated on recorded data; no hardware validation yet.
- The Sleep Quality Score is not clinically validated.

## Dataset citation

Kemp B, et al. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. *IEEE Trans Biomed Eng* 2000.
Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 2000.
