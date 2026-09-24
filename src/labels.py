"""Step 3 - Epoch labels (R&K -> AASM) and wake trimming."""
import mne
import numpy as np

EPOCH_SEC = 30
STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,   # S3 + S4 -> N3
    "Sleep stage R": 4,
}
IGNORED = {"Sleep stage ?", "Movement time"}  # -> label -1
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def epoch_labels(hyp_path, n_epochs, raw=None):
    """One label per 30 s epoch; -1 = unscored / movement / outside hypnogram."""
    ann = mne.read_annotations(hyp_path)

    # align hypnogram to PSG start if both carry timestamps
    offset = 0.0
    if raw is not None and ann.orig_time is not None and raw.info["meas_date"] is not None:
        offset = (ann.orig_time - raw.info["meas_date"]).total_seconds()

    y = np.full(n_epochs, -1, dtype=np.int8)
    unknown = set()
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        if desc in STAGE_MAP:
            lab = STAGE_MAP[desc]
        else:
            lab = -1
            if desc not in IGNORED:
                unknown.add(desc)
        s = int(round((onset + offset) / EPOCH_SEC))
        e = min(s + int(round(dur / EPOCH_SEC)), n_epochs)
        if 0 <= s < n_epochs:
            y[s:e] = lab

    if unknown:
        print(f"[WARN] {hyp_path}: unexpected annotations {unknown}")
    return y


def trim_range(y, margin_epochs=60):
    """[start, end) keeping 30 min (60 epochs) of wake before/after sleep."""
    sleep = np.where((y >= 1) & (y <= 4))[0]
    if len(sleep) == 0:
        raise ValueError("No sleep epochs found in hypnogram")
    start = max(sleep[0] - margin_epochs, 0)
    end = min(sleep[-1] + margin_epochs + 1, len(y))
    return int(start), int(end)
