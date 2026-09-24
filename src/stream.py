"""Step 4 - StreamProcessor: the SAME code is used for dataset building and live/simulated use.

Input : 30 s chunks, shape (3, 3000), in microvolts: [EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal]
Output: scaled epoch signal, handcrafted features, QC flags, status, warm-up flag.

Rule: everything ACROSS epochs is causal (filter state, running baselines).
Operations INSIDE the current buffer (zero-phase filters, Welch, DWT) are allowed,
because the whole epoch is already available when it is processed.
"""
import numpy as np
import pywt
from scipy.signal import butter, find_peaks, sosfilt, sosfilt_zi, sosfiltfilt, welch

FS = 100
EPOCH = 30 * FS                 # 3000 samples
CTX = 5 * FS                    # 5 s extra history for DWT / feature filters
CH_NAMES = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]

SCALE_UV = np.array([50.0, 50.0, 100.0])[:, None]   # fixed physical scaling (keep amplitude info)
CLIP = 10.0
REM_SLOPE_UV_S = 800.0          # EOG slope threshold for eye-movement peaks (tune later)
SW_PTP_UV = 75.0                # slow-wave amplitude criterion

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12),
         "sigma": (11, 16), "beta": (16, 30)}

# QC bit flags per channel
QC_FLAT, QC_CLIP, QC_AMP, QC_HF = 1, 2, 4, 8
STATUS_OK, STATUS_DEGRADED, STATUS_BAD = 0, 1, 2


def _bp(lo, hi, order=4):
    return butter(order, [lo, hi], btype="band", fs=FS, output="sos")


def _eeg_block_names(ch):
    dwt = [f"{ch}_dwt_{b}" for b in ["A5", "D5", "D4", "D3", "D2", "D1"]]
    bands = [f"{ch}_rel_{b}" for b in BANDS] + [f"{ch}_logpow", f"{ch}_theta_alpha"]
    return dwt + bands + [f"{ch}_spindle_per_min"]


FEATURE_NAMES = (
    _eeg_block_names("fpz") + ["fpz_slowwave_frac"]
    + _eeg_block_names("pz")
    + ["eog_sem_rel", "eog_logpow", "eog_movement_count", "eog_logptp", "fpz_eog_corr"]
)
N_FPZ = 15   # 14 + slow-wave
N_PZ = 14
N_EOG = 5
assert len(FEATURE_NAMES) == N_FPZ + N_PZ + N_EOG


class StreamProcessor:
    def __init__(self):
        # causal streaming filters
        self.sos_eeg = _bp(0.3, 45)
        self.sos_eog = _bp(0.1, 15)
        self.zi_eeg = None
        self.zi_eog = None
        # within-buffer feature filters
        self.sos_sigma = _bp(11, 16)
        self.sos_sw = _bp(0.5, 2)
        self.sos_lp5 = _bp(1, 5)   # 1-5 Hz: removes slow-wave leakage from EOG

        self.buf = np.zeros((3, EPOCH + CTX))
        self.n_seen = 0
        self.spindle_base = [None, None]   # running baseline per EEG channel

    # ---------- filtering ----------
    def _filter(self, x):
        if self.zi_eeg is None:   # start in steady state for the first sample -> no DC transient
            z = sosfilt_zi(self.sos_eeg)
            self.zi_eeg = z[:, None, :] * x[:2, 0][None, :, None]
            z = sosfilt_zi(self.sos_eog)
            self.zi_eog = z[:, None, :] * x[2:, 0][None, :, None]
        y = np.empty_like(x)
        y[:2], self.zi_eeg = sosfilt(self.sos_eeg, x[:2], axis=-1, zi=self.zi_eeg)
        y[2:], self.zi_eog = sosfilt(self.sos_eog, x[2:], axis=-1, zi=self.zi_eog)
        return y

    # ---------- quality control ----------
    def _qc(self, raw, filt):
        flags = np.zeros(3, dtype=np.uint8)
        for c in range(3):
            r, f = raw[c], filt[c]
            if r.std() < 0.5:
                flags[c] |= QC_FLAT
            mx, mn = r.max(), r.min()
            if mx > mn and (np.mean(r == mx) + np.mean(r == mn)) > 0.01:
                flags[c] |= QC_CLIP
            if np.ptp(f) > (500.0 if c < 2 else 1000.0):
                flags[c] |= QC_AMP
            if c < 2:
                fr, p = welch(f, FS, nperseg=256)
                if p[fr > 30].sum() / (p.sum() + 1e-12) > 0.5:
                    flags[c] |= QC_HF
        bad = flags > 0
        if bad[0] and bad[1]:
            status = STATUS_BAD
        elif bad.any():
            status = STATUS_DEGRADED
        else:
            status = STATUS_OK
        return flags, status

    # ---------- features ----------
    @staticmethod
    def _dwt(x_ext):
        coeffs = pywt.wavedec(x_ext, "db4", level=5)          # [A5, D5, D4, D3, D2, D1]
        keep = EPOCH / (EPOCH + CTX)                         # use only coeffs of current epoch
        E = np.array([np.sum(c[-max(1, int(round(len(c) * keep))):] ** 2) for c in coeffs])
        return list(np.log(E / (E.sum() + 1e-12) + 1e-12))

    @staticmethod
    def _bands(x):
        fr, p = welch(x, FS, nperseg=400)
        tot = p[(fr >= 0.5) & (fr <= 30)].sum() + 1e-12
        rel = [p[(fr >= lo) & (fr < hi)].sum() / tot for lo, hi in BANDS.values()]
        return rel + [np.log(tot), rel[1] / (rel[2] + 1e-6)]

    def _spindles(self, x_ext, c, update):
        s = sosfiltfilt(self.sos_sigma, x_ext)
        w = int(0.3 * FS)
        env = np.sqrt(np.convolve(s ** 2, np.ones(w) / w, mode="same"))[-EPOCH:]
        med = np.median(env)
        if update:
            b = self.spindle_base[c]
            self.spindle_base[c] = med if b is None else 0.95 * b + 0.05 * med
        base = self.spindle_base[c] if self.spindle_base[c] is not None else med
        above = (env > 2 * base).astype(int)
        d = np.diff(np.r_[0, above, 0])
        dur = (np.where(d == -1)[0] - np.where(d == 1)[0]) / FS
        return float(np.sum((dur >= 0.5) & (dur <= 2.0))) / 0.5   # events per minute

    def _slow_wave_frac(self, x_ext):
        s = sosfiltfilt(self.sos_sw, x_ext)[-EPOCH:]
        return float(np.mean(np.ptp(s.reshape(15, 2 * FS), axis=1) > SW_PTP_UV))

    def _eog(self, e_ext, fpz):
        e = e_ext[-EPOCH:]
        fr, p = welch(e, FS, nperseg=1000)
        tot = p[(fr >= 0.1) & (fr <= 15)].sum() + 1e-12
        sem = p[(fr >= 0.1) & (fr < 1.0)].sum() / tot
        lp = sosfiltfilt(self.sos_lp5, e_ext)[-EPOCH:]
        slope = np.abs(np.diff(lp)) * FS
        peaks, _ = find_peaks(slope, height=REM_SLOPE_UV_S, distance=int(0.3 * FS))
        corr = np.corrcoef(fpz, e)[0, 1] if fpz.std() > 0 and e.std() > 0 else 0.0
        return [sem, np.log(tot), float(len(peaks)), np.log1p(np.ptp(e)), corr]

    # ---------- main entry ----------
    def process(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.shape != (3, EPOCH):
            raise ValueError(f"Expected chunk shape (3, {EPOCH}), got {chunk.shape}")

        filt = self._filter(chunk)
        self.buf = np.concatenate([self.buf[:, EPOCH:], filt], axis=1)
        self.n_seen += 1
        ext = self.buf
        ep = ext[:, -EPOCH:]

        flags, status = self._qc(chunk, ep)
        ok = flags == 0

        f_fpz = self._dwt(ext[0]) + self._bands(ep[0]) + \
            [self._spindles(ext[0], 0, ok[0]), self._slow_wave_frac(ext[0])]
        f_pz = self._dwt(ext[1]) + self._bands(ep[1]) + [self._spindles(ext[1], 1, ok[1])]
        f_eog = self._eog(ext[2], ep[0])

        feats = np.array(f_fpz + f_pz + f_eog, dtype=np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        # zero-out features of failed channels (model sees them as missing)
        if not ok[0]:
            feats[:N_FPZ] = 0
        if not ok[1]:
            feats[N_FPZ:N_FPZ + N_PZ] = 0
        if not ok[2]:
            feats[N_FPZ + N_PZ:] = 0

        signal = np.clip(ep / SCALE_UV, -CLIP, CLIP).astype(np.float32)
        return dict(signal=signal, features=feats, qc_flags=flags,
                    status=status, warmup=self.n_seen <= 1)
