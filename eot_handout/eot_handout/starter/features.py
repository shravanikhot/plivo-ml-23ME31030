"""
Audio utilities + feature extraction for the EOT assignment.

Causality reminder: for a pause at `pause_start`, features may only use
audio[0:pause_start] and previously-COMPLETED pauses (whose pause_end
already occurred before this pause's pause_start). Never pause_end of the
CURRENT pause, and never anything after pause_start.
"""
import numpy as np
import soundfile as sf

FRAME_MS = 25
HOP_MS = 10


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def speech_before(x, sr, pause_start, window_s=1.5):
    end = int(pause_start * sr)
    start = max(0, end - int(window_s * sr))
    return x[start:end]


def frames(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]


def frame_energy_db(x, sr):
    fr = frames(x, sr)
    if len(fr) == 0:
        return np.zeros(0, dtype=np.float32)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def autocorr_f0(frame, sr, fmin=60.0, fmax=400.0, voicing_thresh=0.30):
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo = int(sr / fmax)
    hi = min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < voicing_thresh:
        return 0.0
    return float(sr / lag)


def f0_contour(x, sr, frame_ms=40, hop_ms=HOP_MS):
    fr = frames(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if len(fr) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.array([autocorr_f0(f, sr) for f in fr], dtype=np.float32)


# ---------------------------------------------------------------------
# Turn-final prosodic cue helpers
# ---------------------------------------------------------------------

def linregress_slope(y):
    """Least-squares slope of y over its own sample index. 0 if too short."""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float32)
    x_mean, y_mean = x.mean(), y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def voiced_runs(f0):
    """Contiguous (start, end) index ranges where f0 > 0."""
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(f0):
        voiced = v > 0
        if voiced and not in_run:
            start, in_run = i, True
        elif not voiced and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(f0)))
    return runs


def extract_features(x, sr, pause_start, pause_index, prev_pause_durs):
    """Causal feature vector for one pause.

    x, sr        : full turn audio + sample rate. We only ever slice
                   audio[0:int(pause_start*sr)] below - nothing later.
    pause_start  : seconds, moment speech stops for THIS pause.
    pause_index  : 0,1,2... position of this pause within the turn.
    prev_pause_durs : durations of pauses 0..i-1 in THIS turn. All are
                   in the past relative to pause_start (their pause_end
                   already occurred), so using them is causally legal.
                   The CURRENT pause's own duration is never passed in.
    """
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    whole = x[:int(pause_start * sr)]

    if len(seg) < sr // 10:
        return np.zeros(12, dtype=np.float32)

    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)
    voiced_mask = f0 > 0
    voiced_vals = f0[voiced_mask]

    energy_slope = linregress_slope(e[-8:]) if len(e) >= 8 else 0.0
    energy_final_rel = float(e[-1] - e.mean()) if len(e) else 0.0

    runs = voiced_runs(f0)
    if runs:
        s, en = runs[-1]
        f0_slope_final = linregress_slope(f0[s:en])
        last_run_dur = en - s
    else:
        f0_slope_final = 0.0
        last_run_dur = 0.0

    f0_final_norm = float(voiced_vals[-1] - np.median(voiced_vals)) if len(voiced_vals) else 0.0
    voicing_ratio = float(voiced_mask.mean()) if len(f0) else 0.0

    run_durs = [en - s for s, en in runs]
    mean_run_dur = float(np.mean(run_durs)) if run_durs else 0.0
    final_run_dur_rel = float(last_run_dur - mean_run_dur) if mean_run_dur else 0.0

    if len(f0) and voiced_mask.any():
        last_voiced_idx = np.nonzero(voiced_mask)[0][-1]
        trailing_silence_frac = float((len(f0) - 1 - last_voiced_idx) / len(f0))
    else:
        trailing_silence_frac = 1.0

    if len(whole) > sr // 10:
        whole_f0 = f0_contour(whole, sr)
        speaking_rate = float((whole_f0 > 0).mean()) if len(whole_f0) else 0.0
    else:
        speaking_rate = 0.0

    n_prev = float(pause_index)
    mean_prev_pause = float(np.mean(prev_pause_durs)) if prev_pause_durs else 0.0
    std_prev_pause = float(np.std(prev_pause_durs)) if len(prev_pause_durs) > 1 else 0.0

    return np.array([
        energy_slope, energy_final_rel,
        f0_slope_final, f0_final_norm, voicing_ratio,
        final_run_dur_rel, trailing_silence_frac,
        speaking_rate, n_prev, mean_prev_pause, std_prev_pause,
        pause_start,
    ], dtype=np.float32)


FEATURE_NAMES = [
    "energy_slope", "energy_final_rel",
    "f0_slope_final", "f0_final_norm", "voicing_ratio",
    "final_run_dur_rel", "trailing_silence_frac",
    "speaking_rate", "n_prev_pauses", "mean_prev_pause_dur", "std_prev_pause_dur",
    "pause_start_time",
]


def extract_features_v2(x, sr, pause_start, pause_index, prev_pause_durs):
    """v1 features + rhythm-of-listing features (last pause duration,
    and regularity of recent prior pauses - both purely from COMPLETED
    prior pauses, fully causal)."""
    base = extract_features(x, sr, pause_start, pause_index, prev_pause_durs)

    last_pause_dur = float(prev_pause_durs[-1]) if prev_pause_durs else 0.0
    if len(prev_pause_durs) >= 2:
        recent = prev_pause_durs[-3:]
        recent_pause_regularity = float(np.std(recent))  # low = metronomic listing
    else:
        recent_pause_regularity = 0.0

    return np.concatenate([base, [last_pause_dur, recent_pause_regularity]]).astype(np.float32)


FEATURE_NAMES_V2 = FEATURE_NAMES + ["last_pause_dur", "recent_pause_regularity"]
