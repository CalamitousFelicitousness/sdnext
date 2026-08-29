"""Loudness and level measurement for generated audio, following ITU-R BS.1770-4.

Measurement is the point: a saved file has to be describable, not corrected. Normalization is a
separate opt-in step so the default output stays exactly what the model produced.
"""

import numpy as np
from modules.audio.stream import normalize_waveform


BLOCK_SECONDS = 0.400 # BS.1770 gating block
STEP_SECONDS = 0.100 # 75 percent overlap
ABSOLUTE_GATE = -70.0 # LUFS
RELATIVE_GATE = -10.0 # LU below the ungated level
OFFSET = -0.691 # the recommendation's calibration constant
SURROUND_WEIGHT = 1.41 # Ls and Rs; front channels weigh 1.0

# BS.1770 stage 1 is a high shelf and stage 2 an RLB high pass. The recommendation tabulates
# coefficients for 48k only, so both are designed by bilinear transform of the analogue prototypes
# and the 48k case is pinned against the printed table by the offline tests. The plain RBJ cookbook
# forms do not reproduce that table, they land about 0.06 off in the shelf.
SHELF_GAIN_DB = 3.999843853973347
SHELF_Q = 0.7071752369554196
SHELF_FC = 1681.974450955533
SHELF_VB_EXPONENT = 0.499666774155 # shelf mid-gain exponent that reproduces the tabulated values
HIGHPASS_Q = 0.5003270373238773
HIGHPASS_FC = 38.13547087602444


def shelf_coeffs(rate: int):
    k = np.tan(np.pi * SHELF_FC / rate)
    vh = np.power(10.0, SHELF_GAIN_DB / 20.0)
    vb = np.power(vh, SHELF_VB_EXPONENT)
    norm = 1.0 + k / SHELF_Q + k * k
    b = np.array([
        (vh + vb * k / SHELF_Q + k * k) / norm,
        2.0 * (k * k - vh) / norm,
        (vh - vb * k / SHELF_Q + k * k) / norm,
    ])
    a = np.array([
        1.0,
        2.0 * (k * k - 1.0) / norm,
        (1.0 - k / SHELF_Q + k * k) / norm,
    ])
    return b, a


def highpass_coeffs(rate: int):
    k = np.tan(np.pi * HIGHPASS_FC / rate)
    norm = 1.0 + k / HIGHPASS_Q + k * k
    b = np.array([1.0, -2.0, 1.0])
    a = np.array([
        1.0,
        2.0 * (k * k - 1.0) / norm,
        (1.0 - k / HIGHPASS_Q + k * k) / norm,
    ])
    return b, a


def k_weight(audio: np.ndarray, rate: int) -> np.ndarray:
    from scipy.signal import lfilter
    out = audio.astype(np.float64, copy=True)
    for b, a in (shelf_coeffs(rate), highpass_coeffs(rate)):
        out = lfilter(b, a, out, axis=-1)
    return out


def channel_weights(channels: int) -> np.ndarray:
    weights = np.ones(channels, dtype=np.float64)
    if channels >= 5: # L R C Ls Rs
        weights[3:5] = SURROUND_WEIGHT
    return weights


def block_mean_squares(weighted: np.ndarray, rate: int) -> np.ndarray | None:
    """Mean square per channel for every overlapping gating block, or None when too short."""
    block = int(round(BLOCK_SECONDS * rate))
    step = int(round(STEP_SECONDS * rate))
    samples = weighted.shape[-1]
    if samples < block:
        return None
    cumulative = np.cumsum(np.square(weighted, dtype=np.float64), axis=-1)
    cumulative = np.concatenate([np.zeros((weighted.shape[0], 1)), cumulative], axis=-1)
    starts = np.arange(0, samples - block + 1, step)
    sums = cumulative[:, starts + block] - cumulative[:, starts]
    return sums / block


def integrated_lufs(audio: np.ndarray, rate: int) -> float | None:
    """Gated integrated loudness, or None when the signal is shorter than one gating block.

    Below 400 ms the gating has nothing to run on and ffmpeg answers -70 with a success code,
    which is a number that reads as real and is not.
    """
    squares = block_mean_squares(k_weight(audio, rate), rate)
    if squares is None:
        return None
    weights = channel_weights(audio.shape[0])[:, None]
    per_block = np.sum(weights * squares, axis=0)
    with np.errstate(divide='ignore'):
        levels = OFFSET + 10.0 * np.log10(per_block)
    keep = levels > ABSOLUTE_GATE
    if not np.any(keep):
        return None
    threshold = OFFSET + 10.0 * np.log10(np.mean(per_block[keep])) + RELATIVE_GATE
    keep = keep & (levels > threshold)
    if not np.any(keep):
        return None
    return float(OFFSET + 10.0 * np.log10(np.mean(per_block[keep])))


def oversample_factor(rate: int) -> int:
    """Oversampling for true-peak estimation: 4x at the rates BS.1770-4 specifies, less above them,
    since the point is an effective rate near 192k rather than a fixed multiplier."""
    if rate <= 48000:
        return 4
    if rate <= 96000:
        return 2
    return 1


def true_peak_dbtp(audio: np.ndarray, rate: int) -> float:
    """Inter-sample peak, estimated by oversampling as BS.1770-4 Annex 2 prescribes."""
    from scipy.signal import resample_poly
    factor = oversample_factor(rate)
    upsampled = resample_poly(audio.astype(np.float64), factor, 1, axis=-1) if factor > 1 else audio
    peak = float(np.max(np.abs(upsampled))) if upsampled.size else 0.0
    return to_db(peak)


def to_db(value: float) -> float:
    return float(20.0 * np.log10(value)) if value > 0 else -np.inf


def measure(audio, rate: int, true_peak: bool = True) -> dict:
    """Levels for one waveform. Cheap enough to run on every generation, and every field is a
    measurement rather than a verdict."""
    samples = normalize_waveform(audio).astype(np.float64)
    channels, count = samples.shape
    peak = float(np.max(np.abs(samples))) if count else 0.0
    result = {
        'channels': channels,
        'samples': int(count),
        'duration': round(count / rate, 3) if rate else None,
        'sample_rate': int(rate),
        'peak': round(peak, 6),
        'peak_dbfs': round(to_db(peak), 2) if peak > 0 else None,
        'rms': [round(float(np.sqrt(np.mean(np.square(c)))), 6) for c in samples],
        'dc_offset': [round(float(np.mean(c)), 6) for c in samples],
        'clipped': int(np.count_nonzero(np.abs(samples) >= 1.0)),
    }
    lufs = integrated_lufs(samples, rate) if count else None
    result['lufs'] = round(lufs, 2) if lufs is not None else None
    if true_peak and count:
        result['true_peak_dbtp'] = round(true_peak_dbtp(samples, rate), 2)
    return result
