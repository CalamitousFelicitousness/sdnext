#!/usr/bin/env python
"""
Offline regression tests for the audio track of generated video.

This path fails quietly. A muxed file with a silent, half-level, pitch-shifted or channel-collapsed
soundtrack plays fine and looks correct in every log line, so the checks here measure the decoded
waveform rather than asserting that the writer returned. Two defects it is written against have
already shipped: audio captured after the batch script hooks rewrapped the sample list, which
muxed silence onto every generic-path video, and a hard-coded stereo stream layout, which cost
mono soundtracks 3 dB to ffmpeg's upmix compensation.

Covers:

- ``get_audio_rate`` resolution order, since muxing at the wrong rate shifts the pitch
- the tensor to container path at each rate the shipped models use, checked by decoding it back
- channel handling for mono, stereo and transposed input
- clipping, which must clamp rather than wrap
- the ``AudioFrameList`` attribute that the capture ordering exists to protect

Tolerances come from measuring the codec: aac round-trips a tone to within 1.5 percent of its
input rms and pads the tail by under 25 ms, so the bounds below sit a few times outside that and
still catch a 3 dB error.

No running server required, no model, no network.

Usage:
    python test/test-video-audio.py
"""

import os
import sys
import tempfile

import numpy as np
import torch

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

os.environ['SD_INSTALL_QUIET'] = '1'

import modules.cmd_args  # pylint: disable=wrong-import-position
import installer  # pylint: disable=wrong-import-position
orig_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    modules.cmd_args.parse_args()
finally:
    sys.argv = orig_argv
installer.add_args(modules.cmd_args.parser)
modules.cmd_args.parsed, _ = modules.cmd_args.parser.parse_known_args([])

import av                                                     # pylint: disable=wrong-import-position
from modules.errors import log                                # pylint: disable=wrong-import-position
from modules.audio import stream                              # pylint: disable=wrong-import-position
from modules.processing_diffusers import AudioFrameList, attach_audio  # pylint: disable=wrong-import-position
from modules.video_models import video_save                   # pylint: disable=wrong-import-position


results: dict[str, dict] = {}
FPS = 24
FRAMES = 24
DURATION = FRAMES / FPS
RMS_TOLERANCE = 0.05 # measured worst case 0.0145 across rates and channel counts
DC_TOLERANCE = 1e-3 # measured worst case 4.2e-5
PAD_TOLERANCE = 0.05 # seconds; aac pads the tail, measured 2.7 ms to 24 ms


def category(name: str):
    if name not in results:
        results[name] = {'passed': 0, 'failed': 0, 'tests': []}
    return name


def record(cat: str, passed: bool, name: str, detail: str = ''):
    status = 'PASS' if passed else 'FAIL'
    results[cat]['passed' if passed else 'failed'] += 1
    results[cat]['tests'].append((status, name))
    msg = f'  {status}: {name}'
    if detail:
        msg += f' ({detail})'
    if passed:
        log.info(msg)
    else:
        log.error(msg)


def run_test(cat: str, fn):
    name = fn.__name__
    try:
        ok = fn()
        if ok is False:
            record(cat, False, name)
        else:
            record(cat, True, name)
    except AssertionError as e:
        record(cat, False, name, str(e))
    except Exception as e:  # pylint: disable=broad-except
        record(cat, False, name, f'exception: {e}')


def tone(rate: int, channels: int = 2, amp: float = 0.25, freq: float = 440.0, seconds: float = DURATION):
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    wave = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([wave] * channels)


def mux(audio, rate: int, tmp: str, name: str = 'clip.mp4'):
    fn = os.path.join(tmp, name)
    frames = torch.zeros((FRAMES, 64, 64, 3), dtype=torch.uint8)
    video_save.atomic_save_video(fn, tensor=frames, audio=audio, fps=FPS, codec='libx264', sample_rate=rate)
    return fn


def decode(fn: str):
    """Decoded soundtrack as (channels, samples), plus the rate and layout the container declares."""
    with av.open(fn) as container:
        if not container.streams.audio:
            return None, None, None
        astream = container.streams.audio[0]
        rate = int(astream.codec_context.sample_rate)
        layout = astream.layout.name
        channels = astream.layout.nb_channels
        chunks = [frame.to_ndarray() for frame in container.decode(audio=0)]
    if not chunks:
        return None, rate, layout
    data = np.concatenate(chunks, axis=-1).astype(np.float32)
    if data.ndim == 2 and data.shape[0] == 1 and channels == 2: # packed rather than planar
        data = data.reshape(-1, 2).T
    return data, rate, layout


def rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x is not None and x.size else 0.0


# ============================================================
# Rate resolution
# ============================================================

class FakeVocoderConfig:
    def __init__(self, rate):
        self.output_sampling_rate = rate


class FakeVocoder:
    def __init__(self, rate):
        self.config = FakeVocoderConfig(rate)


class FakeModel:
    def __init__(self, rate):
        self.vocoder = FakeVocoder(rate)


class FakeP:
    def __init__(self, rate=None):
        if rate is not None:
            self.audio_sampling_rate = rate


def with_model(model, fn):
    """Swap the shared model slot for the duration of one call, releasing the lock that would
    otherwise discard the assignment."""
    import modules.modeldata as modeldata
    was_locked = modeldata.model_data.locked
    previous = modeldata.model_data.sd_model
    modeldata.model_data.locked = False
    modeldata.model_data.sd_model = model
    try:
        return fn()
    finally:
        modeldata.model_data.sd_model = previous
        modeldata.model_data.locked = was_locked


def test_rate_prefers_the_processing_object():
    got = with_model(FakeModel(24000), lambda: stream.get_audio_rate(FakeP(48000), default=16000))
    assert got == 48000, f'expected the request rate, got {got}'


def test_rate_falls_back_to_the_vocoder():
    got = with_model(FakeModel(44100), lambda: stream.get_audio_rate(FakeP(), default=16000))
    assert got == 44100, f'expected the vocoder rate, got {got}'


def test_rate_falls_back_to_the_default():
    got = with_model(None, lambda: stream.get_audio_rate(None, default=16000))
    assert got == 16000, f'expected the default, got {got}'


def test_rate_is_an_int():
    got = with_model(FakeModel(48000.0), lambda: stream.get_audio_rate(None, default=24000))
    assert isinstance(got, int), f'expected int, got {type(got).__name__}'


# ============================================================
# Mux round trip
# ============================================================

def roundtrip(rate: int, channels: int, amp: float = 0.25):
    source = tone(rate, channels, amp=amp)
    with tempfile.TemporaryDirectory() as tmp:
        fn = mux(torch.from_numpy(source), rate, tmp)
        return (source,) + decode(fn)


def check_track(rate: int, channels: int):
    source, decoded, out_rate, layout = roundtrip(rate, channels)
    assert decoded is not None, f'{rate}/{channels}ch: no audio decoded'
    assert out_rate == rate, f'{rate}/{channels}ch: rate became {out_rate}, which shifts the pitch'
    assert decoded.shape[0] == channels, f'{rate}/{channels}ch: channels became {decoded.shape[0]} ({layout})'
    src_rms, got_rms = rms(source), rms(decoded)
    assert got_rms > 0.01, f'{rate}/{channels}ch: track is silent (rms {got_rms})'
    relative = abs(got_rms - src_rms) / src_rms
    assert relative < RMS_TOLERANCE, f'{rate}/{channels}ch: level moved {relative:.3f}, {src_rms:.5f} to {got_rms:.5f}'
    dc = float(np.mean(decoded))
    assert abs(dc) < DC_TOLERANCE, f'{rate}/{channels}ch: dc offset {dc}'
    padding = (decoded.shape[-1] - source.shape[-1]) / rate
    assert 0 <= padding < PAD_TOLERANCE, f'{rate}/{channels}ch: length moved by {padding:.4f}s'


def test_mux_stereo_at_every_shipped_rate():
    # 24k is LTX-2.0, 32k is MiniMax H3, 48k is LTX 2.3 and 2.5
    for rate in (24000, 32000, 48000):
        check_track(rate, 2)


def test_mux_mono_keeps_its_level():
    # a stereo stream declared for mono audio makes ffmpeg upmix it, and its power compensation
    # takes 3 dB off, which is quiet rather than broken and so goes unnoticed
    for rate in (24000, 32000, 48000):
        check_track(rate, 1)


def test_mux_accepts_a_transposed_waveform():
    rate = 48000
    source = tone(rate, 2)
    with tempfile.TemporaryDirectory() as tmp:
        fn = mux(torch.from_numpy(source.T.copy()), rate, tmp) # samples first, as some pipelines emit
        decoded, out_rate, _layout = decode(fn)
    assert decoded is not None and decoded.shape[0] == 2, f'transposed input became {decoded.shape if decoded is not None else None}'
    assert out_rate == rate
    assert abs(rms(decoded) - rms(source)) / rms(source) < RMS_TOLERANCE


def test_mux_accepts_a_batch_dimension():
    rate = 48000
    source = tone(rate, 2)
    with tempfile.TemporaryDirectory() as tmp:
        fn = mux(torch.from_numpy(source[None, ...].copy()), rate, tmp)
        decoded, _rate, _layout = decode(fn)
    assert decoded is not None and decoded.shape[0] == 2, 'a leading batch dimension should squeeze away'


def test_mux_clamps_instead_of_wrapping():
    # int16 conversion has to clamp: a wrap turns the loudest part of a track into inverted noise
    rate = 48000
    source = tone(rate, 2, amp=2.0)
    clipped = np.clip(source, -1.0, 1.0)
    with tempfile.TemporaryDirectory() as tmp:
        fn = mux(torch.from_numpy(source), rate, tmp)
        decoded, _rate, _layout = decode(fn)
    assert decoded is not None
    n = min(decoded.shape[-1], clipped.shape[-1])
    correlation = float(np.corrcoef(decoded[0, :n], clipped[0, :n])[0, 1])
    assert correlation > 0.8, f'decoded track does not follow the clipped source, correlation {correlation:.3f}'
    assert float(np.max(np.abs(decoded))) < 1.5, 'decoded samples ran far past full scale'


def test_video_without_audio_has_no_audio_stream():
    with tempfile.TemporaryDirectory() as tmp:
        fn = mux(None, 48000, tmp)
        decoded, rate, _layout = decode(fn)
    assert decoded is None and rate is None, 'a silent request should write no audio stream at all'


# ============================================================
# The attribute the capture ordering protects
# ============================================================

def test_attach_audio_carries_the_track():
    frames = ['a', 'b']
    wrapped = attach_audio(frames, 'waveform')
    assert isinstance(wrapped, AudioFrameList)
    assert list(wrapped) == frames, 'frames must survive the wrapper'
    assert getattr(wrapped, 'audio', None) == 'waveform'


def test_attach_audio_is_a_noop_without_audio():
    frames = ['a', 'b']
    assert attach_audio(frames, None) is frames


def test_a_plain_list_cannot_hold_the_track():
    # this is the mechanism behind the silent-audio defect: the batch script hooks rewrap the
    # sample list, and a plain list has nowhere to keep the attribute, so anything reading it after
    # the hooks reads None. The capture in process_images_inner has to stay above them.
    wrapped = attach_audio(['a'], 'waveform')
    rewrapped = list(wrapped)
    assert getattr(rewrapped, 'audio', None) is None, 'a plain list unexpectedly kept the attribute'
    try:
        rewrapped.audio = 'waveform'
        raise AssertionError('a plain list unexpectedly accepted the attribute')
    except AttributeError:
        pass


# ============================================================
# Runner
# ============================================================

def run_all():
    rates = category('rate resolution')
    for fn in [
        test_rate_prefers_the_processing_object,
        test_rate_falls_back_to_the_vocoder,
        test_rate_falls_back_to_the_default,
        test_rate_is_an_int,
    ]:
        run_test(rates, fn)

    muxing = category('mux round trip')
    for fn in [
        test_mux_stereo_at_every_shipped_rate,
        test_mux_mono_keeps_its_level,
        test_mux_accepts_a_transposed_waveform,
        test_mux_accepts_a_batch_dimension,
        test_mux_clamps_instead_of_wrapping,
        test_video_without_audio_has_no_audio_stream,
    ]:
        run_test(muxing, fn)

    plumbing = category('track plumbing')
    for fn in [
        test_attach_audio_carries_the_track,
        test_attach_audio_is_a_noop_without_audio,
        test_a_plain_list_cannot_hold_the_track,
    ]:
        run_test(plumbing, fn)

    log.warning('=== Results ===')
    total_passed = 0
    total_failed = 0
    for cat_name, info in results.items():
        ok = info['failed'] == 0
        status = 'PASS' if ok else 'FAIL'
        log.info(f"  {cat_name}: {info['passed']} passed, {info['failed']} failed [{status}]")
        total_passed += info['passed']
        total_failed += info['failed']
    log.warning(f'Total: {total_passed} passed, {total_failed} failed')
    return total_failed == 0


if __name__ == '__main__':
    import time
    t0 = time.time()
    success = run_all()
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if success else 1)
