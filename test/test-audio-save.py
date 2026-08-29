#!/usr/bin/env python
"""
Offline unit tests for the audio save path.

Covers the pieces that decide whether a generated track is reproducible:

- ``write_audio`` / ``read_audio`` round trips per container
- parameter tags surviving flac, opus and mp3, and demonstrably not surviving wav
- the merged container-plus-stream reader, which is what makes opus tags readable at all
- unicode and large payloads, the two things container tag readers get wrong most often
- ``retag`` leaving the encoded audio byte-identical
- ``stream_rate`` snapping for encoders that reject the source rate
- ``normalize_waveform`` axis handling
- BS.1770 K-weighting against the coefficient table the recommendation prints
- gated loudness, its short-signal refusal, and the level statistics the smoke checks read

No running server required, no model, no network.

Usage:
    python test/test-audio-save.py
"""

import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

os.environ['SD_INSTALL_QUIET'] = '1'

# Bootstrap cmd_args before any module that pulls in shared.py.
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

from modules.errors import log                                    # pylint: disable=wrong-import-position
from modules.audio import loudness, metadata, save, stream        # pylint: disable=wrong-import-position
from modules.infotext import quote, unquote, re_param             # pylint: disable=wrong-import-position


# ============================================================
# Test infrastructure
# ============================================================

results: dict[str, dict] = {}
SR = 48000

# Coefficients ITU-R BS.1770-4 tabulates for 48 kHz, transcribed from the recommendation.
SPEC_48K_SHELF_B = [1.53512485958697, -2.69169618940638, 1.19839281085285]
SPEC_48K_SHELF_A = [1.0, -1.69065929318241, 0.73248077421585]
SPEC_48K_HIGHPASS_B = [1.0, -2.0, 1.0]
SPEC_48K_HIGHPASS_A = [1.0, -1.99004745483398, 0.99007225036621]

# A stereo 1 kHz sine at amplitude 0.1 measured through torchaudio's independent BS.1770
# implementation on 2026-08-29. Kept as a constant so this suite needs no torchaudio.
TORCHAUDIO_SINE_LUFS = -20.035
QC_TOLERANCE = 0.2 # EBU R128 v5 measurement-error tolerance


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


def sine(seconds: float = 1.0, freq: float = 1000.0, amp: float = 0.1, rate: int = SR, channels: int = 2):
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    wave = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([wave] * channels)


# ============================================================
# Container round trips
# ============================================================

def roundtrip(ext: str, text: str = 'Prompt: test, Steps: 8, Seed: 42'):
    """Write a short tone with a parameters tag, read the tag back through the merged reader."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, f'take.{ext}')
        written = save.write_audio(fn, sine(1.0), SR, ext=ext, metadata={metadata.INFO_KEY: text, 'comment': text})
        assert written is not None, f'{ext}: write returned None'
        assert os.path.exists(fn) and os.path.getsize(fn) > 0, f'{ext}: no bytes written'
        return metadata.read_audio_metadata(fn).get(metadata.INFO_KEY), text


def test_flac_carries_parameters():
    got, want = roundtrip('flac')
    assert got == want, f'flac parameters mismatch: {got!r}'


def test_opus_carries_parameters():
    # opus keeps tags on the stream and leaves the container empty, so this fails the moment the
    # reader stops merging both
    got, want = roundtrip('opus')
    assert got == want, f'opus parameters mismatch: {got!r}'


def test_mp3_carries_parameters():
    got, want = roundtrip('mp3')
    assert got == want, f'mp3 parameters mismatch: {got!r}'


def test_wav_drops_parameters():
    # ffmpeg's RIFF INFO writer takes a fixed key whitelist and discards the rest silently; the
    # sidecar is the only parameter record for wav, and this pins that it really is a limitation
    got, _want = roundtrip('wav')
    assert got is None, f'wav unexpectedly carried parameters: {got!r}'


def test_unicode_parameters_survive():
    text = 'Prompt: sakura さくら, Ünïcodé, эмодзи 🎵, Steps: 8'
    got, want = roundtrip('flac', text)
    assert got == want, f'unicode mangled: {got!r}'


def test_large_parameters_survive():
    text = 'Prompt: ' + ('long lyric line, ' * 4000)
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, 'take.flac')
        save.write_audio(fn, sine(0.5), SR, ext='flac', metadata={metadata.INFO_KEY: text})
        got = metadata.read_audio_metadata(fn).get(metadata.INFO_KEY)
    assert got == text, f'large payload truncated: {len(got or "")} of {len(text)}'


def test_tag_read_is_case_insensitive():
    # ffmpeg's tag dictionary is case-insensitive on write, so the case a file comes back with is
    # not necessarily the case it went in with
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, 'take.flac')
        save.write_audio(fn, sine(0.5), SR, ext='flac', metadata={'PARAMETERS': 'upper case key'})
        tags = metadata.read_audio_metadata(fn)
    assert tags.get('parameters') == 'upper case key', f'case-insensitive read failed: {list(tags)}'


def test_flac_preserves_sample_count():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, 'take.flac')
        source = sine(2.0)
        save.write_audio(fn, source, SR, ext='flac')
        decoded, rate = save.read_audio(fn)
    assert rate == SR, f'rate changed: {rate}'
    assert decoded.shape[-1] == source.shape[-1], f'samples {decoded.shape[-1]} != {source.shape[-1]}'


def test_flac_is_lossless_within_int16():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, 'take.flac')
        source = sine(1.0)
        save.write_audio(fn, source, SR, ext='flac')
        decoded, _rate = save.read_audio(fn)
    delta = float(np.max(np.abs(decoded[:, :source.shape[-1]] - source)))
    assert delta < 1e-4, f'flac round trip lost {delta}, more than int16 quantization'


def test_opus_resamples_to_a_supported_rate():
    # opus only accepts 8/12/16/24/48k, so a 44.1k source has to be resampled rather than refused
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, 'take.opus')
        written = save.write_audio(fn, sine(1.0, rate=44100), 44100, ext='opus')
        assert written is not None, 'opus write at 44.1k returned None'
        decoded, rate = save.read_audio(fn)
    assert rate == 48000, f'expected resample to 48k, got {rate}'
    duration = decoded.shape[-1] / rate
    assert abs(duration - 1.0) < 0.05, f'duration drifted to {duration}'


# ============================================================
# Stream helpers
# ============================================================

def test_stream_rate_snaps_only_when_needed():
    assert stream.stream_rate('libopus', 44100) == 48000
    assert stream.stream_rate('libopus', 48000) == 48000
    assert stream.stream_rate('libopus', 96000) == 48000, 'above the top rate should clamp down'
    assert stream.stream_rate('flac', 44100) == 44100, 'flac takes any rate'
    assert stream.stream_rate('libmp3lame', 48000) == 48000


def test_normalize_waveform_axis_handling():
    n = 800
    assert stream.normalize_waveform(np.zeros(n)).shape == (1, n), 'mono 1d'
    assert stream.normalize_waveform(np.zeros((2, n))).shape == (2, n), 'already channels-first'
    assert stream.normalize_waveform(np.zeros((n, 2))).shape == (2, n), 'samples-first should transpose'
    assert stream.normalize_waveform(np.zeros((1, 2, n))).shape == (2, n), 'batch dim should squeeze'


def test_normalize_waveform_accepts_tensors():
    import torch
    out = stream.normalize_waveform(torch.zeros(2, 400))
    assert isinstance(out, np.ndarray) and out.shape == (2, 400)
    assert out.dtype == np.float32


def test_decodable_check_is_not_just_the_extension():
    # ffmpeg picks the demuxer from the filename, so an empty file named .flac opens cleanly and
    # reports one audio stream. Only a decoded frame tells the two apart, and the audio-info
    # endpoint relies on this to refuse a planted file
    with tempfile.TemporaryDirectory() as tmp:
        empty = os.path.join(tmp, 'empty.flac')
        open(empty, 'wb').close()
        junk = os.path.join(tmp, 'junk.flac')
        with open(junk, 'wb') as f:
            f.write(b'not audio at all' * 100)
        real = os.path.join(tmp, 'real.flac')
        save.write_audio(real, sine(0.5), SR, ext='flac')
        assert stream.is_decodable_audio(real), 'real audio should decode'
        assert not stream.is_decodable_audio(empty), 'an empty file named .flac must not pass'
        assert not stream.is_decodable_audio(junk), 'junk bytes named .flac must not pass'
        assert not stream.is_decodable_audio(os.path.join(tmp, 'missing.flac'))


# ============================================================
# Metadata helpers
# ============================================================

def test_retag_keeps_audio_identical():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, 'take.flac')
        save.write_audio(fn, sine(1.0), SR, ext='flac', metadata={metadata.INFO_KEY: 'before'})
        before, _rate = save.read_audio(fn)
        ok = metadata.retag(fn, {metadata.INFO_KEY: 'after', 'encoder': 'SD.Next'})
        assert ok, 'retag reported failure'
        after, _rate = save.read_audio(fn)
        tags = metadata.read_audio_metadata(fn)
    assert tags.get(metadata.INFO_KEY) == 'after', f'tag not replaced: {tags.get(metadata.INFO_KEY)!r}'
    assert np.array_equal(before, after), 'retag altered the decoded audio'


def test_sidecar_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, 'take.flac')
        save.write_audio(fn, sine(0.5), SR, ext='flac')
        metadata.write_sidecar(fn, {'parameters': 'p'}, extra={'levels': {'lufs': -14.0}})
        data = metadata.read_sidecar(fn)
    assert data is not None and data.get('parameters') == 'p'
    assert data.get('levels', {}).get('lufs') == -14.0


LYRICS = '[verse]\nWalking down the street, thinking of you\nRain on my coat, a colon: here\n[chorus]\nOh, oh, oh'


def infotext_round_trip(value):
    """Build a one-line infotext holding value and parse it back the way a reader does."""
    line = f'Steps: 8, Seed: 42, Lyrics: {quote(value)}, Sample rate: 48000'
    parsed = {k.strip(): unquote(v.strip()) for k, v in re_param.findall(line)}
    return line, parsed


def test_lyrics_survive_the_infotext_round_trip():
    line, parsed = infotext_round_trip(LYRICS)
    assert '\n' not in line, 'infotext must stay on one line'
    assert set(parsed) == {'Steps', 'Seed', 'Lyrics', 'Sample rate'}, f'neighbouring keys lost: {list(parsed)}'
    assert parsed['Lyrics'] == LYRICS, f'lyrics not preserved: {parsed["Lyrics"]!r}'


def test_flattening_newlines_would_lose_the_structure():
    """The lossy alternative, kept as the contrast that gives the test above its meaning."""
    _line, parsed = infotext_round_trip(LYRICS.replace('\n', ' / '))
    assert parsed['Lyrics'] != LYRICS, 'flattened lyrics unexpectedly matched the original'
    assert '\n' not in parsed['Lyrics'], 'flattened lyrics should carry no newlines'


def test_the_writer_hands_over_lyrics_unflattened():
    """Pins the writer itself, so reinstating a flattening step fails here rather than silently."""
    from modules.audio_models import audio_run, models_def  # pylint: disable=import-outside-toplevel
    row = models_def.default_model()
    p = SimpleNamespace(duration=60.0, lyrics=LYRICS)
    params = audio_run.infotext_params(p, row, 'text2music', 48000, {'lufs': -11.0})
    assert params['Lyrics'] == LYRICS, f'writer altered the lyrics: {params["Lyrics"]!r}'
    _line, parsed = infotext_round_trip(params['Lyrics'])
    assert parsed['Lyrics'] == LYRICS, 'writer output did not survive the round trip'
    assert params['Audio task'] == 'text2music' and params['Sample rate'] == 48000


def test_reading_a_missing_file_is_quiet():
    assert metadata.read_audio_info('/nonexistent/nope.flac') is None
    assert metadata.read_audio_metadata('/nonexistent/nope.flac') == {}
    assert metadata.read_sidecar('/nonexistent/nope.flac') is None


def test_save_audio_writes_audio_and_sidecar():
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, 'generated')
        out = save.save_audio(p=None, waveform=sine(1.0), sample_rate=SR, ext='flac',
                              metadata={metadata.INFO_KEY: 'Prompt: x'}, levels={'lufs': -12.0}, filename=base)
        assert out == f'{base}.flac', f'unexpected path {out}'
        assert os.path.exists(out)
        assert metadata.read_audio_info(out) == 'Prompt: x'
        sidecar = metadata.read_sidecar(out)
    assert sidecar is not None and sidecar.get('levels', {}).get('lufs') == -12.0


def test_save_audio_falls_back_to_flac():
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, 'generated')
        out = save.save_audio(p=None, waveform=sine(0.5), sample_rate=SR, ext='aiff', filename=base)
    assert out is not None and out.endswith('.flac'), f'unknown format should fall back to flac, got {out}'


# ============================================================
# Loudness
# ============================================================

def test_kfilter_matches_the_recommendation_table():
    b, a = loudness.shelf_coeffs(48000)
    shelf_err = float(max(np.max(np.abs(b - SPEC_48K_SHELF_B)), np.max(np.abs(a - SPEC_48K_SHELF_A))))
    b2, a2 = loudness.highpass_coeffs(48000)
    hp_err = float(max(np.max(np.abs(b2 - SPEC_48K_HIGHPASS_B)), np.max(np.abs(a2 - SPEC_48K_HIGHPASS_A))))
    assert shelf_err < 1e-12, f'shelf coefficients drifted from the table by {shelf_err}'
    assert hp_err < 1e-12, f'highpass coefficients drifted from the table by {hp_err}'


def test_loudness_agrees_with_an_independent_implementation():
    got = loudness.integrated_lufs(sine(5.0).astype(np.float64), SR)
    assert got is not None
    assert abs(got - TORCHAUDIO_SINE_LUFS) < QC_TOLERANCE, f'{got} vs reference {TORCHAUDIO_SINE_LUFS}'


def test_loudness_refuses_below_one_gating_block():
    # ffmpeg answers -70 LUFS with a success code here, which reads as a real measurement and is
    # not one; normalizing from it would apply about +56 dB
    for seconds in (0.1, 0.399):
        assert loudness.integrated_lufs(sine(seconds).astype(np.float64), SR) is None, f'{seconds}s should refuse'
    assert loudness.integrated_lufs(sine(0.5).astype(np.float64), SR) is not None, '0.5s should measure'


def test_loudness_gating_discounts_silence():
    rng = np.random.default_rng(0)
    burst = rng.normal(0, 0.2, size=(2, SR)).astype(np.float64)
    padded = np.concatenate([burst, np.zeros((2, SR * 4))], axis=-1)
    alone = loudness.integrated_lufs(burst, SR)
    gated = loudness.integrated_lufs(padded, SR)
    assert alone is not None and gated is not None
    assert abs(gated - alone) < 1.0, f'gating failed: {gated} with silence vs {alone} without'


def test_measure_reports_the_stats_the_smoke_checks_read():
    stats = loudness.measure(sine(1.0), SR)
    assert stats['channels'] == 2 and stats['samples'] == SR
    assert stats['duration'] == 1.0 and stats['sample_rate'] == SR
    assert abs(stats['peak'] - 0.1) < 1e-3, f"peak {stats['peak']}"
    assert stats['clipped'] == 0
    assert all(abs(dc) < 1e-3 for dc in stats['dc_offset']), f"dc {stats['dc_offset']}"
    assert stats['lufs'] is not None and stats['true_peak_dbtp'] is not None


def test_measure_detects_dc_offset_and_clipping():
    # the failure this pins is a decoder returning a constant, which sounds like nothing and
    # measures like a healthy peak
    constant = np.full((2, SR), 0.9, dtype=np.float32)
    stats = loudness.measure(constant, SR)
    assert all(abs(dc - 0.9) < 1e-3 for dc in stats['dc_offset']), f"dc not detected: {stats['dc_offset']}"
    clipping = np.full((2, SR), 1.0, dtype=np.float32)
    assert loudness.measure(clipping, SR)['clipped'] == 2 * SR


def test_measure_reads_the_ace_step_peak_invariant():
    # the ace-step pipeline rescales every output to exactly -1 dBFS, so peak is fixed by the
    # pipeline rather than by the content and cannot serve as a health signal
    target = 10.0 ** (-1.0 / 20.0)
    wave = sine(1.0, amp=0.05)
    wave = wave * (target / np.max(np.abs(wave)))
    stats = loudness.measure(wave, SR)
    assert abs(stats['peak_dbfs'] - (-1.0)) < 0.01, f"peak_dbfs {stats['peak_dbfs']}"


def test_measure_handles_silence():
    stats = loudness.measure(np.zeros((2, SR), dtype=np.float32), SR)
    assert stats['peak'] == 0.0 and stats['lufs'] is None, 'digital silence has no defined loudness'
    assert stats['rms'] == [0.0, 0.0]


# ============================================================
# audio-info endpoint guard
#
# The endpoint reads a caller-supplied path, and its sidecar lookup replaces the extension, so
# without these checks any path ending in an audio extension would return the json beside it.
# ============================================================

def call_audio_info(path, allowed):
    from fastapi.exceptions import HTTPException
    from modules import shared
    from modules.api.audio import APIAudio
    shared.demo = type('Demo', (), {'allowed_paths': list(allowed)})()
    try:
        return None, APIAudio(queue_lock=None).get_audio_info(path)
    except HTTPException as e:
        return e.status_code, None


def test_audio_info_refuses_outside_allowed_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        code, _ = call_audio_info('/etc/passwd', [tmp])
    assert code == 403, f'expected 403, got {code}'


def test_audio_info_refuses_unknown_extension():
    with tempfile.TemporaryDirectory() as tmp:
        code, _ = call_audio_info(os.path.join(tmp, 'notes.txt'), [tmp])
    assert code == 403, f'expected 403, got {code}'


def test_audio_info_refuses_a_planted_file():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, 'secret.json'), 'w', encoding='utf8') as f:
            f.write('{"token": "must-not-leak"}')
        open(os.path.join(tmp, 'secret.flac'), 'wb').close() # right extension, no audio in it
        code, res = call_audio_info(os.path.join(tmp, 'secret.flac'), [tmp])
    assert code == 403, f'expected 403, got {code}'
    assert res is None, 'a planted file must not return a sidecar'


def test_audio_info_reads_the_output_folder_when_gradio_does_not_serve_it():
    # the output folder is commonly outside the paths gradio serves, and a guard that refuses the
    # files this endpoint's own generations produce is worse than no guard
    from modules import shared
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, 'take.flac')
        save.write_audio(fn, sine(0.5), SR, ext='flac', metadata={metadata.INFO_KEY: 'Prompt: y'})
        original_samples, original_audio = shared.opts.outdir_samples, shared.opts.outdir_audio
        try:
            shared.opts.outdir_samples = ''
            shared.opts.outdir_audio = tmp
            code, res = call_audio_info(fn, ['/nonexistent/gradio/root'])
        finally:
            shared.opts.outdir_samples, shared.opts.outdir_audio = original_samples, original_audio
    assert code is None, f'file in the audio output folder refused with {code}'
    assert res.get('info') == 'Prompt: y'


def test_audio_info_reads_a_real_file():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, 'take.flac')
        save.write_audio(fn, sine(0.5), SR, ext='flac', metadata={metadata.INFO_KEY: 'Prompt: x'})
        code, res = call_audio_info(fn, [tmp])
    assert code is None, f'legitimate file refused with {code}'
    assert res.get('info') == 'Prompt: x', f"info {res.get('info')!r}"


# ============================================================
# Runner
# ============================================================

def run_all():
    containers = category('containers')
    for fn in [
        test_flac_carries_parameters,
        test_opus_carries_parameters,
        test_mp3_carries_parameters,
        test_wav_drops_parameters,
        test_unicode_parameters_survive,
        test_large_parameters_survive,
        test_tag_read_is_case_insensitive,
        test_flac_preserves_sample_count,
        test_flac_is_lossless_within_int16,
        test_opus_resamples_to_a_supported_rate,
    ]:
        run_test(containers, fn)

    helpers = category('stream helpers')
    for fn in [
        test_stream_rate_snaps_only_when_needed,
        test_normalize_waveform_axis_handling,
        test_normalize_waveform_accepts_tensors,
        test_decodable_check_is_not_just_the_extension,
    ]:
        run_test(helpers, fn)

    guard = category('audio-info guard')
    for fn in [
        test_audio_info_refuses_outside_allowed_dirs,
        test_audio_info_refuses_unknown_extension,
        test_audio_info_refuses_a_planted_file,
        test_audio_info_reads_the_output_folder_when_gradio_does_not_serve_it,
        test_audio_info_reads_a_real_file,
    ]:
        run_test(guard, fn)

    meta = category('metadata')
    for fn in [
        test_retag_keeps_audio_identical,
        test_sidecar_round_trip,
        test_lyrics_survive_the_infotext_round_trip,
        test_flattening_newlines_would_lose_the_structure,
        test_the_writer_hands_over_lyrics_unflattened,
        test_reading_a_missing_file_is_quiet,
        test_save_audio_writes_audio_and_sidecar,
        test_save_audio_falls_back_to_flac,
    ]:
        run_test(meta, fn)

    levels = category('loudness')
    for fn in [
        test_kfilter_matches_the_recommendation_table,
        test_loudness_agrees_with_an_independent_implementation,
        test_loudness_refuses_below_one_gating_block,
        test_loudness_gating_discounts_silence,
        test_measure_reports_the_stats_the_smoke_checks_read,
        test_measure_detects_dc_offset_and_clipping,
        test_measure_reads_the_ace_step_peak_invariant,
        test_measure_handles_silence,
    ]:
        run_test(levels, fn)

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
