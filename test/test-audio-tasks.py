#!/usr/bin/env python
"""
Offline tests for the audio derived operations.

Everything past text2music operates on a track the caller supplies, and the ways that goes wrong
are quiet rather than loud. A source at the wrong sample rate decodes fine and is heard as a pitch
shift; a mono file against a stereo vae is a shape mismatch deep inside an encode; a task asking
for a component the checkpoint does not ship raises from the middle of the pipeline rather than
before the model loads. The checks here measure the converted waveform and read the refusals.

What each task consumes is declared in the registry rather than recovered from its name, so these
tests read the same table the runner does, and a task added without its requirements fails here.

Covers:

- refusals: an unsupported task, a task missing its source track, a task the checkpoint cannot run
- the arguments derived per task, and that values for inputs a task does not take are dropped
- source loading: rate conversion, channel layout, and the duration that must survive both

No running server required, no model, no network.

Usage:
    python test/test-audio-tasks.py
"""

import os
import sys
import tempfile

import numpy as np

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

from modules.errors import log                                # pylint: disable=wrong-import-position
from modules.audio import save as audio_save                  # pylint: disable=wrong-import-position
from modules.audio_models import audio_run, models_def        # pylint: disable=wrong-import-position
from modules.audio_models.audio_error import AudioError       # pylint: disable=wrong-import-position


results: dict[str, dict] = {}
TURBO = models_def.find('ACE-Step 1.5 XL Turbo')
BASE = models_def.find('ACE-Step 1.5 XL Base')
SILENCE = np.zeros((2, 48000), dtype=np.float32)


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


def derive(row, task, **kwargs):
    kwargs.setdefault('source', None)
    kwargs.setdefault('span', None)
    kwargs.setdefault('strength', None)
    kwargs.setdefault('track', '')
    kwargs.setdefault('classes', None)
    return audio_run.task_args_for(row, task, **kwargs)


def refusal(row, task, **kwargs):
    try:
        derive(row, task, **kwargs)
    except AudioError as e:
        return e
    return None


def tone(rate: int, channels: int, seconds: float = 1.0, freq: float = 440.0):
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    wave = (0.25 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.repeat(wave[None, :], channels, axis=0)


# ============================================================
# Refusals
# ============================================================

def test_a_task_the_checkpoint_cannot_run_is_refused():
    """cover needs the tokenizer pair, which the turbo repo does not ship."""
    assert 'cover' not in TURBO.tasks, 'turbo should not offer cover'
    assert 'cover' in BASE.tasks, 'base should offer cover'
    err = refusal(TURBO, 'cover', source=SILENCE, strength=0.5)
    assert err is not None and err.code == 400, err
    assert 'unsupported' in err.msg, err.msg


def test_a_task_without_its_source_track_is_refused():
    for task in ('repaint', 'extract', 'lego', 'complete'):
        err = refusal(TURBO, task, track='vocals', span=(0, 5), classes=['bass'])
        assert err is not None and err.code == 400, f'{task}: {err}'
        assert 'needs a source track' in err.msg, f'{task}: {err.msg}'


def test_an_unknown_task_is_refused():
    err = refusal(TURBO, 'nonsense', source=SILENCE)
    assert err is not None and err.code == 400, err


def test_text2music_needs_nothing():
    assert derive(TURBO, 'text2music') == {}


# ============================================================
# Derived arguments
# ============================================================

def test_each_task_derives_the_arguments_it_declares():
    cases = {
        'repaint': ({'source': SILENCE, 'span': (5.0, 20.0)}, {'src_audio', 'repainting_start', 'repainting_end'}),
        'extract': ({'source': SILENCE, 'track': 'vocals'}, {'src_audio', 'track_name'}),
        'lego': ({'source': SILENCE, 'span': (0.0, -1.0), 'track': 'drums'}, {'src_audio', 'repainting_start', 'repainting_end', 'track_name'}),
        'complete': ({'source': SILENCE, 'classes': ['bass', 'drums']}, {'src_audio', 'complete_track_classes'}),
    }
    for task, (kwargs, expected) in cases.items():
        got = set(derive(TURBO, task, **kwargs))
        assert got == expected, f'{task}: {got} != {expected}'
    cover = derive(BASE, 'cover', source=SILENCE, strength=0.6)
    assert set(cover) == {'src_audio', 'audio_cover_strength'}, set(cover)
    assert cover['audio_cover_strength'] == 0.6


def test_values_a_task_does_not_take_are_dropped():
    """A leftover control on the caller's side must not steer a task that ignores it."""
    got = derive(TURBO, 'extract', source=SILENCE, track='vocals', span=(1.0, 2.0), strength=0.5)
    assert set(got) == {'src_audio', 'track_name'}, set(got)


def test_the_registry_and_the_derivation_agree():
    """Every task a row offers must derive without error once its declared inputs are supplied."""
    for row in (TURBO, BASE):
        for task in row.tasks:
            spec = models_def.task_of(row, task)
            assert spec is not None, f'{row.name}: {task} has no spec'
            kwargs = {
                'source': SILENCE if spec.source else None,
                'span': (0.0, -1.0) if spec.span else None,
                'strength': 0.5 if spec.strength else None,
                'track': 'vocals' if spec.track else '',
                'classes': ['bass'] if spec.classes else None,
            }
            derive(row, task, **kwargs)


# ============================================================
# Source loading
# ============================================================

def test_source_is_converted_to_the_model_rate_and_layout():
    """A 44.1 kHz mono upload against a 48 kHz stereo model, which is the common case."""
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, 'src.flac')
        audio_save.write_audio(fn, tone(44100, 1), 44100, ext='flac')
        tensor = audio_run.load_source(fn, 48000)
    assert tensor.shape[0] == 2, f'not upmixed to stereo: {tuple(tensor.shape)}'
    seconds = tensor.shape[-1] / 48000
    assert abs(seconds - 1.0) < 0.01, f'duration changed by resampling: {seconds}'


def test_source_at_the_model_rate_is_unchanged_in_length():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, 'src.flac')
        audio_save.write_audio(fn, tone(48000, 2, seconds=2.0), 48000, ext='flac')
        tensor = audio_run.load_source(fn, 48000)
    assert tensor.shape == (2, 96000), tuple(tensor.shape)


def test_samples_may_be_passed_instead_of_a_path():
    tensor = audio_run.load_source(np.zeros((2, 4800), dtype=np.float32), 48000)
    assert tuple(tensor.shape) == (2, 4800), tuple(tensor.shape)
    mono = audio_run.load_source(np.zeros(4800, dtype=np.float32), 48000)
    assert mono.dim() == 2 and mono.shape[0] == 1, tuple(mono.shape)


def test_a_missing_or_undecodable_source_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            audio_run.load_source(os.path.join(tmp, 'nope.flac'), 48000)
            raise AssertionError('missing file was accepted')
        except AudioError as e:
            assert e.code == 400, e.code
        planted = os.path.join(tmp, 'planted.flac')
        with open(planted, 'w', encoding='utf-8') as f:
            f.write('not audio')
        try:
            audio_run.load_source(planted, 48000)
            raise AssertionError('undecodable file was accepted')
        except AudioError as e:
            assert e.code == 400, e.code


def test_no_source_stays_none():
    assert audio_run.load_source(None, 48000) is None


# ============================================================
# Runner
# ============================================================

def run_all():
    refusals = category('refusals')
    for fn in [
        test_a_task_the_checkpoint_cannot_run_is_refused,
        test_a_task_without_its_source_track_is_refused,
        test_an_unknown_task_is_refused,
        test_text2music_needs_nothing,
    ]:
        run_test(refusals, fn)

    derivation = category('derivation')
    for fn in [
        test_each_task_derives_the_arguments_it_declares,
        test_values_a_task_does_not_take_are_dropped,
        test_the_registry_and_the_derivation_agree,
    ]:
        run_test(derivation, fn)

    loading = category('source loading')
    for fn in [
        test_source_is_converted_to_the_model_rate_and_layout,
        test_source_at_the_model_rate_is_unchanged_in_length,
        test_samples_may_be_passed_instead_of_a_path,
        test_a_missing_or_undecodable_source_is_refused,
        test_no_source_stays_none,
    ]:
        run_test(loading, fn)

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
