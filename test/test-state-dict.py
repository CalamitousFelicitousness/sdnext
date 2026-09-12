#!/usr/bin/env python
"""
Offline unit tests for modules.sd_models_utils state dict handling.

Covers:

- ``StateDictCache`` default state and the enable/disable scoping the lora
  family chain relies on
- ``read_state_dict`` raising rather than returning None, and keeping a cancel
  distinguishable from a broken file

No running server required.

Usage:
    python test/test-state-dict.py
"""

import os
import sys
import tempfile

import torch
import safetensors.torch

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

# shared first: importing sd_models_utils ahead of it re-enters a partly built module
from modules import shared                              # pylint: disable=wrong-import-position
from modules import sd_models_utils as sdu              # pylint: disable=wrong-import-position
from modules.errors import log                          # pylint: disable=wrong-import-position


results = {'passed': 0, 'failed': 0}


def run_test(fn):
    try:
        fn()
        results['passed'] += 1
        log.info(f'  PASS: {fn.__name__}')
    except Exception as e:  # pylint: disable=broad-except
        results['failed'] += 1
        log.error(f'  FAIL: {fn.__name__} ({e})')


def write_sample(directory: str) -> str:
    path = os.path.join(directory, 'sample.safetensors')
    safetensors.torch.save_file({'a.weight': torch.zeros(4, 4)}, path)
    return path


# ============================================================
# StateDictCache
# ============================================================

def test_cache_is_off_by_default():
    """Only the lora chain pays for pinning; everything else must not cache."""
    sdu.state_dict_cache.set('/fake/path.safetensors', {'k': 1})
    assert sdu.state_dict_cache.get('/fake/path.safetensors') is None


def test_cache_round_trip_while_enabled():
    sdu.state_dict_cache.enable()
    try:
        sdu.state_dict_cache.set('/fake/path.safetensors', {'k': 1})
        assert sdu.state_dict_cache.get('/fake/path.safetensors') == {'k': 1}
    finally:
        sdu.state_dict_cache.disable()
    assert sdu.state_dict_cache.get('/fake/path.safetensors') is None, 'disable must drop the entry'


# ============================================================
# read_state_dict
# ============================================================

def test_read_returns_tensors():
    with tempfile.TemporaryDirectory() as d:
        sd = sdu.read_state_dict(write_sample(d), what='test')
        assert list(sd.keys()) == ['a.weight']
        assert sd['a.weight'].shape == (4, 4)


def test_missing_file_raises_read_error():
    try:
        sdu.read_state_dict('/nonexistent/nope.safetensors', what='test')
        raise AssertionError('expected StateDictReadError')
    except sdu.StateDictReadError as e:
        assert 'nope.safetensors' in str(e), str(e)


def test_cancel_raises_load_interrupted_not_read_error():
    """A cancel must never be reported as a broken file."""
    orig = shared.state.interrupted
    shared.state.interrupted = True
    try:
        with tempfile.TemporaryDirectory() as d:
            sdu.read_state_dict(write_sample(d), what='test')
        raise AssertionError('expected LoadInterrupted')
    except sdu.LoadInterrupted:
        pass
    except sdu.StateDictReadError:
        raise AssertionError('cancel must not surface as a read failure') from None
    finally:
        shared.state.interrupted = orig


def test_read_failure_chains_the_cause():
    """The platform detail (a Windows commit failure) rides on the message and
    stays reachable as __cause__."""
    orig = sdu.safetensors.torch.load_file

    def boom(*_args, **_kwargs):
        raise OSError('The paging file is too small for this operation to complete')

    sdu.safetensors.torch.load_file = boom
    try:
        with tempfile.TemporaryDirectory() as d:
            sdu.read_state_dict(write_sample(d), what='test')
        raise AssertionError('expected StateDictReadError')
    except sdu.StateDictReadError as e:
        assert 'paging file' in str(e), str(e)
        assert isinstance(e.__cause__, OSError), 'original error must be chained'
    finally:
        sdu.safetensors.torch.load_file = orig


if __name__ == '__main__':
    log.warning('=== StateDictCache ===')
    for f in [test_cache_is_off_by_default, test_cache_round_trip_while_enabled]:
        run_test(f)
    log.warning('=== read_state_dict ===')
    for f in [
        test_read_returns_tensors,
        test_missing_file_raises_read_error,
        test_cancel_raises_load_interrupted_not_read_error,
        test_read_failure_chains_the_cause,
    ]:
        run_test(f)
    log.warning(f'Total: {results["passed"]} passed, {results["failed"]} failed')
    sys.exit(1 if results['failed'] else 0)
