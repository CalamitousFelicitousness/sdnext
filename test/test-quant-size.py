#!/usr/bin/env python
"""
Offline unit tests for the SDNQ component size gate.

The gate only works if the size is known before the weights are fetched, and if two components of
the same class are told apart by size rather than by name.

Covers:

- ``min_size_skip``: the threshold comparison, including the disabled and unmeasured cases
- ``estimate_size``: the meta-device count, its dtype accounting, the embedding tie, the config
  object and auto class paths, and its refusals
- ``skip_small_module``: the loader-facing gate, that it does no work while disabled or while
  SDNQ would not quantize the module anyway, and that prequantized repos are left alone
- ``skip_small_file``: the single-file override gate
- ``get_dit_args``: that ``allow_sdnq`` strips only the SDNQ config

Configurations are written to a temporary directory rather than fetched, so the sizes below are
fixed inputs. The two Qwen3 shapes are the ones that share a class in the tree and differ only in
size: a text encoder of the first shape is 1.11 GB and one of the second is 3.20 GB, which is the
pair a class name cannot separate.

No running server required, no model, no network.

Usage:
    python test/test-quant-size.py
"""

import os
import sys
import json
import tempfile

import torch

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

import transformers                                     # pylint: disable=wrong-import-position
from modules.errors import log                          # pylint: disable=wrong-import-position
from modules import model_quant, shared                 # pylint: disable=wrong-import-position


# ============================================================
# Test infrastructure
# ============================================================

results: dict[str, dict] = {}


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
        import traceback
        traceback.print_exc()


class opts_of: # pylint: disable=invalid-name
    """Hold option values for the body of a test and put the old ones back."""

    def __init__(self, **values):
        self.values = values
        self.previous = {}

    def __enter__(self):
        for name, value in self.values.items():
            self.previous[name] = (name in shared.opts.data, shared.opts.data.get(name, None))
            shared.opts.data[name] = value
        return self

    def __exit__(self, *args):
        for name, (existed, value) in self.previous.items():
            if existed:
                shared.opts.data[name] = value
            else:
                shared.opts.data.pop(name, None)
        return False


# the loader gate only decides while SDNQ would otherwise quantize, so its tests pin every option
# that feeds that condition rather than inheriting whatever the local config holds
GATE_OPTS = {
    'sdnq_quantize_mode': 'pre',
    'sdnq_quantize_weights': ['Model', 'TE', 'VAE'],
    'sdnq_quantize_weights_mode': 'int8',
    'sdnq_quantize_weights_mode_te': 'Same as model',
    'models_not_to_quant': '',
    'trt_quantization': [],
}


def gate_at(threshold):
    return opts_of(sdnq_quantize_min_size=threshold, **GATE_OPTS)


# The two Qwen3 encoder shapes in the tree that share the Qwen3Model class. Every field is taken
# from the shipped config of each, so the sizes the tests assert are the sizes the loader sees.
SMALL_QWEN3 = { # ACE-Step 1.5 XL and Anima 1.0 both load this shape
    'architectures': ['Qwen3Model'],
    'model_type': 'qwen3',
    'hidden_size': 1024,
    'intermediate_size': 3072,
    'num_hidden_layers': 28,
    'num_attention_heads': 16,
    'num_key_value_heads': 8,
    'head_dim': 128,
    'vocab_size': 151669,
    'tie_word_embeddings': True,
}
LARGE_QWEN3 = { # Ovis-Image-7B
    **SMALL_QWEN3,
    'hidden_size': 2048,
    'intermediate_size': 6144,
    'vocab_size': 151936,
}

SMALL_GB = 1.110 # 595,776,512 params at bfloat16
LARGE_GB = 3.205 # 1,720,574,976 params at bfloat16
TOLERANCE = 0.002


def config_dir(tmp: str, name: str, config: dict) -> str:
    path = os.path.join(tmp, name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'config.json'), 'w', encoding='utf8') as f:
        json.dump(config, f)
    return path


# ============================================================
# min_size_skip
# ============================================================

def test_min_size_skip_disabled_at_zero():
    with opts_of(sdnq_quantize_min_size=0):
        assert model_quant.min_size_skip(0.1) is False, 'a threshold of zero must not gate anything'


def test_min_size_skip_disabled_when_negative():
    with opts_of(sdnq_quantize_min_size=-1):
        assert model_quant.min_size_skip(0.1) is False, 'a negative threshold must not gate anything'


def test_min_size_skip_below_threshold_skips():
    with opts_of(sdnq_quantize_min_size=2.0):
        assert model_quant.min_size_skip(1.11) is True, 'a component under the threshold must be skipped'


def test_min_size_skip_above_threshold_quantizes():
    with opts_of(sdnq_quantize_min_size=2.0):
        assert model_quant.min_size_skip(3.205) is False, 'a component over the threshold must be quantized'


def test_min_size_skip_at_threshold_quantizes():
    with opts_of(sdnq_quantize_min_size=2.0):
        assert model_quant.min_size_skip(2.0) is False, 'the threshold itself must quantize, not skip'


def test_min_size_skip_unmeasured_size_quantizes():
    # a failed measurement returns 0, which must not read as "smaller than the threshold"
    with opts_of(sdnq_quantize_min_size=2.0):
        assert model_quant.min_size_skip(0) is False, 'an unmeasured component must not be skipped'


# ============================================================
# estimate_size
# ============================================================

def test_estimate_size_measures_small_encoder():
    with tempfile.TemporaryDirectory() as tmp:
        path = config_dir(tmp, 'small', SMALL_QWEN3)
        size = model_quant.estimate_size(transformers.Qwen3Model, path, dtype=torch.bfloat16)
    assert abs(size - SMALL_GB) < TOLERANCE, f'expected {SMALL_GB} GB, measured {size:.4f}'


def test_estimate_size_measures_large_encoder():
    with tempfile.TemporaryDirectory() as tmp:
        path = config_dir(tmp, 'large', LARGE_QWEN3)
        size = model_quant.estimate_size(transformers.Qwen3Model, path, dtype=torch.bfloat16)
    assert abs(size - LARGE_GB) < TOLERANCE, f'expected {LARGE_GB} GB, measured {size:.4f}'


def test_estimate_size_separates_two_encoders_of_one_class():
    # the case a class name cannot decide: same cls, sizes either side of a 2 GB threshold
    with tempfile.TemporaryDirectory() as tmp:
        small = model_quant.estimate_size(transformers.Qwen3Model, config_dir(tmp, 'small', SMALL_QWEN3), dtype=torch.bfloat16)
        large = model_quant.estimate_size(transformers.Qwen3Model, config_dir(tmp, 'large', LARGE_QWEN3), dtype=torch.bfloat16)
    assert small < 2.0 < large, f'a 2 GB threshold must separate them, got {small:.3f} and {large:.3f}'


def test_estimate_size_counts_the_load_dtype_not_the_config_dtype():
    # the config declares its own dtype, but the component is loaded at the device dtype and that
    # is what quantization saves against, so float32 must come out at twice bfloat16
    with tempfile.TemporaryDirectory() as tmp:
        path = config_dir(tmp, 'small', SMALL_QWEN3)
        half = model_quant.estimate_size(transformers.Qwen3Model, path, dtype=torch.bfloat16)
        full = model_quant.estimate_size(transformers.Qwen3Model, path, dtype=torch.float32)
    assert abs(full - 2 * half) < TOLERANCE, f'float32 must be twice bfloat16, got {full:.4f} and {half:.4f}'


def test_estimate_size_reads_a_config_without_a_model_type():
    # the component's own config class reads this and so does the loader, so the measurement has to
    # as well: routing through AutoConfig instead would leave the gate silently off for the load
    config = {k: v for k, v in SMALL_QWEN3.items() if k != 'model_type'}
    with tempfile.TemporaryDirectory() as tmp:
        size = model_quant.estimate_size(transformers.Qwen3Model, config_dir(tmp, 'bare', config), dtype=torch.bfloat16)
    assert abs(size - SMALL_GB) < TOLERANCE, f'expected {SMALL_GB} GB, measured {size:.4f}'


def test_estimate_size_reads_a_config_with_an_unknown_model_type():
    config = {**SMALL_QWEN3, 'model_type': 'qwen9_unreleased'}
    with tempfile.TemporaryDirectory() as tmp:
        size = model_quant.estimate_size(transformers.Qwen3Model, config_dir(tmp, 'future', config), dtype=torch.bfloat16)
    assert abs(size - SMALL_GB) < TOLERANCE, f'expected {SMALL_GB} GB, measured {size:.4f}'


def test_estimate_size_refuses_missing_class():
    assert model_quant.estimate_size(None, 'any/repo') == 0, 'no class means no measurement'


def test_estimate_size_refuses_missing_repo():
    assert model_quant.estimate_size(transformers.Qwen3Model, None) == 0, 'no repo means no measurement'


def test_estimate_size_refuses_none_repo_string():
    assert model_quant.estimate_size(transformers.Qwen3Model, 'None') == 0, 'the none placeholder means no measurement'


def test_estimate_size_survives_unreadable_config():
    # a config class asked for a repo holding no config file hands back the class defaults rather
    # than raising, and those describe a different model entirely, so an absent config has to be
    # caught as absent instead of measured
    with tempfile.TemporaryDirectory() as tmp:
        empty = os.path.join(tmp, 'empty')
        os.makedirs(empty)
        size = model_quant.estimate_size(transformers.Qwen3Model, empty)
    assert size == 0, f'a missing config must measure as 0 rather than as the class defaults, got {size:.3f}'


def test_estimate_size_ties_embeddings_before_counting():
    # meta init skips the embedding tie, so without an explicit tie the head is counted twice
    config = {**SMALL_QWEN3, 'architectures': ['Qwen3ForCausalLM']}
    head_gb = SMALL_QWEN3['vocab_size'] * SMALL_QWEN3['hidden_size'] * 2 / 1024**3
    with tempfile.TemporaryDirectory() as tmp:
        tied = model_quant.estimate_size(transformers.Qwen3ForCausalLM, config_dir(tmp, 'tied', config), dtype=torch.bfloat16)
        untied = model_quant.estimate_size(transformers.Qwen3ForCausalLM, config_dir(tmp, 'untied', {**config, 'tie_word_embeddings': False}), dtype=torch.bfloat16)
    assert abs(tied - SMALL_GB) < TOLERANCE, f'tied embeddings must count once, expected {SMALL_GB} GB, measured {tied:.4f}'
    assert abs(untied - SMALL_GB - head_gb) < TOLERANCE, f'untied embeddings must count the head, expected {SMALL_GB + head_gb:.4f} GB, measured {untied:.4f}'


def test_estimate_size_uses_a_provided_config_object():
    # a shared text encoder entry can carry its config as an object, which must be measured as
    # handed over rather than fetched; a repo that would fail to fetch proves no fetch happened
    config = transformers.Qwen3Config(**{k: v for k, v in SMALL_QWEN3.items() if k != 'model_type'})
    size = model_quant.estimate_size(transformers.Qwen3Model, 'unused/never-fetched', config=config, dtype=torch.bfloat16)
    assert abs(size - SMALL_GB) < TOLERANCE, f'expected {SMALL_GB} GB, measured {size:.4f}'


def test_estimate_size_resolves_an_auto_class():
    # a repo override loads through AutoModel, so the estimate must resolve the class the same way
    with tempfile.TemporaryDirectory() as tmp:
        size = model_quant.estimate_size(transformers.AutoModel, config_dir(tmp, 'auto', SMALL_QWEN3), dtype=torch.bfloat16)
    assert abs(size - SMALL_GB) < TOLERANCE, f'expected {SMALL_GB} GB, measured {size:.4f}'


# ============================================================
# skip_small_module
# ============================================================

def test_skip_small_module_does_nothing_while_disabled():
    # a bogus repo proves no measurement was attempted: reaching the estimator would still return
    # False, but it would have to fail first
    with gate_at(0):
        assert model_quant.skip_small_module(transformers.Qwen3Model, 'not-a-real/repo', module='TE') is False


def test_skip_small_module_skips_a_small_component():
    with tempfile.TemporaryDirectory() as tmp:
        path = config_dir(tmp, 'small', SMALL_QWEN3)
        with gate_at(2.0):
            assert model_quant.skip_small_module(transformers.Qwen3Model, path, module='TE', dtype=torch.bfloat16) is True


def test_skip_small_module_keeps_a_large_component():
    with tempfile.TemporaryDirectory() as tmp:
        path = config_dir(tmp, 'large', LARGE_QWEN3)
        with gate_at(2.0):
            assert model_quant.skip_small_module(transformers.Qwen3Model, path, module='TE', dtype=torch.bfloat16) is False


def test_skip_small_module_quantizes_when_size_is_unknown():
    # an unreadable config must leave the component quantized rather than skipped, so a source the
    # gate cannot measure behaves as it did before the setting existed. An absolute path that does
    # not exist fails repo-id validation locally, so no fetch is attempted
    with gate_at(2.0):
        assert model_quant.skip_small_module(transformers.Qwen3Model, '/nonexistent/sdnext-test-repo', module='TE') is False


def test_skip_small_module_keeps_everything_at_a_low_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        path = config_dir(tmp, 'small', SMALL_QWEN3)
        with gate_at(0.5):
            assert model_quant.skip_small_module(transformers.Qwen3Model, path, module='TE', dtype=torch.bfloat16) is False


def test_skip_small_module_inactive_in_post_mode():
    # post mode measures the loaded model in sdnq_quantize_model, so the loader gate must not
    # measure or decide anything
    with tempfile.TemporaryDirectory() as tmp:
        path = config_dir(tmp, 'small', SMALL_QWEN3)
        with gate_at(2.0), opts_of(sdnq_quantize_mode='post'):
            assert model_quant.skip_small_module(transformers.Qwen3Model, path, module='TE', dtype=torch.bfloat16) is False


def test_skip_small_module_inactive_when_bucket_unselected():
    with tempfile.TemporaryDirectory() as tmp:
        path = config_dir(tmp, 'small', SMALL_QWEN3)
        with gate_at(2.0), opts_of(sdnq_quantize_weights=['Model']):
            assert model_quant.skip_small_module(transformers.Qwen3Model, path, module='TE', dtype=torch.bfloat16) is False


def test_skip_small_module_inactive_when_dtype_none():
    with tempfile.TemporaryDirectory() as tmp:
        path = config_dir(tmp, 'small', SMALL_QWEN3)
        with gate_at(2.0), opts_of(sdnq_quantize_weights_mode='none'):
            assert model_quant.skip_small_module(transformers.Qwen3Model, path, module='TE', dtype=torch.bfloat16) is False


def test_skip_small_module_inactive_for_excluded_model_family():
    with tempfile.TemporaryDirectory() as tmp:
        path = config_dir(tmp, 'small', SMALL_QWEN3)
        with gate_at(2.0), opts_of(models_not_to_quant=str(shared.sd_model_type)):
            assert model_quant.skip_small_module(transformers.Qwen3Model, path, module='Model', dtype=torch.bfloat16) is False


def test_skip_small_module_leaves_prequantized_repos_alone():
    # a repo that is already sdnq-quantized is not a quantization candidate, so the gate must not
    # claim a decision over it even when its size sits under the threshold
    with tempfile.TemporaryDirectory() as tmp:
        path = config_dir(tmp, 'sdnq-uint4-small', SMALL_QWEN3)
        with gate_at(2.0):
            assert model_quant.skip_small_module(transformers.Qwen3Model, path, module='TE', dtype=torch.bfloat16) is False


# ============================================================
# skip_small_file
# ============================================================

def test_skip_small_file_skips_a_small_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'small.safetensors')
        with open(path, 'wb') as f:
            f.write(b'0' * 1024)
        with gate_at(2.0):
            assert model_quant.skip_small_file(path, module='Model', cls_name='Test') is True


def test_skip_small_file_keeps_a_large_file():
    # a sparse file reports its logical size, so a threshold-crossing file costs no disk
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'large.safetensors')
        with open(path, 'wb') as f:
            f.truncate(3 * 1024**3)
        with gate_at(2.0):
            assert model_quant.skip_small_file(path, module='Model', cls_name='Test') is False


def test_skip_small_file_quantizes_when_file_is_missing():
    with gate_at(2.0):
        assert model_quant.skip_small_file('/nonexistent/override.safetensors', module='Model', cls_name='Test') is False


def test_skip_small_file_inactive_in_post_mode():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'small.safetensors')
        with open(path, 'wb') as f:
            f.write(b'0' * 1024)
        with gate_at(2.0), opts_of(sdnq_quantize_mode='post'):
            assert model_quant.skip_small_file(path, module='Model', cls_name='Test') is False


# ============================================================
# get_dit_args wiring
# ============================================================

def test_get_dit_args_allow_sdnq_gates_only_sdnq():
    # the gate suppresses the sdnq config through allow_sdnq rather than allow_quant, so other
    # quantization backends keep their own decision
    with gate_at(2.0):
        _, args_on = model_quant.get_dit_args({}, module='TE', allow_quant=True, allow_sdnq=True)
        _, args_off = model_quant.get_dit_args({}, module='TE', allow_quant=True, allow_sdnq=False)
    assert args_on.get('quantization_config', None) is not None, 'sdnq config must be created while allowed'
    assert 'quantization_config' not in args_off, 'allow_sdnq=False must strip the sdnq config'


# ============================================================
# Runner
# ============================================================

def run_all():
    cat = category('threshold')
    for fn in [
        test_min_size_skip_disabled_at_zero,
        test_min_size_skip_disabled_when_negative,
        test_min_size_skip_below_threshold_skips,
        test_min_size_skip_above_threshold_quantizes,
        test_min_size_skip_at_threshold_quantizes,
        test_min_size_skip_unmeasured_size_quantizes,
    ]:
        run_test(cat, fn)

    cat = category('size estimate')
    for fn in [
        test_estimate_size_measures_small_encoder,
        test_estimate_size_measures_large_encoder,
        test_estimate_size_separates_two_encoders_of_one_class,
        test_estimate_size_counts_the_load_dtype_not_the_config_dtype,
        test_estimate_size_reads_a_config_without_a_model_type,
        test_estimate_size_reads_a_config_with_an_unknown_model_type,
        test_estimate_size_refuses_missing_class,
        test_estimate_size_refuses_missing_repo,
        test_estimate_size_refuses_none_repo_string,
        test_estimate_size_survives_unreadable_config,
        test_estimate_size_ties_embeddings_before_counting,
        test_estimate_size_uses_a_provided_config_object,
        test_estimate_size_resolves_an_auto_class,
    ]:
        run_test(cat, fn)

    cat = category('loader gate')
    for fn in [
        test_skip_small_module_does_nothing_while_disabled,
        test_skip_small_module_skips_a_small_component,
        test_skip_small_module_keeps_a_large_component,
        test_skip_small_module_quantizes_when_size_is_unknown,
        test_skip_small_module_keeps_everything_at_a_low_threshold,
        test_skip_small_module_inactive_in_post_mode,
        test_skip_small_module_inactive_when_bucket_unselected,
        test_skip_small_module_inactive_when_dtype_none,
        test_skip_small_module_inactive_for_excluded_model_family,
        test_skip_small_module_leaves_prequantized_repos_alone,
    ]:
        run_test(cat, fn)

    cat = category('file gate')
    for fn in [
        test_skip_small_file_skips_a_small_file,
        test_skip_small_file_keeps_a_large_file,
        test_skip_small_file_quantizes_when_file_is_missing,
        test_skip_small_file_inactive_in_post_mode,
    ]:
        run_test(cat, fn)

    cat = category('quant args wiring')
    for fn in [
        test_get_dit_args_allow_sdnq_gates_only_sdnq,
    ]:
        run_test(cat, fn)

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
    ok = run_all()
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if ok else 1)
