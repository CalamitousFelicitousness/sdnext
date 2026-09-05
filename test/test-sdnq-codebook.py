"""SDNQ codebook dtypes: the sdnext-side glue around the fork's cb2-cb6 family.

The dtype family itself is covered by the sdnq package's own suite
(extensions-builtin/sdnq/tests/test_codebook.py). This file covers what sdnext
adds on top:

- Ingest: comfy asym_w4a8_int8 layers adopted as cb4 (nibble order, folded
  grouped scales, verbatim fp32 codebook, convrot, sidecar validation) and the
  format gate that hides the mapping when the sdnq build has no cb4 dtype.
- LoRA routing: the grid step of a codebook layer is the scale times the mean
  adjacent gap of the int8 book.
- UI: every codebook dtype sits in the quant mode dropdown right after its
  affine counterpart.

All tensors are synthetic; no model files or running server required.

Usage:
    python test/test-sdnq-codebook.py
"""

import os
import sys
import time

import torch

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

os.environ['SD_INSTALL_QUIET'] = '1'

# Bootstrap cmd_args before any module that pulls in shared.py.
import modules.cmd_args  # pylint: disable=wrong-import-position
import installer  # pylint: disable=wrong-import-position
_orig_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    modules.cmd_args.parse_args()
finally:
    sys.argv = _orig_argv
installer.add_args(modules.cmd_args.parser)
modules.cmd_args.parsed, _ = modules.cmd_args.parser.parse_known_args([])

from modules.errors import log  # pylint: disable=wrong-import-position
from modules.lora import lora_sdnq  # pylint: disable=wrong-import-position
from modules.shared_items import sdnq_quant_modes  # pylint: disable=wrong-import-position
from pipelines.native_transformer import COMFY_QUANT_FORMATS, adopt_asym_w4a8_layer, detect_comfy_quant, OverrideArchMismatch  # pylint: disable=wrong-import-position
from sdnq.common import dtype_dict  # pylint: disable=wrong-import-position
from sdnq.dequantizer import dequantize_weight  # pylint: disable=wrong-import-position
from sdnq.packed_int import unpack_int  # pylint: disable=wrong-import-position
from sdnq.quant_utils import get_hadamard, rotate_hadamard  # pylint: disable=wrong-import-position
from sdnq.quantizer import sdnq_quantize_layer, SDNQConfig  # pylint: disable=wrong-import-position

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_F, IN_F = 256, 512
CB_DTYPES = ('cb2', 'cb3', 'cb4', 'cb5', 'cb6')

results: dict[str, dict] = {}


def category(name: str):
    if name not in results:
        results[name] = {'passed': 0, 'failed': 0}
    return name


def run_test(cat: str, fn):
    try:
        fn()
        results[cat]['passed'] += 1
        log.info(f'  PASS: {fn.__name__}')
    except Exception as e:  # pylint: disable=broad-except
        results[cat]['failed'] += 1
        log.error(f'  FAIL: {fn.__name__} ({type(e).__name__}: {e})')


def build_layer(weights_dtype='cb4', seed=0):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(IN_F, OUT_F, bias=False, dtype=torch.bfloat16, device=DEVICE)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(OUT_F, IN_F, device=DEVICE) * 0.04)
    cfg = SDNQConfig(weights_dtype=weights_dtype, group_size=0, use_hadamard=True, dequantize_fp32=False, quantization_device=str(DEVICE), return_device=str(DEVICE))
    layer, _ = sdnq_quantize_layer(lin, cfg, torch_dtype=torch.bfloat16, param_name='test.weight')
    return layer


# --- ingest ---

def pack_kijai(idx: torch.Tensor) -> torch.Tensor:
    # the container packs the even flat index into the HIGH nibble (hi_first)
    pairs = idx.reshape(idx.shape[0], -1, 2)
    return (pairs[..., 0] << 4 | pairs[..., 1]).to(torch.uint8).view(torch.int8)


def make_w4a8_sidecars(out_f: int, in_f: int, group_size: int, seed: int):
    g = torch.Generator(device='cpu').manual_seed(seed)
    idx = torch.randint(0, 16, (out_f, in_f), generator=g, dtype=torch.uint8)
    codebook = torch.sort(torch.randn(16, generator=g)).values
    s_channel = torch.rand(out_f, generator=g) * 0.05 + 0.01
    s_rel = (torch.rand(out_f, in_f // group_size, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)
    return idx, codebook, s_channel, s_rel


def test_ingest_adopt_matches_container_dequant():
    out_f, in_f, group_size = 32, 64, 16
    idx, codebook, s_channel, s_rel = make_w4a8_sidecars(out_f, in_f, group_size, seed=16)
    name = 'blocks.0.mlp.fc1'
    sd = {
        f'{name}.weight': pack_kijai(idx),
        f'{name}.weight_codebook': codebook.clone(),
        f'{name}.weight_s_channel': s_channel.clone(),
        f'{name}.weight_s_rel': s_rel.clone(),
    }
    lin = torch.nn.Linear(in_f, out_f, bias=False, device='meta')
    adopt_asym_w4a8_layer(sd, name, lin, 'test', {'format': 'asym_w4a8_int8', 'group_size': group_size})
    assert f'{name}.weight_codebook' not in sd and f'{name}.weight_s_rel' not in sd and f'{name}.weight_s_channel' not in sd, 'sidecars must be consumed'
    assert sd[f'{name}.codebook'].dtype == torch.float32, 'ingested codebook must stay fp32'
    assert torch.equal(sd[f'{name}.codebook'], codebook.float()), 'ingested codebook must be adopted verbatim'
    # container dequant reference: codebook[idx] * s_channel * s_rel per group of 16
    scales_full = (s_rel.float() * s_channel.reshape(-1, 1)).repeat_interleave(group_size, dim=1)
    ref = codebook.float()[idx.int()] * scales_full
    groups = in_f // group_size
    out = dequantize_weight(
        'cb4', sd[f'{name}.weight'], sd[f'{name}.scale'], codebook=sd[f'{name}.codebook'],
        quantized_weight_shape=torch.Size((out_f, groups, group_size)), result_shape=torch.Size((out_f, in_f)), dtype=torch.float32,
    )
    assert torch.equal(out, ref), f'ingested dequant differs from container reference by {float((out - ref).abs().max()):.3e}'


def test_ingest_nibble_order_roundtrip():
    idx = torch.arange(256, dtype=torch.uint8).reshape(16, 16) % 16
    name = 'l'
    sd = {
        f'{name}.weight': pack_kijai(idx),
        f'{name}.weight_codebook': torch.linspace(-1.0, 1.0, 16),
        f'{name}.weight_s_channel': torch.ones(16),
        f'{name}.weight_s_rel': torch.ones(16, 1).to(torch.float8_e4m3fn),
    }
    lin = torch.nn.Linear(16, 16, bias=False, device='meta')
    adopt_asym_w4a8_layer(sd, name, lin, 'test', {'format': 'asym_w4a8_int8', 'group_size': 16})
    unpacked = unpack_int(sd[f'{name}.weight'], 'cb4', torch.Size((16, 1, 16))).reshape(16, 16)
    assert torch.equal(unpacked.to(torch.uint8), idx), 'nibble swap does not recover the original index order'


def test_ingest_convrot_roundtrip():
    # convrot weights are stored rotated; the dequantizer re-applies the same regular
    # Hadamard, so rotating the reference identically must reproduce it bit-exact
    out_f, in_f, group_size = 16, 256, 16
    idx, codebook, s_channel, s_rel = make_w4a8_sidecars(out_f, in_f, group_size, seed=17)
    name = 'l'
    sd = {
        f'{name}.weight': pack_kijai(idx),
        f'{name}.weight_codebook': codebook.clone(),
        f'{name}.weight_s_channel': s_channel.clone(),
        f'{name}.weight_s_rel': s_rel.clone(),
    }
    lin = torch.nn.Linear(in_f, out_f, bias=False, device='meta')
    adopt_asym_w4a8_layer(sd, name, lin, 'test', {'format': 'asym_w4a8_int8', 'group_size': group_size, 'convrot': True, 'convrot_groupsize': 256})
    scales_full = (s_rel.float() * s_channel.reshape(-1, 1)).repeat_interleave(group_size, dim=1)
    stored = codebook.float()[idx.int()] * scales_full
    hadamard = get_hadamard(256, dtype=torch.float32, device=torch.device('cpu'))
    ref = rotate_hadamard(stored, hadamard=hadamard)
    out = dequantize_weight(
        'cb4', sd[f'{name}.weight'], sd[f'{name}.scale'], codebook=sd[f'{name}.codebook'], hadamard=hadamard,
        quantized_weight_shape=torch.Size((out_f, in_f // group_size, group_size)), result_shape=torch.Size((out_f, in_f)), dtype=torch.float32,
    )
    assert torch.equal(out, ref), 'convrot dequant does not match the rotated reference'


def test_ingest_rejects_missing_sidecars():
    name = 'l'
    lin = torch.nn.Linear(64, 32, bias=False, device='meta')
    sd = {f'{name}.weight': torch.zeros(32, 32, dtype=torch.int8)}
    try:
        adopt_asym_w4a8_layer(sd, name, lin, 'test', {'format': 'asym_w4a8_int8', 'group_size': 16})
    except OverrideArchMismatch as e:
        assert 'missing' in str(e), f'unexpected message: {e}'
        return
    raise AssertionError('expected OverrideArchMismatch for missing sidecars')


def test_detect_gates_formats_on_the_sdnq_build():
    # the marker-detection step only admits formats whose target dtype the loaded sdnq carries
    assert COMFY_QUANT_FORMATS['asym_w4a8_int8'] == 'cb4', 'w4a8 must map onto cb4'
    marker = torch.frombuffer(bytearray(b'{"format": "asym_w4a8_int8", "group_size": 16}'), dtype=torch.uint8)
    sd = {'l.weight': torch.zeros(16, 8, dtype=torch.int8), 'l.comfy_quant': marker}
    marked, fmt = detect_comfy_quant(sd, 'test')
    assert fmt == 'asym_w4a8_int8' and 'l' in marked, f'detection returned {fmt} {sorted(marked)}'
    saved = dtype_dict.pop('cb4')
    try:
        try:
            detect_comfy_quant(sd, 'test')
        except OverrideArchMismatch as e:
            assert 'not supported' in str(e), f'unexpected message: {e}'
        else:
            raise AssertionError('a build without cb4 must report w4a8 as unsupported')
    finally:
        dtype_dict['cb4'] = saved


# --- lora + ui ---

def test_grid_step_scales_with_level_gap():
    cb_layer = build_layer('cb4')
    int8_layer = build_layer('int8')
    assert abs(lora_sdnq.grid_step(int8_layer) - float(int8_layer.scale.detach().float().mean())) < 1e-12, 'int8 step must equal the mean scale'
    gaps = cb_layer.codebook.detach().float().sort().values.diff().mean()
    expected = float(cb_layer.scale.detach().float().mean()) * float(gaps)
    assert abs(lora_sdnq.grid_step(cb_layer) - expected) < 1e-9, 'cb step must be the scale times the mean level gap'
    assert lora_sdnq.grid_step(cb_layer) > float(cb_layer.scale.detach().float().mean()), 'cb step must exceed the raw scale unit'


def test_dropdown_lists_codebook_dtypes_after_affine():
    for wdt in CB_DTYPES:
        assert wdt in sdnq_quant_modes, f'{wdt} missing from sdnq_quant_modes'
        assert sdnq_quant_modes.index(wdt) == sdnq_quant_modes.index(f'uint{wdt[2:]}') + 1, f'{wdt} must directly follow uint{wdt[2:]}'


def run_all() -> bool:
    started = time.time()
    suites = [
        ('ingest', [test_ingest_adopt_matches_container_dequant, test_ingest_nibble_order_roundtrip, test_ingest_convrot_roundtrip, test_ingest_rejects_missing_sidecars, test_detect_gates_formats_on_the_sdnq_build]),
        ('lora-ui', [test_grid_step_scales_with_level_gap, test_dropdown_lists_codebook_dtypes_after_affine]),
    ]
    with torch.inference_mode():
        for cat, tests in suites:
            log.warning(f'=== {cat} ===')
            category(cat)
            for fn in tests:
                run_test(cat, fn)
    total_passed = sum(d['passed'] for d in results.values())
    total_failed = sum(d['failed'] for d in results.values())
    log.warning(f'Total: {total_passed} passed, {total_failed} failed in {time.time() - started:.1f}s')
    return total_failed == 0


if __name__ == '__main__':
    sys.exit(0 if run_all() else 1)
