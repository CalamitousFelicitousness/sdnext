#!/usr/bin/env python
"""
Offline tests for prompt enhance modality selection.

Which instructions the enhancer receives used to be a single boolean, ``module == 'video'``, and
``module`` came from the tab the script instance was built on. Two consequences this file pins:

- the api advertised a video enhancement mode that never reached the template, because the request
  always ran against the control tab's instance, so ``type='video'`` returned image instructions
- a third output type could not be expressed at all without a second boolean

Selection is now a lookup keyed on the module, and the request may name it. The composition tests
compare against the previous logic rebuilt inline, so a refactor that changes image or video output
fails here rather than being noticed in generated prompts.

No running server required, no model, no network.

Usage:
    python test/test-prompt-enhance.py
"""

import os
import sys

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
from scripts.prompt_enhance import template                   # pylint: disable=wrong-import-position
from scripts.prompt_enhance.options import Options            # pylint: disable=wrong-import-position


results: dict[str, dict] = {}
opts = Options()
TAB_MODULES = ('txt2img', 'img2img', 'control', 'video')


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


def previous_system(options, is_video, nsfw, has_prompt, has_image):
    """The composition before the modality lookup, kept as the reference for image and video."""
    if not has_image:
        system = options.t2v_prompt if is_video else options.t2i_prompt
    elif is_video:
        system = options.i2v_prompt if has_prompt else options.i2v_noprompt
    else:
        system = options.i2i_prompt if has_prompt else options.i2i_noprompt
    system += options.nsfw_ok if nsfw else options.nsfw_no
    system += options.details_prompt
    system += options.details_format
    return system


def shapes():
    for nsfw in (True, False):
        for has_prompt in (True, False):
            for has_image in (True, False):
                yield nsfw, has_prompt, has_image


# ============================================================
# Modality selection
# ============================================================

def test_video_selects_the_video_instructions():
    assert template.modality_of('video').text == 't2v_prompt'


def test_audio_selects_the_music_instructions():
    modality = template.modality_of('audio')
    assert modality.text == 't2a_prompt', modality.text
    assert modality.details == 'details_audio', modality.details


def test_unknown_and_absent_modules_fall_back_to_images():
    for module in (None, '', 'txt2img', 'img2img', 'control', 'nonsense'):
        assert template.modality_of(module) is template.IMAGE, f'{module} did not fall back'


# ============================================================
# Composition
# ============================================================

def test_image_and_video_composition_is_unchanged():
    for module in TAB_MODULES:
        is_video = module == 'video'
        for nsfw, has_prompt, has_image in shapes():
            got = template.build_system(opts, template.modality_of(module), nsfw, has_prompt, has_image)
            want = previous_system(opts, is_video, nsfw, has_prompt, has_image)
            assert got == want, f'changed for module={module} nsfw={nsfw} prompt={has_prompt} image={has_image}'


def test_music_instructions_carry_no_image_guidance():
    system = template.build_system(opts, template.modality_of('audio'), nsfw=False, has_prompt=True, has_image=False)
    assert 'music prompt engineer' in system, 'music instruction missing'
    assert opts.details_prompt not in system, 'image detail guidance leaked into the music instructions'
    assert opts.details_audio in system, 'music detail guidance missing'


def test_shared_blocks_still_apply_to_music():
    modality = template.modality_of('audio')
    allowed = template.build_system(opts, modality, nsfw=True, has_prompt=True, has_image=False)
    refused = template.build_system(opts, modality, nsfw=False, has_prompt=True, has_image=False)
    assert opts.nsfw_ok in allowed and opts.nsfw_no in refused, 'nsfw block not applied'
    assert opts.details_format in allowed, 'output format block not applied'


def test_music_has_no_separate_image_variant():
    """Recorded rather than assumed: an image sent with a music request keeps the music instruction."""
    modality = template.modality_of('audio')
    assert modality.image == modality.text and modality.image_only == modality.text


# ============================================================
# Request type reaching the template
# ============================================================

def test_request_type_selects_the_instructions():
    """The api bug: type named the output but only picked a default model, so video never reached here."""
    expected = {'video': 't2v_prompt', 'audio': 't2a_prompt', 'text': 't2i_prompt', 'image': 't2i_prompt'}
    for req_type, field in expected.items():
        assert template.modality_of(req_type).text == field, f'type={req_type} selected {template.modality_of(req_type).text}'


def test_set_template_builds_a_music_system_prompt():
    chat = template.set_template(system=None, prompt='sad piano song', image=None, options=opts,
                                 model='google/gemma-3-1b-it', nsfw=False, has_processor=False, module='audio')
    assert [m['role'] for m in chat] == ['system', 'user'], chat
    assert 'music prompt engineer' in chat[0]['content'], 'music instruction missing from the system turn'
    assert chat[1]['content'] == 'sad piano song', chat[1]['content']


def test_an_explicit_system_prompt_still_wins():
    given = 'you are a very specific thing indeed'
    chat = template.set_template(system=given, prompt='sad piano song', image=None, options=opts,
                                 model='google/gemma-3-1b-it', nsfw=False, has_processor=False, module='audio')
    assert chat[0]['content'] == given, chat[0]['content']


# ============================================================
# Runner
# ============================================================

def run_all():
    selection = category('modality selection')
    for fn in [
        test_video_selects_the_video_instructions,
        test_audio_selects_the_music_instructions,
        test_unknown_and_absent_modules_fall_back_to_images,
    ]:
        run_test(selection, fn)

    composition = category('composition')
    for fn in [
        test_image_and_video_composition_is_unchanged,
        test_music_instructions_carry_no_image_guidance,
        test_shared_blocks_still_apply_to_music,
        test_music_has_no_separate_image_variant,
    ]:
        run_test(composition, fn)

    routing = category('request routing')
    for fn in [
        test_request_type_selects_the_instructions,
        test_set_template_builds_a_music_system_prompt,
        test_an_explicit_system_prompt_still_wins,
    ]:
        run_test(routing, fn)

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
