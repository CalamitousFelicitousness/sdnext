"""Running audio generation.

The core is keyword-only and returns a result object; the gradio adapter is a thin wrapper around
it so the UI, the API and any in-process caller take the same path.

Audio does not go through ``process_images``. Its pipelines return ``AudioPipelineOutput``, which
carries ``audios`` and neither ``images`` nor ``frames``, so the shared decode step reads nothing
from it and hands back an empty result. Calling the pipeline here and building ``Processed``
directly also keeps this path out of the two files upstream touches most.
"""

import time
from dataclasses import dataclass, field
import torch
from modules import errors, processing, processing_class, processing_helpers, shared, timer
from modules.audio import loudness
from modules.audio import metadata as audio_metadata
from modules.audio import save as audio_save
from modules.audio_models import audio_load, models_def
from modules.audio_models.audio_error import AudioError
from modules.logger import log


@dataclass
class AudioResult:
    path: str | None = None
    waveform: object = None
    sample_rate: int = 48000
    duration: float = 0.0
    levels: dict = field(default_factory=dict)
    info: str = ''
    processed: object = None


def ignored_scheduler_opts() -> list[str]:
    """Image-side sampler settings that must not reach an audio pipeline.

    Sigma adjust rescales the schedule in place, which silently retunes a distilled model's eight
    steps, and a custom timestep string is accepted by any pipeline whose call signature takes one.
    Both are reported rather than dropped in silence.
    """
    ignored = []
    if getattr(shared.opts, 'schedulers_sigma_adjust', 1.0) != 1.0:
        ignored.append('schedulers_sigma_adjust')
    timesteps = getattr(shared.opts, 'schedulers_timesteps', '') or ''
    if len([t for t in timesteps.split(',') if t.strip()]) > 2:
        ignored.append('schedulers_timesteps')
    return ignored


def make_callback():
    """Step ticks and interrupt checks, and nothing else.

    The shared image callback stores latents into ``state.current_latent``, where the preview path
    renders them as an image grid. Audio latents are not an image, so this one never sets it.
    """
    def callback(pipe, _step_index, _timestep, kwargs):
        shared.state.step()
        if shared.state.interrupted or shared.state.skipped:
            pipe._interrupt = True # pylint: disable=protected-access
        return kwargs
    return callback


def resolve_model(engine: str | None, model: str | None) -> models_def.Model:
    row = models_def.find(model, engine) if model else models_def.default_model()
    if row is None:
        raise AudioError(f'audio: model="{model}" not found in engine="{engine}"', 404)
    return row


def run(
    *,
    engine: str | None = None,
    model: str | None = None,
    prompt: str = '',
    lyrics: str = '',
    negative_prompt: str = '',
    duration: float | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int = -1,
    task: str = 'text2music',
    reference_audio=None,
    audio_format: str | None = None,
    save: bool = True,
    n_iter: int = 1,
    override_settings: dict | None = None,
    **task_args,
) -> list[AudioResult]:
    """Generate one or more takes from a single model load. Raises AudioError on refusal."""
    row = resolve_model(engine, model)
    if task not in row.tasks:
        raise AudioError(f'audio: task="{task}" unsupported by model="{row.name}", supported: {list(row.tasks)}', 400)
    duration = float(duration if duration is not None else row.duration)
    if duration < row.duration_min or duration > row.duration_max:
        raise AudioError(f'audio: duration={duration} outside the model range {row.duration_min} to {row.duration_max}', 400)
    steps = int(steps if steps is not None else row.steps)
    cfg_scale = float(cfg_scale if cfg_scale is not None else row.cfg)
    if row.distilled and cfg_scale != row.cfg:
        log.info(f'Audio: model="{row.name}" is guidance distilled and ignores cfg={cfg_scale}')
    if negative_prompt and not row.negative:
        log.info(f'Audio: model="{row.name}" takes no negative prompt, ignoring it')
    ignored = ignored_scheduler_opts()
    if ignored:
        log.info(f'Audio: sampler settings not applied to audio: {", ".join(ignored)}')

    audio_load.load_model(row)
    if shared.sd_model is None:
        raise AudioError(f'audio: model="{row.name}" not loaded', 500)

    p = processing_class.StableDiffusionProcessingAudio(
        prompt=prompt,
        negative_prompt=negative_prompt if row.negative else '',
        lyrics=lyrics,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        audio_engine=models_def.engine_of(row),
        audio_model=row.name,
        audio_task=task,
        audio_format=audio_format or audio_save.DEFAULT_FORMAT,
        audio_sampling_rate=row.sample_rate,
        sampler_name='Default', # the pipeline keeps its own scheduler; Default is what the video path records for the same case
        override_settings=override_settings or {},
    )
    processing.fix_seed(p)

    results = []
    try:
        for iteration in range(max(1, int(n_iter))):
            results.append(generate_one(p, row, iteration, task, reference_audio, save, task_args))
    finally:
        p.close()
    return results


def infotext_params(p, row: models_def.Model, task: str, rate: int, levels: dict) -> dict:
    """The audio fields the infotext carries, built without touching the pipeline so it stays checkable.

    Lyrics are written raw: create_infotext quotes any value holding a comma, colon or newline, so
    the section structure survives a round trip and the infotext still occupies one line.
    """
    params = {
        'Audio model': row.name,
        'Audio task': task,
        'Duration': round(p.duration, 2),
        'Sample rate': rate,
    }
    if p.lyrics:
        params['Lyrics'] = p.lyrics
    if (levels or {}).get('lufs') is not None:
        params['Loudness'] = levels['lufs']
    return params


def generate_one(p, row: models_def.Model, iteration: int, task: str, reference_audio, save: bool, task_args: dict) -> AudioResult:
    seed = int(p.seed) + iteration
    p.seeds = [seed]
    shared.state.sampling_step = 0
    shared.state.sampling_steps = p.steps
    shared.state.textinfo = 'Generate'
    args = {
        'prompt': p.prompt,
        'audio_duration': p.duration,
        'num_inference_steps': p.steps,
        'guidance_scale': p.cfg_scale,
        'task_type': task,
        'output_type': 'pt',
        'callback_on_step_end': make_callback(),
    }
    generator = processing_helpers.get_generator(p) # honors the generator device setting, and is None when it is Unset
    if generator is not None:
        args['generator'] = generator
    if row.lyrics:
        args['lyrics'] = p.lyrics or ''
    if row.negative and p.negative_prompt:
        args['negative_prompt'] = p.negative_prompt
    if reference_audio is not None:
        args['reference_audio'] = reference_audio
    args.update(task_args or {})

    t0 = time.time()
    try:
        output = shared.sd_model(**args)
    except AssertionError as e: # the shared interrupt path raises this out of a callback
        raise AudioError('audio: interrupted', 499) from e
    except Exception as e:
        log.error(f'Audio generate: model="{row.name}" {e}')
        errors.display(e, 'audio')
        raise AudioError(f'audio: generation failed: {e}', 500) from e
    if shared.state.interrupted or shared.state.skipped:
        raise AudioError('audio: interrupted', 499)
    timer.process.add('generate', time.time() - t0)

    waveform = getattr(output, 'audios', None)
    if waveform is None:
        raise AudioError('audio: pipeline returned no audio', 500)
    if torch.is_tensor(waveform) and waveform.ndim == 3:
        waveform = waveform[0]

    shared.state.textinfo = 'Save'
    rate = int(getattr(output, 'sampling_rate', None) or row.sample_rate)
    levels = loudness.measure(waveform, rate)
    p.seed = seed
    p.extra_generation_params.update(infotext_params(p, row, task, rate, levels))
    info = processing.create_infotext(p)

    path = None
    if save:
        path = audio_save.save_audio(p=p, waveform=waveform, sample_rate=rate, ext=p.audio_format, levels=levels)
    processed = processing.Processed(p, [], audio=waveform, info=info)
    return AudioResult(
        path=path,
        waveform=waveform,
        sample_rate=rate,
        duration=round(levels.get('samples', 0) / rate, 3) if rate else 0.0,
        levels=levels,
        info=info,
        processed=processed,
    )


def read_info(path: str) -> str | None:
    """Embedded parameters of a previously generated file, for paste-back."""
    return audio_metadata.read_audio_info(path)
