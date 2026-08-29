import html
import os
from urllib.parse import quote
import gradio as gr
from modules import call_queue, infotext
from modules.audio import save as audio_save
from modules.audio_models import audio_run, models_def
from modules.audio_models.audio_error import AudioError
from modules.logger import log


def download_links(results) -> str:
    """Download anchors pointing at the audio file route.

    The player's own download button names the file after gradio's temp copy, whose path is
    flattened into the filename, so saving a take yields _tmp_gradio_<hash>_<name>. The file route
    answers with a content disposition carrying the real name.
    """
    rows = []
    for index, result in enumerate(results or [], start=1):
        if not result.path:
            continue
        name = html.escape(os.path.basename(result.path), quote=True)
        url = f'/sdapi/v1/audio/file?file={quote(result.path)}'
        rows.append(f'<a class="audio-download" href="{url}" download="{name}">Take {index}: {name}</a>')
    return '<div class="audio-downloads">' + '<br>'.join(rows) + '</div>' if rows else ''


def model_choices(engine: str) -> list[str]:
    return models_def.model_names(engine)


RESTORE_OUTPUTS = 11


def enhance_tags(prompt_text: str):
    """Rewrite the style prompt through the shared enhancer, asking it for music instructions.

    The enhancer is a script bound to a tab it was built on, and audio runs no script runner, so the
    module is passed explicitly rather than inherited.
    """
    if not (prompt_text or '').strip():
        return gr.update()
    try:
        from modules.scripts_manager import scripts_control
        loaded = getattr(scripts_control, 'scripts', None) or [] # the runner is absent until the server builds its ui
        instance = next((s for s in loaded if 'prompt_enhance_ext.py' in s.filename), None)
        if instance is None:
            log.warning('Audio enhance: prompt enhance script is not loaded')
            return gr.update()
        enhanced = instance.enhance(prompt=prompt_text, module='audio')
    except Exception as e:
        log.error(f'Audio enhance: {e}')
        return gr.update()
    if not enhanced or not str(enhanced).strip():
        log.warning('Audio enhance: enhancer returned nothing, prompt left as it was')
        return gr.update()
    return gr.update(value=str(enhanced).strip())


def as_number(params: dict, key: str, cast):
    value = params.get(key, None)
    if value is None:
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def restore_params(info_text: str):
    """Repopulate the tab from a take's parameters.

    Not wired through the shared copypaste plumbing: that path coerces each value with
    type(component.value), so it cannot repopulate the model and task dropdowns whose choices follow
    the selected model, and it calls a per-tab javascript token recount this tab has no counter for.
    """
    blank = [gr.update() for _ in range(RESTORE_OUTPUTS)]
    params = infotext.parse(info_text or '')
    if not params:
        return blank
    row = models_def.find(str(params.get('Audio model') or ''))
    engine_name = models_def.engine_of(row) if row is not None else None
    task_name = str(params.get('Audio task') or '') or None
    duration = as_number(params, 'Duration', float)
    if row is not None and duration is not None:
        duration = min(max(duration, row.duration_min), row.duration_max)

    def text(key):
        value = params.get(key, None)
        return gr.update(value=str(value)) if value is not None else gr.update(value='')

    if row is None: # an unknown model leaves the selectors alone rather than clearing them
        log.debug(f'Audio restore: model="{params.get("Audio model")}" not in the registry, selectors left as they are')
        engine_update, model_update, task_update, info_update = gr.update(), gr.update(), gr.update(), gr.update()
        duration_update = gr.update(value=duration) if duration is not None else gr.update()
    else:
        engine_update = gr.update(value=engine_name)
        model_update = gr.update(choices=model_choices(engine_name), value=row.name)
        task_update = gr.update(choices=list(row.tasks), value=task_name if task_name in row.tasks else row.tasks[0])
        info_update = describe(engine_name, row.name)
        duration_update = gr.update(minimum=row.duration_min, maximum=row.duration_max, value=duration if duration is not None else row.duration)

    seed = as_number(params, 'Seed', int)
    steps = as_number(params, 'Steps', int)
    cfg = as_number(params, 'CFG scale', float)
    return [
        engine_update,
        model_update,
        info_update,
        text('Prompt'),
        text('Lyrics'),
        text('Negative prompt'),
        duration_update,
        gr.update(value=seed) if seed is not None else gr.update(),
        task_update,
        gr.update(value=steps) if steps is not None else gr.update(),
        gr.update(value=cfg) if cfg is not None else gr.update(),
    ]


def describe(engine: str, name: str) -> str:
    row = models_def.find(name, engine)
    if row is None:
        return ''
    bits = [f'{row.steps} steps', f'{row.sample_rate // 1000} kHz', f'up to {int(row.duration_max)}s']
    if row.distilled:
        bits.append('guidance distilled, cfg ignored')
    if not row.negative:
        bits.append('no negative prompt')
    bits.append(f'tasks: {", ".join(row.tasks)}')
    return ' | '.join(bits)


def task_hint(engine: str, name: str) -> str:
    """What each of the model's tasks needs, so the requirement is readable without a failed run."""
    row = models_def.find(name, engine)
    if row is None:
        return ''
    labels = {'source': 'source track', 'span': 'time range', 'strength': 'cover strength', 'track': 'track name', 'classes': 'track classes'}
    lines = []
    for task in row.tasks:
        spec = models_def.task_of(row, task)
        needs = [labels[i] for i in spec.inputs()] if spec else []
        lines.append(f'<b>{task}</b>: {", ".join(needs) if needs else "prompt only"}')
    return '<span>' + ' &nbsp;|&nbsp; '.join(lines) + '</span>'


def generate(engine, model, prompt, lyrics, negative, duration, seed, takes, task, steps, cfg, fmt,
             source, repaint_start, repaint_end, cover_strength, track_name, track_classes):
    """Gradio adapter. The core is keyword-only; this binds the tab's controls to it.

    The task controls are always visible, so every run would otherwise hand the core values the
    selected task does not take and collect a log line refusing each one. The registry answers what
    the task consumes, and only that is passed; a request arriving over the api is still checked
    there, since it can carry anything.
    """
    spec = models_def.task_of(models_def.find(model, engine), task) if models_def.find(model, engine) else None
    takes_input = spec.inputs() if spec else ()
    try:
        results = audio_run.run(
            engine=engine,
            model=model,
            prompt=prompt,
            lyrics=lyrics,
            negative_prompt=negative,
            duration=duration,
            steps=int(steps) if steps else None,
            cfg_scale=float(cfg) if cfg else None,
            seed=int(seed),
            task=task,
            source_audio=(source or None) if 'source' in takes_input else None,
            repaint_start=float(repaint_start) if 'span' in takes_input else None,
            repaint_end=float(repaint_end) if 'span' in takes_input else None,
            cover_strength=float(cover_strength) if 'strength' in takes_input else None,
            track_name=(track_name or '').strip() if 'track' in takes_input else '',
            track_classes=[t.strip() for t in (track_classes or '').replace(',', ' ').split() if t.strip()] if 'classes' in takes_input else None,
            n_iter=int(takes),
            audio_format=fmt,
        )
    except AudioError as e:
        log.error(f'Audio: {e.msg}')
        return None, None, f'Error: {e.msg}', ''
    first = results[0].path if results else None
    second = results[1].path if len(results) > 1 else None
    info = results[0].info if results else ''
    return first, second, info, download_links(results)


def create_ui():
    log.debug('UI initialize: tab=audio')
    engines = models_def.engines()
    default_engine = engines[0] if engines else ''
    default_models = model_choices(default_engine)
    default_model = default_models[0] if default_models else ''

    with gr.Blocks(analytics_enabled=False) as _audio_interface:
        with gr.Row(elem_id='audio_interface', equal_height=False):
            with gr.Column(scale=3):
                # both prompt boxes stay visible: every music model splits style from lyrics, and a
                # model without lyrics simply leaves the second box empty
                prompt = gr.Textbox(label='Tags', elem_id='audio_prompt', lines=2, placeholder='genre, instrumentation, mood, tempo')
                enhance_btn = gr.Button('Enhance tags', elem_id='audio_enhance', size='sm')
                lyrics = gr.Textbox(label='Lyrics', elem_id='audio_lyrics', lines=8, placeholder='[verse]\n...\n[chorus]\n...')
                negative = gr.Textbox(label='Negative prompt', elem_id='audio_negative', lines=1)
                with gr.Row():
                    engine = gr.Dropdown(label='Engine', elem_id='audio_engine', choices=engines, value=default_engine)
                    model = gr.Dropdown(label='Model', elem_id='audio_model', choices=default_models, value=default_model)
                model_info = gr.HTML(value=describe(default_engine, default_model), elem_id='audio_model_info')
                # duration leads: it is the one control that scales cost linearly
                duration = gr.Slider(label='Duration', elem_id='audio_duration', minimum=1, maximum=600, step=1, value=60)
                with gr.Row():
                    seed = gr.Number(label='Seed', elem_id='audio_seed', value=-1, precision=0)
                    takes = gr.Slider(label='Takes', elem_id='audio_takes', minimum=1, maximum=8, step=1, value=2)
                with gr.Accordion('Advanced', elem_id='audio_advanced', open=False):
                    with gr.Row():
                        task = gr.Dropdown(label='Task', elem_id='audio_task', choices=list(models_def.ACE_STEP_TASKS), value='text2music')
                        fmt = gr.Dropdown(label='Format', elem_id='audio_format', choices=list(audio_save.AUDIO_FORMATS), value=audio_save.DEFAULT_FORMAT)
                    with gr.Row():
                        steps = gr.Slider(label='Steps', elem_id='audio_steps', minimum=0, maximum=200, step=1, value=0)
                        cfg = gr.Slider(label='Guidance scale', elem_id='audio_cfg', minimum=0, maximum=30, step=0.1, value=0)
                    gr.HTML(value='<span>Steps and guidance at zero use the model defaults.</span>')
                # every task control stays visible and the model decides what applies, the same way
                # guidance stays visible on a distilled model
                with gr.Accordion('Source track', elem_id='audio_source_panel', open=False):
                    source = gr.Audio(label='Source track', elem_id='audio_source', type='filepath', source='upload')
                    task_info = gr.HTML(value=task_hint(default_engine, default_model), elem_id='audio_task_info')
                    with gr.Row():
                        repaint_start = gr.Number(label='Repaint from', elem_id='audio_repaint_start', value=0, precision=2)
                        repaint_end = gr.Number(label='Repaint to', elem_id='audio_repaint_end', value=-1, precision=2)
                    with gr.Row():
                        cover_strength = gr.Slider(label='Cover strength', elem_id='audio_cover_strength', minimum=0, maximum=1, step=0.05, value=1.0)
                        track_name = gr.Dropdown(label='Track', elem_id='audio_track', choices=list(models_def.ACE_STEP_TRACKS), value='vocals', allow_custom_value=True)
                    track_classes = gr.Textbox(label='Track classes', elem_id='audio_track_classes', lines=1, placeholder='drums bass')
                generate_btn = gr.Button('Generate', elem_id='audio_generate', variant='primary')

            with gr.Column(scale=2):
                # audio has no thumbnail, so comparing takes means hearing both: each stays loaded
                take_a = gr.Audio(label='Take 1', elem_id='audio_take_a', type='filepath', interactive=False, show_download_button=False)
                take_b = gr.Audio(label='Take 2', elem_id='audio_take_b', type='filepath', interactive=False, show_download_button=False)
                downloads = gr.HTML(value='', elem_id='audio_downloads')
                info = gr.Textbox(label='Parameters', elem_id='audio_info', lines=6, interactive=True)
                restore_btn = gr.Button('Restore parameters', elem_id='audio_restore')

        def on_engine(engine_name):
            choices = model_choices(engine_name)
            value = choices[0] if choices else ''
            return gr.update(choices=choices, value=value), describe(engine_name, value)

        def on_model(engine_name, model_name):
            row = models_def.find(model_name, engine_name)
            if row is None:
                return describe(engine_name, model_name), gr.update(), gr.update(), gr.update()
            return (
                describe(engine_name, model_name),
                gr.update(minimum=row.duration_min, maximum=row.duration_max, value=min(max(row.duration, row.duration_min), row.duration_max)),
                gr.update(choices=list(row.tasks), value=row.tasks[0] if row.tasks else 'text2music'),
                task_hint(engine_name, model_name),
            )

        engine.change(fn=on_engine, inputs=[engine], outputs=[model, model_info])
        model.change(fn=on_model, inputs=[engine, model], outputs=[model_info, duration, task, task_info])
        enhance_btn.click(fn=enhance_tags, inputs=[prompt], outputs=[prompt])
        restore_btn.click(
            fn=restore_params,
            inputs=[info],
            outputs=[engine, model, model_info, prompt, lyrics, negative, duration, seed, task, steps, cfg],
            show_progress='hidden',
        )
        generate_btn.click(
            fn=call_queue.wrap_gradio_gpu_call(generate, extra_outputs=[None, None, '', ''], name='Audio'),
            inputs=[engine, model, prompt, lyrics, negative, duration, seed, takes, task, steps, cfg, fmt,
                    source, repaint_start, repaint_end, cover_strength, track_name, track_classes],
            outputs=[take_a, take_b, info, downloads],
        )
