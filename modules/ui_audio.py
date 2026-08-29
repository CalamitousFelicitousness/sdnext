import html
import os
from urllib.parse import quote
import gradio as gr
from modules import call_queue
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


def generate(engine, model, prompt, lyrics, negative, duration, seed, takes, task, steps, cfg, fmt):
    """Gradio adapter. The core is keyword-only; this binds the tab's controls to it."""
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
                generate_btn = gr.Button('Generate', elem_id='audio_generate', variant='primary')

            with gr.Column(scale=2):
                # audio has no thumbnail, so comparing takes means hearing both: each stays loaded
                take_a = gr.Audio(label='Take 1', elem_id='audio_take_a', type='filepath', interactive=False, show_download_button=False)
                take_b = gr.Audio(label='Take 2', elem_id='audio_take_b', type='filepath', interactive=False, show_download_button=False)
                downloads = gr.HTML(value='', elem_id='audio_downloads')
                info = gr.Textbox(label='Parameters', elem_id='audio_info', lines=6, interactive=False)

        def on_engine(engine_name):
            choices = model_choices(engine_name)
            value = choices[0] if choices else ''
            return gr.update(choices=choices, value=value), describe(engine_name, value)

        def on_model(engine_name, model_name):
            row = models_def.find(model_name, engine_name)
            if row is None:
                return describe(engine_name, model_name), gr.update(), gr.update()
            return (
                describe(engine_name, model_name),
                gr.update(minimum=row.duration_min, maximum=row.duration_max, value=min(max(row.duration, row.duration_min), row.duration_max)),
                gr.update(choices=list(row.tasks), value=row.tasks[0] if row.tasks else 'text2music'),
            )

        engine.change(fn=on_engine, inputs=[engine], outputs=[model, model_info])
        model.change(fn=on_model, inputs=[engine, model], outputs=[model_info, duration, task])
        generate_btn.click(
            fn=call_queue.wrap_gradio_gpu_call(generate, extra_outputs=[None, None, '', ''], name='Audio'),
            inputs=[engine, model, prompt, lyrics, negative, duration, seed, takes, task, steps, cfg, fmt],
            outputs=[take_a, take_b, info, downloads],
        )
