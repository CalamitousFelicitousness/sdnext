"""Audio model registry.

A row declares how to load a model and how to run it. The run half is the part the video registry
lacks: without it, per-architecture behavior has to be recovered downstream from display-name
substrings, where a miss is indistinguishable from "not applicable" and yields a plausible wrong
result. Everything a caller needs to build a request is a declared field here.

Rows are frozen and hold class names as strings; the loader resolves them into local variables
rather than writing resolved classes back onto the row.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    name: str
    repo: str
    url: str = ''
    repo_cls: str = None # diffusers pipeline class name
    repo_revision: str = None
    dit: str = None
    dit_cls: str = None
    dit_folder: str = 'transformer'
    dit_revision: str = None
    te: str = None
    te_cls: str = None
    te_folder: str = 'text_encoder'
    te_revision: str = None
    vae_tiling: bool = True # AutoencoderOobleck exposes use_tiling as an attribute and no enable_tiling method
    base: bool = False

    # run spec
    steps: int = 50
    cfg: float = 7.0
    distilled: bool = False # guidance is baked in and the pipeline ignores cfg
    sample_rate: int = 48000
    duration: float = 60.0
    duration_min: float = 1.0
    duration_max: float = 600.0
    tasks: tuple = ('text2music',)
    task_spec: dict = None # name to Task; a second architecture brings its own table
    lyrics: bool = True
    lyrics_format: str = 'tagged' # tagged for [verse]/[chorus], lrc for timestamped lines
    negative: bool = False
    reference: bool = False # accepts reference audio for the cover task
    languages: tuple = ()

    def __str__(self):
        return f'name="{self.name}" repo="{self.repo}" cls="{self.repo_cls}" steps={self.steps} cfg={self.cfg} distilled={self.distilled} rate={self.sample_rate} tasks={list(self.tasks)}'


@dataclass(frozen=True)
class Task:
    """What a task consumes, so callers read the requirement instead of matching on the task name.

    A task that needs a source track cannot run without one, which is a refusal the caller should
    get before a model load rather than a failure inside the pipeline.
    """
    name: str
    source: bool = False # operates on a track the caller supplies
    span: bool = False # takes a start and end time within that track
    strength: bool = False # takes a blend strength between the source and the prompt
    track: bool = False # takes the name of one stem
    classes: bool = False # takes a list of stem names to add

    def inputs(self) -> tuple:
        fields = ('source', 'span', 'strength', 'track', 'classes')
        return tuple(f for f in fields if getattr(self, f))


# Requirements read from the pipeline rather than from the model cards, which document only
# text2music. cover is the one task gated on components: both of its conditioning paths require
# audio_tokenizer and audio_token_detokenizer and raise without them, and the turbo repo ships
# neither. The rest are driven by an instruction string and need only the source track, so they are
# mechanically available wherever the pipeline is.
ACE_STEP_TASK_SPEC = {
    'text2music': Task('text2music'),
    'repaint': Task('repaint', source=True, span=True),
    'cover': Task('cover', source=True, strength=True),
    'extract': Task('extract', source=True, track=True),
    'lego': Task('lego', source=True, span=True, track=True),
    'complete': Task('complete', source=True, classes=True),
}

ACE_STEP_TASKS = ('text2music', 'repaint', 'extract', 'lego', 'complete')
ACE_STEP_BASE_TASKS = ('text2music', 'repaint', 'cover', 'extract', 'lego', 'complete')

# Stems the model names in its own task instructions; free text is accepted for anything else.
ACE_STEP_TRACKS = ('vocals', 'drums', 'bass', 'guitar', 'piano', 'strings', 'synth', 'other')

# Both repos ship the same Qwen3-Embedding-0.6B text encoder, byte identical by hash and 1.19 GB
# each, and the hub caches blobs per repo. Every row loads the one in the base repo, which is the
# checkpoint the others derive from: diffusers drops an explicitly passed component from its
# download patterns, so the copy in the other repo is never fetched rather than fetched and then
# deduplicated. The condition encoder differs between the two and is not shared.
ACE_STEP_TE = 'ACE-Step/acestep-v15-xl-base-diffusers'
ACE_STEP_TE_CLS = 'Qwen3Model' # the class model_index.json names

models: dict[str, list[Model]] = {
    'ACE-Step': [
        Model(
            name='ACE-Step 1.5 XL Turbo',
            repo='ACE-Step/acestep-v15-xl-turbo-diffusers',
            url='https://huggingface.co/ACE-Step/acestep-v15-xl-turbo-diffusers',
            repo_cls='AceStepPipeline',
            dit_cls='AceStepTransformer1DModel',
            te=ACE_STEP_TE,
            te_cls=ACE_STEP_TE_CLS,
            steps=8,
            cfg=1.0,
            distilled=True, # the pipeline warns and ignores any other guidance value
            sample_rate=48000,
            duration=60.0,
            duration_max=600.0,
            tasks=ACE_STEP_TASKS, # no cover: this repo ships neither the tokenizer nor the detokenizer
            task_spec=ACE_STEP_TASK_SPEC,
            reference=True,
        ),
        Model(
            name='ACE-Step 1.5 XL Base',
            repo='ACE-Step/acestep-v15-xl-base-diffusers',
            url='https://huggingface.co/ACE-Step/acestep-v15-xl-base-diffusers',
            repo_cls='AceStepPipeline',
            dit_cls='AceStepTransformer1DModel',
            te=ACE_STEP_TE,
            te_cls=ACE_STEP_TE_CLS,
            steps=50,
            cfg=6.0,
            sample_rate=48000,
            duration=60.0,
            duration_max=600.0,
            tasks=ACE_STEP_BASE_TASKS, # the only repo carrying the tokenizer pair, so the only one that covers
            task_spec=ACE_STEP_TASK_SPEC,
            reference=True,
        ),
    ],
}


def is_model(row: Model) -> bool:
    """Rows that name a loadable model; placeholders and dropdown separators do not."""
    return row.name != 'None' and not row.name.startswith('─')


def engines() -> list[str]:
    return [engine for engine, rows in models.items() if any(is_model(row) for row in rows)]


def model_names(engine: str) -> list[str]:
    return [row.name for row in models.get(engine, []) if is_model(row)]


def all_models() -> list[Model]:
    return [row for rows in models.values() for row in rows if is_model(row)]


def engine_of(row: Model) -> str | None:
    for engine, rows in models.items():
        if row in rows:
            return engine
    return None


def find(name: str, engine: str | None = None) -> Model | None:
    """Case-insensitive exact-name lookup, optionally scoped to one engine."""
    for family, rows in models.items():
        if engine is not None and family.lower() != engine.lower():
            continue
        for row in rows:
            if is_model(row) and row.name.lower() == (name or '').lower():
                return row
    return None


def default_model() -> Model | None:
    rows = all_models()
    return rows[0] if rows else None


def pipeline_classes() -> set[str]:
    return {row.repo_cls for row in all_models() if row.repo_cls is not None}


def supports(row: Model, task: str) -> bool:
    return task in (row.tasks or ())


def task_of(row: Model, task: str) -> Task | None:
    """What the named task consumes on this model, or None when the model does not offer it."""
    if not supports(row, task):
        return None
    return (row.task_spec or {}).get(task)


def needs_source(row: Model, task: str) -> bool:
    spec = task_of(row, task)
    return bool(spec and spec.source)
