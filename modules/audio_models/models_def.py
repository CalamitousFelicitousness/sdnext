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
    lyrics: bool = True
    lyrics_format: str = 'tagged' # tagged for [verse]/[chorus], lrc for timestamped lines
    negative: bool = False
    reference: bool = False # accepts reference audio for the cover task
    languages: tuple = ()

    def __str__(self):
        return f'name="{self.name}" repo="{self.repo}" cls="{self.repo_cls}" steps={self.steps} cfg={self.cfg} distilled={self.distilled} rate={self.sample_rate} tasks={list(self.tasks)}'


# The code-driven tasks need the audio_tokenizer and audio_token_detokenizer components, which the
# base repo carries and the turbo repo does not, so the shorter list is a property of the weights
# rather than a policy.
ACE_STEP_TASKS = ('text2music', 'repaint', 'cover')
ACE_STEP_BASE_TASKS = ACE_STEP_TASKS + ('extract', 'lego', 'complete')

models: dict[str, list[Model]] = {
    'ACE-Step': [
        Model(
            name='ACE-Step 1.5 XL Turbo',
            repo='ACE-Step/acestep-v15-xl-turbo-diffusers',
            url='https://huggingface.co/ACE-Step/acestep-v15-xl-turbo-diffusers',
            repo_cls='AceStepPipeline',
            dit_cls='AceStepTransformer1DModel',
            steps=8,
            cfg=1.0,
            distilled=True, # the pipeline warns and ignores any other guidance value
            sample_rate=48000,
            duration=60.0,
            duration_max=600.0,
            tasks=ACE_STEP_TASKS,
            reference=True,
        ),
        Model(
            name='ACE-Step 1.5 XL Base',
            repo='ACE-Step/acestep-v15-xl-base-diffusers',
            url='https://huggingface.co/ACE-Step/acestep-v15-xl-base-diffusers',
            repo_cls='AceStepPipeline',
            dit_cls='AceStepTransformer1DModel',
            steps=50,
            cfg=6.0,
            sample_rate=48000,
            duration=60.0,
            duration_max=600.0,
            tasks=ACE_STEP_BASE_TASKS, # stem extraction and track completion are base only
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
