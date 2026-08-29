from threading import Lock
from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
from fastapi.exceptions import HTTPException
from modules import errors, shared
from modules.api import helpers
from modules.audio import metadata as audio_metadata
from modules.audio import save as audio_save
from modules.audio_models import audio_load, audio_run, models_def
from modules.audio_models.audio_error import AudioError
from modules.paths import resolve_output_path


errors.install()


class ReqAudio(BaseModel):
    engine: str | None = Field(default=None, title="Engine", description="Audio engine family; omit with model to use the first registered model")
    model: str | None = Field(default=None, title="Model", description="Audio model name within the engine; see GET /sdapi/v1/audio/models")
    prompt: str = Field(default="", title="Prompt", description="Style tags describing genre, instrumentation, mood and tempo")
    lyrics: str = Field(default="", title="Lyrics", description="Lyrics for models that accept them; newlines are passed through, structure markers depend on the model's lyrics format")
    negative_prompt: str = Field(default="", title="Negative prompt", description="Ignored by models that do not accept one; see the model's negative field")
    duration: float | None = Field(default=None, title="Duration", description="Length in seconds; defaults to the model's own default and is clamped to its range")
    steps: int | None = Field(default=None, title="Steps", description="Denoising steps; defaults to the model's own value")
    cfg_scale: float | None = Field(default=None, title="CFG scale", description="Guidance scale; ignored by guidance-distilled models")
    seed: int = Field(default=-1, title="Seed", description="Random seed, -1 for random. Takes past the first increment from it")
    task: str = Field(default="text2music", title="Task", description="Generation task; see the model's tasks list")
    n_iter: int = Field(default=1, ge=1, le=16, title="Takes", description="Number of takes from one model load; each uses the next seed")
    audio_format: str | None = Field(default=None, title="Format", description="Output container: flac, opus, mp3 or wav. wav cannot store parameters")
    save: bool = Field(default=True, title="Save", description="Write the audio to the output folder")
    send_audio: bool = Field(default=False, title="Send audio", description="Return the file base64 encoded; prefer GET /sdapi/v1/audio/file for anything long")
    override_settings: dict = Field(default={}, title="Override settings", description="Settings applied for this request")


class ItemAudioTake(BaseModel):
    audio: str | None = Field(default=None, title="Audio", description="Base64-encoded audio file; populated only when send_audio is set and the file is under the size cap")
    audio_path: str | None = Field(default=None, title="Audio path", description="Server path of the saved audio; fetch via GET /sdapi/v1/audio/file")
    seed: int = Field(default=-1, title="Seed", description="Seed this take used")
    duration: float = Field(default=0.0, title="Duration", description="Measured duration in seconds")
    sample_rate: int = Field(default=0, title="Sample rate", description="Output sample rate")
    levels: dict = Field(default={}, title="Levels", description="Measured levels: peak, rms, dc offset, clipped count, integrated loudness and true peak")
    info: str = Field(default="", title="Info", description="Generation parameters, the same string embedded in the file")


class ResAudio(BaseModel):
    takes: list[ItemAudioTake] = Field(default=[], title="Takes", description="One entry per generated take, in seed order")
    params: dict = Field(default={}, title="Parameters", description="Values actually used, after model defaults and clamping")
    info: str = Field(default="", title="Info", description="Generation info of the first take")


class ItemAudioModel(BaseModel):
    engine: str = Field(title="Engine", description="Audio engine family")
    name: str = Field(title="Name", description="Model name; pass with engine to select it")
    repo: str = Field(default="", title="Repo", description="Model repository")
    url: str = Field(default="", title="URL", description="Model information page")
    steps: int = Field(default=0, title="Steps", description="Default denoising steps for this model")
    cfg: float = Field(default=0.0, title="CFG scale", description="Default guidance scale")
    distilled: bool = Field(default=False, title="Distilled", description="Guidance is baked in and cfg_scale is ignored")
    sample_rate: int = Field(default=0, title="Sample rate", description="Output sample rate")
    duration: float = Field(default=0.0, title="Duration", description="Default duration in seconds")
    duration_min: float = Field(default=0.0, title="Minimum duration", description="Shortest accepted duration")
    duration_max: float = Field(default=0.0, title="Maximum duration", description="Longest accepted duration")
    tasks: list[str] = Field(default=[], title="Tasks", description="Accepted task values")
    lyrics: bool = Field(default=False, title="Lyrics", description="Accepts a lyrics input")
    lyrics_format: str = Field(default="", title="Lyrics format", description="tagged for [verse] and [chorus] markers, lrc for timestamped lines")
    negative: bool = Field(default=False, title="Negative", description="Accepts a negative prompt")
    reference: bool = Field(default=False, title="Reference", description="Accepts reference audio")
    loaded: bool = Field(default=False, title="Loaded", description="Currently loaded")


class APIAudio:
    def __init__(self, queue_lock: Lock):
        self.queue_lock = queue_lock

    def post_audio(self, req: ReqAudio):
        """Generate audio from a text prompt and optional lyrics.

        Defaults come from the selected model rather than from this schema: omitting `steps`,
        `cfg_scale` or `duration` uses the model's own values, which
        `GET /sdapi/v1/audio/models` publishes. A request naming a task the model does not
        support, or a duration outside its range, is refused before anything loads.

        Files are written to the audio output folder and reported as `audio_path`; fetch them with
        `GET /sdapi/v1/audio/file`. `send_audio` inlines the bytes instead, which is fine for short
        takes and wasteful for long ones. Every take reports its measured levels, and nothing is
        loudness normalized on the way to disk.
        """
        with self.queue_lock:
            jobid = shared.state.begin('API-AUD', api=True)
            try:
                results = audio_run.run(
                    engine=req.engine,
                    model=req.model,
                    prompt=req.prompt,
                    lyrics=req.lyrics,
                    negative_prompt=req.negative_prompt,
                    duration=req.duration,
                    steps=req.steps,
                    cfg_scale=req.cfg_scale,
                    seed=req.seed,
                    task=req.task,
                    n_iter=req.n_iter,
                    audio_format=req.audio_format,
                    save=req.save,
                    override_settings=req.override_settings,
                )
            except AudioError as e:
                raise HTTPException(status_code=e.code, detail=e.msg) from e
            except Exception as e:
                errors.display(e, 'api audio')
                raise HTTPException(status_code=500, detail=str(e)) from e
            finally:
                shared.state.end(jobid)

        takes = []
        for result in results:
            encoded = None
            if req.send_audio and result.path:
                encoded = helpers.encode_file_to_base64(result.path)
            takes.append(ItemAudioTake(
                audio=encoded,
                audio_path=result.path,
                seed=int(result.processed.seed) if result.processed is not None else -1,
                duration=result.duration,
                sample_rate=result.sample_rate,
                levels=result.levels,
                info=result.info,
            ))
        first = results[0] if results else None
        params = {
            'model': first.processed.audio_model if first is not None and hasattr(first.processed, 'audio_model') else req.model,
            'duration': first.duration if first is not None else 0.0,
            'sample_rate': first.sample_rate if first is not None else 0,
            'takes': len(takes),
        }
        return ResAudio(takes=takes, params=params, info=first.info if first is not None else '')

    def get_audio_models(self, engine: str | None = None):
        """List audio engines and models with the defaults and limits each one accepts."""
        items = []
        for family, rows in models_def.models.items():
            if engine is not None and family.lower() != engine.lower():
                continue
            for m in rows:
                if not models_def.is_model(m):
                    continue
                items.append(ItemAudioModel(
                    engine=family,
                    name=m.name,
                    repo=m.repo or '',
                    url=m.url or '',
                    steps=m.steps,
                    cfg=m.cfg,
                    distilled=m.distilled,
                    sample_rate=m.sample_rate,
                    duration=m.duration,
                    duration_min=m.duration_min,
                    duration_max=m.duration_max,
                    tasks=list(m.tasks),
                    lyrics=m.lyrics,
                    lyrics_format=m.lyrics_format,
                    negative=m.negative,
                    reference=m.reference,
                    loaded=(m.name == audio_load.loaded_model),
                ))
        return items

    def get_audio_file(self, file: str):
        """Serve audio produced by this endpoint; the path must resolve inside the audio output directory."""
        import mimetypes
        from pathlib import Path
        from starlette.responses import FileResponse
        if not file or not file.strip():
            raise HTTPException(status_code=400, detail="file path is required")
        root = Path(resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_audio)).resolve()
        target = Path(file).resolve()
        if root not in target.parents:
            raise HTTPException(status_code=403, detail=f"file {file}: must be inside the audio output directory")
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"file not found: {file}")
        media_type = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
        return FileResponse(str(target), media_type=media_type, filename=target.name)

    def get_audio_info(self, file: str):
        """Read the generation parameters embedded in an audio file, the audio counterpart of png-info.

        Tags are merged from the container and the audio stream, since ogg and opus carry them on
        the stream only. The json sidecar answers for formats that store no tags at all.
        """
        import os
        if not file or not file.strip():
            raise HTTPException(status_code=400, detail="file path is required")
        if not os.path.isfile(file):
            raise HTTPException(status_code=404, detail=f"file not found: {file}")
        info = audio_metadata.read_audio_info(file)
        sidecar = audio_metadata.read_sidecar(file)
        if info is None and sidecar is not None:
            info = sidecar.get(audio_metadata.INFO_KEY)
        return {
            'info': info or '',
            'tags': audio_metadata.read_audio_metadata(file),
            'sidecar': sidecar,
            'formats': {ext: {'tags': spec.tags, 'lossless': spec.lossless} for ext, spec in audio_save.AUDIO_FORMATS.items()},
        }
