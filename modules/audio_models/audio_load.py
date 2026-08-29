"""Loading audio models into the shared pipeline slot.

Mirrors the video loader's shape: components come from the generic loaders so quantization and the
single-file overrides apply, and the post-load tail opts into the same options, quant and offload
passes the image and video paths use.
"""

import os
import time
import diffusers
import transformers
from modules import devices, errors, model_quant, sd_checkpoint, sd_models, shared
from modules.audio_models import models_def
from modules.audio_models.audio_error import AudioError
from modules.logger import log
from pipelines import generic


loaded_model = None


def resolve_classes(row: models_def.Model):
    """Class names to classes, as locals. Rows are frozen, so nothing is written back."""
    repo_cls = getattr(diffusers, row.repo_cls, None) if row.repo_cls else None
    dit_cls = getattr(diffusers, row.dit_cls, None) if row.dit_cls else None
    te_cls = getattr(transformers, row.te_cls, None) if row.te_cls else None
    if row.repo_cls and repo_cls is None:
        raise AudioError(f'audio: pipeline class not found: {row.repo_cls}', 500)
    return repo_cls, dit_cls, te_cls


def preflight(repo: str):
    """Reject an unreachable or gated repo before the loader starts pulling gigabytes.

    A gated repo otherwise fails deep inside the download with an error that names neither the repo
    nor the thing the user has to do about it.
    """
    if shared.opts.offline_mode:
        return
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo)
    except Exception as e:
        name = type(e).__name__
        if 'Gated' in name or 'gated' in str(e).lower():
            raise AudioError(f'audio: model="{repo}" is gated, accept its license at https://huggingface.co/{repo}', 400) from e
        if 'RepositoryNotFound' in name or '404' in str(e):
            raise AudioError(f'audio: model="{repo}" not found on the hub', 400) from e
        log.debug(f'Load audio: preflight inconclusive repo="{repo}" {e}')
        return
    if getattr(info, 'gated', False):
        raise AudioError(f'audio: model="{repo}" is gated, accept its license at https://huggingface.co/{repo}', 400)


def load_model(selected: models_def.Model) -> str:
    global loaded_model # pylint: disable=global-statement
    if selected is None or selected.repo is None:
        raise AudioError('audio: no model selected', 400)
    repo_cls, dit_cls, te_cls = resolve_classes(selected)

    if not shared.sd_loaded:
        loaded_model = None
    elif loaded_model == selected.name and repo_cls is not None and not isinstance(shared.sd_model, repo_cls):
        # the shared slot auto-reloads the default checkpoint when it is emptied, which swaps the
        # pipe class behind the name-based cache; a class mismatch is the reliable signal
        log.warning(f'Load audio: cached model="{selected.name}" cls={type(shared.sd_model).__name__} mismatch forcing reload')
        loaded_model = None
    if loaded_model == selected.name:
        return ''
    if shared.sd_loaded:
        sd_models.unload_model_weights()

    t0 = time.time()
    preflight(selected.repo)
    jobid = shared.state.begin('Load model')
    try:
        offline_args = {}
        if shared.opts.offline_mode:
            offline_args['local_files_only'] = True
            os.environ['HF_HUB_OFFLINE'] = '1'
        else:
            os.environ.pop('HF_HUB_OFFLINE', None)

        kwargs = {}
        sd_models.hf_auth_check(selected.repo)
        if te_cls is not None:
            if model_quant.check_quant('TE'):
                # the setting is honored rather than gated, and the cost is stated: measured on
                # ACE-Step turbo at one seed, quantizing this encoder to uint4 took output rms from
                # 0.2375 to 0.1411, and a Qwen3 encoder under sdnq has produced NaN embeddings on
                # another architecture
                log.warning(f'Load audio: text encoder is quantized to {shared.opts.sdnq_quantize_weights_mode}, which degrades this model; untick TE in quantization settings to keep it at full precision')
            # allow_shared is off deliberately. The shared text encoder registry matches on the
            # model class, and its Qwen3Model entry carries no identifier, so it captures any
            # Qwen3 encoder from any repo: it swapped this one for Anima's and generation carried
            # on with the wrong weights at roughly half the output level.
            text_encoder = generic.load_text_encoder(
                selected.te or selected.repo,
                cls_name=te_cls,
                subfolder=selected.te_folder,
                revision=selected.te_revision or selected.repo_revision,
                allow_shared=False,
            )
            if text_encoder is None:
                raise AudioError(f'audio: text encoder failed to load from "{selected.te or selected.repo}"', 500)
            kwargs['text_encoder'] = text_encoder
        if dit_cls is not None:
            kwargs['transformer'] = generic.load_transformer(
                selected.dit or selected.repo,
                cls_name=dit_cls,
                subfolder=selected.dit_folder,
                revision=selected.dit_revision or selected.repo_revision,
            )

        sd_models.hf_prefetch_configs(selected.repo, {}, 'audio')
        try:
            shared.sd_model = repo_cls.from_pretrained(
                pretrained_model_name_or_path=selected.repo,
                revision=selected.repo_revision,
                cache_dir=shared.opts.hfcache_dir,
                torch_dtype=devices.dtype,
                **kwargs,
                **offline_args,
            )
        except Exception as e:
            log.error(f'Load audio: repo="{selected.repo}" cls={selected.repo_cls} {e}')
            errors.display(e, 'audio')
            raise AudioError(f'audio: model="{selected.name}" failed to load', 500) from e
        if shared.sd_model is None:
            from modules.modeldata import model_data
            if model_data.locked: # assignment to the shared slot is discarded while locked, with no error
                raise AudioError('audio: model slot is locked, load ran before the server finished starting', 500)
            raise AudioError(f'audio: model="{selected.name}" failed to load', 500)

        post_load(selected)
    finally:
        shared.state.end(jobid)

    loaded_model = selected.name
    msg = f'Load audio: cls={shared.sd_model.__class__.__name__} model="{selected.name}" time={time.time()-t0:.2f}'
    log.info(msg)
    return msg


def post_load(selected: models_def.Model):
    """The tail the image and video paths share, plus the one audio-specific placement."""
    shared.sd_model.sd_checkpoint_info = sd_checkpoint.CheckpointInfo(selected.repo)
    shared.sd_model.sd_model_hash = None
    sd_models.set_diffuser_options(shared.sd_model, offload=False)
    vae = getattr(shared.sd_model, 'vae', None)
    tiling = False
    if selected.vae_tiling and vae is not None and hasattr(vae, 'use_tiling'):
        # AutoencoderOobleck exposes the flag and no enable_tiling method, so the generic hasattr
        # check the video loader uses would skip it. Untiled, a four minute decode allocates 27 GB
        # and takes minutes; tiled it is seconds, and the flag is inert below the tile length.
        vae.use_tiling = True
        tiling = True
    if hasattr(shared.sd_model, 'set_progress_bar_config'):
        shared.sd_model.set_progress_bar_config(bar_format='Progress {rate_fmt}{postfix} {bar:15} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining} ' + '\x1b[38;5;71m', ncols=120, colour='#327fba')
    shared.sd_model = model_quant.do_post_load_quant(shared.sd_model, allow=False)
    sd_models.set_diffuser_offload(shared.sd_model)
    log.debug(f'Audio model: tiling={tiling} rate={selected.sample_rate}')


def unload():
    global loaded_model # pylint: disable=global-statement
    loaded_model = None
    if shared.sd_loaded:
        sd_models.unload_model_weights()
