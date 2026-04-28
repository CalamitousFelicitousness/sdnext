"""Loader shim for NanoGPT cloud image pipeline.

Implementation lives in ``modules/cloud/nanogpt_image.py``. The class name
``NanoGPTImagePipeline`` is preserved so literal-name dispatch in
``modules/sd_models.py`` and ``modules/processing_args.py`` keeps working.
"""


def load_nanogpt_image(checkpoint_info, diffusers_load_config):  # pylint: disable=unused-argument
    from modules import sd_models  # pylint: disable=import-outside-toplevel
    from modules.cloud.nanogpt_image import build_pipeline  # pylint: disable=import-outside-toplevel
    repo_id = sd_models.path_to_repo(checkpoint_info)
    model_id = repo_id.replace('nanogpt:', '') if repo_id.startswith('nanogpt:') else repo_id
    return build_pipeline(model_id)
