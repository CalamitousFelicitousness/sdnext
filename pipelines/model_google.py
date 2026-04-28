"""Loader shim for the Google NanoBanana cloud pipeline.

Implementation lives in ``modules/cloud/google_image.py`` (Phase 2 cloud
framework). The class name ``GoogleNanoBananaPipeline`` is preserved so the
literal-name dispatch in ``modules/sd_models.py:54``,
``modules/modeldata.py:138-144`` and ``modules/processing_args.py:158`` keeps
working unchanged.
"""


def load_nanobanana(checkpoint_info, diffusers_load_config):  # pylint: disable=unused-argument
    from modules import sd_models  # pylint: disable=import-outside-toplevel
    from modules.cloud.google_image import build_pipeline  # pylint: disable=import-outside-toplevel
    repo_id = sd_models.path_to_repo(checkpoint_info)
    return build_pipeline(repo_id)
