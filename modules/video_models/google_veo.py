"""Loader shim for the Google Veo cloud video pipeline.

Implementation lives in ``modules/cloud/google_video.py`` (Phase 2 cloud
framework). This file is preserved as a thin loader to keep
``modules.video_models.video_load.load_custom`` import paths stable.
"""


def load_veo(model_name: str):
    from modules.cloud.google_video import build_pipeline  # pylint: disable=import-outside-toplevel
    return build_pipeline(model_name)
