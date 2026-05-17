"""Cloud size-constraint probe tooling.

Dev-time tooling for discovering and codifying per-model image size
constraints into modules/cloud/size_constraints.json. Not loaded at runtime
by sdnext.

Entry point: `python -m modules.cloud.probe.probe_image_sizes --help`.
"""
