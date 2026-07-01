"""Compatibility shim for the root-level entry point created at project
start. The real implementation lives in src/mtl/models/backbone.py as part
of the installable `mtl` package - import from there directly in new code.
"""
from mtl.models.backbone import build_backbone  # noqa: F401
