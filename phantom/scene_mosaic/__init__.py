"""
Planar scene mosaic: a stitched top-down photo of the (flat) workspace built from a
wrist-camera scan video + robot poses, and a renderer that synthesizes the wrist
camera's view of it at any end-effector pose.

Vendored from diffusion_policy_THI (branch scene-reconstruction-scan,
scene_reconstruction/video_mosaic); see README.md for what was changed locally and
README_upstream.md for the original description of how the pipeline works.

The synthetic pick-and-place pipeline consumes only `MosaicModel` (via
phantom.synthetic.wrist_background); `build_mosaic` / `evaluate` are the offline
tools that produce and validate a model directory.
"""

from phantom.scene_mosaic.common import Canvas, Plane, matrix_to_pose6, pose6_to_matrix

__all__ = ["Canvas", "MosaicModel", "Plane", "matrix_to_pose6", "pose6_to_matrix"]


def __getattr__(name):
    """
    `MosaicModel` lives in render_view, which is also a CLI: importing it eagerly here
    would make `python -m phantom.scene_mosaic.render_view` load that module twice and
    warn about it. Re-exporting it lazily keeps `from phantom.scene_mosaic import
    MosaicModel` working without that.
    """
    if name == "MosaicModel":
        from phantom.scene_mosaic.render_view import MosaicModel
        return MosaicModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
