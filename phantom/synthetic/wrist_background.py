"""
Wrist-camera background rendering from a scene mosaic.

The external camera's synthetic background comes from random frames of a clean-plate
video, which works because that camera never moves. The wrist camera does move with
the robot, so there is no single clean plate for it: its background has to be
*synthesized at the pose the robot is actually in*. That is what the scene mosaic
(`phantom.scene_mosaic`) provides -- a stitched top-down photo of the flat workspace,
plus the geometry to project it back into the wrist camera at any end-effector pose.

This module is the bridge between that model and the MuJoCo twin:

- `WristMosaicBackground.camera_params()` returns the `MujocoCameraParams` for the
  twin's flange-mounted camera, derived from `mosaic_meta.json`;
- `WristMosaicBackground.render_at_camera(T_base_cam)` returns the RGB background for
  the camera pose MuJoCo actually rendered from (`TwinRobot.get_wrist_camera_pose`),
  at the twin's render resolution and with the same square crop, ready for
  `overlay_robot`. `render(eef_pose6)` is the standalone equivalent that composes the
  camera pose from an end-effector pose itself.

Both come from one file, which is the point: the mosaic's self-calibrated `T_eef_cam`
differs from the raw hand-eye calibration by 165.6 deg about the optical axis for the
current scan (see phantom/scene_mosaic/README.md), and any disagreement between the
camera MuJoCo renders the robot from and the camera the background is rendered for
shows up directly as the robot sitting wrong on its background. Taking the render pose
from the simulation closes that loop: whatever frame offsets sit between the pose the
twin reports and the body the camera hangs off, the two images are the same camera.

Conventions
-----------
- `eef_pose6` is [x, y, z, rx, ry, rz], robot-base frame, meters + rotation vector --
  the same convention as the replay buffer's `robot_eef_pose` and as the poses the
  mosaic was built from.
- `T_eef_cam` is flange -> camera in the OpenCV optical frame (x right, y down,
  z forward). MuJoCo cameras look down -z with +y up, hence the y/z flip in
  `convert_real_camera_ori_to_mujoco`.
- The flange here is the twin's eef *body* (`robot0_right_hand`), which is where the
  camera is mounted -- not the grip site that `robot0_eef_pos/quat` reports.
- The returned background is RGB uint8, matching `load_background_frames`; the mosaic
  itself is BGR (OpenCV), so gain is applied before the conversion, in the channel
  order it was estimated in.
"""
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

from phantom.scene_mosaic import MosaicModel, pose6_to_matrix
from phantom.scene_mosaic import common, mosaic as mz
from phantom.utils.image_utils import center_crop_square


class WristMosaicBackground:
    """
    Renders wrist-camera backgrounds for arbitrary end-effector poses from a mosaic
    model directory built by `phantom.scene_mosaic.build_mosaic`.

    Args:
        model_dir: mosaic model directory (mosaic.png, mosaic_valid.png,
            mosaic_meta.json, gain.npy).
        img_w, img_h: resolution the twin renders at. The mosaic's intrinsics are
            scaled to it, so it must have the same aspect ratio as the calibration.
        square: apply the same centre square crop `TwinRobot.get_cam_image` applies,
            so background and overlay stay pixel-aligned.
        apply_distortion: re-apply the real lens distortion, as the upstream renderer
            does. False (the default) renders a pinhole image instead, which is what
            MuJoCo's camera model produces -- with it True the background is more
            faithful to raw camera frames but the composited robot is geometrically
            wrong near the edges, where distortion is largest.
        apply_gain: re-apply the camera-fixed shading (vignetting, the arm's own
            shadow) that was divided out when the mosaic was built.
        fill_color: RGB value for pixels the scan never saw (outside the mosaic).
        mount_translation: replaces `T_eef_cam`'s translation (the camera's lever arm
            in the flange frame, meters). The scan cannot measure it -- all its poses
            share one orientation, so the lever arm is very nearly unobservable, as
            upstream's Limitations note says -- so unlike the rotation it is only as
            good as the hand-eye file. None keeps the hand-eye value.
        T_eef_cam: replaces the whole flange -> camera transform (4x4, OpenCV optical
            convention) instead of using the mosaic's self-calibrated one -- e.g. to
            test an independent hand-eye calibration against the scan's. The twin's
            camera is configured from the same value (`camera_params`) and the
            background is rendered for the pose the twin reports, so the two stay
            consistent whichever transform is used; what changes is *where on the
            mosaic* the camera is deemed to be looking. `mount_translation` still
            overrides its translation if both are given.
    """

    def __init__(self, model_dir: str, img_w: int, img_h: int, square: bool = False,
                 apply_distortion: bool = False, apply_gain: bool = True,
                 fill_color: Sequence[int] = (0, 0, 0),
                 mount_translation: Optional[Sequence[float]] = None,
                 T_eef_cam: Optional[np.ndarray] = None) -> None:
        self.model = MosaicModel(model_dir)
        self.model_dir = str(model_dir)
        self.img_w, self.img_h = int(img_w), int(img_h)
        self.square = bool(square)
        self.apply_distortion = bool(apply_distortion)
        self.apply_gain = bool(apply_gain)
        self.fill_color = np.array(fill_color, dtype=np.uint8)

        self.T_eef_cam = (self.model.T_eef_cam if T_eef_cam is None
                          else np.asarray(T_eef_cam, dtype=float)).copy()
        if self.T_eef_cam.shape != (4, 4):
            raise ValueError(f"T_eef_cam must be 4x4, got {self.T_eef_cam.shape}")
        self.T_eef_cam_source = "mosaic self-calibration" if T_eef_cam is None else "caller-supplied"
        if mount_translation is not None:
            self.T_eef_cam[:3, 3] = np.asarray(mount_translation, dtype=float)

        # mosaic.png / mosaic_valid.png are not in the upstream repo's git, so a model
        # directory is often assembled by hand from two places; catch a mosaic paired
        # with another build's meta here rather than rendering silent nonsense.
        canvas_shape = (self.model.canvas.height, self.model.canvas.width)
        for fname, arr in (("mosaic.png", self.model.mosaic), ("mosaic_valid.png", self.model.valid)):
            if arr.shape[:2] != canvas_shape:
                raise ValueError(
                    f"{model_dir}/{fname} is {arr.shape[1]}x{arr.shape[0]} but its "
                    f"mosaic_meta.json describes a {canvas_shape[1]}x{canvas_shape[0]} "
                    "canvas -- they are from different builds of the mosaic."
                )

        aspect_model = self.model.w / self.model.h
        aspect_target = self.img_w / self.img_h
        if abs(aspect_model - aspect_target) > 1e-3:
            raise ValueError(
                f"Wrist mosaic was calibrated at {self.model.w}x{self.model.h} "
                f"(aspect {aspect_model:.4f}) but the twin renders at "
                f"{self.img_w}x{self.img_h} (aspect {aspect_target:.4f}). Scaling the "
                "intrinsics between different aspect ratios would need a crop "
                "convention this code does not assume; re-render the twin at a "
                "matching aspect ratio instead."
            )

        # Intrinsics scaled from the calibration resolution to the render resolution.
        # Distortion coefficients act on normalised coordinates, so they are
        # scale-invariant and carry over unchanged.
        self.scale = self.img_w / self.model.w
        self.K = self.model.K.copy()
        self.K[:2, :] *= self.scale
        self.dist = self.model.dist if self.apply_distortion else np.zeros(5)

        # The ray for every pixel depends only on (K, dist, w, h), so it is computed
        # once here rather than per frame -- at 1920x1080 undistorting 2M points
        # dominated the cost of a render.
        self.rays = common.pixel_rays(self.K, self.dist, self.img_w, self.img_h)

        self._coarse_ray_cache: dict = {}

        self.gain = None
        if self.apply_gain and self.model.gain is not None:
            self.gain = cv2.resize(self.model.gain, (self.img_w, self.img_h),
                                   interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------ geometry

    def camera_pose(self, eef_pose6: np.ndarray) -> np.ndarray:
        """
        T_base_cam for a *flange* pose. Note this is not the frame the twin reports as
        its end effector (that is the grip site); inside the pipeline the camera pose
        comes from `TwinRobot.get_wrist_camera_pose` instead, so no flange pose has to
        be reconstructed.
        """
        return pose6_to_matrix(np.asarray(eef_pose6, dtype=float)) @ self.T_eef_cam

    def camera_params(self, name: str = "wristview") -> Any:
        """
        `MujocoCameraParams` for the twin's flange-mounted camera, from the mosaic's
        self-calibrated extrinsic and the scan's intrinsics.

        `pos`/`ori_wxyz` are local to the robot's eef body (that is how TwinRobot
        interprets `wrist_camera_params`), i.e. exactly `T_eef_cam`. The principal
        point is expressed for the *full* render resolution, as the fixed camera's is:
        the square crop happens afterwards, on the rendered image.
        """
        # Imported lazily: pulls in robosuite/robomimic, which callers that only want
        # backgrounds (e.g. a quick render check) don't need.
        from phantom.twin_bimanual_robot import MujocoCameraParams
        from phantom.twin_robot import convert_real_camera_ori_to_mujoco

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        # convert_real_camera_ori_to_mujoco negates columns in place, so hand it a copy.
        ori_wxyz = convert_real_camera_ori_to_mujoco(self.T_eef_cam[:3, :3].copy())

        return MujocoCameraParams(
            name=name,
            pos=self.T_eef_cam[:3, 3].copy(),
            ori_wxyz=ori_wxyz,
            fov=float(np.degrees(2 * np.arctan(self.img_h / (2 * fy)))),
            resolution=(self.img_h, self.img_w),
            # Same (fx/fy-swapped) sensor-size convention as the fixed camera's
            # params in BaseProcessor._calculate_sensor_size, kept identical on
            # purpose so both cameras are configured the same way.
            sensorsize=np.array([self.img_w / fy / 1000, self.img_h / fx / 1000]),
            principalpixel=np.array([self.img_w / 2 - cx, cy - self.img_h / 2]),
            focalpixel=np.array([fx, fy]),
        )

    # -------------------------------------------------------------------- render

    def render(self, eef_pose6: np.ndarray) -> Tuple[np.ndarray, float]:
        """Background for one flange pose (see `camera_pose`)."""
        return self.render_at_camera(self.camera_pose(eef_pose6))

    def render_at_camera(self, T_base_cam: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Background as seen from an explicit camera pose (base frame, OpenCV optical
        convention), which is what `TwinRobot.get_wrist_camera_pose` returns.

        Returns (RGB uint8 image, coverage), where coverage is the fraction of the
        image the mosaic actually covers -- it drops as the camera looks outside the
        scanned area, and the uncovered pixels are `fill_color`.
        """
        img, cov = mz.render_view(self.model.mosaic, self.model.valid, self.model.canvas,
                                  self.K, self.dist, self.img_w, self.img_h,
                                  T_base_cam, rays=self.rays)
        if self.gain is not None:
            img = img * self.gain
        img = np.clip(img, 0, 255).astype(np.uint8)
        img[~cov] = self.fill_color[::-1]  # still BGR here
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.square:
            img = center_crop_square(img)
            cov = center_crop_square(cov)
        return img, float(cov.mean())

    # ------------------------------------------------------------------ coverage

    def _coarse_rays(self, step: int) -> np.ndarray:
        """
        Pixel rays on a `step`-subsampled grid, over the square crop when one is
        applied, cached per step. Used to probe coverage without paying for a full
        render -- a preflight over a randomization region needs hundreds of probes.
        """
        cached = self._coarse_ray_cache.get(step)
        if cached is not None:
            return cached
        x0, x1 = 0, self.img_w
        if self.square:
            x0 = (self.img_w - self.img_h) // 2
            x1 = x0 + self.img_h
        xs = np.arange(x0, x1, step, dtype=np.float64)
        ys = np.arange(0, self.img_h, step, dtype=np.float64)
        pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, self.K, self.dist).reshape(len(ys), len(xs), 2)
        rays = np.concatenate([und, np.ones((len(ys), len(xs), 1))], axis=-1)
        self._coarse_ray_cache[step] = rays
        return rays

    def coverage_at_camera(self, T_base_cam: np.ndarray, step: int = 16) -> float:
        """
        Fraction of the view the mosaic covers from this camera pose, without rendering
        the image. Same quantity `render_at_camera` returns as its second value, to
        within the subsampling.
        """
        rays = self._coarse_rays(step)
        canvas = self.model.canvas
        R, c = T_base_cam[:3, :3], T_base_cam[:3, 3]
        X, depth = canvas.plane.intersect(np.broadcast_to(c, rays.shape), rays @ R.T)
        px = canvas.st_to_px(canvas.plane.to_st(X)).astype(np.float32)
        px[depth <= 0] = -1  # plane behind the camera
        cov = cv2.remap(self.model.valid.astype(np.float32), px[..., 0], px[..., 1],
                        cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return float((cov > 0.5).mean())

    def scan_bounds(self) -> dict:
        """
        Where the scan that built this mosaic actually went, in the robot base frame:
        the flange positions it visited, the camera heights those produced, and the
        plane height. A randomization region far outside this is the usual reason
        synthetic wrist frames come out half black.
        """
        used = [self.model.meta['dwells'][i] for i in self.model.meta['used']]
        P = np.array([d['eef_pose'] for d in used], dtype=float)
        cams = np.array([pose6_to_matrix(p) @ self.T_eef_cam for p in P])
        plane = self.model.canvas.plane
        plane_z = float(plane.d / plane.n[2])
        return {
            'flange_low': P[:, :3].min(axis=0), 'flange_high': P[:, :3].max(axis=0),
            'camera_z_low': float(cams[:, 2, 3].min()), 'camera_z_high': float(cams[:, 2, 3].max()),
            'plane_z': plane_z, 'n_dwells': len(P),
        }

    def describe(self) -> str:
        """One-line summary for the processor's log."""
        return (f"wrist mosaic {self.model_dir}: canvas "
                f"{self.model.canvas.width}x{self.model.canvas.height} @ "
                f"{self.model.canvas.res:g} px/m, calibrated at "
                f"{self.model.w}x{self.model.h} -> rendering at "
                f"{self.img_w}x{self.img_h} "
                f"(T_eef_cam: {self.T_eef_cam_source}, "
                f"{'distorted' if self.apply_distortion else 'pinhole'}"
                f"{', gain re-applied' if self.gain is not None else ''})")
