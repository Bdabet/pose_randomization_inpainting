# phantom/scene_mosaic

Vendored copy of `scene_reconstruction/video_mosaic` from
[diffusion_policy_THI](https://github.com/Bdabet/diffusion_policy_THI/tree/scene-reconstruction-scan/scene_reconstruction),
which builds one stitched top-down RGB image of the flat workspace from a wrist-camera
scan recording and renders the wrist camera's view of it at an arbitrary end-effector
pose. `README_upstream.md` is the original README (how the pipeline works, results,
limitations) and still describes this code.

It is used here as the **wrist-camera background source** for the synthetic
pick-and-place pipeline: where the external camera draws its background from random
frames of a clean-plate video, the wrist camera's background is rendered from the
mosaic at the robot's current pose (`phantom/synthetic/wrist_background.py`).

## Local changes relative to upstream

Kept deliberately small so the files stay diffable against upstream:

- imports are package-absolute (`from phantom.scene_mosaic import common`) instead of
  the flat `import common`, so the modules work as a package rather than only with
  `video_mosaic` as the working directory;
- `mosaic.render_view` takes an optional precomputed `rays` argument, so rendering
  many poses for one camera does not re-undistort the full pixel grid every frame
  (at 1920x1080 that dominated the per-frame cost);
- `build_mosaic.py`'s `--intrinsics` / `--extrinsics` defaults point next to the code
  rather than one directory up, and the two calibration files they name
  (`camera_intrinsics.json`, the scan camera's calibration, and
  `camera_extrinsics.json`, the raw hand-eye `fTc`) are vendored alongside them;
- `__init__.py` and this README are new.

Everything else, including the CLIs, is upstream code. Run them as modules from the
repo root:

```bash
python -m phantom.scene_mosaic.build_mosaic --data <scan_data> --episode 1 --camera 1 --out phantom/scene_mosaic/out_ep1_full
python -m phantom.scene_mosaic.render_view --model phantom/scene_mosaic/out_ep1_full --random --seed 2 --out /tmp/random.png
python -m phantom.scene_mosaic.evaluate --model phantom/scene_mosaic/out_ep1_full --out /tmp/eval
```

## Model directory

`build_mosaic.py` writes the directory the synthetic pipeline consumes
(`synthetic.wrist.mosaic_model`):

| file | role |
|---|---|
| `mosaic.png` | the stitched top-down workspace image |
| `mosaic_valid.png` | which canvas pixels the scan actually saw |
| `mosaic_meta.json` | canvas/plane geometry, camera intrinsics, `T_eef_cam`, dwell poses |
| `gain.npy` | camera-fixed shading (vignetting + the arm's shadow) |

`mosaic.png`, `mosaic_valid.png` and the per-dwell images are **not** in the upstream
repo's git history (its `data/` is ignored), so a model directory has to be copied
over from wherever the scan was built, or rebuilt here from the scan recording.

## Calibration: `mosaic_meta.json` is the source of truth

`mosaic_meta.json` holds two flange->camera extrinsics: `T_eef_cam_handeye`, straight
from the hand-eye calibration file, and `T_eef_cam`, the one the scan's own
self-calibration corrected (upstream step 2) and which its held-out evaluation
validated to 0.42 px. For this scan they differ by **165.6 deg about the optical
axis**, so the wrist camera in MuJoCo is configured from `mosaic_meta.json` rather
than from `phantom/camera/camera_*_wrist_example.json`; see
`phantom/synthetic/wrist_background.py`. The files under `phantom/camera/` are not
consulted by the synthetic pipeline and currently disagree with this scan in both
rotation and intrinsics.

The lever arm -- the translation part of `T_eef_cam` -- is the one piece the scan does
not *correct*, since every scan pose shares one orientation. It does constrain it,
though, along the plane normal: step 2 measures the camera-surface distance from
parallax (image shifts against the robot's own metric translations), which never uses
the lever arm. For this scan that distance is 0.2866 m, and the hand-eye lever arm
puts the camera 0.2853 m above the fitted plane -- agreement to **1.3 mm**. Negating
it, as `phantom/camera/camera_extrinsics_xyzrpy_wrist_example.json` does, puts the
camera at 0.1452 m, 141 mm out. The hand-eye lever arm is therefore the default
(`synthetic.wrist.mount_translation: null`).

What stays unconstrained is the lever arm's *in-plane* part: an error there slides the
whole mosaic sideways without making any view disagree with another. Only real wrist
frames settle that -- `python -m phantom.scene_mosaic.render_view --model <dir>
--eef-pose <...> --real-image <frame.png>` scores a rendering against one.

## Frames

Three frames are easy to confuse, and two of them are *not* interchangeable:

| frame | what it is |
|---|---|
| flange | the real robot's `robot_eef_pose`, and what `T_eef_cam` is relative to. In the twin this is the eef **body**, `robot0_right_hand`. |
| grip site | what the twin reports as `robot0_eef_pos` / `robot0_eef_quat`, and what the synthetic trajectory commands. For a Robotiq85 it is 0.155 m out along the tool axis from the flange and rotated 45 deg about it. |
| camera | `T_base_cam = T_base_flange @ T_eef_cam`, OpenCV optical convention. |

MuJoCo attaches the wrist camera to the flange body and applies the `pos`/`quat` it is
given literally -- it does not flip anything. Because of that gap between the flange
and the grip site, though, the pipeline does not reconstruct the camera pose from the
pose the twin reports: it reads the camera's actual pose back out of the simulation
(`TwinRobot.get_wrist_camera_pose`) and renders the background from that, so the
background and the composited robot are the same camera whatever sits in between.
