# video_mosaic: stitched workspace photo from a wrist-camera scan video + robot poses

Standalone pipeline (it doesn't use the older `fuse_scan.py` / `build_orthomosaic.py` code). It takes a scan recording
(`replay_buffer.zarr` + `videos/<episode>/<camera>.mp4`), builds one stitched top-down
RGB image of the (flat) workspace, and renders the camera view for any robot pose from it.

```
build_mosaic.py   video + poses -> mosaic.png + mosaic_meta.json   (steps 1-4 below)
render_view.py    mosaic + eef pose (given / random / recorded) -> synthesized camera image
evaluate.py       render at held-out recorded poses and compare with the real images
sync.py           step 1: video <-> pose association
calib.py          step 2: camera yaw + surface distance self-calibration
mosaic.py         steps 3-4: plane refinement, warping/blending, rendering
common.py         IO, pose math, plane / canvas geometry
```

## How it works

1. **Sync (`sync.py`).** The mp4 has no timestamps, and frame index / 30 fps is not a usable
   clock: the stream stalls (the same image repeats for many frames) and then jumps.
   The scan is stop-and-go, though, so the pipeline splits the robot trajectory into **dwells**
   (eef speed ≈ 0, pose constant and exactly known) and the video into **static segments**,
   then aligns them in order with dynamic programming (the offset between them may drift
   slowly). Each dwell's frames are averaged into one low-noise image.
   Episode 1 result: 30/32 dwells matched; offsets stay within ±0.4 s and vary smoothly.
2. **Self-calibration (`calib.py`).** Phase correlation between overlapping dwell images gives
   pixel shifts, and the robot gives the metric translations. One 2×2 linear model fits all
   100 view pairs to 0.66 px RMS. Its scale gives the camera–surface distance (**0.287 m**)
   and its rotation gives the camera's true yaw about the optical axis.
   **`camera_extrinsics.json` is off by +165.6° about the optical axis** for this video
   (≈180° flip plus ~15°). The corrected extrinsic is stored in `mosaic_meta.json`
   (`T_eef_cam`; the original is kept as `T_eef_cam_handeye`).
3. **Plane.** Height and tilt are refined by multi-view photometric consistency, starting
   from the step-2 distance. Result: z ≈ 0.06 m in the base frame, tilt 2–3°.
   No depth is needed. The OAK stereo depth from episode 2 says ~1 m, which is wrong at this range.
4. **Mosaic.** Camera-fixed shading (vignetting, the arm's shadow) is estimated as a
   flat-field gain and divided out. Each image is undistorted and projected onto the plane
   by its exact homography; each canvas pixel comes from the view that sees it most
   centrally.
5. **Render.** For a query pose, every raw camera pixel is undistorted to a ray, intersected
   with the plane and sampled from the mosaic. Then the gain is re-applied, so the output
   looks like a real, distorted frame from that camera.

## Usage

```bash
cd scene_reconstruction/video_mosaic
# build from all dwells
python build_mosaic.py --data ../latest_scan_data --episode 1 --camera 1 --out out_ep1_full
# build with 3 random grid positions held out (every visit to them), then evaluate
python build_mosaic.py --data ../latest_scan_data --episode 1 --camera 1 --out out_ep1_holdout --holdout 3 --seed 1
python evaluate.py --model out_ep1_holdout --out eval_holdout
# random pose (prints the eef pose; yaw ±45°, tilt ≤10°, height ±6 cm by default)
python render_view.py --model out_ep1_full --random --seed 2 --out test_random/random_2.png
# move the robot to that printed pose, grab a frame, and score the rendering against it
python render_view.py --model out_ep1_full --eef-pose X Y Z RX RY RZ --real-image frame.png --out check.png
```

## Results (episode 1, camera 1)

Held-out evaluation: 3 grid positions = 6 dwells, never seen by calibration, plane fit or mosaic.

| metric | mean | worst |
|---|---|---|
| global misalignment (phase corr.) | 0.42 px | 0.53 px |
| red tape agreement within 3 px | 0.85 | 0.72 |
| PSNR | 27.1 dB | 23.2 dB |

Remaining error is a few px of local misalignment near the image borders, plus
view-dependent lighting on the paper.

## Limitations

- Assumes a **flat** workspace. Objects that stick out of the plane will be smeared.
- All scan poses have the same orientation, so the scan cannot calibrate the camera's lever
  arm or tilt relative to the flange; those come from the hand-eye file. Renders at strongly
  rotated or tilted poses therefore rely on that part of the hand-eye calibration being right.
  Check this with `--real-image` at a few such poses.
- Only the parts of the surface that the scan saw are rendered; the rest is black.
