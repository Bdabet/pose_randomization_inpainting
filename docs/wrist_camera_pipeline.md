# Wrist-camera synthetic data: scan to episodes

How to go from an empty workspace to synthetic training episodes that have a *moving*
wrist-camera view, and how to check at each step that what you generated matches what
the real robot would see.

The external camera is easy: it never moves, so one clean-plate video of the empty
workspace is a valid background for any robot pose. The wrist camera is not. It is
bolted to the flange, so its background changes with every pose, and no single
recording can stand in for it. Instead the workspace is photographed once as a
**scene mosaic** -- a stitched top-down image of the flat table -- and the wrist
background is *rendered* from it at whatever pose the robot is in.

Everything below assumes the `phantom` conda environment (see `install.sh`) and that
`process_data.py` is run from `phantom/`, which is what its config paths are relative
to. Other commands are run from the repository root.

---

## 0. What you need

| | |
|---|---|
| A flat workspace | The mosaic is a plane projection. Texture helps: the tape markings in the sample data are what the alignment checks key on. |
| A wrist camera, calibrated | Intrinsics + distortion (`phantom/scene_mosaic/camera_intrinsics.json`) and a hand-eye `fTc` (`phantom/scene_mosaic/camera_extrinsics.json`). The hand-eye value only has to be close; step 2 corrects its rotation. |
| A clean plate for the external camera | A video of the empty workspace from the fixed camera, e.g. `pick_and_place/Empty_workspace.mp4`. |
| The external camera's calibration | `phantom/camera/camera_extrinsics_xyzrpy_example.json` + `camera_intrinsics_example.json`, as `configs/ur7e.yaml` points at them. |

---

## 1. Record a scan

Drive the arm over the workspace in a **stop-and-go** pattern, recording the wrist
video and the robot poses into a diffusion_policy replay buffer
(`replay_buffer.zarr` + `videos/<episode>/<camera>.mp4`). `phantom/scene_mosaic/test_data`
is an example of the layout.

What the sync in `phantom/scene_mosaic/sync.py` needs:

- **Dwell at each waypoint for about a second.** The video has no per-frame
  timestamps and its clock drifts against the robot's, so frames are matched to poses
  by pairing still stretches of video with still stretches of the trajectory. Keep
  moving and nothing matches.
- **Keep one orientation for the whole scan.** Everything downstream assumes it.
- **Keep one height.** You can vary it, but see the limitation at the end.
- **Overlap the views.** Neighbouring dwells should share a good fraction of their
  field of view.

Cover the area you intend to randomize over, plus margin. The scan's footprint is a
hard limit on the usable workspace -- step 5 will shrink your randomization region to
fit it, not the other way round.

---

## 2. Build the mosaic

```bash
cd phantom/scene_mosaic
python build_mosaic.py --data <scan_dir> --episode 1 --camera 1 --out out_ep1_full
```

This syncs dwells to video segments, averages each dwell's frames into one low-noise
image, estimates the table plane, self-calibrates the flange->camera rotation, removes
camera-fixed shading, and stitches everything into `out_ep1_full/`:

```
mosaic.png  mosaic_valid.png  mosaic_meta.json  gain.npy  dwells/
```

`mosaic_meta.json` is the file the rest of the pipeline reads. It holds the
intrinsics, the canvas geometry, and two extrinsics: `T_eef_cam_handeye` (what you
supplied) and `T_eef_cam` (what the scan's own alignment says). **They differ** -- by
165.6 deg about the optical axis for the sample scan -- and `T_eef_cam` is the one
everything uses. See `phantom/scene_mosaic/README.md`.

Hold out a few dwells (`--holdout N`) and `evaluate.py` will score the mosaic against
the real images at poses it never saw.

---

## 3. Confirm the extrinsic against real frames

Before generating anything, check that the mosaic places the camera where the camera
really is. This needs a *second* recording with its own wrist video -- any episode in
the same workspace will do.

```bash
cd phantom/scene_mosaic
python compare_extrinsics.py --data test_data --episode 0 --camera 0 \
    --model out_ep1_full --out /tmp/cmp
```

For every dwell it renders the mosaic under each candidate transform and compares with
the real frame. On the sample data:

| candidate | tape agreement | MAE | PSNR |
|---|---|---|---|
| mosaic `T_eef_cam` | **0.282** | **4.75** | **30.2 dB** |
| hand-eye | 0.000 | 12.35 | 21.0 dB |

Read `tape_f3px` and `mae`, not `shift_px`: phase correlation returns a confident-looking
zero shift when the render is blank, which is exactly what a wrong candidate produces.

`--twin` goes further and replays each pose through MuJoCo, comparing the camera pose
the simulation reaches with the one the recording implies. On the sample data the
rotation residual is 0.00-0.01 deg and the translation residual equals the controller's
tracking error pose for pose -- i.e. the twin's flange frame agrees with the robot's.
If that ever stops being true, nothing downstream is trustworthy and this is where you
find out.

---

## 4. Option A -- replay a recorded trajectory

The best sanity check, because the answer is on film:

```bash
python scripts/replay_real_trajectory.py \
    --data phantom/scene_mosaic/test_data --episode 0 --camera 0 \
    --model phantom/scene_mosaic/out_ep1_full \
    --background pick_and_place/Empty_workspace.mp4 \
    --out data/processed_data/replay_test_data --compare-video
```

It drives the twin along the recorded flange poses and composites both cameras, writing
`wrist_replay.mp4`, `fixed_replay.mp4`, `replay_stats.json`, and -- with
`--compare-video` -- `wrist_vs_real.mp4`, the recreated wrist frame beside the real one
(aligned by apython scripts/replay_real_trajectory.py \
    --data phantom/scene_mosaic/test_data --episode 0 --camera 0 \
    --model phantom/scene_mosaic/out_ep1_full \
    --background pick_and_place/Empty_workspace.mp4 \
    --out replay_out --compare-video
 single offset fitted over the matched dwells, so close but not
frame-exact).

What to look at in `replay_stats.json`:

- `ik_err_mm` -- how well the controller tracked the recorded poses. On the sample
  episode, 1.1 mm mean / 10.7 mm max over 535 poses.
- `camera_vs_recorded_mm` -- the wrist camera versus where the recording puts it. This
  should **equal** the IK error; anything more is a frame error, not a tracking error.
- `wrist_coverage` -- how much of the view the mosaic covers.

Use `--stride N --max-frames N` for a quick pass, but note a large stride inflates the
IK error (each step becomes a bigger jump): the sample episode shows 7.4 mm mean at
`--stride 10` against 1.1 mm at full rate.

---

## 5. Option B -- generate synthetic episodes

```bash
cd phantom
python process_data.py --config-name=ur7e mode=synthetic_pickplace \
    demo_name=<name> synthetic.num_episodes=10
```

Writes a diffusion_policy replay buffer and per-episode QA videos for both cameras to
`data/processed_data/<name>/synthetic_processor/`.

### The preflight

Before any episode is generated, the processor checks the randomization region against
the scan and prints what it found:

```
scan covered flange xyz [0.219,-0.386,0.25]..[0.45,-0.206,0.25] (30 dwells),
  camera at z 0.318 over a plane at z 0.062; tool length 0.155,
  commanded-vs-settled twist 45.0 deg
scan orientation corresponds to trajectory yaw +84.3 deg (residual 0.0 deg);
  configured yaw band is +0.0 +- 60 deg
configured region: xy [0.35,0.65] x [-0.25,0.25] ... -> worst 0.0%, mean 31.0%
region_from_mosaic: xy [0.261,0.408] x [-0.353,-0.239] at z 0.095,
  yaw +84.3 +- 60 deg -> worst 90.0%, mean 99.9%
```

This exists because the failure it catches is silent: episodes generate fine, and the
wrist frames are simply part fill colour. Three traps, all visible above:

- **`region_low/high` are grip-site coordinates; the scan is in flange coordinates**,
  one tool length apart (0.155 m for a Robotiq85). A region that looks like it sits on
  the scan can be a tool length above it.
- **Yaw matters.** The camera turns with the flange, so a yaw band centred away from
  the orientation the scan was flown at points the camera off the scanned patch no
  matter how small the region is.
- **Height matters more than width.** The camera's footprint grows as it rises, so
  lifting runs off the scanned patch long before translating does.

With `synthetic.wrist.region_from_mosaic: true` (the default in `configs/ur7e.yaml`)
the region, lift height, place position and yaw centre are all derived from the scan.
With it off, a region below `min_region_coverage` is refused, and the error names the
values to use instead.

### Settings worth knowing

| key | |
|---|---|
| `synthetic.wrist.region_from_mosaic` | derive the randomization envelope from the scan |
| `synthetic.wrist.min_region_coverage` | coverage floor the envelope must hold (0.9) |
| `synthetic.wrist.apply_distortion` | `false` renders a pinhole background, matching MuJoCo's camera, so the composited robot is geometrically consistent; real wrist frames then have to be undistorted at inference time |
| `synthetic.wrist.mount_translation` | override the camera's lever arm, which the scan cannot measure |
| `synthetic.eef_pose_frame` | `flange` (default) writes `robot_eef_pose` the way the real robot records it; `grip` reproduces datasets made before this was configurable |

---

## How the pieces fit

```
scan recording ──build_mosaic──> mosaic model ──compare_extrinsics──> trust it?
                                      │
                      ┌───────────────┴────────────────┐
                      │                                │
              recorded poses                    random trajectories
      scripts/replay_real_trajectory.py   processors/synthetic_processor.py
                      │                                │
                      └──────────── both go through ───┘
                         phantom/synthetic/twin_frames.py   (flange <-> grip site <-> camera)
                         phantom/synthetic/wrist_background.py (mosaic -> background)
                         phantom/utils/image_utils.overlay_robot
```

The replay script and the processor differ only in where the poses come from. Both take
the camera pose from `TwinRobot.get_wrist_camera_pose()` -- read back out of the
simulation rather than recomputed from the commanded pose -- so the background is
always rendered for exactly the camera MuJoCo rendered the robot from.

| file | |
|---|---|
| `phantom/scene_mosaic/` | mosaic building, rendering, evaluation, extrinsic comparison |
| `phantom/synthetic/twin_frames.py` | flange / grip site / commanded orientation, measured from the sim |
| `phantom/synthetic/wrist_background.py` | mosaic -> wrist background and camera params |
| `phantom/synthetic/wrist_region.py` | coverage prediction and region fitting |
| `phantom/synthetic/trajectory.py` | pick-and-place trajectory generation |
| `phantom/processors/synthetic_processor.py` | the generation pipeline |
| `scripts/replay_real_trajectory.py` | replay a recorded trajectory |

---

## Limitations

- **A single-height scan buys almost no lift.** On the sample mosaic, holding 90%
  coverage leaves a 0 mm lift, so the trajectories slide in the picking plane. Dropping
  `min_region_coverage` to 0.75 buys about 10 cm. Scanning at two or three heights is
  the real fix.
- **The scan's footprint caps the randomization region.** The sample scan covers
  23 x 18 cm and yields a 15 x 11 cm region at 90% coverage.
- **Scan with the tool you will deploy with.** The sample scan was flown with a bare
  flange, so the generated frames contain a Robotiq85 (about 5% of the view) that has
  no counterpart in the real plate -- and its occlusion and shadow are missing from the
  mosaic.
- **No lens distortion in the composite** by default, so synthetic wrist frames are
  pinhole and real ones have to be undistorted to match.
- **The coverage floor is approximate.** It is predicted over a probe grid, while an
  episode interpolates between waypoints; the per-episode coverage printed during
  generation is the authority.
