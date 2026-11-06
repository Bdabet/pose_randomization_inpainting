"""
Which flange -> camera transform is right? Decide it against real wrist frames.

The mosaic's self-calibrated `T_eef_cam` and the hand-eye calibration in
phantom/camera/camera_extrinsics_xyzrpy_wrist_example.json disagree -- by 165.6 deg
about the optical axis, and by the sign of the translation. Looking at a composited
twin frame can only say which one *looks* plausible; this says which one actually
reproduces what the camera recorded.

Method: take a robot episode with its wrist video (poses + mp4, the same format
build_mosaic.py consumes), sync it the same way -- dwells where the eef is still,
matched to static runs of video frames, averaged into one low-noise image per pose --
then, for every candidate transform, render the mosaic at that pose and compare with
the real image using evaluate.py's metrics. The candidate that puts the camera where
it really is wins on shift_px / tape agreement; a wrong one is off by a visible shift
or renders a different part of the workspace entirely.

Candidates (-c/--candidates, default all three):
    mosaic   mosaic_meta.json's self-calibrated T_eef_cam
    handeye  mosaic_meta.json's T_eef_cam_handeye (the raw hand-eye calibration)
    json     an {xyz, rpy, degrees} file, as the twin test scripts read it
             (--extrinsics-json, default the repo's wrist example)

Rendering uses the real lens distortion, since the comparison is against raw camera
frames -- unlike the twin's backgrounds, which are pinhole to match MuJoCo.

--twin additionally replays each pose through the MuJoCo twin. Scoring a transform
against the mosaic only tests it against the *real* robot's flange frame; the twin
applies the same transform to robosuite's eef body, so the synthetic wrist view is
faithful only if those two frames agree. The replay closes that gap: the real flange
pose is converted to the grip-site target `TwinRobot.move_to_target_state` expects
(the grip site sits 0.155 m out along the flange's tool axis), the arm is driven
there, and the camera pose the simulation ends up with is compared -- as a transform
and as a rendered image -- with the one the real data implies. A residual of ~0 means
a pose replayed in sim puts the wrist camera where the real robot put it; anything
else is the sim-to-real offset, reported as a rotation and a translation so it can be
folded back into the twin.

Example
  python compare_extrinsics.py --data test_data --episode 0 --camera 0 \
      --model out_ep1_full --out ../../../cmp_extrinsics
"""

import argparse
import json
import pathlib

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from phantom.scene_mosaic import common, sync
from phantom.scene_mosaic import mosaic as mz
from phantom.scene_mosaic.evaluate import compare, tape_mask
from phantom.scene_mosaic.render_view import MosaicModel, label

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_JSON = HERE.parent / 'camera' / 'camera_extrinsics_xyzrpy_wrist_example.json'


def load_xyzrpy(path):
    """{"xyz": [...], "rpy": [...], "degrees": bool} -> 4x4, as the twin test reads it."""
    d = json.load(open(path))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', d['rpy'], degrees=d.get('degrees', True)).as_matrix()
    T[:3, 3] = d['xyz']
    return T


def render_at(model, T_base_cam, apply_gain=True):
    """Mosaic view from an explicit camera pose (BGR uint8, coverage mask)."""
    img, cov = mz.render_view(model.mosaic, model.valid, model.canvas, model.K, model.dist,
                              model.w, model.h, T_base_cam)
    if apply_gain and model.gain is not None:
        img = img * model.gain
    return np.clip(img, 0, 255).astype(np.uint8), cov


def render(model, T_eef_cam, pose6, apply_gain=True):
    """Mosaic view at `pose6` for a given flange -> camera transform (BGR uint8, cov)."""
    return render_at(model, common.pose6_to_matrix(np.asarray(pose6, float)) @ T_eef_cam, apply_gain)


def twin_replay(model, T_eef_cam, scored, out, args):
    """
    Drive the MuJoCo twin to each real pose and compare the camera pose it reaches with
    T_base_flange_real @ T_eef_cam. Returns the per-dwell record written to compare.json.
    """
    # Imported here: robosuite/robomimic are only needed for --twin.
    import sys
    sys.path.insert(0, str(HERE.parent.parent))
    from test_ur7e_single_pose_overlay import (build_camera_params, DEFAULT_CAMERA_EXTRINSICS_PATH,
                                               DEFAULT_CAMERA_INTRINSICS_PATH)
    from phantom.synthetic.wrist_background import WristMosaicBackground
    from phantom.twin_robot import TwinRobot

    w, h = model.w, model.h
    # `bg` only supplies the twin's wrist MujocoCameraParams here; the comparison images
    # are rendered by `render_at` with the real lens distortion, since they are scored
    # against raw camera frames rather than against a MuJoCo render.
    bg = WristMosaicBackground(str(model.dir), w, h, T_eef_cam=T_eef_cam)
    fixed = build_camera_params(DEFAULT_CAMERA_EXTRINSICS_PATH, DEFAULT_CAMERA_INTRINSICS_PATH, w, h)
    robot = TwinRobot('UR7e', 'Robotiq85', fixed, camera_height=h, camera_width=w, render=False,
                      n_steps_short=3, n_steps_long=75, square=False,
                      wrist_camera_params=bg.camera_params(), show_wrist_camera=False)

    if args.twin_dwells:
        want = {int(x) for x in args.twin_dwells.split(',')}
        items = [it for it in scored if it[0] in want]
    else:
        items = scored if not args.twin_max_dwells else scored[:args.twin_max_dwells]
    records, rows = {}, []
    try:
        # `state["ori_xyzw"]` is not the grip site's world orientation: the controller
        # right-multiplies a fixed rotation onto it (get_action_from_ee_pose's -135 deg
        # about z). Measure that constant once from a settled pose and invert it, rather
        # than hard-coding a convention that lives in another file.
        probe_ori = np.array([0.0, 1.0, 0.0, 0.0])
        robot.move_to_target_state({'pos': np.array([0.45, -0.30, 0.30]), 'ori_xyzw': probe_ori,
                                    'gripper_pos': 0.04}, init=True)
        obs = robot.env.env._get_observations()
        C = (Rotation.from_quat(probe_ori).inv() * Rotation.from_quat(obs['robot0_eef_quat'])).as_matrix()

        # The grip site the controller tracks is not the flange the camera hangs off, so
        # measure flange -> grip site from the settled probe pose instead of assuming it
        # (it comes out as a pure 0.155 m offset along the tool axis for a Robotiq85).
        sim = robot.env.env.sim
        bid = sim.model.body_name2id(robot.env.env.robots[0].robot_model.eef_name)
        T_bf_probe = np.eye(4)
        T_bf_probe[:3, :3] = sim.data.body_xmat[bid].reshape(3, 3)
        T_bf_probe[:3, 3] = sim.data.body_xpos[bid] - robot.robot_base_pos
        T_bg_probe = np.eye(4)
        T_bg_probe[:3, :3] = Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
        T_bg_probe[:3, 3] = obs['robot0_eef_pos'] - robot.robot_base_pos
        T_flange_grip = np.linalg.inv(T_bf_probe) @ T_bg_probe
        print(f'[twin] flange -> grip site: trans {np.round(T_flange_grip[:3, 3], 4).tolist()} m, '
              f'rot {np.degrees(Rotation.from_matrix(T_flange_grip[:3, :3]).magnitude()):.2f} deg')

        first = True
        for k, pose6, real in items:
            T_bf_real = common.pose6_to_matrix(np.asarray(pose6, float))
            T_bg_des = T_bf_real @ T_flange_grip
            cmd_ori = Rotation.from_matrix(T_bg_des[:3, :3] @ np.linalg.inv(C)).as_quat()
            res = robot.move_to_target_state({'pos': T_bg_des[:3, 3], 'ori_xyzw': cmd_ori,
                                              'gripper_pos': 0.04}, init=first)
            first = False

            T_bf_sim = np.eye(4)
            T_bf_sim[:3, :3] = sim.data.body_xmat[bid].reshape(3, 3)
            T_bf_sim[:3, 3] = sim.data.body_xpos[bid] - robot.robot_base_pos
            d_flange = np.linalg.inv(T_bf_real) @ T_bf_sim

            T_cam_sim = robot.get_wrist_camera_pose()
            T_cam_real = T_bf_real @ T_eef_cam
            d_cam = np.linalg.inv(T_cam_real) @ T_cam_sim
            rot_deg = float(np.degrees(Rotation.from_matrix(d_cam[:3, :3]).magnitude()))
            trans_mm = float(np.linalg.norm(d_cam[:3, 3]) * 1000)

            # Rendered here rather than through `bg` because the comparison needs the
            # coverage mask, which the twin's background API reduces to a fraction.
            rendered, cov = render_at(model, T_cam_sim)
            scores = compare(real, rendered, cov) if cov.mean() >= args.min_coverage else None

            # The scan was recorded with a bare flange, so the real frames contain no
            # tool at all; the twin always renders one. Report how much of the wrist
            # view it covers -- that part of a synthetic frame has no real counterpart.
            tool_frac = float(((res['wrist_robot_mask'] > 0) | (res['wrist_gripper_mask'] > 0)).mean())
            records[k] = {
                'pos_err': float(res['pos_err']),
                'twin_tool_frac': tool_frac,
                'flange_rot_deg': float(np.degrees(Rotation.from_matrix(d_flange[:3, :3]).magnitude())),
                'flange_trans_mm': float(np.linalg.norm(d_flange[:3, 3]) * 1000),
                'cam_rot_deg': rot_deg, 'cam_trans_mm': trans_mm,
                'coverage': float(cov.mean()), 'scores': scores,
            }
            s = 'coverage too low to score' if scores is None else \
                f'MAE {scores["mae"]:5.2f}  tape@3px ' + \
                ('n/a' if scores['tape_f3px'] is None else f'{scores["tape_f3px"]:.3f}')
            print(f'  dwell {k:2d}  IK err {res["pos_err"] * 1000:6.1f}mm  '
                  f'camera vs real: {rot_deg:6.2f}deg {trans_mm:6.1f}mm  '
                  f'tool {tool_frac * 100:4.1f}% of view  {s}')
            rows.append(np.concatenate([
                label(real, f'real (dwell {k})'),
                label(rendered, f'twin replay: {rot_deg:.1f}deg {trans_mm:.0f}mm off'),
            ], axis=1))
    finally:
        robot.close()

    if rows:
        cv2.imwrite(str(out / 'twin_replay.png'), np.concatenate(rows[:6], axis=0))
    ok = [r for r in records.values() if r['pos_err'] < 0.02]
    summary = {
        'n': len(records), 'n_reached': len(ok),
        'cam_rot_deg_median': float(np.median([r['cam_rot_deg'] for r in ok])) if ok else None,
        'cam_trans_mm_median': float(np.median([r['cam_trans_mm'] for r in ok])) if ok else None,
        'mae_median': float(np.median([r['scores']['mae'] for r in ok if r['scores']])) if ok else None,
        'twin_tool_frac_median': float(np.median([r['twin_tool_frac'] for r in ok])) if ok else None,
    }
    print(f'\n[twin] {summary["n_reached"]}/{summary["n"]} poses reached (IK err < 20mm)')
    if ok:
        print(f'[twin] camera pose vs real data: median {summary["cam_rot_deg_median"]:.2f} deg, '
              f'{summary["cam_trans_mm_median"]:.1f} mm; rendered-vs-real MAE median '
              f'{summary["mae_median"]:.2f}')
        print(f'[twin] twin renders a tool over {summary["twin_tool_frac_median"] * 100:.1f}% of the '
              'wrist view; the scan frames have none')
    print(f'[twin] wrote {out}/twin_replay.png')
    return {'summary': summary, 'per_dwell': records}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='dir with replay_buffer.zarr and videos/')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--camera', type=int, default=0, help='video file videos/<episode>/<camera>.mp4')
    ap.add_argument('--model', required=True, help='mosaic model dir from build_mosaic.py')
    ap.add_argument('--extrinsics-json', default=str(DEFAULT_JSON))
    ap.add_argument('--candidates', '-c', default='mosaic,handeye,json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--time-offset', type=float, default=0.0,
                    help='video clock - robot clock, seconds (see sync.py)')
    ap.add_argument('--max-dwells', type=int, default=0, help='0 = all matched dwells')
    ap.add_argument('--twin', action='store_true',
                    help='also replay each pose through the MuJoCo twin and compare the camera pose it '
                         'reaches with the one the real data implies (needs the robosuite deps)')
    ap.add_argument('--twin-candidate', default='mosaic',
                    help='which candidate transform to mount the twin\'s wrist camera with (--twin)')
    ap.add_argument('--twin-max-dwells', type=int, default=8, help='poses to replay (--twin); 0 = all')
    ap.add_argument('--twin-dwells', default=None,
                    help='comma separated dwell ids to replay (--twin), instead of the first N')
    ap.add_argument('--min-coverage', type=float, default=0.15,
                    help='skip a (dwell, candidate) pair whose render covers less than this '
                         'fraction of the view -- too little overlap to score fairly')
    args = ap.parse_args()

    data = pathlib.Path(args.data)
    video = data / 'videos' / str(args.episode) / f'{args.camera}.mp4'
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = MosaicModel(args.model)
    cands = {}
    for name in args.candidates.split(','):
        name = name.strip()
        if name == 'mosaic':
            cands[name] = model.T_eef_cam
        elif name == 'handeye':
            cands[name] = np.array(model.meta['T_eef_cam_handeye'])
        elif name == 'json':
            cands[name] = load_xyzrpy(args.extrinsics_json)
        else:
            raise SystemExit(f'unknown candidate {name!r}')

    ep, ep_idx = common.load_episode(data / 'replay_buffer.zarr', args.episode)
    n_frames, fps, w, h = common.video_info(video)
    if (w, h) != (model.w, model.h):
        raise SystemExit(f'video is {w}x{h} but the mosaic was calibrated at {model.w}x{model.h}')
    print(f'[cmp] episode {ep_idx}: {len(ep["timestamp"])} pose samples over '
          f'{ep["timestamp"][-1] - ep["timestamp"][0]:.1f}s; video {video.name}: {n_frames} frames '
          f'@ {fps:.0f}fps, {w}x{h}')

    dwells = sync.robot_dwells(ep['timestamp'], ep['robot_eef_pose_vel'])
    segments = sync.video_static_segments(video)
    matches = sync.match_dwells_to_segments(dwells, ep['timestamp'], segments, fps, args.time_offset)
    images = sync.average_segment_frames(video, matches)
    print(f'[cmp] {len(dwells)} robot dwells, {len(segments)} video static segments, '
          f'{len(images)} matched')

    results = {name: {} for name in cands}
    scored = []
    for m in matches:
        k = m['dwell']
        if m['frames'] is None or k not in images:
            continue
        if args.max_dwells and len(results[next(iter(cands))]) >= args.max_dwells:
            break
        a, b = m['pose_range']
        pose6 = ep['robot_eef_pose'][(a + b) // 2]
        real = np.clip(images[k], 0, 255).astype(np.uint8)

        scored.append((k, pose6, real))
        panels = [label(real, f'real (dwell {k}) xyz={np.round(pose6[:3], 3).tolist()}')]
        line = []
        for name, T in cands.items():
            rendered, cov = render(model, T, pose6)
            if cov.mean() < args.min_coverage:
                line.append(f'{name}: coverage {cov.mean() * 100:4.1f}% -- skipped')
                panels.append(label(rendered, f'{name}: coverage {cov.mean() * 100:.0f}% (skipped)'))
                continue
            r = compare(real, rendered, cov)
            results[name][k] = r
            f3 = 'n/a' if r['tape_f3px'] is None else f'{r["tape_f3px"]:.3f}'
            line.append(f'{name}: shift {r["shift_px"]:6.2f}px  tape@3px {f3}  '
                        f'MAE {r["mae"]:5.2f}  cov {r["coverage"] * 100:4.1f}%')
            panels.append(label(rendered, f'{name}: shift {r["shift_px"]:.1f}px, MAE {r["mae"]:.1f}'))
        print(f'  dwell {k:2d}  ' + '\n            '.join(line))
        cv2.imwrite(str(out / f'dwell_{k:03d}.png'), np.concatenate(panels, axis=1))

    summary = {}
    for name, per in results.items():
        vals = list(per.values())
        if not vals:
            summary[name] = {'n': 0}
            continue
        f3 = [v['tape_f3px'] for v in vals if v['tape_f3px'] is not None]
        summary[name] = {
            'n': len(vals),
            'shift_px_median': float(np.median([v['shift_px'] for v in vals])),
            'shift_px_mean': float(np.mean([v['shift_px'] for v in vals])),
            'tape_f3px_mean': float(np.mean(f3)) if f3 else None,
            'mae_mean': float(np.mean([v['mae'] for v in vals])),
            'psnr_mean': float(np.mean([v['psnr'] for v in vals])),
            'coverage_mean': float(np.mean([v['coverage'] for v in vals])),
        }
    print('\n[cmp] summary (lower shift_px / MAE and higher tape_f3px = better)')
    for name, s in summary.items():
        if not s['n']:
            print(f'  {name:8s}: no dwell had enough coverage to score')
            continue
        f3 = 'n/a' if s['tape_f3px_mean'] is None else f'{s["tape_f3px_mean"]:.3f}'
        print(f'  {name:8s}: n={s["n"]:2d}  shift median {s["shift_px_median"]:6.2f}px  '
              f'mean {s["shift_px_mean"]:6.2f}px  tape@3px {f3}  MAE {s["mae_mean"]:5.2f}  '
              f'PSNR {s["psnr_mean"]:5.2f}dB  coverage {s["coverage_mean"] * 100:4.1f}%')
    payload = {'summary': summary, 'per_dwell': results,
               'candidates': {k: v.tolist() for k, v in cands.items()}}
    if args.twin:
        payload['twin'] = twin_replay(model, cands[args.twin_candidate], scored, out, args)
    json.dump(payload, open(out / 'compare.json', 'w'), indent=1)
    print(f'[cmp] wrote {out}/compare.json and per-dwell panels')


if __name__ == '__main__':
    main()
