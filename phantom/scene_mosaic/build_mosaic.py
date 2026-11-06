"""
Build a stitched top-down RGB mosaic of a flat workspace from a wrist-camera scan video
(mp4) + the robot poses in the replay buffer.

Pipeline
  1. sync      : robot dwells <-> static video segments (see sync.py), one averaged
                 image per dwell with an exactly known camera pose
                 T_base_cam = T_base_eef @ T_eef_cam.
  2. calib     : camera yaw about the optical axis + camera-surface distance from the
                 image shifts between overlapping views vs the robot's translation
                 (calib.py); corrects the hand-eye extrinsic's yaw.
     plane     : work-surface plane (height + tilt in the base frame) refined from
                 photometric consistency of the overlapping views (no depth needed),
                 or fixed with --plane-z.
  3. mosaic    : every dwell image is undistorted and projected onto the plane
                 (homography), blended with a "most central view wins" weighting.
  4. outputs   : <out>/mosaic.png, <out>/mosaic_meta.json (plane, canvas, calibration,
                 per-dwell poses), <out>/dwells/<k>.png (the averaged real images,
                 used as ground truth by evaluate.py).

Example
  python build_mosaic.py --data ../latest_scan_data --episode 1 --camera 1 \
      --out out_ep1 --holdout 6
"""

import argparse
import json
import pathlib

import cv2
import numpy as np

from phantom.scene_mosaic import common
from phantom.scene_mosaic import sync
from phantom.scene_mosaic import mosaic as mz
from phantom.scene_mosaic import calib


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = pathlib.Path(__file__).resolve().parent
    ap.add_argument('--data', required=True, help='dir with replay_buffer.zarr and videos/')
    ap.add_argument('--episode', type=int, default=1)
    ap.add_argument('--camera', type=int, default=1, help='video file videos/<episode>/<camera>.mp4')
    ap.add_argument('--intrinsics', default=str(here / 'camera_intrinsics.json'))
    ap.add_argument('--extrinsics', default=str(here / 'camera_extrinsics.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--time-offset', type=float, default=0.0,
                    help='video time (frame/fps) minus pose time, seconds; only used to '
                         'roughly associate dwells with static video segments')
    ap.add_argument('--plane-z', type=float, default=None, help='fix a horizontal plane at this base-frame z instead of estimating it')
    ap.add_argument('--no-self-calib', action='store_true',
                    help='trust the hand-eye extrinsic as is (skip the yaw/distance self-calibration)')
    ap.add_argument('--no-flatfield', action='store_true', help='do not remove camera-fixed shading (vignetting, arm shadow) before stitching')
    ap.add_argument('--no-tilt', action='store_true', help='estimate plane height only (horizontal plane)')
    ap.add_argument('--px-per-m', type=float, default=2000.0)
    ap.add_argument('--margin', type=float, default=0.02)
    ap.add_argument('--sharpness', type=float, default=40.0, help='blend weight exponent; higher = harder seams, less ghosting')
    ap.add_argument('--holdout', type=int, default=0, help='number of dwells to exclude (random, --seed) for evaluation')
    ap.add_argument('--holdout-ids', type=str, default=None, help='comma separated dwell ids to exclude')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / 'dwells').mkdir(parents=True, exist_ok=True)
    data = pathlib.Path(args.data)
    video = data / 'videos' / str(args.episode) / f'{args.camera}.mp4'

    ep, ep_idx = common.load_episode(data / 'replay_buffer.zarr', args.episode)
    K, dist = common.load_intrinsics(args.intrinsics)
    T_eef_cam = common.load_extrinsics(args.extrinsics)
    n_frames, fps, w, h = common.video_info(video)
    print(f'[build] episode {ep_idx}: {len(ep["timestamp"])} pose samples over '
          f'{ep["timestamp"][-1] - ep["timestamp"][0]:.1f}s; video {video.name}: {n_frames} frames '
          f'@ {fps:.0f}fps ({n_frames / fps:.1f}s), {w}x{h}')

    # ---- 1. sync
    dwells = sync.robot_dwells(ep['timestamp'], ep['robot_eef_pose_vel'])
    segments = sync.video_static_segments(video)
    matches = sync.match_dwells_to_segments(dwells, ep['timestamp'], segments, fps, args.time_offset)
    images = sync.average_segment_frames(video, matches)
    print(f'[build] {len(dwells)} robot dwells, {len(segments)} video static segments, '
          f'{len(images)} dwells matched')

    t0 = ep['timestamp'][0]
    dwell_info = []
    for m in matches:
        a, b = m['pose_range']
        pose6 = ep['robot_eef_pose'][(a + b) // 2]
        T_cam = common.pose6_to_matrix(pose6) @ T_eef_cam
        info = {'id': m['dwell'], 'pose_t': [float(ep['timestamp'][a] - t0), float(ep['timestamp'][b - 1] - t0)],
                'eef_pose': pose6.tolist(), 'T_base_cam': T_cam.tolist(),
                'frames': None if m['frames'] is None else list(m['frames'])}
        dwell_info.append(info)
        if m['frames'] is None:
            print(f'  dwell {m["dwell"]:2d} t={info["pose_t"][0]:5.1f}s  NO MATCH')
        else:
            s, e = m['frames']
            print(f'  dwell {m["dwell"]:2d} t={info["pose_t"][0]:5.1f}-{info["pose_t"][1]:4.1f}s  '
                  f'<- frames {s:4d}-{e:4d} (video t={s / fps:5.1f}-{e / fps:4.1f}s, '
                  f'offset {(s + e) / 2 / fps - sum(info["pose_t"]) / 2:+.2f}s)  '
                  f'eef xyz={np.round(pose6[:3], 3).tolist()}')
            cv2.imwrite(str(out / 'dwells' / f'{m["dwell"]:03d}.png'), np.clip(images[m['dwell']], 0, 255).astype(np.uint8))

    ids = [d['id'] for d in dwell_info if d['frames'] is not None]
    if args.holdout_ids:
        holdout = sorted(int(x) for x in args.holdout_ids.split(','))
    elif args.holdout > 0:
        holdout = sorted(np.random.default_rng(args.seed).choice(ids, args.holdout, replace=False).tolist())
    else:
        holdout = []
    # the scan may revisit the same waypoint (e.g. two passes over the same grid): hold
    # out every dwell at a held-out position, otherwise its twin makes the test trivial
    pos = {d['id']: np.array(d['eef_pose'][:3]) for d in dwell_info}
    holdout = sorted({j for i in holdout for j in ids if np.linalg.norm(pos[j] - pos[i]) < 0.005})
    used = [i for i in ids if i not in holdout]
    print(f'[build] using {len(used)} dwells for the mosaic, holding out {holdout}')

    # half resolution "structure" images (red tape + texture, shading removed) are
    # plenty for calibration and plane fitting
    Kh = K.copy() 
    Kh[:2] *= 0.5 # intrinsics for half-res images
    wh = mz.Warper(Kh, dist, w // 2, h // 2)
    structs = [wh.undistort(mz.structure_image(
        cv2.resize(images[i], (w // 2, h // 2), interpolation=cv2.INTER_AREA))) for i in used]

    def cam_poses(T_ec):
        for d in dwell_info:
            d['T_base_cam'] = (common.pose6_to_matrix(np.array(d['eef_pose'])) @ T_ec).tolist()
        return [np.array(dwell_info[i]['T_base_cam']) for i in used]

    # ---- 2a. self-calibration of camera yaw + surface distance
    poses = cam_poses(T_eef_cam)
    calib_report = None
    if args.no_self_calib:
        distance = None
    else:
        yaw, distance, calib_report = calib.self_calibrate(Kh, structs, poses)
        T_eef_cam = T_eef_cam @ calib.rot_z(yaw)
        poses = cam_poses(T_eef_cam)
        # re-run: the corrected extrinsic must leave (almost) no residual yaw
        yaw2, _, _ = calib.self_calibrate(Kh, structs, poses, log=lambda *a: None)
        print(f'[calib] residual yaw after correction: {np.degrees(yaw2):+.3f} deg')

    # ---- 2b. plane
    if args.plane_z is not None:
        plane = common.Plane([0, 0, 1], args.plane_z)
        print(f'[plane] fixed z={args.plane_z}')
    else:
        if distance is None:
            raise SystemExit('--no-self-calib needs --plane-z')
        plane, (zs, costs) = mz.estimate_plane(wh, structs, poses, distance, fit_tilt=not args.no_tilt)
        np.savetxt(out / 'plane_height_search.txt', np.stack([zs, costs], 1), header='z cost')
    cam_heights = [plane.n @ T[:3, 3] - plane.d for T in poses]
    print(f'[plane] camera distance to the surface: {min(cam_heights):.3f}-{max(cam_heights):.3f} m')

    # ---- 3. mosaic
    st = np.concatenate([common.footprint_st(plane, K, dist, w, h, T) for T in poses])
    canvas = common.Canvas.covering(plane, st, args.px_per_m, args.margin)
    print(f'[build] canvas {canvas.width}x{canvas.height} px '
          f'({canvas.width / args.px_per_m:.2f} x {canvas.height / args.px_per_m:.2f} m)')
    warper = mz.Warper(K, dist, w, h)
    gain = None
    if not args.no_flatfield:
        gain = mz.estimate_gain([images[i] for i in used])
        np.save(out / 'gain.npy', gain)
        cv2.imwrite(str(out / 'gain.png'), np.clip(gain / gain.max() * 255, 0, 255).astype(np.uint8))
        print(f'[build] flat-field gain range {gain.min():.2f}-{gain.max():.2f} (saved gain.npy)')
    undist = [warper.undistort(images[i] / gain if gain is not None else images[i]) for i in used]
    mosaic, valid = mz.blend(warper, undist, poses, canvas, args.sharpness)
    mosaic_u8 = np.clip(mosaic, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out / 'mosaic.png'), mosaic_u8)
    cv2.imwrite(str(out / 'mosaic_valid.png'), valid.astype(np.uint8) * 255)

    meta = {
        'canvas': canvas.to_dict(),
        'intrinsics': {'K': K.tolist(), 'dist': dist.tolist(), 'width': w, 'height': h},
        'T_eef_cam': T_eef_cam.tolist(),  # after self-calibration
        'T_eef_cam_handeye': common.load_extrinsics(args.extrinsics).tolist(),
        'self_calibration': calib_report,
        'gain_file': None if gain is None else 'gain.npy',  # raw image = flat image * gain
        'source': {'data': str(data.resolve()), 'episode': ep_idx, 'video': str(video.resolve())},
        'dwells': dwell_info,
        'used': used,
        'holdout': holdout,
    }
    with open(out / 'mosaic_meta.json', 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'[build] wrote {out / "mosaic.png"} ({valid.mean() * 100:.1f}% covered) and mosaic_meta.json')


if __name__ == '__main__':
    main()
