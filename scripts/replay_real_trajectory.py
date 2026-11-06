#!/usr/bin/env python3
"""
Replay a *recorded* robot trajectory through the twin and synthesize both camera views.

The synthetic pipeline invents trajectories; this one takes the poses a real robot
actually drove (a diffusion_policy replay buffer, the same format build_mosaic.py
consumes) and recreates that episode in simulation: the arm follows the recorded
flange poses, the fixed camera's robot is composited onto a clean plate, and the wrist
camera's is composited onto the mosaic rendered at the pose the simulation reached.

It is the end-to-end version of the single-pose check in
phantom/scene_mosaic/compare_extrinsics.py --twin, and it is worth running on a
dataset that also has its wrist video, because then the recreated frames can be put
next to what the camera really recorded (--compare-video). Alignment for that
comparison comes from the scan's dwells: the video clock drifts against the robot
clock, so a single offset fitted over the matched dwells maps pose time to frame
index, which is good enough to look at but not frame-exact.

Frames
------
`robot_eef_pose` in the recording is the flange. The twin tracks the grip site, one
tool length further along the tool axis, and its controller applies a fixed twist to
the commanded orientation; both are measured from the simulation here rather than
assumed (see phantom/synthetic/wrist_region.py for why that matters).

Example
    python scripts/replay_real_trajectory.py --data phantom/scene_mosaic/test_data \
        --episode 0 --camera 0 --model phantom/scene_mosaic/out_ep1_full \
        --background pick_and_place/Empty_workspace.mp4 --out replay_out
"""
import argparse
import json
import os
import pathlib
import sys

import cv2
import mediapy as media
import numpy as np
from scipy.spatial.transform import Rotation

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from phantom.scene_mosaic import common, sync  # noqa: E402
from phantom.synthetic.background import load_background_frames  # noqa: E402
# The frame conventions, the background renderer and the compositing are the synthetic
# processor's own, not re-derived here: this script differs from it only in where the
# poses come from, so anything it shows about the geometry holds for generated episodes
# too (and vice versa).
from phantom.synthetic.twin_frames import (describe_tool_frames, flange_to_command,  # noqa: E402
                                           measure_tool_frames)
from phantom.synthetic.wrist_background import WristMosaicBackground  # noqa: E402
from phantom.twin_robot import TwinRobot  # noqa: E402
from phantom.utils.image_utils import overlay_robot  # noqa: E402


def build_fixed_camera(img_w, img_h, extrinsics, intrinsics):
    """
    The fixed camera, from the same {xyz, rpy} + camera-matrix pair `configs/*.yaml`
    hands the processor (which reads them through BaseProcessor instead, since it has
    a whole cfg to work from).
    """
    from test_ur7e_single_pose_overlay import build_camera_params
    return build_camera_params(extrinsics, intrinsics, img_w, img_h)


def video_offset_from_dwells(data_dir, episode, camera, ep, fps):
    """
    Seconds to add to a pose time to get the video time, fitted over the dwells the
    scan sync matches. Returns None when the video or the match is unusable.
    """
    video = pathlib.Path(data_dir) / 'videos' / str(episode) / f'{camera}.mp4'
    if not video.exists():
        return None, None
    dwells = sync.robot_dwells(ep['timestamp'], ep['robot_eef_pose_vel'])
    segments = sync.video_static_segments(video)
    matches = sync.match_dwells_to_segments(dwells, ep['timestamp'], segments, fps)
    t0 = ep['timestamp'][0]
    offs = []
    for m in matches:
        if m['frames'] is None:
            continue
        a, b = m['pose_range']
        d_mid = (ep['timestamp'][a] + ep['timestamp'][b - 1]) / 2 - t0
        offs.append(sum(m['frames']) / 2 / fps - d_mid)
    if not offs:
        return None, None
    return float(np.median(offs)), len(offs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='dir with replay_buffer.zarr and videos/')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--model', required=True, help='mosaic model dir (wrist background)')
    ap.add_argument('--background', default=None,
                    help='clean-plate video for the fixed camera; without it only the wrist view is written')
    ap.add_argument('--camera-extrinsics', default=str(REPO / 'phantom/camera/camera_extrinsics_xyzrpy_example.json'))
    ap.add_argument('--camera-intrinsics', default=str(REPO / 'phantom/camera/camera_intrinsics_example.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--robot', default='UR7e')
    ap.add_argument('--gripper', default='Robotiq85')
    ap.add_argument('--resolution', type=int, nargs=2, default=(1280, 720), metavar=('W', 'H'))
    ap.add_argument('--square', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--stride', type=int, default=1, help='replay every Nth recorded sample')
    ap.add_argument('--max-frames', type=int, default=0, help='0 = the whole episode')
    ap.add_argument('--ik-threshold', type=float, default=0.02,
                    help='flag frames whose grip site ends up further than this from the recorded pose (m)')
    ap.add_argument('--compare-video', action='store_true',
                    help='also write a side-by-side of each recreated wrist frame and the real one')
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img_w, img_h = args.resolution

    ep, ep_idx = common.load_episode(pathlib.Path(args.data) / 'replay_buffer.zarr', args.episode)
    poses = ep['robot_eef_pose'][::args.stride]
    times = ep['timestamp'][::args.stride] - ep['timestamp'][0]
    if args.max_frames:
        poses, times = poses[:args.max_frames], times[:args.max_frames]
    fps = len(times) / max(times[-1], 1e-6)
    print(f'[replay] episode {ep_idx}: {len(poses)} poses over {times[-1]:.1f}s -> {fps:.1f} fps')

    wrist_bg = WristMosaicBackground(args.model, img_w, img_h, square=args.square)
    print('[replay] ' + wrist_bg.describe())

    bg_frames = None
    if args.background:
        bg_frames = load_background_frames(args.background, 2000, img_w, img_h, args.square)
        print(f'[replay] clean plate: {len(bg_frames)} frames from {args.background}')

    fixed = build_fixed_camera(img_w, img_h, args.camera_extrinsics, args.camera_intrinsics)
    robot = TwinRobot(args.robot, args.gripper, fixed, camera_height=img_h, camera_width=img_w,
                      render=False, n_steps_short=3, n_steps_long=75, square=args.square,
                      wrist_camera_params=wrist_bg.camera_params(), show_wrist_camera=False)

    real_frames = None
    if args.compare_video:
        offset, n_match = video_offset_from_dwells(args.data, args.episode, args.camera, ep,
                                                   common.video_info(pathlib.Path(args.data) / 'videos' /
                                                                     str(args.episode) / f'{args.camera}.mp4')[1])
        if offset is None:
            print('[replay] no usable video/dwell match; skipping the real-frame comparison')
        else:
            v_fps = common.video_info(pathlib.Path(args.data) / 'videos' / str(args.episode) / f'{args.camera}.mp4')[1]
            want = {int(round((t + offset) * v_fps)): i for i, t in enumerate(times)}
            print(f'[replay] video offset {offset:+.2f}s from {n_match} matched dwells; '
                  f'decoding {len(want)} real frames')
            real_frames = {}
            for idx, frame in common.iter_video(pathlib.Path(args.data) / 'videos' / str(args.episode) /
                                                f'{args.camera}.mp4', indices=want.keys()):
                real_frames[want[idx]] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    T_fg, C = measure_tool_frames(robot)
    print(f'[replay] {describe_tool_frames(T_fg, C)}')

    wrist_imgs, fixed_imgs, cmp_imgs = [], [], []
    ik_errs, cam_errs, covs = [], [], []
    try:
        for i, pose6 in enumerate(poses):
            pos, ori = flange_to_command(pose6, T_fg, C)
            res = robot.move_to_target_state({'pos': pos, 'ori_xyzw': ori, 'gripper_pos': 0.04},
                                             init=(i == 0))
            ik_errs.append(res['pos_err'])

            T_cam_sim = robot.get_wrist_camera_pose()
            T_cam_rec = common.pose6_to_matrix(np.asarray(pose6, float)) @ wrist_bg.T_eef_cam
            cam_errs.append(np.linalg.norm((np.linalg.inv(T_cam_rec) @ T_cam_sim)[:3, 3]))

            bgr, cov = wrist_bg.render_at_camera(T_cam_sim)
            covs.append(cov)
            wrist = overlay_robot(bgr, {'rgb_img': res['wrist_rgb_img'],
                                        'robot_mask': res['wrist_robot_mask'],
                                        'gripper_mask': res['wrist_gripper_mask']})
            wrist_imgs.append(wrist)
            if bg_frames is not None:
                fixed_imgs.append(overlay_robot(bg_frames[i % len(bg_frames)], res))
            if real_frames is not None and i in real_frames:
                cmp_imgs.append(np.concatenate([real_frames[i], wrist], axis=1))
            if (i + 1) % 50 == 0:
                print(f'  {i + 1}/{len(poses)} frames, IK err {np.mean(ik_errs) * 1000:.1f} mm mean, '
                      f'coverage {np.mean(covs) * 100:.0f}% mean')
    finally:
        robot.close()

    media.write_video(str(out / 'wrist_replay.mp4'), wrist_imgs, fps=fps)
    if fixed_imgs:
        media.write_video(str(out / 'fixed_replay.mp4'), fixed_imgs, fps=fps)
    if cmp_imgs:
        media.write_video(str(out / 'wrist_vs_real.mp4'), cmp_imgs, fps=fps)

    ik, cam, cov = np.array(ik_errs), np.array(cam_errs), np.array(covs)
    stats = {
        'episode': ep_idx, 'frames': len(poses), 'fps': fps,
        'ik_err_mm': {'mean': float(ik.mean() * 1000), 'max': float(ik.max() * 1000),
                      'over_threshold': int((ik > args.ik_threshold).sum())},
        'camera_vs_recorded_mm': {'mean': float(cam.mean() * 1000), 'max': float(cam.max() * 1000)},
        'wrist_coverage': {'mean': float(cov.mean()), 'min': float(cov.min())},
    }
    json.dump(stats, open(out / 'replay_stats.json', 'w'), indent=1)
    print(f"\n[replay] IK error: mean {stats['ik_err_mm']['mean']:.1f} mm, max "
          f"{stats['ik_err_mm']['max']:.1f} mm ({stats['ik_err_mm']['over_threshold']} frames over "
          f"{args.ik_threshold * 1000:.0f} mm)")
    print(f"[replay] wrist camera vs recorded pose: mean "
          f"{stats['camera_vs_recorded_mm']['mean']:.1f} mm, max {stats['camera_vs_recorded_mm']['max']:.1f} mm")
    print(f"[replay] wrist mosaic coverage: mean {cov.mean() * 100:.1f}%, worst {cov.min() * 100:.1f}%")
    print(f'[replay] wrote {out}/wrist_replay.mp4'
          + (', fixed_replay.mp4' if fixed_imgs else '')
          + (', wrist_vs_real.mp4' if cmp_imgs else '') + ', replay_stats.json')


if __name__ == '__main__':
    main()
