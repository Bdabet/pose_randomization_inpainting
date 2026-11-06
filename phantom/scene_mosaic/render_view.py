"""
Render the camera view at a robot pose from a mosaic built by build_mosaic.py.

The pose is a robot eef pose [x, y, z, rx, ry, rz] (base frame, meters + rotation
vector, same convention as replay_buffer.zarr 'robot_eef_pose'); the camera pose is
T_base_eef @ T_eef_cam (self-calibrated extrinsic stored in mosaic_meta.json). The
output is a raw-camera-like image: lens distortion and the camera-fixed shading
(flat-field gain) are re-applied, so it is directly comparable to recorded frames.

Examples
  # a random pose around the scanned area (yaw / tilt / height perturbed)
  python render_view.py --model out_ep1 --random --seed 3 --out random3.png
  # an explicit eef pose
  python render_view.py --model out_ep1 --eef-pose 0.33 -0.30 0.25 -1.057 -2.958 0 --out view.png
  # check a random pose for real: move the robot to the printed eef pose, grab a frame,
  python render_view.py --model out_ep1 --eef-pose <printed pose> --real-image frame.png --out check.png
  # a recorded dwell (also shows the real image next to the rendering)
  python render_view.py --model out_ep1 --dwell 12 --out dwell12.png
"""

import argparse
import json
import pathlib

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from phantom.scene_mosaic import common
from phantom.scene_mosaic import mosaic as mz


class MosaicModel:
    def __init__(self, model_dir):
        d = pathlib.Path(model_dir)
        self.dir = d
        self.meta = json.load(open(d / 'mosaic_meta.json'))
        self.canvas = common.Canvas.from_dict(self.meta['canvas'])
        intr = self.meta['intrinsics']
        self.K = np.array(intr['K'])
        self.dist = np.array(intr['dist'])
        self.w, self.h = intr['width'], intr['height']
        self.T_eef_cam = np.array(self.meta['T_eef_cam'])
        self.mosaic = cv2.imread(str(d / 'mosaic.png')).astype(np.float32)
        self.valid = cv2.imread(str(d / 'mosaic_valid.png'), cv2.IMREAD_GRAYSCALE) > 127
        g = self.meta.get('gain_file')
        self.gain = np.load(d / g) if g else None

    def cam_pose(self, eef_pose6):
        return common.pose6_to_matrix(np.asarray(eef_pose6, float)) @ self.T_eef_cam

    def render(self, eef_pose6, apply_gain=True):
        """Returns (BGR uint8 image, coverage mask)."""
        img, cov = mz.render_view(self.mosaic, self.valid, self.canvas, self.K, self.dist,
                                  self.w, self.h, self.cam_pose(eef_pose6))
        if apply_gain and self.gain is not None:
            img = img * self.gain
        return np.clip(img, 0, 255).astype(np.uint8), cov

    def footprint_px(self, eef_pose6):
        st = common.footprint_st(self.canvas.plane, self.K, self.dist, self.w, self.h,
                                 self.cam_pose(eef_pose6), n_edge=32)
        return self.canvas.st_to_px(st)

    def dwell(self, k):
        return self.meta['dwells'][k]

    def real_image(self, k):
        p = self.dir / 'dwells' / f'{k:03d}.png'
        return cv2.imread(str(p)) if p.exists() else None


def random_eef_pose(model, rng, max_yaw_deg=45.0, max_tilt_deg=10.0, dz=0.06, xy_margin=0.03):
    """Random pose around the scan: xy uniform over the scanned eef positions (+margin),
    z = scan height +- dz, orientation = scan orientation rotated by a random yaw about
    base z and a random tilt about a random horizontal axis."""
    used = [model.dwell(i) for i in model.meta['used']]
    P = np.array([d['eef_pose'] for d in used])
    lo, hi = P[:, :3].min(0), P[:, :3].max(0)
    xyz = np.array([rng.uniform(lo[0] - xy_margin, hi[0] + xy_margin),
                    rng.uniform(lo[1] - xy_margin, hi[1] + xy_margin),
                    rng.uniform(lo[2] - dz, hi[2] + dz)])
    R0 = Rotation.from_rotvec(P[0, 3:])
    yaw = Rotation.from_euler('z', rng.uniform(-max_yaw_deg, max_yaw_deg), degrees=True)
    axis = rng.normal(size=2)
    axis = np.append(axis / np.linalg.norm(axis), 0.0)
    tilt = Rotation.from_rotvec(axis * np.radians(rng.uniform(0, max_tilt_deg)))
    R = tilt * yaw * R0
    return np.concatenate([xyz, R.as_rotvec()])


def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def mosaic_with_footprint(model, eef_pose6, height):
    m = model.mosaic.clip(0, 255).astype(np.uint8).copy()
    for i in model.meta['used']:  # faint outlines of the scan views
        fp = model.footprint_px(model.dwell(i)['eef_pose']).astype(np.int32)
        cv2.polylines(m, [fp], True, (200, 200, 200), 1, cv2.LINE_AA)
    fp = model.footprint_px(eef_pose6)
    cv2.polylines(m, [fp.astype(np.int32)], True, (255, 160, 0), 5, cv2.LINE_AA)
    # mark the image's top-left corner so orientation is visible
    cv2.circle(m, tuple(int(v) for v in fp[0]), 14, (255, 160, 0), -1)
    s = height / m.shape[0]
    return cv2.resize(m, (int(m.shape[1] * s), height), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='build_mosaic.py output dir')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--eef-pose', type=float, nargs=6, metavar=('X', 'Y', 'Z', 'RX', 'RY', 'RZ'))
    g.add_argument('--random', action='store_true')
    g.add_argument('--dwell', type=int, help='render at a recorded dwell pose and compare to the real image')
    ap.add_argument('--real-image', default=None,
                    help='a real raw camera frame captured at this pose (e.g. after moving the robot '
                         'to the printed random pose): shown next to the rendering and scored')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--max-yaw', type=float, default=45.0)
    ap.add_argument('--max-tilt', type=float, default=10.0)
    ap.add_argument('--out', required=True, help='output panel .png (the bare rendering is saved as <out>_view.png)')
    args = ap.parse_args()

    model = MosaicModel(args.model)
    real = None
    if args.dwell is not None:
        pose = np.array(model.dwell(args.dwell)['eef_pose'])
        real = model.real_image(args.dwell)
        tag = f'dwell {args.dwell}' + (' (held out)' if args.dwell in model.meta['holdout'] else ' (used in mosaic)')
    elif args.random:
        pose = random_eef_pose(model, np.random.default_rng(args.seed), args.max_yaw, args.max_tilt)
        tag = f'random pose (seed {args.seed})'
    else:
        pose = np.array(args.eef_pose)
        tag = 'given pose'

    if args.real_image:
        real = cv2.imread(args.real_image)
    view, cov = model.render(pose)
    print(f'[render] {tag}: eef pose {np.round(pose, 4).tolist()}')
    print(f'[render] {cov.mean() * 100:.1f}% of the view is covered by the mosaic')

    out = pathlib.Path(args.out)
    cv2.imwrite(str(out.with_name(out.stem + '_view.png')), view)
    h = model.h
    panels = [label(mosaic_with_footprint(model, pose, h), 'mosaic + view footprint'),
              label(view, f'rendered: {tag}')]
    if real is not None:
        from phantom.scene_mosaic.evaluate import compare, overlay
        r = compare(real, view, cov)
        print(f'[render] vs real: shift {r["shift_px"]:.2f}px, tape@3px {r["tape_f3px"]}, '
              f'MAE {r["mae"]:.2f}, PSNR {r["psnr"]:.2f}dB')
        panels += [label(real, 'real camera image'),
                   label(overlay(real, view, cov), 'tape: real=magenta rendered=green')]
    cv2.imwrite(str(out), np.concatenate(panels, axis=1))
    print(f'[render] wrote {out}')


if __name__ == '__main__':
    main()
