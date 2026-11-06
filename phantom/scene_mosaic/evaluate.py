"""
Quantitative check of a mosaic: render the view at robot poses whose real images were
NOT used to build the mosaic (build_mosaic.py --holdout N), and compare with the real
camera image recorded at that pose.

Metrics (inside the part of the view covered by the mosaic):
  shift_px   residual misalignment real vs rendered (phase correlation on the tape /
             texture structure image, full resolution). The main geometric metric.
  tape_iou   IoU of the red tape masks (harsh: the tape is only ~10px wide).
  tape_f@3px fraction of tape pixels (real and rendered) lying within 3px of tape in
             the other image -- a tolerant symmetric agreement score.
  mae        mean absolute color error (0..255).
  psnr       dB.

Writes <out>/eval.json and one panel per pose: real | rendered | tape overlay
(magenta = real only, green = rendered only, white = both).

Example
  python evaluate.py --model out_ep1_holdout --out eval_holdout
"""

import argparse
import json
import pathlib

import cv2
import numpy as np

from phantom.scene_mosaic import mosaic as mz
from phantom.scene_mosaic.render_view import MosaicModel, label


def tape_mask(img):
    f = img.astype(np.float32)
    return (f[..., 2] - 0.5 * (f[..., 0] + f[..., 1])) > 40


def compare(real, rendered, cov):
    m = cov & (real.sum(-1) > 0)
    win = cv2.createHanningWindow(real.shape[1::-1], cv2.CV_64F)
    sr = mz.structure_image(real.astype(np.float32)) * m
    sv = mz.structure_image(rendered.astype(np.float32)) * m
    (dx, dy), resp = cv2.phaseCorrelate(sr.astype(np.float64), sv.astype(np.float64), win)
    tr, tv = tape_mask(real) & m, tape_mask(rendered) & m
    union = (tr | tv).sum()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    near_r = cv2.dilate(tr.astype(np.uint8), k) > 0
    near_v = cv2.dilate(tv.astype(np.uint8), k) > 0
    n_tape = tr.sum() + tv.sum()
    tape_f3 = float(((tr & near_v).sum() + (tv & near_r).sum()) / n_tape) if n_tape else None
    err = np.abs(real.astype(np.float32) - rendered.astype(np.float32))[m]
    mse = float((err ** 2).mean())
    return {'shift_px': float(np.hypot(dx, dy)), 'shift_xy': [float(dx), float(dy)],
            'corr_response': float(resp),
            'tape_iou': float((tr & tv).sum() / union) if union else None,
            'tape_f3px': tape_f3,
            'mae': float(err.mean()), 'psnr': float(10 * np.log10(255 ** 2 / max(mse, 1e-9))),
            'coverage': float(m.mean())}


def overlay(real, rendered, cov):
    tr, tv = tape_mask(real), tape_mask(rendered)
    base = (0.5 * cv2.cvtColor(cv2.cvtColor(real, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)).astype(np.uint8)
    base[tr & ~tv] = (255, 0, 255)
    base[tv & ~tr] = (0, 255, 0)
    base[tr & tv] = (255, 255, 255)
    base[~cov] //= 3
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--dwells', type=str, default=None, help='comma separated dwell ids (default: the model\'s holdout set)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    model = MosaicModel(args.model)
    ids = [int(x) for x in args.dwells.split(',')] if args.dwells else model.meta['holdout']
    if not ids:
        raise SystemExit('no dwells to evaluate: build with --holdout N or pass --dwells')
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for k in ids:
        real = model.real_image(k)
        if real is None:
            print(f'dwell {k}: no real image (unmatched), skipped')
            continue
        rendered, cov = model.render(model.dwell(k)['eef_pose'])
        r = compare(real, rendered, cov)
        r['held_out'] = k in model.meta['holdout']
        results[k] = r
        iou = 'n/a' if r['tape_iou'] is None else f'{r["tape_iou"]:.3f}'
        print(f'dwell {k:2d} ({"held out" if r["held_out"] else "in mosaic"}): '
              f'shift {r["shift_px"]:5.2f}px  tape IoU {iou}  tape@3px {r["tape_f3px"]:.3f}  MAE {r["mae"]:5.2f}  '
              f'PSNR {r["psnr"]:5.2f}dB  coverage {r["coverage"] * 100:5.1f}%')
        panel = np.concatenate([label(real, f'real (dwell {k})'), label(rendered, 'rendered from mosaic'),
                                label(overlay(real, rendered, cov), 'tape: real=magenta rendered=green')], axis=1)
        cv2.imwrite(str(out / f'dwell_{k:03d}.png'), panel)

    vals = list(results.values())
    summary = {
        'n': len(vals),
        'shift_px_mean': float(np.mean([v['shift_px'] for v in vals])),
        'shift_px_max': float(np.max([v['shift_px'] for v in vals])),
        'tape_iou_mean': float(np.mean([v['tape_iou'] for v in vals if v['tape_iou'] is not None] or [np.nan])),
        'tape_f3px_mean': float(np.mean([v['tape_f3px'] for v in vals if v['tape_f3px'] is not None] or [np.nan])),
        'mae_mean': float(np.mean([v['mae'] for v in vals])),
        'psnr_mean': float(np.mean([v['psnr'] for v in vals])),
    }
    print('[eval] ' + ', '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in summary.items()))
    json.dump({'summary': summary, 'per_dwell': results}, open(out / 'eval.json', 'w'), indent=1)
    print(f'[eval] wrote {out}/eval.json and per-dwell panels')


if __name__ == '__main__':
    main()
