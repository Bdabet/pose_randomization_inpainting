"""
Self-calibration from the scan itself: camera yaw (rotation about the optical axis)
and camera-to-surface distance.

For a (near) top-down camera translating parallel to a plane at distance D, a camera
translation d (base frame, xy) shifts the image content by
      shift_px = -(f / D) * R_cw[:2, :2] @ d          (R_cw = camera <- base rotation)
Phase correlation between overlapping dwell images measures shift_px; the robot gives
d. Fitting the 2x2 map A (shift = A d) over all view pairs yields
  - its scale            -> f / D           (surface distance)
  - its rotation         -> the camera's true yaw about the optical axis, which is
                            compared with what the hand-eye extrinsic predicts.
The resulting correction is a rotation of the camera frame about its own z axis:
T_eef_cam_corrected = T_eef_cam @ Rz(yaw_correction).

All scan poses share one orientation, so the scan can calibrate this yaw (and the
plane) but NOT the camera's lever arm / tilt relative to the flange; those stay as
given by the hand-eye calibration.
"""

import numpy as np
import cv2


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def pairwise_shifts(images, positions, max_dist=0.12, min_response=0.3):
    """Phase-correlation image shifts for all pairs of views closer than `max_dist`
    (meters). `images`: same-size single-channel float images (undistorted).
    Returns (d_xy (N,2) base-frame camera translations, shifts (N,2) px, pairs)."""
    win = cv2.createHanningWindow(images[0].shape[::-1], cv2.CV_64F)
    D, S, P = [], [], []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            d = positions[j] - positions[i]
            if np.linalg.norm(d) > max_dist:
                continue
            (dx, dy), resp = cv2.phaseCorrelate(images[i].astype(np.float64),
                                                images[j].astype(np.float64), win)
            if resp < min_response:
                continue
            D.append(d[:2])
            S.append([dx, dy])
            P.append((i, j))
    return np.array(D), np.array(S), P


def fit_shift_model(d_xy, shifts, n_iter=3, inlier_px=3.0):
    """Least-squares 2x2 A with shifts ~ A @ d, with a couple of outlier-rejection
    rounds. Returns (A, inlier mask, rms px)."""
    keep = np.ones(len(d_xy), bool)
    for _ in range(n_iter):
        At, *_ = np.linalg.lstsq(d_xy[keep], shifts[keep], rcond=None)
        r = np.linalg.norm(d_xy @ At - shifts, axis=1)
        keep = r < max(inlier_px, 3 * np.median(r))
    A = At.T
    rms = float(np.sqrt(np.mean(np.linalg.norm(d_xy[keep] @ At - shifts[keep], axis=1) ** 2)))
    return A, keep, rms


def self_calibrate(K, images_undist, cam_poses, log=print):
    """Returns (yaw_correction_rad, distance_m, report dict)."""
    pos = np.array([T[:3, 3] for T in cam_poses])
    d_xy, shifts, _ = pairwise_shifts(images_undist, pos)
    if len(d_xy) < 3:
        raise RuntimeError('too few overlapping view pairs for self-calibration')
    A, keep, rms = fit_shift_model(d_xy, shifts)
    # translations must span both directions, otherwise A is under-determined
    # (e.g. only moves along the texture's dominant direction correlate: aperture problem)
    dsv = np.linalg.svd(d_xy[keep], compute_uv=False)
    if dsv[1] < 0.2 * dsv[0]:
        raise RuntimeError(f'[calib] view-pair translations are (nearly) collinear '
                           f'(singular values {dsv}); cannot self-calibrate')
    sv = np.linalg.svd(A, compute_uv=False)
    if sv[0] / sv[1] > 1.1:
        raise RuntimeError(f'[calib] image-shift model is strongly anisotropic ({sv[0] / sv[1]:.2f}); '
                           f'the camera is not looking roughly perpendicular at a plane, or pairs are mismatched')
    f = 0.5 * (K[0, 0] + K[1, 1])
    D = f / sv.mean()

    # model: A = -(f/D) * Rz2(-yaw) @ Rcw_xy  ->  Rz2(-yaw) ~ -(D/f) A Rcw_xy^-1
    R_cw = np.mean([T[:3, :3].T for T in cam_poses], axis=0)
    C = -(D / f) * A @ np.linalg.pinv(R_cw[:2, :2])
    neg_yaw = np.arctan2(C[1, 0] - C[0, 1], C[0, 0] + C[1, 1])
    yaw = -neg_yaw
    # check the model is a rotation (not a reflection) and reproduces A
    Rz2 = rot_z(-yaw)[:2, :2]
    A_model = -(f / D) * Rz2 @ R_cw[:2, :2]
    model_rms = float(np.sqrt(np.mean(np.linalg.norm(d_xy[keep] @ A_model.T - shifts[keep], axis=1) ** 2)))
    report = {'pairs': int(len(d_xy)), 'inliers': int(keep.sum()), 'fit_rms_px': rms,
              'model_rms_px': model_rms, 'singular_values': sv.tolist(),
              'anisotropy': float(sv[0] / sv[1]), 'distance_m': float(D),
              'yaw_correction_deg': float(np.degrees(yaw)), 'A': A.tolist()}
    log(f'[calib] {keep.sum()}/{len(d_xy)} view pairs, linear fit rms {rms:.2f}px, '
        f'scale anisotropy {sv[0] / sv[1]:.4f}')
    log(f'[calib] camera-surface distance {D:.4f} m; camera yaw correction '
        f'{np.degrees(yaw):+.2f} deg about the optical axis (model rms {model_rms:.2f}px)')
    return yaw, D, report
