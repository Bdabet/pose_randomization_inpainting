"""
Planar mosaic core: warp posed images onto the work-surface plane, blend them, estimate
the plane from photometric consistency, and render the plane back into an arbitrary
camera pose.
"""

import numpy as np
import cv2
from scipy.optimize import minimize

from phantom.scene_mosaic.common import Plane, Canvas, footprint_st, undistort_maps, pixel_rays


def center_weight(w, h):
    """1 at the image center, -> 0 at the border (Chebyshev distance)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    wgt = 1.0 - np.maximum(np.abs(xx - (w - 1) / 2) / (w / 2), np.abs(yy - (h - 1) / 2) / (h / 2))
    return np.clip(wgt, 0.0, 1.0)


class Warper:
    """Projects raw (distorted) camera images onto a canvas."""

    def __init__(self, K, dist, w, h):
        self.K, self.dist, self.w, self.h = K, dist, w, h
        self.map1, self.map2 = undistort_maps(K, dist, w, h)
        valid = cv2.remap(np.ones((h, w), np.float32), self.map1, self.map2, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        self.base_weight = center_weight(w, h) * valid

    def undistort(self, img):
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def warp(self, img_undist, T_base_cam, canvas, pad=2):
        """Warp an already-undistorted image into canvas coordinates. Returns
        (x0, y0, warped_img, warped_weight) for the canvas sub-rectangle it covers,
        or None if it doesn't touch the canvas."""
        st = footprint_st(canvas.plane, self.K, self.dist, self.w, self.h, T_base_cam)
        px = canvas.st_to_px(st)
        x0 = max(int(np.floor(px[:, 0].min())) - pad, 0)
        y0 = max(int(np.floor(px[:, 1].min())) - pad, 0)
        x1 = min(int(np.ceil(px[:, 0].max())) + pad, canvas.width)
        y1 = min(int(np.ceil(px[:, 1].max())) + pad, canvas.height)
        if x1 <= x0 or y1 <= y0:
            return None
        H = canvas.homography_canvas_to_image(self.K, T_base_cam)
        shift = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=np.float64)
        Hs = H @ shift  # sub-canvas pixel -> image pixel
        size = (x1 - x0, y1 - y0)
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        warped = cv2.warpPerspective(img_undist, Hs, size, flags=flags,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        weight = cv2.warpPerspective(self.base_weight, Hs, size, flags=flags,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return x0, y0, warped, weight


def blend(warper, images_undist, cam_poses, canvas, sharpness=40.0):
    """Weighted blend of all images on the canvas. weight = center_weight ** sharpness,
    so each canvas pixel is effectively taken from the frame that sees it most
    centrally, with smooth transitions only where two frames tie (seams)."""
    C = images_undist[0].shape[2] if images_undist[0].ndim == 3 else 1
    acc = np.zeros((canvas.height, canvas.width, C), np.float64)
    wsum = np.zeros((canvas.height, canvas.width), np.float64)
    for img, T in zip(images_undist, cam_poses):
        r = warper.warp(img, T, canvas)
        if r is None:
            continue
        x0, y0, wi, ww = r
        ww = ww.astype(np.float64) ** sharpness
        if wi.ndim == 2:
            wi = wi[..., None]
        hh, wd = ww.shape
        acc[y0:y0 + hh, x0:x0 + wd] += wi * ww[..., None]
        wsum[y0:y0 + hh, x0:x0 + wd] += ww
    valid = wsum > 1e-30
    out = np.zeros_like(acc, dtype=np.float32)
    out[valid] = acc[valid] / wsum[valid, None]
    return out, valid


def estimate_gain(images, sigma=40.0):
    """Camera-fixed shading (vignetting, the arm's own shadow/illumination falloff):
    per-pixel median over many views of a mostly uniform surface, heavily smoothed,
    normalised to mean 1. Scene content (tape, marks) differs between views and
    drops out in the median. Returns (h, w, 3) float32."""
    med = np.median(np.stack(images), axis=0).astype(np.float32)
    gain = cv2.GaussianBlur(med, (0, 0), sigma)
    gain /= gain.reshape(-1, gain.shape[-1]).mean(axis=0)
    return np.clip(gain, 0.2, None).astype(np.float32)


# --------------------------------------------------------------- plane estimate

def structure_image(img_bgr):
    """Single-channel image used for plane fitting: emphasises the red tape / any
    texture and suppresses smooth shading (lighting, vignetting)."""
    f = img_bgr.astype(np.float32)
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    redness = np.clip(f[..., 2] - 0.5 * (f[..., 0] + f[..., 1]), 0, None)
    hp = g - cv2.GaussianBlur(g, (0, 0), 15)
    s = redness + np.abs(hp) * 2.0
    return cv2.GaussianBlur(s, (0, 0), 3)


def consistency_cost(warper, structs_undist, cam_poses, plane, px_per_m=300.0):
    """Mean (over canvas pixels seen by >=2 frames) of the per-pixel variance of the
    structure images across frames, normalised by their mean energy. Low when the
    plane hypothesis makes overlapping views agree."""
    st = np.concatenate([footprint_st(plane, warper.K, warper.dist, warper.w, warper.h, T)
                         for T in cam_poses])
    if not np.all(np.isfinite(st)) or np.ptp(st, axis=0).max() > 10:
        return 1e3
    canvas = Canvas.covering(plane, st, px_per_m, margin=0.01)
    s1 = np.zeros((canvas.height, canvas.width))
    s2 = np.zeros_like(s1)
    cnt = np.zeros_like(s1)
    for img, T in zip(structs_undist, cam_poses):
        r = warper.warp(img, T, canvas)
        if r is None:
            continue
        x0, y0, wi, ww = r
        m = ww > 0.05  # ignore the extreme border
        hh, wd = ww.shape
        s1[y0:y0 + hh, x0:x0 + wd] += wi * m
        s2[y0:y0 + hh, x0:x0 + wd] += wi ** 2 * m
        cnt[y0:y0 + hh, x0:x0 + wd] += m
    ov = cnt >= 2
    if ov.sum() < 500:
        return 1e3
    mean = s1[ov] / cnt[ov]
    var = s2[ov] / cnt[ov] - mean ** 2
    return float(var.mean() / (np.mean(mean ** 2) + 1e-9))


def estimate_plane(warper, structs_undist, cam_poses, distance, fit_tilt=True, log=print):
    """Refine (height, tilt_x, tilt_y) by minimizing the multi-view consistency cost,
    starting from the parallax camera-surface `distance` (calib.self_calibrate) and
    restricted to +-25% of it so the optimizer can't run into degenerate planes (e.g.
    right in front of the camera, where views stop overlapping)."""
    D = distance
    cam_z = float(np.mean([T[2, 3] for T in cam_poses]))
    z_lo, z_hi = cam_z - 1.25 * D, cam_z - 0.75 * D
    zs = np.linspace(z_lo, z_hi, 41)
    costs = [consistency_cost(warper, structs_undist, cam_poses, Plane.from_height_tilt(z)) for z in zs]
    z0 = zs[int(np.argmin(costs))]
    log(f'[plane] height search in [{z_lo:.3f}, {z_hi:.3f}]: best z={z0:.3f} m (cost {min(costs):.4f})')

    def cost(p):
        z, tx, ty = (p[0], 0.0, 0.0) if len(p) == 1 else p
        if not z_lo <= z <= z_hi or max(abs(tx), abs(ty)) > np.radians(10):
            return 1e3
        return consistency_cost(warper, structs_undist, cam_poses, Plane.from_height_tilt(z, tx, ty))

    if fit_tilt:
        x0 = [z0, 0.0, 0.0]
        simplex = np.array([x0, [z0 + 0.02, 0, 0], [z0, 0.02, 0], [z0, 0, 0.02]])
    else:
        x0 = [z0]
        simplex = np.array([[z0], [z0 + 0.02]])
    res = minimize(cost, x0, method='Nelder-Mead',
                   options={'xatol': 1e-4, 'fatol': 1e-7, 'initial_simplex': simplex})
    z, tx, ty = (res.x[0], 0.0, 0.0) if not fit_tilt else res.x
    log(f'[plane] refined: z={z:.4f} m, tilt_x={np.degrees(tx):.2f} deg, '
        f'tilt_y={np.degrees(ty):.2f} deg (cost {res.fun:.4f})')
    return Plane.from_height_tilt(z, tx, ty), (zs, np.array(costs))


# ------------------------------------------------------------------- render

def render_view(mosaic, valid, canvas, K, dist, w, h, T_base_cam, rays=None):
    """Synthesize the raw (distorted, as the real camera records it) w x h image seen
    from camera pose T_base_cam, by casting each pixel's ray onto the plane and
    sampling the mosaic. Returns (image float32, coverage mask).

    `rays` is the (h, w, 3) output of pixel_rays(K, dist, w, h); pass it in to avoid
    re-undistorting the whole pixel grid when rendering many poses for one camera
    (VENDOR: local addition, see README.md)."""
    if rays is None:
        rays = pixel_rays(K, dist, w, h)  # (h, w, 3) camera frame
    R, c = T_base_cam[:3, :3], T_base_cam[:3, 3]
    X, depth = canvas.plane.intersect(np.broadcast_to(c, rays.shape), rays @ R.T)
    px = canvas.st_to_px(canvas.plane.to_st(X)).astype(np.float32)
    px[depth <= 0] = -1  # plane behind the camera
    img = cv2.remap(mosaic, px[..., 0], px[..., 1], cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cov = cv2.remap(valid.astype(np.float32), px[..., 0], px[..., 1], cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
    return img, cov
