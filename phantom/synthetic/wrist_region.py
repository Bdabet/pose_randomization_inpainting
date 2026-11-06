"""
Keeping the randomization region inside what the wrist scan actually saw.

The fixed camera's background is a clean plate of the whole workspace, so any pose the
arm can reach has a valid background. The wrist camera's does not work that way: its
background is a projection of the scene mosaic, which only covers the patch of table
the scan swept, seen from roughly the height the scan was flown at. Sample start poses
outside that patch -- or lift far above it -- and the wrist frames come out part black,
because those pixels are workspace nobody ever photographed.

That failure is quiet: the pipeline runs, the videos are written, and the robot sits
correctly on a background that is half fill colour. These helpers make it loud, and
optionally fix it, by answering two questions before the episodes are generated:

- `region_coverage` -- for a configured region, how much of the wrist view the mosaic
  covers at the worst pose the trajectory can reach;
- `fit_region` -- the largest region (and lift height) centred on the scan that stays
  above a coverage floor.

Frames
------
Trajectories are generated in *grip-site* coordinates (what `TwinRobot` tracks), the
mosaic lives in *flange* coordinates (what `T_eef_cam` is relative to, and what a real
robot reports as `robot_eef_pose`). `phantom.synthetic.twin_frames` holds that
conversion -- including the controller's commanded-vs-settled twist, which coverage
depends on -- and is shared with the processor and the replay script so all three
place the camera the same way.
"""
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from phantom.synthetic.twin_frames import BASE_ORI_XYZW, grip_to_camera  # noqa: F401  (re-exported)


def scan_yaw_center(bg, ori_correction: np.ndarray = None) -> tuple:
    """
    The trajectory yaw whose settled orientation matches the one the scan was flown at,
    in degrees, with the residual that yaw cannot explain.

    The wrist camera is bolted to the flange, so yaw turns its footprint over the
    table. The mosaic covers one patch at one orientation; a randomization band centred
    somewhere else points the camera off that patch no matter how small the region is,
    which is why this is worth deriving rather than leaving at zero.
    """
    used = bg.model.meta['used']
    R_scan = Rotation.from_rotvec(np.asarray(bg.model.meta['dwells'][used[0]]['eef_pose'][3:], float))
    C = Rotation.identity() if ori_correction is None else Rotation.from_matrix(ori_correction)
    # want base * Rz(y) * C == R_scan, so Rz(y) == base^-1 * R_scan * C^-1
    M = Rotation.from_quat(BASE_ORI_XYZW).inv() * R_scan * C.inv()
    yaw = float(M.as_euler('zyx', degrees=True)[0])
    residual = float(np.degrees((Rotation.from_euler('z', yaw, degrees=True).inv() * M).magnitude()))
    return yaw, residual


def _yaw_quats(yaw_range_deg: float, n: int = 5, yaw_center_deg: float = 0.0) -> np.ndarray:
    base = Rotation.from_quat(BASE_ORI_XYZW)
    yaws = (yaw_center_deg + np.linspace(-yaw_range_deg, yaw_range_deg, n) if yaw_range_deg
            else np.array([yaw_center_deg]))
    return np.array([(base * Rotation.from_euler('z', y, degrees=True)).as_quat() for y in yaws])


def region_poses(region_low, region_high, place_pos, lift_height, yaw_range_deg: float,
                 n_xy: int = 5, n_path: int = 5, yaw_center_deg: float = 0.0) -> list:
    """
    Grip-site poses spanning everywhere a pick-and-place trajectory in this region can
    go: the region box (corners and interior grid) at its own height and at the lift
    height, the transport column, and the place position -- each over a range of yaws.

    It is a grid, not the true reachable set, so a generated episode can still land a
    little below the coverage the grid reports; the per-episode coverage check in
    SyntheticPickPlaceProcessor._run_episode is what catches that.
    """
    lo, hi = np.asarray(region_low, float), np.asarray(region_high, float)
    place = np.asarray(place_pos, float)
    xs, ys = np.linspace(lo[0], hi[0], n_xy), np.linspace(lo[1], hi[1], n_xy)
    zs = np.unique(np.concatenate([np.linspace(lo[2], hi[2], 2), [lift_height]]))
    pts = [np.array([x, y, z]) for x in xs for y in ys for z in zs]
    pts += [place, np.array([place[0], place[1], lift_height])]
    # The transport leg runs from each possible start, at the lift height, to the place
    # position, and an episode spends most of its frames on it. Sampling only the box
    # corners misses it: coverage is not monotone along the way, so the worst pose can
    # sit in the middle of a leg whose endpoints both pass.
    for x in xs:
        for y in ys:
            a = np.array([x, y, lift_height])
            b = np.array([place[0], place[1], lift_height])
            pts += [a + t * (b - a) for t in np.linspace(0, 1, n_path)[1:-1]]
    return [(p, q) for p in pts for q in _yaw_quats(yaw_range_deg, yaw_center_deg=yaw_center_deg)]


def region_coverage(bg, T_flange_grip: np.ndarray, region_low, region_high, place_pos,
                    lift_height: float, yaw_range_deg: float, step: int = 16,
                    ori_correction: np.ndarray = None, yaw_center_deg: float = 0.0) -> dict:
    """
    Mosaic coverage over everywhere the trajectories can reach. Returns the worst and
    mean coverage and the grip-site pose where it is worst.
    """
    poses = region_poses(region_low, region_high, place_pos, lift_height, yaw_range_deg,
                         yaw_center_deg=yaw_center_deg)
    covs = np.array([bg.coverage_at_camera(
        grip_to_camera(T_flange_grip, bg.T_eef_cam, p, q, ori_correction), step) for p, q in poses])
    k = int(covs.argmin())
    return {'min': float(covs.min()), 'mean': float(covs.mean()),
            'worst_pos': poses[k][0], 'n_poses': len(poses)}


def fit_region(bg, T_flange_grip: np.ndarray, yaw_range_deg: float, min_coverage: float = 0.95,
               max_lift: float = 0.40, margin: float = 0.0, step: int = 16,
               z_offset: float = 0.0, ori_correction: np.ndarray = None,
               yaw_center_deg: float = None) -> dict:
    """
    Largest region centred on the scan whose worst reachable pose still has at least
    `min_coverage` of its wrist view covered by the mosaic.

    The scan is a single sweep at one height, so both knobs trade coverage for range:
    moving outward in xy walks off the edge of the stitched patch, and lifting widens
    the camera's footprint until it overruns that patch. Both are bisected (xy first,
    at the scan height; then the lift, with that xy box), which is why the result is a
    box and not the true reachable set -- a box is what `synthetic.region_low/high`
    can express.

    `z_offset` raises the picking height above the scan plane; the default puts the
    grip site exactly where the scan's flange was, minus the tool length.

    Returns region_low/high, lift_height and place_pos in grip-site coordinates, ready
    to drop into the synthetic config.
    """
    if yaw_center_deg is None:
        yaw_center_deg = scan_yaw_center(bg, ori_correction)[0]
    b = bg.scan_bounds()
    # Flange -> grip site: the trajectory commands the grip site, the scan recorded the
    # flange. The offset lies along the tool axis, which points straight down in base
    # frame for the gripper-down orientation these trajectories use, so at the scan's
    # height the grip site sits one tool length below the flange.
    tool_length = float(np.linalg.norm(T_flange_grip[:3, 3]))
    grip_z = float(b['flange_low'][2]) - tool_length + z_offset
    centre = (b['flange_low'][:2] + b['flange_high'][:2]) / 2
    half_full = np.maximum((b['flange_high'][:2] - b['flange_low'][:2]) / 2 - margin, 0.0)

    def worst(half, lift):
        lo = np.array([centre[0] - half[0], centre[1] - half[1], grip_z])
        hi = np.array([centre[0] + half[0], centre[1] + half[1], grip_z])
        place = np.array([centre[0], centre[1], grip_z])
        return region_coverage(bg, T_flange_grip, lo, hi, place, lift, yaw_range_deg, step,
                               ori_correction, yaw_center_deg)['min']

    def bisect(fn, lo, hi, iters=12):
        """Largest value in [lo, hi] that still passes; lo must pass."""
        if not fn(hi):
            for _ in range(iters):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if fn(mid) else (lo, mid)
            return lo
        return hi

    s = bisect(lambda s: worst(half_full * s, grip_z) >= min_coverage, 0.0, 1.0)
    half = half_full * s
    lift = bisect(lambda z: worst(half, z) >= min_coverage, grip_z, grip_z + max_lift)

    lo = np.array([centre[0] - half[0], centre[1] - half[1], grip_z])
    hi = np.array([centre[0] + half[0], centre[1] + half[1], grip_z])
    place = np.array([centre[0], centre[1], grip_z])
    cov = region_coverage(bg, T_flange_grip, lo, hi, place, lift, yaw_range_deg, step,
                          ori_correction, yaw_center_deg)
    return {'region_low': lo, 'region_high': hi, 'lift_height': float(lift),
            'place_pos': place, 'xy_scale': float(s), 'coverage': cov,
            'yaw_center_deg': float(yaw_center_deg),
            'scan_bounds': b, 'tool_length': tool_length}


def describe_region(name: str, region_low, region_high, lift_height: float, place_pos) -> str:
    lo, hi = np.round(np.asarray(region_low, float), 3), np.round(np.asarray(region_high, float), 3)
    return (f"{name}: xy [{lo[0]}, {hi[0]}] x [{lo[1]}, {hi[1]}] at z {lo[2]}"
            f"{'' if lo[2] == hi[2] else f'-{hi[2]}'}, lift to {lift_height:.3f}, "
            f"place {np.round(np.asarray(place_pos, float), 3).tolist()}")
