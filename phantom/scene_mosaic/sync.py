"""
Associate video frames with robot poses.

The recorded mp4 has no per-frame timestamps, and its frame index is NOT a reliable
clock: the camera stream stalls (the same image is repeated for many frames) and then
jumps, so a frame at index i was not necessarily captured at t0 + i/fps.

The scan, however, is a stop-and-go trajectory: the arm moves between waypoints and
dwells ~1 s at each one. During a dwell the pose is constant and exactly known, and
the video shows a long run of (near) identical frames. So instead of per-frame time
matching we:
  1. split the robot trajectory into dwells (eef speed ~ 0),
  2. split the video into static segments (runs of frames that barely change),
  3. match each dwell to the longest static segment overlapping it in (approximate)
     time, and
  4. average the segment's frames into one low-noise image for that dwell's pose.
Short static segments are stalls during motion and are discarded.
"""

import numpy as np
import cv2

from phantom.scene_mosaic.common import iter_video


def robot_dwells(timestamps, eef_vel, max_speed=0.003, min_duration=0.4):
    """List of (start_idx, end_idx) into the episode arrays where the eef is still."""
    speed = np.linalg.norm(eef_vel[:, :3], axis=1)
    still = speed < max_speed
    out, i = [], 0
    while i < len(still):
        if still[i]:
            j = i
            while j < len(still) and still[j]:
                j += 1
            if timestamps[j - 1] - timestamps[i] >= min_duration:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def video_static_segments(video_path, change_thresh=1.0, thumb=(160, 90)):
    """Split the video into runs of frames whose mean abs gray difference to the run's
    first frame stays below `change_thresh` (on 0..255, at thumbnail resolution).
    Returns list of (start_frame, end_frame) covering the whole video."""
    anchor, segs, start, n = None, [], 0, 0
    for i, f in iter_video(video_path):
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), thumb,
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        if anchor is None:
            anchor = g
        elif np.abs(g - anchor).mean() > change_thresh:
            segs.append((start, i))
            start, anchor = i, g
        n = i + 1
    segs.append((start, n))
    return segs


def match_dwells_to_segments(dwells, pose_t, segments, fps, time_offset=0.0,
                             min_frames=15, max_offset=3.0, drift_weight=4.0,
                             skip_cost=1.5, trim=2):
    """Monotonic alignment of robot dwells to candidate video segments (static runs of
    >= `min_frames` frames) by dynamic programming.

    The video clock drifts and jumps relative to the robot clock (stalls / dropped
    frames), so a fixed offset doesn't hold for the whole scan, but it changes slowly
    from one dwell to the next. For a dwell k matched to segment j, with
    offset_kj = video_mid(j) - dwell_mid(k):
        cost = drift_weight * |offset_kj - offset of previous match|  (smooth drift)
             - duration overlap reward (prefer long segments of dwell-like length)
    Dwells may stay unmatched at `skip_cost`. Each segment is used at most once and
    order is preserved.
    Returns list of dicts {dwell, pose_range, frames=(start, end) or None}."""
    t0 = pose_t[0]
    cands = [(s, e) for s, e in segments if e - s >= min_frames]
    v_mid = np.array([(s + e) / 2 / fps + time_offset for s, e in cands])
    v_len = np.array([(e - s) / fps for s, e in cands])
    d_mid = np.array([(pose_t[a] + pose_t[b - 1]) / 2 - t0 for a, b in dwells])
    d_len = np.array([pose_t[b - 1] - pose_t[a] for a, b in dwells])
    nD, nC = len(dwells), len(cands)

    def unary(k, j):
        off = v_mid[j] - d_mid[k]
        if abs(off) > max_offset:
            return np.inf
        return -min(v_len[j], d_len[k] + 0.3) / max(d_len[k], 1e-3)  # in [-1.3, 0]

    # best[k, j]: min cost of an alignment whose last match is dwell k <-> cand j.
    # Predecessor (kp, jp) with kp < k, jp < j; dwells strictly between are skipped.
    best = np.full((nD, nC), np.inf)
    back = {}
    for k in range(nD):
        for j in range(nC):
            u = unary(k, j)
            if not np.isfinite(u):
                continue
            c_best, src = k * skip_cost, None  # all earlier dwells skipped
            off = v_mid[j] - d_mid[k]
            for kp in range(k):
                for jp in range(j):
                    if not np.isfinite(best[kp, jp]):
                        continue
                    c = (best[kp, jp] + (k - kp - 1) * skip_cost
                         + drift_weight * abs(off - (v_mid[jp] - d_mid[kp])))
                    if c < c_best:
                        c_best, src = c, (kp, jp)
            best[k, j] = c_best + u
            back[(k, j)] = src

    total = best + (nD - 1 - np.arange(nD))[:, None] * skip_cost
    k, j = np.unravel_index(np.argmin(total), total.shape)
    assign = [None] * nD
    node = (int(k), int(j))
    while node is not None:
        assign[node[0]] = node[1]
        node = back[node]

    matches = []
    for k, (a, b) in enumerate(dwells):
        if assign[k] is None:
            matches.append({'dwell': k, 'pose_range': (a, b), 'frames': None})
            continue
        s, e = cands[assign[k]]
        matches.append({'dwell': k, 'pose_range': (a, b), 'frames': (s + trim, e - trim)})
    return matches


def average_segment_frames(video_path, matches):
    """Decode the video once and return {dwell_idx: float32 BGR mean image}."""
    want = {}
    for m in matches:
        if m['frames'] is not None:
            for i in range(*m['frames']):
                want[i] = m['dwell']
    sums, counts = {}, {}
    for i, f in iter_video(video_path, indices=want.keys()):
        k = want[i]
        if k not in sums:
            sums[k] = np.zeros(f.shape, np.float64)
            counts[k] = 0
        sums[k] += f
        counts[k] += 1
    return {k: (sums[k] / counts[k]).astype(np.float32) for k in sums}
