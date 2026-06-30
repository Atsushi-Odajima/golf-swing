import argparse
import json
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pydeps"))

import cv2
import numpy as np


@dataclass
class Detection:
    frame: int
    t: float
    x: float
    y: float
    area: int
    score: float


@dataclass
class Candidate:
    frame: int
    t: float
    x: float
    y: float
    area: int
    score: float
    kind: str


def rotate_frame(frame, rotate):
    if rotate == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def parse_xy(value):
    x, y = value.split(",", 1)
    return float(x), float(y)


def point_line_metrics(px, py, launch, direction):
    rx = px - launch[0]
    ry = py - launch[1]
    along = rx * direction[0] + ry * direction[1]
    perp = abs(rx * -direction[1] + ry * direction[0])
    return along, perp


def make_direction(launch, aim):
    dx = aim[0] - launch[0]
    dy = aim[1] - launch[1]
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


def compact_white_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    white = (v > 170) & (s < 72)
    yellow = (v > 130) & (s > 70) & (h > 15) & (h < 45)
    # Once the ball is in the sky it often appears as a tiny dark dot rather
    # than a white ball because it is backlit against the clouds.
    dark_dot = (v < 95) & (s < 110)
    return (white | yellow | dark_dot).astype(np.uint8) * 255


def classify_blob(frame, labels, label_id, x, y, w, h):
    roi = frame[y:y + h, x:x + w]
    mask = labels[y:y + h, x:x + w] == label_id
    if roi.size == 0 or not np.any(mask):
        return "unknown", 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    vals = hsv[mask]
    sat = float(np.mean(vals[:, 1]))
    val = float(np.mean(vals[:, 2]))
    if val > 150 and sat < 95:
        return "bright", val - sat * 0.25
    if val < 115 and sat < 125:
        return "dark", 180 - val - sat * 0.15
    return "color", 40.0


def extract_motion_candidates(frame, prev_gray, frame_idx, fps, diff_threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    _, motion = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    color = compact_white_mask(frame)
    mask = cv2.bitwise_and(motion, color)
    mask = cv2.medianBlur(mask, 3)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    h_frame, w_frame = frame.shape[:2]
    candidates = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 2 or area > 120:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > 7.0:
            continue
        cx, cy = centroids[i]
        # Ignore obvious shoe/body noise and bottom grass specks. The actual
        # launch still sits above this lower band in typical tee-shot framing.
        if cy > h_frame * 0.78:
            continue
        if cx < w_frame * 0.12 and cy > h_frame * 0.42:
            continue
        kind, color_score = classify_blob(frame, labels, i, x, y, w, h)
        compactness = area / max(1, w * h)
        score = color_score + min(area, 18) * 2.0 + compactness * 30.0
        candidates.append(Candidate(
            frame=frame_idx,
            t=frame_idx / fps,
            x=float(cx),
            y=float(cy),
            area=int(area),
            score=float(score),
            kind=kind,
        ))
    return gray, candidates, mask


def estimate_impact_frame(video_path, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start = int((args.scan_start or 0.0) * fps)
    end = int((args.scan_end * fps) if args.scan_end else total - 1)
    start = max(1, min(start, total - 2))
    end = max(start + 1, min(end, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)

    ok, raw = cap.read()
    if not ok:
        raise RuntimeError("Could not read scan frame")
    prev = rotate_frame(raw, args.rotate)
    prev_small = cv2.resize(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (270, 480))
    energies = []

    for frame_idx in range(start, end + 1):
        ok, raw = cap.read()
        if not ok:
            break
        frame = rotate_frame(raw, args.rotate)
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (270, 480))
        diff = cv2.absdiff(gray, prev_small)
        # Lower-middle ROI: club, hands, divot, and launch area. This avoids
        # tree/cloud flicker dominating the motion estimate.
        roi = diff[185:365, 40:205]
        energy = float(np.sum(roi > 22))
        energies.append((frame_idx, energy))
        prev_small = gray
    cap.release()

    if not energies:
        return start
    vals = np.array([e for _i, e in energies], dtype=np.float64)
    kernel = np.ones(5, dtype=np.float64) / 5.0
    smooth = np.convolve(vals, kernel, mode="same")
    best_i = int(np.argmax(smooth))
    return int(energies[best_i][0])


def collect_candidates(video_path, args, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    end_frame = min(total - 1, end_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - 1))

    ok, raw = cap.read()
    if not ok:
        raise RuntimeError("Could not read candidate seed frame")
    first = rotate_frame(raw, args.rotate)
    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    h, w = first.shape[:2]

    by_frame = {}
    debug_items = []
    for frame_idx in range(start_frame, end_frame + 1):
        ok, raw = cap.read()
        if not ok:
            break
        frame = rotate_frame(raw, args.rotate)
        prev_gray, candidates, mask = extract_motion_candidates(
            frame, prev_gray, frame_idx, fps, args.diff_threshold
        )
        by_frame[frame_idx] = candidates
        if args.debug_masks and len(debug_items) < 30 and candidates:
            debug_items.append((frame.copy(), mask.copy(), candidates, None))
    cap.release()
    return by_frame, (w, h), fps, debug_items


def select_initial_chain(by_frame, impact_frame, frame_size):
    w, h = frame_size
    best_chain = []
    best_score = -1e9
    # Impact-energy peak can be a little before the actual ball leaves the club,
    # especially in slow-motion footage. Search roughly the next second.
    start_frames = range(impact_frame, impact_frame + 36)

    for f in start_frames:
        for start in by_frame.get(f, []):
            if not (h * 0.48 <= start.y <= h * 0.73):
                continue
            # In rear-view tee-shot videos the club/body motion is often on
            # the player side, while the ball emerges into the open fairway.
            # Avoid starting a track on the clubhead or the player's hands.
            if not (w * 0.42 <= start.x <= w * 0.82):
                continue
            chain = [start]
            vx, vy = 0.0, -60.0
            last = start
            for nf in range(f + 1, f + 22):
                options = by_frame.get(nf, [])
                if not options:
                    continue
                gap = nf - last.frame
                pred_x = last.x + vx * gap
                pred_y = last.y + vy * gap
                chosen = None
                chosen_score = 1e9
                for c in options:
                    # The first visible ball points should generally climb in
                    # the frame. Allow tiny regressions for compression noise.
                    if c.y > last.y + 24:
                        continue
                    if c.y > h * 0.74:
                        continue
                    if c.x < w * 0.38:
                        continue
                    if c.y < last.y - 165 * gap:
                        continue
                    if abs(c.x - last.x) > 105 * gap:
                        continue
                    if abs(c.x - start.x) > w * 0.24:
                        continue
                    dist = math.hypot(c.x - pred_x, c.y - pred_y)
                    if dist > 125 + 25 * gap:
                        continue
                    upward = max(0.0, last.y - c.y)
                    score = dist - upward * 0.75 - min(c.score, 120) * 0.08
                    if score < chosen_score:
                        chosen_score = score
                        chosen = c
                if chosen:
                    nvx = (chosen.x - last.x) / max(1, chosen.frame - last.frame)
                    nvy = (chosen.y - last.y) / max(1, chosen.frame - last.frame)
                    vx = vx * 0.45 + nvx * 0.55
                    vy = vy * 0.45 + nvy * 0.55
                    chain.append(chosen)
                    last = chosen
                    if len(chain) >= 5:
                        break
            if len(chain) < 3:
                continue
            vertical_gain = chain[0].y - chain[-1].y
            horizontal_span = abs(chain[-1].x - chain[0].x)
            if vertical_gain < 105:
                continue
            score = min(vertical_gain, h * 0.33) * 2.0 + len(chain) * 80.0 - horizontal_span * 0.45
            if score > best_score:
                best_score = score
                best_chain = chain
    return best_chain


def find_upper_anchors(by_frame, chain, frame_size):
    if len(chain) < 2:
        return []
    w, h = frame_size
    p0 = chain[0]
    p1 = chain[min(len(chain) - 1, 3)]
    dx = p1.x - p0.x
    dy = p1.y - p0.y
    n = math.hypot(dx, dy) or 1.0
    direction = (dx / n, dy / n)
    anchors = []

    for f in range(chain[-1].frame + 1, chain[0].frame + 80):
        best = None
        best_score = -1e9
        for c in by_frame.get(f, []):
            if c.x < w * 0.30 or c.x > w * 0.90:
                continue
            if c.y > chain[-1].y - 80:
                continue
            rx, ry = c.x - p0.x, c.y - p0.y
            along = rx * direction[0] + ry * direction[1]
            perp = abs(rx * -direction[1] + ry * direction[0])
            if along < 120:
                continue
            if perp > 230 + along * 0.10:
                continue
            # Prefer high, compact sky dots close to the inferred launch line.
            score = along * 0.6 - perp * 1.4 + (h - c.y) * 0.25 + min(c.score, 140) * 0.25
            if c.kind == "dark" and c.y < h * 0.55:
                score += 90
            if c.kind == "bright":
                score += 30
            if score > best_score:
                best_score = score
                best = c
        if best:
            anchors.append(best)
            if len(anchors) >= 2:
                break
    return anchors


def back_project_launch(chain):
    if len(chain) < 2:
        p = chain[0]
        return p.x, p.y
    p0, p1 = chain[0], chain[1]
    vx = p1.x - p0.x
    vy = p1.y - p0.y
    return p0.x - vx * 0.70, p0.y - vy * 0.70


def make_visual_curve(launch, detections, frame_size, args):
    w, h = frame_size
    pts = [(d.x, d.y) for d in detections]
    if not pts:
        return [launch]

    # Use the real first points to establish direction, then turn that into a
    # SmoothSwing-style display arc. This keeps the line believable even when
    # later ball pixels are too faint to track reliably.
    first = pts[0]
    ref = pts[min(len(pts) - 1, 4)]
    vx = ref[0] - launch[0]
    vy = ref[1] - launch[1]
    if abs(vx) + abs(vy) < 1:
        vx, vy = 20.0, -450.0

    apex = min(pts, key=lambda p: p[1])
    if apex[1] > h * 0.45:
        scale = (launch[1] - h * 0.34) / max(1.0, launch[1] - ref[1])
        apex = (launch[0] + vx * scale, h * 0.34)

    # SmoothSwing draws a display trajectory rather than a literal pixel track.
    # The visible tail should suggest depth, so it bends away and fades while it
    # is still high in the frame instead of dropping all the way back to the tee.
    landing_y = min(max(apex[1] + h * 0.18, h * 0.38), h * 0.50)
    side = np.sign(apex[0] - launch[0]) or np.sign(vx) or 1.0
    landing_x = apex[0] + side * max(168.0, abs(apex[0] - launch[0]) * 1.05)
    landing_x = float(np.clip(landing_x, w * 0.08, w * 0.92))
    landing = (landing_x, landing_y)

    # A single cubic keeps the apex continuous. Two joined curves looked like a
    # mid-air bend when the incoming and outgoing tangents did not match.
    out = []
    c1 = (
        launch[0] + (apex[0] - launch[0]) * 0.24,
        launch[1] + (apex[1] - launch[1]) * 1.05,
    )
    c2 = (
        apex[0] + (landing[0] - apex[0]) * 0.18,
        apex[1] - h * 0.10,
    )
    for k in range(110):
        q = k / 109.0
        x = (
            (1 - q) ** 3 * launch[0]
            + 3 * (1 - q) ** 2 * q * c1[0]
            + 3 * (1 - q) * q ** 2 * c2[0]
            + q ** 3 * landing[0]
        )
        y = (
            (1 - q) ** 3 * launch[1]
            + 3 * (1 - q) ** 2 * q * c1[1]
            + 3 * (1 - q) * q ** 2 * c2[1]
            + q ** 3 * landing[1]
        )
        out.append((x, y))
    return out


def make_curve_times(curve, detections, launch_t, args):
    if not curve:
        return []
    if not detections:
        return [launch_t + args.draw_seconds * i / max(1, len(curve) - 1) for i in range(len(curve))]

    first_t = launch_t
    last_detect_t = max(d.t for d in detections)
    if last_detect_t - first_t < args.draw_seconds:
        last_detect_t = first_t + max(0.15, args.draw_seconds)
    draw_speed = max(0.1, float(args.line_draw_speed))
    last_detect_t = first_t + (last_detect_t - first_t) / draw_speed
    detected_portion = max(0.42, min(0.72, args.detected_curve_portion))
    detected_count = max(2, min(len(curve), int(round(len(curve) * detected_portion))))
    projected_count = len(curve) - detected_count

    times = []
    for i in range(detected_count):
        q = i / max(1, detected_count - 1)
        times.append(first_t + (last_detect_t - first_t) * q)
    projection_seconds = max(args.projection_seconds, 0.05) / draw_speed
    for i in range(projected_count):
        q = (i + 1) / max(1, projected_count)
        times.append(last_detect_t + projection_seconds * q)
    return times


def auto_track_ball(video_path, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    impact_frame = int(args.start * fps) if args.start is not None else estimate_impact_frame(video_path, args)
    scan_end = int(args.end * fps) if args.end is not None else impact_frame + int(fps * 3.0)
    by_frame, frame_size, fps, debug_items = collect_candidates(
        video_path,
        args,
        max(1, impact_frame - 3),
        min(total - 1, scan_end),
    )
    launch_search_frame = impact_frame + int(round(fps * args.launch_delay))
    chain = select_initial_chain(by_frame, launch_search_frame, frame_size)
    if not chain:
        raise RuntimeError("Could not auto-detect the launch ball path")
    anchors = find_upper_anchors(by_frame, chain, frame_size)
    selected = (chain + anchors)[:6]
    launch = args.launch if args.launch else back_project_launch(chain)
    curve = make_visual_curve(launch, selected, frame_size, args)
    impact_t = impact_frame / fps
    if args.line_delay > 0:
        launch_t = chain[0].t + args.line_delay * max(1.0, float(args.speed))
    else:
        raw_launch_t = chain[0].t - args.line_preroll
        launch_t = max(0.0, min(raw_launch_t, impact_t + 0.12))
    curve_times = make_curve_times(curve, selected, launch_t, args)
    detections = [
        Detection(c.frame, c.t, c.x, c.y, c.area, c.score)
        for c in selected
    ]
    return {
        "fps": fps,
        "frame_count": total,
        "width": frame_size[0],
        "height": frame_size[1],
        "impact_frame": impact_frame,
        "impact_t": launch_t,
        "launch": launch,
        "aim": (curve[-1][0], curve[-1][1]) if curve else launch,
        "detections": detections,
        "curve": curve,
        "curve_times": curve_times,
        "debug_masks": debug_items,
    }


def find_candidates(frame, prev_gray, launch, direction, radius, diff_threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    _, motion = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    color = compact_white_mask(frame)
    mask = cv2.bitwise_and(motion, color)
    mask = cv2.medianBlur(mask, 3)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 2 or area > 140:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > 5.5:
            continue
        cx, cy = centroids[i]
        along, perp = point_line_metrics(cx, cy, launch, direction)
        if along <= -8:
            continue
        corridor = radius * (1.0 + max(0.0, along) / 900.0)
        if perp > corridor:
            continue
        candidates.append((float(cx), float(cy), int(area), float(along), float(perp)))
    return gray, candidates, mask


def track_ball(video_path, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(args.start * fps))
    end_frame = min(total - 1, int(args.end * fps)) if args.end else total - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, raw = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame")
    first = rotate_frame(raw, args.rotate)
    height, width = first.shape[:2]

    launch = args.launch
    aim = args.aim
    direction = make_direction(launch, aim)
    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    detections = []
    last = launch
    velocity = (direction[0] * 22.0, direction[1] * 22.0)
    lost = 0
    best_masks = []

    frame_idx = start_frame + 1
    while frame_idx <= end_frame:
        ok, raw = cap.read()
        if not ok:
            break
        frame = rotate_frame(raw, args.rotate)
        t = frame_idx / fps
        gray, candidates, mask = find_candidates(
            frame, prev_gray, launch, direction, args.corridor, args.diff_threshold
        )
        prev_gray = gray

        predicted = (last[0] + velocity[0], last[1] + velocity[1])
        search_radius = args.search_radius * (1.0 + min(lost, 8) * 0.35)
        best = None
        best_score = 1e9
        for cx, cy, area, along, perp in candidates:
            dist = math.hypot(cx - predicted[0], cy - predicted[1])
            if dist > search_radius:
                continue
            score = dist + perp * 0.22 - min(area, 25) * 0.45
            if score < best_score:
                best_score = score
                best = (cx, cy, area, score)

        if best:
            cx, cy, area, score = best
            velocity = (cx - last[0], cy - last[1])
            last = (cx, cy)
            lost = 0
            detections.append(Detection(frame_idx, t, cx, cy, area, score))
        elif detections:
            lost += 1
            if lost > args.max_lost:
                break

        if args.debug_masks and len(best_masks) < 24 and (best or candidates):
            best_masks.append((frame.copy(), mask.copy(), candidates, best))

        frame_idx += 1

    cap.release()
    return {
        "fps": fps,
        "frame_count": total,
        "width": width,
        "height": height,
        "launch": launch,
        "aim": aim,
        "detections": detections,
        "debug_masks": best_masks,
    }


def fit_curve(track, args):
    pts = [(args.launch[0], args.launch[1], args.start)] + [(d.x, d.y, d.t) for d in track]
    if len(pts) < 2:
        return [(args.launch[0], args.launch[1]), args.aim]

    t0 = pts[0][2]
    us = np.array([p[2] - t0 for p in pts], dtype=np.float64)
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    deg_y = 2 if len(pts) >= 3 else 1
    try:
        cx = np.polyfit(us, xs, 1)
        cy = np.polyfit(us, ys, deg_y)
    except Exception:
        return [(p[0], p[1]) for p in pts]

    u_max = max(float(us[-1]), 0.12)
    if len(track) < args.min_points:
        # The ball is often visible for only a few frames. Fall back to a
        # SmoothSwing-style projected arc using the launch/aim direction.
        dx = args.aim[0] - args.launch[0]
        dy = args.aim[1] - args.launch[1]
        out = []
        for k in range(80):
            q = k / 79.0
            bend = math.sin(q * math.pi) * args.arc_bend
            x = args.launch[0] + dx * q + bend
            y = args.launch[1] + dy * q - math.sin(q * math.pi) * args.arc_lift
            out.append((x, y))
        return out

    out = []
    for k in range(80):
        u = u_max * k / 79.0
        out.append((float(np.polyval(cx, u)), float(np.polyval(cy, u))))

    # Extend the fitted tangent so the line reads like a ball flight, not just
    # the few visible pixels immediately after impact.
    if len(out) >= 2:
        vx = out[-1][0] - out[-2][0]
        vy = out[-1][1] - out[-2][1]
        x, y = out[-1]
        for _ in range(80):
            vy += args.gravity
            x += vx
            y += vy
            if x < -80 or x > args.frame_width + 80 or y < -80 or y > args.frame_height + 80:
                break
            out.append((x, y))
    return out


def draw_polyline_alpha(frame, points, color, width, alpha):
    if alpha <= 0:
        return
    if len(points) < 2:
        return
    overlay = frame.copy()
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(
        overlay,
        [pts],
        False,
        color,
        width,
        cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_curve(frame, curve, upto=1.0, fade_start=0.68):
    if len(curve) < 2:
        return
    n = max(2, int(len(curve) * upto))
    shown = curve[:n]
    draw_faded_path(frame, shown, len(curve), fade_start)


def draw_curve_timed(frame, curve, curve_times, t, fade_start=0.68):
    if len(curve) < 2 or not curve_times:
        return
    if t < curve_times[0]:
        return

    shown = []
    for i, point_t in enumerate(curve_times):
        if point_t <= t:
            shown.append(curve[i])
        else:
            if i > 0:
                prev_t = curve_times[i - 1]
                q = (t - prev_t) / max(1e-6, point_t - prev_t)
                q = float(np.clip(q, 0.0, 1.0))
                p0, p1 = curve[i - 1], curve[i]
                shown.append((p0[0] + (p1[0] - p0[0]) * q, p0[1] + (p1[1] - p0[1]) * q))
            break
    if len(shown) < 2:
        return
    draw_faded_path(frame, shown, len(curve), fade_start)


def draw_faded_path(frame, shown, full_count, fade_start):
    if len(shown) < 2:
        return
    fade_idx = max(2, min(len(shown), int(round(full_count * fade_start))))
    solid = shown[:fade_idx]
    draw_polyline_alpha(frame, solid, (20, 28, 205), 26, 0.78)
    draw_polyline_alpha(frame, solid, (42, 48, 255), 9, 0.92)

    if fade_idx >= len(shown):
        return

    tail = shown[fade_idx - 1:]
    bins = 5
    for b in range(bins):
        a = max(0, int(round(b * (len(tail) - 1) / bins)) - 1)
        z = min(len(tail), int(round((b + 1) * (len(tail) - 1) / bins)) + 2)
        part = tail[a:z]
        if len(part) < 2:
            continue
        q = (b + 1) / bins
        outer_width = max(5, int(round(18 - q * 10)))
        inner_width = max(2, int(round(7 - q * 4)))
        outer_alpha = max(0.08, 0.34 * (1.0 - q * 0.72))
        inner_alpha = max(0.10, 0.40 * (1.0 - q * 0.74))
        draw_polyline_alpha(frame, part, (20, 28, 205), outer_width, outer_alpha)
        draw_polyline_alpha(frame, part, (42, 48, 255), inner_width, inner_alpha)


def render_video(video_path, output_path, curve, args, curve_times=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ok, raw = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame for render")
    first = rotate_frame(raw, args.rotate)
    h, w = first.shape[:2]
    speed = max(1.0, float(args.speed))
    out_fps = fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {output_path}")

    frame_idx = 0
    next_write_time = 0.0
    output_step = speed / fps
    while ok:
        frame = rotate_frame(raw, args.rotate)
        t = frame_idx / fps
        if t + 1e-9 >= next_write_time:
            if curve_times:
                draw_curve_timed(frame, curve, curve_times, t, args.fade_start)
            elif t >= args.start:
                upto = min(1.0, max(0.0, (t - args.start) / max(0.35, args.draw_seconds)))
                draw_curve(frame, curve, upto, args.fade_start)
            writer.write(frame)
            next_write_time += output_step
        ok, raw = cap.read()
        frame_idx += 1

    cap.release()
    writer.release()


def save_debug_sheet(path, debug_items):
    if not debug_items:
        return
    tiles = []
    for frame, mask, candidates, best in debug_items:
        small = cv2.resize(frame, (216, 384))
        sx = 216 / frame.shape[1]
        sy = 384 / frame.shape[0]
        for c in candidates:
            if isinstance(c, Candidate):
                cx, cy = c.x, c.y
            else:
                cx, cy = c[0], c[1]
            cv2.circle(small, (int(cx * sx), int(cy * sy)), 3, (255, 255, 0), 1)
        if best:
            if isinstance(best, Candidate):
                bx, by = best.x, best.y
            else:
                bx, by = best[0], best[1]
            cv2.circle(small, (int(bx * sx), int(by * sy)), 7, (0, 0, 255), 2)
        tiles.append(small)
    cols = 6
    rows = math.ceil(len(tiles) / cols)
    blank = np.zeros_like(tiles[0])
    while len(tiles) < rows * cols:
        tiles.append(blank)
    sheet = np.vstack([
        np.hstack(tiles[r * cols:(r + 1) * cols])
        for r in range(rows)
    ])
    cv2.imwrite(path, sheet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--debug-sheet", default=None)
    parser.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none")
    parser.add_argument("--start", type=float, default=None, help="optional impact/launch search start time")
    parser.add_argument("--end", type=float, default=None, help="optional tracking search end time")
    parser.add_argument("--scan-start", type=float, default=None, help="optional full-video impact scan start")
    parser.add_argument("--scan-end", type=float, default=None, help="optional full-video impact scan end")
    parser.add_argument("--launch-delay", type=float, default=0.55, help="seconds after motion peak to begin ball search")
    parser.add_argument("--launch", type=parse_xy, default=None, help="optional x,y override in output pixels")
    parser.add_argument("--aim", type=parse_xy, default=None, help="optional x,y override in output pixels")
    parser.add_argument("--corridor", type=float, default=190.0)
    parser.add_argument("--search-radius", type=float, default=260.0)
    parser.add_argument("--diff-threshold", type=int, default=18)
    parser.add_argument("--max-lost", type=int, default=12)
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--draw-seconds", type=float, default=1.0)
    parser.add_argument("--projection-seconds", type=float, default=2.2)
    parser.add_argument("--detected-curve-portion", type=float, default=0.62)
    parser.add_argument("--line-preroll", type=float, default=1.05)
    parser.add_argument("--line-delay", type=float, default=0.0, help="output seconds after first detected ball before the trajectory starts")
    parser.add_argument("--line-draw-speed", type=float, default=1.0, help="trajectory drawing speed multiplier; 0.8 draws 20 percent slower")
    parser.add_argument("--fade-start", type=float, default=0.58)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed-up for exported video; use 8 for iPhone 240fps slow-mo")
    parser.add_argument("--arc-bend", type=float, default=70.0)
    parser.add_argument("--arc-lift", type=float, default=190.0)
    parser.add_argument("--gravity", type=float, default=0.08)
    parser.add_argument("--debug-masks", action="store_true")
    args = parser.parse_args()

    if args.launch and args.aim:
        if args.start is None:
            args.start = 0.0
        result = track_ball(args.input, args)
        args.frame_width = result["width"]
        args.frame_height = result["height"]
        curve = fit_curve(result["detections"], args)
        result["impact_t"] = args.start
        result["impact_frame"] = int(args.start * result["fps"])
        result["curve"] = curve
        result["curve_times"] = make_curve_times(curve, result["detections"], args.start, args)
    else:
        result = auto_track_ball(args.input, args)
        curve = result["curve"]
        args.start = result["impact_t"]
        args.frame_width = result["width"]
        args.frame_height = result["height"]
    render_video(args.input, args.output, curve, args, result.get("curve_times"))

    if args.debug_sheet:
        save_debug_sheet(args.debug_sheet, result["debug_masks"])
    if args.json:
        payload = {
            "fps": result["fps"],
            "frame_count": result["frame_count"],
            "width": result["width"],
            "height": result["height"],
            "impact_frame": result.get("impact_frame"),
            "impact_t": result.get("impact_t"),
            "launch": result["launch"],
            "aim": result["aim"],
            "detections": [d.__dict__ for d in result["detections"]],
            "curve_points": len(curve),
            "speed": args.speed,
            "curve_times": result.get("curve_times"),
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
