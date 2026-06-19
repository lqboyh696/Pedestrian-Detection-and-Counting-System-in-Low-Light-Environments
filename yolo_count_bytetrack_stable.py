"""
Stable in-memory person counting based on ByteTrack.

This module is an alternative to yolo_count_from_labels.py for the web app.
It does not modify the original B counting code. The main differences are:
- use ByteTrack IDs instead of greedy centroid matching;
- add a hysteresis band around the counting line to avoid boundary jitter;
- count each track only once after a clear directional line crossing.
"""

import cv2
import numpy as np
from types import SimpleNamespace
from ultralytics.utils.plotting import Annotator
from ultralytics.trackers import BYTETracker

from yolo_count_from_labels import (
    compute_zone_bounds,
    draw_cctv_header,
    draw_count_panel,
    draw_counting_line,
)


DEFAULT_HYSTERESIS_RATIO = 0.035


class StableByteTrackSession:
    """ByteTrack tuned for low-light people counting from per-frame YOLO boxes."""

    def __init__(self):
        self.tracker = BYTETracker(
            args=SimpleNamespace(
                track_low_thresh=0.05,
                track_high_thresh=0.25,
                new_track_thresh=0.25,
                track_buffer=60,
                match_thresh=0.8,
                fuse_score=True,
            )
        )

    def update(self, boxes, frame_shape):
        if boxes:
            xyxy = np.array([[b[0], b[1], b[2], b[3]] for b in boxes], dtype=np.float32)
            conf = np.array([b[4] for b in boxes], dtype=np.float32)
            cls_arr = np.zeros(len(boxes), dtype=np.float32)
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            conf = np.empty((0,), dtype=np.float32)
            cls_arr = np.empty((0,), dtype=np.float32)

        class ResultsForTracker:
            def __init__(self, xyxy_arr, conf_arr, cls_arr):
                self.xyxy = xyxy_arr
                self.xywh = self._xyxy2xywh(xyxy_arr)
                self.conf = conf_arr
                self.cls = cls_arr

            @staticmethod
            def _xyxy2xywh(x):
                r = np.zeros_like(x)
                if len(x):
                    r[:, 0] = (x[:, 0] + x[:, 2]) / 2
                    r[:, 1] = (x[:, 1] + x[:, 3]) / 2
                    r[:, 2] = x[:, 2] - x[:, 0]
                    r[:, 3] = x[:, 3] - x[:, 1]
                return r

            def __getitem__(self, idx):
                return ResultsForTracker(self.xyxy[idx], self.conf[idx], self.cls[idx])

            def __len__(self):
                return len(self.xyxy)

        tracked = self.tracker.update(ResultsForTracker(xyxy, conf, cls_arr), img=None)
        result = []
        for t in tracked:
            x1, y1, x2, y2 = float(t[0]), float(t[1]), float(t[2]), float(t[3])
            track_id = int(t[4])
            score = float(t[5])
            result.append((x1, y1, x2, y2, score, "person", track_id))
        return result

    def reset(self):
        self.tracker.reset()


def _ensure_tracker(state):
    if state.get("tracker") is None:
        state["tracker"] = StableByteTrackSession()
    state.setdefault("count", 0)
    state.setdefault("counted_ids", set())
    state.setdefault("track_sides", {})
    state.setdefault("raw_to_stable", {})
    state.setdefault("stable_tracks", {})
    state.setdefault("next_stable_id", 1)
    state.setdefault("frame_index", 0)
    return state["tracker"]


def _center(box):
    x1, y1, x2, y2 = box[:4]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _foot_point(box):
    x1, _y1, x2, y2 = box[:4]
    return (x1 + x2) / 2.0, y2


def _point_in_bounds(point, bounds):
    x, y = point
    x1, y1, x2, y2 = bounds
    return x1 <= x < x2 and y1 <= y < y2


def _box_touches_image_edge(box, width, height, margin_ratio=0.035):
    x1, y1, x2, y2 = box[:4]
    margin = max(8.0, min(width, height) * margin_ratio)
    return x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin


def _axis_value(cx, cy, is_horizontal):
    return cy if is_horizontal else cx


def _assign_stable_ids(state, tracked_boxes, max_age=30):
    """Reconnect short ByteTrack ID switches using recent center distance."""
    state["frame_index"] = state.get("frame_index", 0) + 1
    frame_index = state["frame_index"]
    assigned_stable = set()
    resolved = []

    for x1, y1, x2, y2, conf, cls_name, raw_id in tracked_boxes:
        cx, cy = _center((x1, y1, x2, y2))
        fx, fy = _foot_point((x1, y1, x2, y2))
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        stable_id = state["raw_to_stable"].get(raw_id)

        if stable_id is None or stable_id in assigned_stable:
            best_id = None
            best_dist = None
            for candidate_id, info in state["stable_tracks"].items():
                if candidate_id in assigned_stable:
                    continue
                age = frame_index - info["last_seen"]
                if age > max_age:
                    continue
                px, py = info.get("foot", info["center"])
                foot_dist = ((fx - px) ** 2 + (fy - py) ** 2) ** 0.5
                cx0, cy0 = info["center"]
                center_dist = ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5
                dist = 0.7 * foot_dist + 0.3 * center_dist
                ref_diag = ((max(w, info["size"][0])) ** 2 + (max(h, info["size"][1])) ** 2) ** 0.5
                gate = max(110.0, 0.95 * ref_diag)
                if dist <= gate and (best_dist is None or dist < best_dist):
                    best_id = candidate_id
                    best_dist = dist

            if best_id is not None:
                stable_id = best_id
            else:
                stable_id = state["next_stable_id"]
                state["next_stable_id"] += 1
            state["raw_to_stable"][raw_id] = stable_id

        assigned_stable.add(stable_id)
        first_seen = state["stable_tracks"].get(stable_id, {}).get("first_seen", frame_index)
        state["stable_tracks"][stable_id] = {
            "center": (cx, cy),
            "foot": (fx, fy),
            "size": (w, h),
            "last_seen": frame_index,
            "first_seen": first_seen,
        }
        resolved.append((x1, y1, x2, y2, conf, cls_name, stable_id))

    stale_ids = [
        stable_id for stable_id, info in state["stable_tracks"].items()
        if frame_index - info["last_seen"] > max_age * 2
    ]
    for stable_id in stale_ids:
        state["stable_tracks"].pop(stable_id, None)
        state["track_sides"].pop(stable_id, None)

    return resolved


def _track_age(state, stable_id):
    info = state.get("stable_tracks", {}).get(stable_id)
    if not info:
        return 0
    return state.get("frame_index", 0) - info.get("first_seen", state.get("frame_index", 0)) + 1


def _is_recent_duplicate_crossing(state, stable_id, cx, cy, zone_size,
                                  window_frames=60, distance_ratio=0.35):
    frame_index = state.get("frame_index", 0)
    recent = state.setdefault("recent_crossings", [])
    max_dist = max(45.0, zone_size * distance_ratio)

    kept = []
    duplicate = False
    for item in recent:
        age = frame_index - item["frame"]
        if age <= window_frames:
            kept.append(item)
            dx = cx - item["center"][0]
            dy = cy - item["center"][1]
            dist = (dx * dx + dy * dy) ** 0.5
            if item["id"] != stable_id and dist <= max_dist:
                duplicate = True

    state["recent_crossings"] = kept
    return duplicate


def _remember_crossing(state, stable_id, cx, cy):
    state.setdefault("recent_crossings", []).append({
        "id": stable_id,
        "center": (cx, cy),
        "frame": state.get("frame_index", 0),
    })


def _is_recent_duplicate_zone_event(state, stable_id, point, event_type,
                                    window_frames=72, max_dist=80.0):
    frame_index = state.get("frame_index", 0)
    recent = state.setdefault("recent_zone_events", [])

    kept = []
    duplicate = False
    for item in recent:
        age = frame_index - item["frame"]
        if age <= window_frames:
            kept.append(item)
            if item["type"] != event_type or item["id"] == stable_id:
                continue
            dx = point[0] - item["point"][0]
            dy = point[1] - item["point"][1]
            if (dx * dx + dy * dy) ** 0.5 <= max_dist:
                duplicate = True

    state["recent_zone_events"] = kept
    return duplicate


def _remember_zone_event(state, stable_id, point, event_type):
    state.setdefault("recent_zone_events", []).append({
        "id": stable_id,
        "point": point,
        "type": event_type,
        "frame": state.get("frame_index", 0),
    })


def _ensure_io_counts(state):
    state.setdefault("in_count", 0)
    state.setdefault("out_count", 0)
    state.setdefault("current_count", 0)
    state.setdefault("effective_inside_ids", set())
    state.setdefault("initial_counted_ids", set())
    state.setdefault("entry_seen_frames", 0)
    state["count"] = state.get("in_count", 0)


def _update_io_counts(state, current_inside_ids, warmup_active):
    previous_inside_ids = state.setdefault("effective_inside_ids", set())
    state["current_count"] = len(current_inside_ids)
    if not warmup_active:
        state["in_count"] += len(current_inside_ids - previous_inside_ids)
        state["out_count"] += len(previous_inside_ids - current_inside_ids)
    state["effective_inside_ids"] = set(current_inside_ids)
    state["count"] = state["in_count"]


def _append_foot_trail(state, stable_id, point, max_len=24):
    trails = state.setdefault("foot_trails", {})
    trail = trails.setdefault(stable_id, [])
    trail.append((int(point[0]), int(point[1])))
    if len(trail) > max_len:
        del trail[:-max_len]
    return trail


def _entered_from_direction(prev_center, curr_center, bounds, direction):
    """Return True when a center crosses into bounds from the selected side."""
    if prev_center is None:
        return False

    x1, y1, x2, y2 = bounds
    prev_cx, prev_cy = prev_center
    cx, cy = curr_center

    if direction == "top_to_bottom":
        return x1 <= cx < x2 and prev_cy < y1 <= cy
    if direction == "bottom_to_top":
        return x1 <= cx < x2 and prev_cy >= y2 > cy
    if direction == "left_to_right":
        return y1 <= cy < y2 and prev_cx < x1 <= cx
    if direction == "right_to_left":
        return y1 <= cy < y2 and prev_cx >= x2 > cx
    return False


def _line_side(value, line_pos, margin):
    if value < line_pos - margin:
        return -1
    if value > line_pos + margin:
        return 1
    return 0


def _crossed_direction(prev_side, curr_side, direction):
    if prev_side == 0 or curr_side == 0:
        return False
    if direction in ("top_to_bottom", "left_to_right"):
        return prev_side < 0 and curr_side > 0
    if direction in ("bottom_to_top", "right_to_left"):
        return prev_side > 0 and curr_side < 0
    return False


def _line_for_zone(width, height, zone_id, direction, line_percent):
    zx1, zy1, zx2, zy2 = compute_zone_bounds(width, height, zone_id)
    zone_w, zone_h = zx2 - zx1, zy2 - zy1
    ratio = line_percent / 100.0

    if direction == "top_to_bottom":
        return (zx1, zy1, zx2, zy2), zy1 + zone_h * ratio, True
    if direction == "bottom_to_top":
        return (zx1, zy1, zx2, zy2), zy2 - zone_h * ratio, True
    if direction == "left_to_right":
        return (zx1, zy1, zx2, zy2), zx1 + zone_w * ratio, False
    if direction == "right_to_left":
        return (zx1, zy1, zx2, zy2), zx2 - zone_w * ratio, False
    return (zx1, zy1, zx2, zy2), zy1 + zone_h * ratio, True


def _draw_zone_overlay(frame, zone_id, zone_bounds, line_pos, is_horizontal, count, fps_val, show_fps):
    zx1, zy1, zx2, zy2 = zone_bounds
    zone_color = (0, 180, 255)

    ov_zone = frame.copy()
    cv2.rectangle(ov_zone, (zx1, zy1), (zx2, zy2), zone_color, 2)
    cv2.putText(ov_zone, f"Z{zone_id}", (zx1 + 6, zy1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_color, 2)
    cv2.addWeighted(ov_zone, 0.5, frame, 0.5, 0, frame)

    ov_line = frame.copy()
    lp = int(line_pos)
    if is_horizontal:
        cv2.line(ov_line, (zx1, lp), (zx2, lp), (60, 60, 255), 3)
    else:
        cv2.line(ov_line, (lp, zy1), (lp, zy2), (60, 60, 255), 3)
    cv2.addWeighted(ov_line, 0.4, frame, 0.6, 0, frame)

    badge_text = f"Z{zone_id} | {count}"
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ov = frame.copy()
    cv2.rectangle(ov, (6, 38), (tw + 18, th + 50), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, badge_text, (12, th + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    draw_cctv_header(frame, f"CAM 01-Z{zone_id}", fps_val, show_fps=show_fps)


def _draw_zone_entry_overlay(frame, zone_id, zone_bounds, count, fps_val, show_fps):
    zx1, zy1, zx2, zy2 = zone_bounds
    zone_color = (0, 180, 255)
    zone_label = "FULL" if zone_id is None else f"Z{zone_id}"

    ov_zone = frame.copy()
    cv2.rectangle(ov_zone, (zx1, zy1), (zx2, zy2), zone_color, 2)
    cv2.putText(ov_zone, zone_label, (zx1 + 6, zy1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_color, 2)
    cv2.addWeighted(ov_zone, 0.5, frame, 0.5, 0, frame)

    if isinstance(count, dict):
        badge_text = (
            f"{zone_label} CUR {count.get('current', 0)} "
            f"IN {count.get('in', 0)} OUT {count.get('out', 0)}"
        )
    else:
        badge_text = f"{zone_label} Entry | {count}"
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ov = frame.copy()
    cv2.rectangle(ov, (6, 38), (tw + 18, th + 50), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, badge_text, (12, th + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    draw_cctv_header(frame, f"CAM 01-{zone_label}", fps_val, show_fps=show_fps)


def process_frame_memory_bytetrack(frame, curr_boxes, state,
                                   fps_val=0.0, line_ratio=0.5,
                                   is_video=True, show_fps=True,
                                   hysteresis_ratio=DEFAULT_HYSTERESIS_RATIO):
    height, width = frame.shape[:2]
    line_y = int(height * line_ratio)

    if is_video:
        tracker = _ensure_tracker(state)
        tracked_boxes = _assign_stable_ids(state, tracker.update(curr_boxes, (height, width)))
    else:
        tracked_boxes = [(b[0], b[1], b[2], b[3], b[4], b[5], i) for i, b in enumerate(curr_boxes)]
        state["count"] = len(curr_boxes)

    annotator = Annotator(frame)
    margin = max(6.0, height * hysteresis_ratio)

    for x1, y1, x2, y2, conf, cls_name, track_id in tracked_boxes:
        cx, cy = _center((x1, y1, x2, y2))
        label = f"{cls_name}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (200, 50, 50), 2)

        if is_video:
            side = _line_side(cy, line_y, margin)
            prev_side = state["track_sides"].get(track_id)
            if prev_side is not None and _crossed_direction(prev_side, side, "top_to_bottom"):
                if track_id not in state["counted_ids"]:
                    state["count"] += 1
                    state["counted_ids"].add(track_id)
            if side != 0:
                state["track_sides"][track_id] = side

    if is_video:
        draw_counting_line(frame, line_y)
        draw_cctv_header(frame, "CAM 01", fps_val, show_fps=show_fps)
    draw_count_panel(frame, state["count"])
    return frame, state


def process_frame_memory_full_lines(frame, curr_boxes, state, line_id="h1",
                                    direction="top_to_bottom", fps_val=0.0,
                                    is_video=True, show_fps=True,
                                    hysteresis_ratio=DEFAULT_HYSTERESIS_RATIO):
    """Full-frame ByteTrack counting on one selected global line."""
    height, width = frame.shape[:2]
    line_id = line_id or "h1"
    direction = direction or ("left_to_right" if line_id.startswith("v") else "top_to_bottom")

    if line_id == "v1":
        line_pos, is_horizontal, line_name = width / 3.0, False, "V1"
    elif line_id == "v2":
        line_pos, is_horizontal, line_name = width * 2.0 / 3.0, False, "V2"
    else:
        line_pos, is_horizontal, line_name = height / 2.0, True, "H1"

    if is_video:
        tracker = _ensure_tracker(state)
        tracked_boxes = _assign_stable_ids(state, tracker.update(curr_boxes, (height, width)))
    else:
        tracked_boxes = [(b[0], b[1], b[2], b[3], b[4], b[5], i) for i, b in enumerate(curr_boxes)]
        state["count"] = len(curr_boxes)

    annotator = Annotator(frame)
    margin_base = height if is_horizontal else width
    margin = max(6.0, margin_base * hysteresis_ratio)

    for x1, y1, x2, y2, conf, cls_name, track_id in tracked_boxes:
        foot = _foot_point((x1, y1, x2, y2))
        label = f"{cls_name}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (200, 50, 50), 2)

        trail = _append_foot_trail(state, track_id, foot)
        if len(trail) >= 2:
            cv2.polylines(frame, [np.array(trail, dtype=np.int32)], False, (0, 255, 255), 2)
        cv2.circle(frame, (int(foot[0]), int(foot[1])), 4, (0, 255, 255), -1)

        if is_video:
            value = foot[1] if is_horizontal else foot[0]
            side = _line_side(value, line_pos, margin)
            prev_side = state["track_sides"].get(track_id)
            if prev_side is not None and _crossed_direction(prev_side, side, direction):
                if track_id not in state["counted_ids"]:
                    state["count"] += 1
                    state["counted_ids"].add(track_id)
            if side != 0:
                state["track_sides"][track_id] = side

    if is_horizontal:
        y = int(line_pos)
        cv2.line(frame, (0, y), (width, y), (0, 0, 255), 3)
    else:
        x = int(line_pos)
        cv2.line(frame, (x, 0), (x, height), (0, 0, 255), 3)

    badge_text = f"{line_name} {direction} | {state.get('count', 0)}"
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ov = frame.copy()
    cv2.rectangle(ov, (6, 38), (tw + 18, th + 50), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, badge_text, (12, th + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    draw_cctv_header(frame, f"CAM 01-{line_name}", fps_val, show_fps=show_fps)
    return frame, state


def process_frame_memory_zones_bytetrack(frame, curr_boxes, zone_id, direction, zone_state,
                                         fps_val=0.0, is_video=True, line_percent=12,
                                         show_fps=True,
                                         hysteresis_ratio=DEFAULT_HYSTERESIS_RATIO):
    height, width = frame.shape[:2]
    zone_bounds, line_pos, is_horizontal = _line_for_zone(width, height, zone_id, direction, line_percent)
    zx1, zy1, zx2, zy2 = zone_bounds
    zone_size = (zy2 - zy1) if is_horizontal else (zx2 - zx1)
    margin = max(6.0, zone_size * hysteresis_ratio)

    if is_video:
        tracker = _ensure_tracker(zone_state)
        tracked_boxes = _assign_stable_ids(zone_state, tracker.update(curr_boxes, (height, width)))
    else:
        tracked_boxes = [(b[0], b[1], b[2], b[3], b[4], b[5], i) for i, b in enumerate(curr_boxes)]
        zone_state["count"] = sum(
            1 for b in curr_boxes
            if zx1 <= (b[0] + b[2]) / 2.0 < zx2 and zy1 <= (b[1] + b[3]) / 2.0 < zy2
        )

    annotator = Annotator(frame)
    for x1, y1, x2, y2, conf, cls_name, track_id in tracked_boxes:
        cx, cy = _center((x1, y1, x2, y2))
        in_zone_now = zx1 <= cx < zx2 and zy1 <= cy < zy2
        label = f"{cls_name}"
        color = (30, 180, 60) if in_zone_now else (200, 50, 50)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        if is_video and in_zone_now:
            value = _axis_value(cx, cy, is_horizontal)
            side = _line_side(value, line_pos, margin)
            prev_side = zone_state["track_sides"].get(track_id)
            hold_side = False
            if prev_side is not None and _crossed_direction(prev_side, side, direction):
                if track_id not in zone_state["counted_ids"]:
                    is_duplicate = _is_recent_duplicate_crossing(
                        zone_state, track_id, cx, cy, zone_size
                    )
                    is_too_new = _track_age(zone_state, track_id) < 3
                    if not is_duplicate and not is_too_new:
                        zone_state["count"] += 1
                        zone_state["counted_ids"].add(track_id)
                        _remember_crossing(zone_state, track_id, cx, cy)
                    elif is_too_new:
                        hold_side = True
            if side != 0:
                if not hold_side:
                    zone_state["track_sides"][track_id] = side

    _draw_zone_overlay(frame, zone_id, zone_bounds, line_pos, is_horizontal,
                       zone_state["count"], fps_val, show_fps)
    return frame, zone_state


def process_frame_memory_zones_entry(frame, curr_boxes, zone_id, direction, zone_state,
                                     fps_val=0.0, is_video=True, line_percent=12,
                                     show_fps=True, warmup_frames=5):
    """Count once when a tracked person clearly enters the selected zone.

    The UI can present this as "enter zone counting". Internally we use the
    existing line_percent as a hidden inward margin from the entry boundary.
    """
    height, width = frame.shape[:2]
    zone_bounds = (0, 0, width, height) if zone_id is None else compute_zone_bounds(width, height, zone_id)
    zx1, zy1, zx2, zy2 = zone_bounds
    zone_w, zone_h = zx2 - zx1, zy2 - zy1
    margin_ratio = max(1, min(50, int(line_percent))) / 100.0

    if direction == "top_to_bottom":
        inner_x1, inner_y1, inner_x2, inner_y2 = zx1, zy1 + zone_h * margin_ratio, zx2, zy2
    elif direction == "bottom_to_top":
        inner_x1, inner_y1, inner_x2, inner_y2 = zx1, zy1, zx2, zy2 - zone_h * margin_ratio
    elif direction == "left_to_right":
        inner_x1, inner_y1, inner_x2, inner_y2 = zx1 + zone_w * margin_ratio, zy1, zx2, zy2
    elif direction == "right_to_left":
        inner_x1, inner_y1, inner_x2, inner_y2 = zx1, zy1, zx2 - zone_w * margin_ratio, zy2
    else:
        inner_x1, inner_y1, inner_x2, inner_y2 = zx1, zy1, zx2, zy2

    if is_video:
        tracker = _ensure_tracker(zone_state)
        tracked_boxes = _assign_stable_ids(zone_state, tracker.update(curr_boxes, (height, width)))
        _ensure_io_counts(zone_state)
        zone_state["entry_seen_frames"] = zone_state.get("entry_seen_frames", 0) + 1
    else:
        tracked_boxes = [(b[0], b[1], b[2], b[3], b[4], b[5], i) for i, b in enumerate(curr_boxes)]
        zone_state["in_count"] = sum(
            1 for b in curr_boxes
            if inner_x1 <= (b[0] + b[2]) / 2.0 < inner_x2
            and inner_y1 <= (b[1] + b[3]) / 2.0 < inner_y2
        )
        zone_state["out_count"] = 0
        zone_state["count"] = zone_state["in_count"]

    annotator = Annotator(frame)
    current_inside = set()
    warmup_active = is_video and zone_state["entry_seen_frames"] <= warmup_frames
    for x1, y1, x2, y2, conf, cls_name, stable_id in tracked_boxes:
        cx, cy = _center((x1, y1, x2, y2))
        in_zone_now = zx1 <= cx < zx2 and zy1 <= cy < zy2
        clearly_entered = inner_x1 <= cx < inner_x2 and inner_y1 <= cy < inner_y2
        if clearly_entered:
            current_inside.add(stable_id)

        label = f"{cls_name}"
        color = (30, 180, 60) if in_zone_now else (200, 50, 50)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    if is_video:
        _update_io_counts(zone_state, current_inside, warmup_active)

    _draw_zone_entry_overlay(
        frame, zone_id, zone_bounds,
        {"in": zone_state.get("in_count", zone_state.get("count", 0)),
         "out": zone_state.get("out_count", 0)},
        fps_val, show_fps
    )
    return frame, zone_state


def process_frame_memory_zones_direct_entry(frame, curr_boxes, zone_id, direction, zone_state,
                                            fps_val=0.0, is_video=True, line_percent=12,
                                            show_fps=True, warmup_frames=5):
    """Count once when a tracked person's center enters the selected zone boundary."""
    height, width = frame.shape[:2]
    zone_bounds = (0, 0, width, height) if zone_id is None else compute_zone_bounds(width, height, zone_id)
    zx1, zy1, zx2, zy2 = zone_bounds

    if is_video:
        tracker = _ensure_tracker(zone_state)
        tracked_boxes = _assign_stable_ids(zone_state, tracker.update(curr_boxes, (height, width)))
        _ensure_io_counts(zone_state)
        zone_state["entry_seen_frames"] = zone_state.get("entry_seen_frames", 0) + 1
    else:
        tracked_boxes = [(b[0], b[1], b[2], b[3], b[4], b[5], i) for i, b in enumerate(curr_boxes)]
        zone_state["in_count"] = sum(
            1 for b in curr_boxes
            if zx1 <= (b[0] + b[2]) / 2.0 < zx2 and zy1 <= (b[1] + b[3]) / 2.0 < zy2
        )
        zone_state["out_count"] = 0
        zone_state["count"] = zone_state["in_count"]

    annotator = Annotator(frame)
    current_inside = set()
    warmup_active = is_video and zone_state["entry_seen_frames"] <= warmup_frames

    for x1, y1, x2, y2, conf, cls_name, stable_id in tracked_boxes:
        cx, cy = _center((x1, y1, x2, y2))
        in_zone_now = zx1 <= cx < zx2 and zy1 <= cy < zy2
        if in_zone_now:
            current_inside.add(stable_id)

        label = f"{cls_name}"
        color = (30, 180, 60) if in_zone_now else (200, 50, 50)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    if is_video:
        _update_io_counts(zone_state, current_inside, warmup_active)

    _draw_zone_entry_overlay(
        frame, zone_id, zone_bounds,
        {"in": zone_state.get("in_count", zone_state.get("count", 0)),
         "out": zone_state.get("out_count", 0)},
        fps_val, show_fps
    )
    return frame, zone_state


def process_frame_memory_zones_foot_region(frame, curr_boxes, zone_id, direction, zone_state,
                                           fps_val=0.0, is_video=True, line_percent=12,
                                           show_fps=True, warmup_frames=5,
                                           confirm_frames=2):
    """Region IN/OUT counting using bottom-center foot-point trajectories."""
    height, width = frame.shape[:2]
    zone_bounds = (0, 0, width, height) if zone_id is None else compute_zone_bounds(width, height, zone_id)
    zx1, zy1, zx2, zy2 = zone_bounds

    if is_video:
        tracker = _ensure_tracker(zone_state)
        tracked_boxes = _assign_stable_ids(zone_state, tracker.update(curr_boxes, (height, width)))
        _ensure_io_counts(zone_state)
        zone_state.setdefault("foot_states", {})
        zone_state["entry_seen_frames"] = zone_state.get("entry_seen_frames", 0) + 1
    else:
        tracked_boxes = [(b[0], b[1], b[2], b[3], b[4], b[5], i) for i, b in enumerate(curr_boxes)]
        zone_state["current_count"] = sum(1 for b in curr_boxes if _point_in_bounds(_foot_point(b), zone_bounds))
        zone_state["in_count"] = zone_state["current_count"]
        zone_state["out_count"] = 0
        zone_state["count"] = zone_state["in_count"]

    annotator = Annotator(frame)
    warmup_active = is_video and zone_state["entry_seen_frames"] <= warmup_frames
    current_inside_ids = set()

    for x1, y1, x2, y2, conf, cls_name, stable_id in tracked_boxes:
        box = (x1, y1, x2, y2)
        foot = _foot_point((x1, y1, x2, y2))
        in_zone_now = _point_in_bounds(foot, zone_bounds)
        born_from_image_edge = in_zone_now and _box_touches_image_edge(box, width, height)
        if in_zone_now:
            current_inside_ids.add(stable_id)

        label = f"{cls_name}"
        color = (30, 180, 60) if in_zone_now else (200, 50, 50)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        trail = _append_foot_trail(zone_state, stable_id, foot)
        if len(trail) >= 2:
            cv2.polylines(frame, [np.array(trail, dtype=np.int32)], False, (0, 255, 255), 2)
        cv2.circle(frame, (int(foot[0]), int(foot[1])), 4, (0, 255, 255), -1)

        if not is_video:
            continue

        foot_states = zone_state["foot_states"]
        state = foot_states.setdefault(stable_id, {
            "inside": in_zone_now,
            "inside_streak": 0,
            "outside_streak": 0,
            "pending_in": False,
            "pending_out": False,
            "edge_birth_pending_in": born_from_image_edge,
            # A track born inside the zone has no observed boundary crossing.
            # Ignore its first exit; start counting after it is observed outside.
            "eligible": (not in_zone_now) or born_from_image_edge,
        })

        if warmup_active:
            if in_zone_now and stable_id not in zone_state["initial_counted_ids"]:
                zone_state["in_count"] += 1
                zone_state["initial_counted_ids"].add(stable_id)
                _remember_zone_event(zone_state, stable_id, foot, "in")
            state["inside"] = in_zone_now
            state["inside_streak"] = confirm_frames if in_zone_now else 0
            state["outside_streak"] = confirm_frames if not in_zone_now else 0
            state["pending_in"] = False
            state["pending_out"] = False
            state["edge_birth_pending_in"] = False
            state["eligible"] = True
            continue

        if in_zone_now:
            state["inside_streak"] += 1
            state["outside_streak"] = 0
            if not state["inside"]:
                state["pending_in"] = True
            if state["pending_in"] and state["inside_streak"] >= confirm_frames:
                if state.get("eligible", True) and not _is_recent_duplicate_zone_event(zone_state, stable_id, foot, "in"):
                    zone_state["in_count"] += 1
                    _remember_zone_event(zone_state, stable_id, foot, "in")
                state["inside"] = True
                state["pending_in"] = False
                state["eligible"] = True
            elif state.get("edge_birth_pending_in") and state["inside_streak"] >= confirm_frames:
                if not _is_recent_duplicate_zone_event(zone_state, stable_id, foot, "in"):
                    zone_state["in_count"] += 1
                    _remember_zone_event(zone_state, stable_id, foot, "in")
                state["edge_birth_pending_in"] = False
                state["inside"] = True
                state["eligible"] = True
        else:
            state["outside_streak"] += 1
            state["inside_streak"] = 0
            if state["inside"]:
                state["pending_out"] = True
            if state["pending_out"] and state["outside_streak"] >= confirm_frames:
                if state.get("eligible", True) and not _is_recent_duplicate_zone_event(zone_state, stable_id, foot, "out"):
                    zone_state["out_count"] += 1
                    _remember_zone_event(zone_state, stable_id, foot, "out")
                state["inside"] = False
                state["pending_out"] = False
                state["eligible"] = True

    if is_video:
        zone_state["count"] = zone_state.get("in_count", 0)
        zone_state["current_count"] = len(current_inside_ids)

    _draw_zone_entry_overlay(
        frame, zone_id, zone_bounds,
        {"current": zone_state.get("current_count", 0),
         "in": zone_state.get("in_count", zone_state.get("count", 0)),
         "out": zone_state.get("out_count", 0)},
        fps_val, show_fps
    )
    return frame, zone_state
