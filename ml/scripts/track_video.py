"""Legacy golf-ball tracking baseline.

This preserves the Kalman + ROI-cropped YOLO + RANSAC experiment for measured
comparison with the replacement architecture in Master Plan.md. Its accepted
detections are hypotheses, not known-genuine ball observations.

Usage:
    ml/.venv/bin/python ml/scripts/track_video.py path/to/swing.mp4

Click the ball's position in the first frame when prompted (Ball
Initialization / Tap-to-Start — see Master Plan.md Phase 2).

Two passes:
  1. Collection — the Kalman + ROI-crop + YOLO loop runs across the whole
     video and records every frame's genuine detection. No per-frame trust
     decisions here; the point of this pass is just to gather candidates.
  2. Trajectory fit — a golf ball in flight follows a parabola in screen
     space. RANSAC fits x(t) and y(t) as independent quadratics against the
     candidates collected from the point flight begins onward, keeping only
     the points consistent with the best-supported curve. Background
     clutter that momentarily fooled the detector (a lamp post, a flag,
     dust kicked up at impact, clouds) shows up as scattered outliers that
     don't lie on any consistent parabola with the real ball's points, and
     gets discarded. The final trace is the fitted curve, not raw per-frame
     positions, so it's smooth by construction and can't show the hard
     direction-reversals a single bad detection used to cause.

Address-phase frames (before the ball is struck) are excluded from the fit
entirely — a single quadratic can't represent "stationary for hundreds of
frames, then a sudden parabola," and with far more address frames than
flight frames, RANSAC would otherwise just converge on "the ball barely
moved" as its largest consensus.

Output: an annotated video plus a machine-readable diagnostics JSON containing
the accepted candidate hypotheses and rendered trajectory.
"""

import argparse
import os
import random
import sys
import warnings

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tracker.kalman import KalmanFilter2D
from tracker.annotation_io import POINT_VISIBILITIES, atomic_write_json, load_annotation_document

# RANSAC's minimal samples (3 points) are frequently close together in time,
# which np.polyfit correctly but noisily flags as poorly conditioned. Fitting
# still works; suppress the expected warning spam.
warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")

ROI_SIZE = 300
CONF_THRESHOLD = 0.1
# On a miss, widen the search crop geometrically (capped at MAX_ROI_MULTIPLIER
# x the base size) instead of continuing to scan the same now-empty patch.
# This used to be unsafe — an earlier version of this script trusted
# whatever a widened crop found, so it could wander onto and lock onto real
# background clutter (a cart, a lamp post). It's safe now: this pass only
# *collects* candidates, it doesn't trust any of them, so a wider net just
# means more chances to re-catch the real ball later in the flight. Anything
# false that a wider search sweeps in gets sorted out by RANSAC in pass 2.
MISS_GROWTH_RATE = 1.5
MAX_ROI_MULTIPLIER = 5
# Crop growth only makes sense once the ball is actually moving fast enough
# to outrun the base crop — during address, a miss usually just means the
# detector didn't fire that frame, not that the ball went anywhere. Widening
# on address misses was actively harmful: it swept in far-away false
# positives and corrupted launch detection (see COARSE_LAUNCH_DISPLACEMENT).
# This is a rough, single-frame trigger only used to gate crop growth — the
# real launch point for the trajectory fit is decided later, more robustly,
# by find_launch_frame() over the cleaned-up candidate list.
COARSE_LAUNCH_DISPLACEMENT = 50.0
# Launch detection: the first stretch of LAUNCH_WINDOW consecutive
# candidates whose combined displacement exceeds LAUNCH_DISPLACEMENT, held
# for LAUNCH_CONFIRM consecutive windows, marks where address ends and real
# flight begins.
LAUNCH_WINDOW = 8
LAUNCH_DISPLACEMENT = 150.0
LAUNCH_CONFIRM = 3
# RANSAC parabola fit over the flight segment.
RANSAC_ITERS = 500
RANSAC_INLIER_THRESHOLD = 30.0
RANSAC_MIN_INLIERS = 6
# Motion filter: a golf ball in flight always displaces frame to frame; a
# lamp post, flag, cloud, or dust settling never does for long. Once
# airborne, a detection only counts if the region it covers actually shows
# frame-to-frame pixel change — gated on `airborne` because the ball itself
# is genuinely stationary during address, right up until it's struck, and
# would fail a motion check for no good reason during that phase.
# A small ball against a near-uniform sky produces much weaker frame-to-frame
# contrast than one crossing grass/trees, so these have to stay forgiving —
# clouds drift far too slowly to register on a single-frame diff regardless,
# so there's little risk in being lenient here.
MOTION_DIFF_THRESHOLD = 8
MOTION_MIN_FRACTION = 0.015
DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "models", "golf_ball_yolov8n", "weights", "best.pt"
)


def pick_initial_point(frame):
    """Ball Initialization: let the user click the ball's position in frame 1."""
    picked = {}

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            picked["point"] = (x, y)

    window = "Click the ball, then press any key"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)

    while "point" not in picked:
        display = frame.copy()
        cv2.putText(display, "Click the ball's position", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window, display)
        if cv2.waitKey(20) != -1 and "point" in picked:
            break
    cv2.waitKey(1)
    cv2.destroyWindow(window)
    return picked["point"]


def initial_point_from_annotations(annotation_path):
    """Use the median visible address point as a reproducible tap equivalent."""
    document = load_annotation_document(annotation_path)
    address_points = [
        frame
        for frame in document["frames"]
        if frame["shot_state"] == "address"
        and frame["visibility"] in POINT_VISIBILITIES
    ]
    if not address_points:
        raise ValueError("Annotation file has no visible/blurred address points")
    return (
        float(np.median([frame["x"] for frame in address_points])),
        float(np.median([frame["y"] for frame in address_points])),
    )


def crop_roi(frame, center_x, center_y, size):
    h, w = frame.shape[:2]
    size = min(size, w, h)
    half = size // 2
    x1 = int(np.clip(center_x - half, 0, w - size))
    y1 = int(np.clip(center_y - half, 0, h - size))
    return frame[y1:y1 + size, x1:x1 + size], x1, y1


def compute_motion_mask(prev_frame, curr_frame, diff_threshold=MOTION_DIFF_THRESHOLD):
    """Binary mask of pixels that changed between two consecutive frames."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    return cv2.dilate(mask, None, iterations=2)


def box_has_motion(motion_mask_crop, x1, y1, x2, y2, min_fraction=MOTION_MIN_FRACTION):
    h, w = motion_mask_crop.shape[:2]
    bx1, by1 = max(0, int(x1)), max(0, int(y1))
    bx2, by2 = min(w, int(x2)), min(h, int(y2))
    if bx2 <= bx1 or by2 <= by1:
        return False
    region = motion_mask_crop[by1:by2, bx1:bx2]
    return float(np.count_nonzero(region)) / region.size >= min_fraction


def detect_ball_in_roi(model, roi, conf_threshold, offset_x, offset_y, pred_x, pred_y,
                        motion_mask_crop=None):
    """Among all confident candidates in the crop, pick the one closest to
    the Kalman-predicted position — not simply the highest-confidence one.
    A driving range has other real, similarly-confident golf-ball-like
    objects in view (loose balls on the ground, a cart, a lamp post); taking
    the top-confidence box regardless of location lets the tracker jump onto
    any of those instead of the one actually being tracked.

    If motion_mask_crop is given, the chosen candidate is also required to
    overlap real frame-to-frame pixel motion — a lamp post, flag, cloud, or
    settled dust can be confidently (mis)detected as a ball, but none of
    them are actually moving the way a ball in flight is.
    """
    results = model.predict(roi, conf=conf_threshold, verbose=False)[0]
    if len(results.boxes) == 0:
        return None

    def dist_to_pred(box):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = offset_x + (x1 + x2) / 2, offset_y + (y1 + y2) / 2
        return (cx - pred_x) ** 2 + (cy - pred_y) ** 2

    box = min(results.boxes, key=dist_to_pred)
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    if motion_mask_crop is not None and not box_has_motion(motion_mask_crop, x1, y1, x2, y2):
        return None

    return (x1 + x2) / 2, (y1 + y2) / 2


def collect_candidates(model, cap, first_frame, init_x, init_y, roi_size, conf_threshold):
    """Pass 1: follow the Kalman-guided crop across the whole video and
    record every accepted YOLO hypothesis. No outlier rejection here — that's
    the fitting pass's job. Returns (candidates, last_frame_idx),
    where candidates is a list of (frame_idx, x, y).
    """
    kf = KalmanFilter2D(init_x, init_y)
    candidates = [(0, init_x, init_y)]
    current_roi_size = roi_size
    consecutive_misses = 0
    airborne = False
    prev_x, prev_y = init_x, init_y
    prev_frame_full = None

    frame = first_frame
    frame_idx = 0
    while True:
        pred_x, pred_y = kf.predict()
        roi, offset_x, offset_y = crop_roi(frame, pred_x, pred_y, current_roi_size)

        motion_mask_crop = None
        if airborne and prev_frame_full is not None:
            motion_mask = compute_motion_mask(prev_frame_full, frame)
            h, w = roi.shape[:2]
            motion_mask_crop = motion_mask[offset_y:offset_y + h, offset_x:offset_x + w]

        detection = detect_ball_in_roi(model, roi, conf_threshold, offset_x, offset_y, pred_x, pred_y,
                                        motion_mask_crop)

        if detection is not None:
            local_x, local_y = detection
            meas_x, meas_y = offset_x + local_x, offset_y + local_y
            kf.update(meas_x, meas_y)
            if frame_idx != 0:
                candidates.append((frame_idx, meas_x, meas_y))
            if not airborne and np.hypot(meas_x - prev_x, meas_y - prev_y) > COARSE_LAUNCH_DISPLACEMENT:
                airborne = True
            prev_x, prev_y = meas_x, meas_y
            consecutive_misses = 0
            current_roi_size = roi_size
        else:
            # No genuine detection this frame. Before liftoff, that just
            # means the detector didn't fire — the ball hasn't gone
            # anywhere, so the crop stays put. Once airborne, it may really
            # have outrun the crop, so widen the search for next frame.
            consecutive_misses += 1
            if airborne:
                current_roi_size = min(
                    int(roi_size * (MISS_GROWTH_RATE ** consecutive_misses)),
                    roi_size * MAX_ROI_MULTIPLIER,
                )

        prev_frame_full = frame
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

    return candidates, frame_idx


def find_launch_frame(candidates, window=LAUNCH_WINDOW, threshold=LAUNCH_DISPLACEMENT, confirm=LAUNCH_CONFIRM):
    """First candidate index where the ball sustains real movement for
    `confirm` consecutive windows — the end of address and the start of the
    segment worth trajectory-fitting. Requiring several consecutive windows
    (not just one) rejects a single noisy blip (e.g. a stray detection
    during address) from being mistaken for the actual launch.
    """
    def window_displacement(i):
        total = 0.0
        for j in range(i, i + window - 1):
            _, x1, y1 = candidates[j]
            _, x2, y2 = candidates[j + 1]
            total += float(np.hypot(x2 - x1, y2 - y1))
        return total

    n = len(candidates)
    for i in range(n - window - confirm):
        if all(window_displacement(i + k) > threshold for k in range(confirm)):
            return candidates[i][0]
    return None


def ransac_parabola_fit(points, iters=RANSAC_ITERS, inlier_threshold=RANSAC_INLIER_THRESHOLD,
                         min_inliers=RANSAC_MIN_INLIERS, seed=0):
    """Fit x(t) and y(t) as independent quadratics via RANSAC. Returns
    (px, py, n_inliers, n_outliers), or None if no fit had enough support.
    """
    if len(points) < 3:
        return None

    ts = np.array([p[0] for p in points], dtype=np.float64)
    xs = np.array([p[1] for p in points], dtype=np.float64)
    ys = np.array([p[2] for p in points], dtype=np.float64)
    n = len(points)

    rng = random.Random(seed)
    best_mask = None
    best_count = 0
    for _ in range(iters):
        sample = rng.sample(range(n), 3)
        try:
            px = np.polyfit(ts[sample], xs[sample], 2)
            py = np.polyfit(ts[sample], ys[sample], 2)
        except np.linalg.LinAlgError:
            continue
        errors = np.hypot(xs - np.polyval(px, ts), ys - np.polyval(py, ts))
        mask = errors < inlier_threshold
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask

    if best_mask is None or best_count < min_inliers:
        return None

    px = np.polyfit(ts[best_mask], xs[best_mask], 2)
    py = np.polyfit(ts[best_mask], ys[best_mask], 2)
    return px, py, int(best_mask.sum()), int((~best_mask).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to input swing video")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--roi-size", type=int, default=ROI_SIZE)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--inlier-threshold", type=float, default=RANSAC_INLIER_THRESHOLD)
    parser.add_argument(
        "--annotations",
        help="Dense annotation JSON used only to reproduce the initial address tap",
    )
    parser.add_argument(
        "--diagnostics-output",
        help="Diagnostics JSON path (default: ml/output/<video>.diagnostics.json)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}. Run train.py first, or pass --model.")
        sys.exit(1)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, first_frame = cap.read()
    if not ret:
        print("Video has no frames.")
        sys.exit(1)

    if args.annotations:
        init_x, init_y = initial_point_from_annotations(args.annotations)
        print(f"Initial point from address annotations: ({init_x:.1f}, {init_y:.1f})")
    else:
        init_x, init_y = pick_initial_point(first_frame)

    print("Pass 1/2: collecting detections across the video...")
    candidates, last_frame_idx = collect_candidates(
        model, cap, first_frame, init_x, init_y, args.roi_size, args.conf
    )
    cap.release()
    print(f"Collected {len(candidates)} accepted detection hypotheses "
          f"across {last_frame_idx + 1} frames.")
    midpoint = last_frame_idx // 2
    n_before = sum(1 for c in candidates if c[0] < midpoint)
    n_after = len(candidates) - n_before
    print(f"  ({n_before} in first half, {n_after} in second half of the video)")

    launch_frame = find_launch_frame(candidates)
    px = py = None
    if launch_frame is None:
        print("No sustained launch detected — nothing to trace beyond address.")
    else:
        flight_points = [c for c in candidates if c[0] >= launch_frame]
        fit = ransac_parabola_fit(flight_points, inlier_threshold=args.inlier_threshold)
        if fit is None:
            print(f"Launch detected at frame {launch_frame}, but RANSAC found no confident "
                  f"trajectory ({len(flight_points)} candidate points) — nothing will be traced.")
        else:
            px, py, n_inliers, n_outliers = fit
            print(f"Launch detected at frame {launch_frame}. "
                  f"Trajectory fit: {n_inliers} inliers, {n_outliers} outliers discarded.")

    print("Pass 2/2: rendering traced video...")
    cap = cv2.VideoCapture(args.video)
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "output"), exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    out_path = os.path.join(os.path.dirname(__file__), "..", "output", f"{video_stem}_traced.mp4")
    diagnostics_path = args.diagnostics_output or os.path.join(
        os.path.dirname(__file__), "..", "output", f"{video_stem}.diagnostics.json"
    )
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    trace_points = []
    trajectory = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if px is not None and frame_idx >= launch_frame:
            curve_x = float(np.clip(np.polyval(px, frame_idx), 0, width - 1))
            curve_y = float(np.clip(np.polyval(py, frame_idx), 0, height - 1))
            trace_points.append((curve_x, curve_y))
            trajectory.append(
                {
                    "frame_index": frame_idx,
                    "x": curve_x,
                    "y": curve_y,
                    "source": "legacy_ransac_fit",
                }
            )

            for i in range(1, len(trace_points)):
                cv2.line(frame,
                          (int(trace_points[i - 1][0]), int(trace_points[i - 1][1])),
                          (int(trace_points[i][0]), int(trace_points[i][1])),
                          (0, 255, 255), 2)
            cv2.circle(frame, (int(curve_x), int(curve_y)), 4, (0, 0, 255), -1)
        else:
            cv2.circle(frame, (int(init_x), int(init_y)), 4, (0, 0, 255), -1)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    diagnostics = {
        "schema_version": 1,
        "tracker": "legacy_yolo_kalman_motion_ransac",
        "video": {
            "path": os.path.abspath(args.video),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": last_frame_idx + 1,
        },
        "settings": {
            "roi_size": args.roi_size,
            "confidence_threshold": args.conf,
            "ransac_inlier_threshold": args.inlier_threshold,
            "initialization": "annotation_address_median" if args.annotations else "manual_tap",
        },
        "launch_frame": launch_frame,
        "candidates": [
            {"frame_index": frame_index, "x": x, "y": y, "source": "accepted_yolo_output"}
            for frame_index, x, y in candidates
        ],
        "trajectory": trajectory,
    }
    atomic_write_json(diagnostics_path, diagnostics)

    print(f"Traced video saved to: {out_path}")
    print(f"Diagnostics saved to: {diagnostics_path}")


if __name__ == "__main__":
    main()
