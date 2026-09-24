"""
Speed climbing pose extraction and ground truth comparison.

This script does 2 tasks:
1. It runs a YOLO grip detector and a MediaPipe pose model on a climbing
   video. It uses the detected grips as reference points. It projects a
   small set of body landmarks onto the climbing board coordinate system.
2. It loads ground truth motion capture data from a JSON file. It compares
   the extracted landmark positions to the ground truth, after removing
   the mean from each signal.
3. It can temporally smooth every extracted board-coordinate trajectory with
   the original constant-acceleration Kalman model and Butterworth low-pass.
"""

import json
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import mediapipe as mp
from pykalman import KalmanFilter
from scipy.signal import butter, lfilter, savgol_filter
from ultralytics import YOLO
from ultralytics.engine.results import Results

# Import the ground truth grip layout, computed from official reference boards.
from grip_ground_data import *

mp_pose = mp.solutions.pose

# ---------------------------------------------------------------------------
# Global settings. Change these paths and values to match your setup.
# ---------------------------------------------------------------------------

# Note: move these to a config file for production use.
_PATH_TO_MODEL = "" # PATH TO .PT
_PATH_TO_FOOTAGE = "" # PATH TO MP4 VIDEO
_PATH_TO_GROUND_TRUTH = "" # PATH TO GROUND TRUTH JSON

# Board bounds used to clip the estimated position (in board units).
BOUND_X = [0.0, 3.0]
BOUND_Y = [0.0, 15.0]

# The body landmarks to extract and plot. "Center of Mass" has no single
# MediaPipe landmark. The script builds it from 2 hips and 2 shoulders.
LANDMARKS_OF_INTEREST = {
    "Center of Mass": None,
    "Left Wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "Right Wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "Left Ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "Right Ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
}

# Matching keys in the ground truth JSON file. The script averages the
# keys listed for each landmark. "Center of Mass" has no direct ground
# truth key, so the script builds it the same way as the pose estimate.
GROUND_TRUTH_KEYS = {
    "Center of Mass": ["1 Left Hip", "1 Right Hip", "1 Left Shoulder", "1 Right Shoulder"],
    "Left Wrist": ["1 Left Wrist"],
    "Right Wrist": ["1 Right Wrist"],
    "Left Ankle": ["1 Left Ankle"],
    "Right Ankle": ["1 Right Ankle"],
}

# Axis mapping between the board coordinate system and the ground truth
# file. Change this if your capture rig uses a different axis convention.
# Board x is the horizontal position on the wall. Board y is the height.
GT_AXIS_FOR_BOARD_X = "x"
GT_AXIS_FOR_BOARD_Y = "z"

# Set True to bypass the geometric transform and use the legacy
# bounding-box / grip-size heuristic for every frame.
USE_HEURISTIC = True

# Grip size used by the legacy heuristic exactly as in var_n_pnp_solve.
HEURISTIC_GRIP_SIZE = 0.35

# ---------------------------------------------------------------------------
# Temporal filtering settings. These reproduce the filtering structure used
# by the original prototype, but apply it to every extracted landmark.
# ---------------------------------------------------------------------------

# Master switch for board-coordinate filtering.
USE_FILTERING = False

# Individual filter stages. With both True, the Kalman-smoothed position is
# passed through the Butterworth low-pass filter.
FILTER_USE_KALMAN = True
FILTER_USE_BUTTERWORTH = True

# The original script also applied Savitzky-Golay after Kalman on Y. Keep it
# available, but disabled by default so Kalman + Butterworth is the standard
# output.
FILTER_USE_SAVGOL = False

# In non-heuristic mode, color extracted samples by the number of unique
# grips visible in that frame. The continuous extracted trajectory remains
# visible underneath the colored markers.
COLOR_BY_GRIP_COUNT = True
GRIP_COUNT_COLORS = {
    1: "tab:red",
    2: "tab:orange",
    3: "tab:blue",
    4: "tab:green",  # 4 means 4 or more grips
}

# Same Kalman model/training count as the original script.
KALMAN_EM_ITERATIONS = 140

# Same Butterworth settings as the original script.
BUTTERWORTH_ORDER = 2
BUTTERWORTH_CUTOFF_HZ = 1.0

# Original Savitzky-Golay window rule: max(25, int(0.2 / dt)).
SAVGOL_MIN_WINDOW = 25
SAVGOL_WINDOW_SECONDS = 0.2
SAVGOL_POLYORDER = 2


# ---------------------------------------------------------------------------
# Detection and pose helper functions.
# ---------------------------------------------------------------------------

def pack_into_points(i_xyxy: torch.Tensor, classes: torch.Tensor, pad_zero_z_src: bool = False):
    """Build the source and destination point arrays for the board transform.

    :param i_xyxy: Detected grip boxes, in pixel coordinates.
    :param classes: The class index of each detected grip.
    :param pad_zero_z_src: If True, use the flat reference layout (z = 0).
    :return: A pair (pixel centroids, matching board positions).
    """
    classes_cpu = classes.cpu().numpy().astype(int)
    if pad_zero_z_src:
        source = np.array(GRIP_VALUES_Z_EQ_ZERO[classes_cpu])
    else:
        source = np.array(GRIP_VALUES[classes_cpu])

    xyxy_cpu = i_xyxy.cpu().numpy()
    dst_centroids = (xyxy_cpu[:, :2] + xyxy_cpu[:, 2:]) / 2
    return dst_centroids, source


def suppress_border_detection(boxes: np.ndarray, border_ratio: float, image_dim: Tuple[int, int]) -> np.ndarray:
    """Remove grip detections that are too close to the image border.

    :param boxes: Detected boxes, in pixel coordinates.
    :param border_ratio: Fraction of the image width or height to treat as
        the border.
    :param image_dim: The image size, as (height, width).
    :return: A boolean mask. True keeps the detection.
    """
    h, w = image_dim
    x_min, y_min, x_max, y_max = boxes.T
    margin_x, margin_y = border_ratio * w, border_ratio * h
    mask = (x_min > margin_x) & (y_min > margin_y) & (x_max < w - margin_x) & (y_max < h - margin_y)
    return mask


def estimate_board_transform(
        i_src: np.ndarray,
        i_dst: np.ndarray,
        i_boxes: np.ndarray,
        use_heuristic: bool = False,
):
    """Build a function that maps one pixel point to a board point.

    When ``use_heuristic`` is True, bypass the geometric solver and use the
    legacy bounding-box / grip-size heuristic for any number of visible grips.
    Each grip produces an independent board-position estimate using its own
    bounding-box size, and the estimates are averaged.

    When ``use_heuristic`` is False, pick the transform from the number of
    visible grips: a homography for 4 or more grips, an affine transform for
    3 grips, and the existing scale-based fallback for 1 or 2 grips.

    :param i_src: True board positions of the reference grips.
    :param i_dst: Pixel positions of the reference grips.
    :param i_boxes: Pixel bounding boxes of the reference grips.
    :param use_heuristic: If True, always use the legacy bounding-box heuristic.
        If False, use homography / affine whenever enough grips are visible.
    :return: A function ``point -> board_point``, or None if no grip is
        visible.
    """
    n = i_src.shape[0]
    if n == 0:
        return None

    # Explicit legacy bypass. This reproduces the old var_n_pnp_solve
    # heuristic: estimate pixels-per-board-unit separately from each grip
    # bounding box, map the target point relative to that grip, then average.
    if use_heuristic:
        boxes = i_boxes.cpu().numpy() if hasattr(i_boxes, "cpu") else np.asarray(i_boxes)

        def transform(point):
            x_points = []
            for i in range(len(boxes)):
                size = [
                    boxes[i][2] - boxes[i][0],
                    boxes[i][3] - boxes[i][1],
                ]
                scale = (sum(size) / 2) / HEURISTIC_GRIP_SIZE
                v0_to_h = -(np.array(point) - i_dst[i]) / scale
                x_points.append(i_src[i] + v0_to_h)

            return np.average(x_points, axis=0)

        return transform

    if n >= 4:
        h_inv, _ = cv2.findHomography(i_dst, i_src, cv2.RANSAC)
        if h_inv is None:
            h_inv, _ = cv2.findHomography(i_dst, i_src, cv2.LMEDS)
        if h_inv is None:
            # The fit failed. Try again with only 3 points.
            return estimate_board_transform(
                i_src[:3],
                i_dst[:3],
                i_boxes[:3],
                use_heuristic=False,
            )

        def transform(point):
            w = h_inv @ np.array([point[0], point[1], 1.0])
            return np.array([w[0] / w[2], w[1] / w[2]])

        return transform

    if n == 3:
        aff = cv2.getAffineTransform(i_src.astype(np.float32), i_dst.astype(np.float32))
        a = aff[:, :2]
        t = aff[:, 2:].T
        a_inv = np.linalg.inv(a)

        def transform(point):
            return (a_inv @ (np.array(point) - t).T).ravel()

        return transform

    # 1 or 2 grips visible: use the known grip size to recover the scale.
    widths = i_boxes[:, 2] - i_boxes[:, 0]
    heights = i_boxes[:, 3] - i_boxes[:, 1]
    scale = float(np.mean(np.concatenate([widths, heights]))) / GRIP_SIZE

    def transform(point):
        offsets = -(np.array(point) - i_dst) / scale
        return np.mean(i_src + offsets, axis=0)

    return transform


def clip_point(point: np.ndarray, bound_x=None, bound_y=None) -> np.ndarray:
    """Clip a board point to the given range."""
    if bound_x is not None:
        point[0] = np.clip(point[0], bound_x[0], bound_x[1])
    if bound_y is not None:
        point[1] = np.clip(point[1], bound_y[0], bound_y[1])
    return point


def get_pixel_points(landmarks, width: int, height: int) -> Dict[str, Tuple[float, float]]:
    """Read the pixel position of each landmark of interest from a MediaPipe result.

    :param landmarks: The ``pose_landmarks`` field of a MediaPipe result.
    :param width: The image width, in pixels.
    :param height: The image height, in pixels.
    :return: A dict from landmark name to a (x, y) pixel position.
    """
    lm = landmarks.landmark
    points = {}

    hip_l = lm[mp_pose.PoseLandmark.LEFT_HIP]
    hip_r = lm[mp_pose.PoseLandmark.RIGHT_HIP]
    sho_l = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    sho_r = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    com_x = (hip_l.x + hip_r.x + sho_l.x + sho_r.x) / 4
    com_y = (hip_l.y + hip_r.y + sho_l.y + sho_r.y) / 4
    points["Center of Mass"] = (com_x * width, com_y * height)

    for name, landmark_id in LANDMARKS_OF_INTEREST.items():
        if landmark_id is None:
            continue
        p = lm[landmark_id]
        points[name] = (p.x * width, p.y * height)

    return points


# ---------------------------------------------------------------------------
# Ground truth loading.
# ---------------------------------------------------------------------------

def load_ground_truth(path: str) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Load the ground truth JSON file into a lookup table.

    :param path: Path to the ground truth JSON file.
    :return: A dict from key name (e.g. "1 Left Wrist") to a dict from
        frame number to a dict with keys "x", "y", "z".
    """
    with open(path, "r") as f:
        raw = json.load(f)

    lookup = {}
    for key, entries in raw.items():
        lookup[key] = {entry["frame"]: entry for entry in entries}
    return lookup


def ground_truth_point(lookup: Dict[str, Dict[int, Dict[str, float]]], keys: List[str], frame: int) -> Optional[Dict[str, float]]:
    """Return the average ground truth point for a list of keys, at one frame.

    :param lookup: The ground truth lookup table, from ``load_ground_truth``.
    :param keys: The ground truth keys to average.
    :param frame: The frame number to read.
    :return: A dict with keys "x", "y", "z", or None if no key has data for
        this frame.
    """
    values = []
    for key in keys:
        entry = lookup.get(key, {}).get(frame)
        if entry is not None:
            values.append(entry)
    if not values:
        return None
    return {
        "x": float(np.mean([v["x"] for v in values])),
        "y": float(np.mean([v["y"] for v in values])),
        "z": float(np.mean([v["z"] for v in values])),
    }


# ---------------------------------------------------------------------------
# Main extraction pipeline.
# ---------------------------------------------------------------------------

def run_extraction(model: YOLO, video_path: str, max_frame: int = 0, use_heuristic: bool = False):
    """Run the grip detector and the pose model over one video.

    :param model: The loaded YOLO grip detector.
    :param video_path: Path to the input video.
    :param max_frame: Stop after this many frames. 0 means process the
        full video.
    :param use_heuristic: If True, use the legacy bounding-box heuristic
        instead of homography / affine estimation.
    :return: A tuple (extracted, fps, grip_counts).
        ``extracted`` is a dict from landmark name to a dict from frame
        number to a (x, y) board position.
        ``fps`` is the frame rate of the video.
        ``grip_counts`` maps frame number to the number of unique grips used
        to build the board transform in that frame.
    """
    extracted: Dict[str, Dict[int, np.ndarray]] = {name: {} for name in LANDMARKS_OF_INTEREST}
    prev_points: Dict[str, np.ndarray] = {}
    grip_counts: Dict[int, int] = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"ERROR: Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                       static_image_mode=False, smooth_landmarks=True) as pose:
        frame_idx = 0
        while cap.isOpened() and (not max_frame or frame_idx < max_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]

            results_list: List[Results] = model.track(frame, persist=True, iou=0.40, agnostic_nms=False)
            if not results_list:
                continue
            data = results_list[0]

            pose_result = pose.process(image_rgb)
            landmarks = pose_result.pose_landmarks
            if landmarks is None:
                # No pose in this frame. Hold the last known position.
                for name in LANDMARKS_OF_INTEREST:
                    if name in prev_points:
                        extracted[name][frame_idx] = prev_points[name]
                continue

            boxes = data.boxes
            xyxy = boxes.xyxy.cpu()
            cls = boxes.cls.cpu()
            confs = boxes.conf.cpu().numpy()

            border_mask = suppress_border_detection(xyxy.numpy(), 0.03, (height, width))
            if not np.all(border_mask):
                xyxy = xyxy[border_mask]
                cls = cls[border_mask]
                confs = confs[border_mask]

            if len(cls) == 0:
                for name in LANDMARKS_OF_INTEREST:
                    if name in prev_points:
                        extracted[name][frame_idx] = prev_points[name]
                continue

            # Keep only the highest confidence detection for each grip class.
            cls_np = cls.numpy()
            keep_mask = np.array([confs[i] == confs[cls_np == c].max() for i, c in enumerate(cls_np)])
            xyxy = xyxy[keep_mask]
            cls = cls[keep_mask]

            # Store how many unique reference grips are available in this frame.
            # This is recorded before any temporal filtering so the plot can show
            # which geometric solver regime produced each measurement.
            grip_counts[frame_idx] = int(len(cls))

            dst, src = pack_into_points(xyxy, cls)
            transform = estimate_board_transform(
                src,
                dst,
                xyxy.numpy(),
                use_heuristic=use_heuristic,
            )
            if transform is None:
                for name in LANDMARKS_OF_INTEREST:
                    if name in prev_points:
                        extracted[name][frame_idx] = prev_points[name]
                continue

            pixel_points = get_pixel_points(landmarks, width, height)
            for name, pixel_point in pixel_points.items():
                board_point = clip_point(transform(pixel_point), BOUND_X, BOUND_Y)
                extracted[name][frame_idx] = board_point
                prev_points[name] = board_point

    cap.release()
    return extracted, fps, grip_counts


# ---------------------------------------------------------------------------
# Temporal filtering.
# ---------------------------------------------------------------------------

def butter_lowpass(cutoff: float, fs: float, order: int = 5):
    """Return coefficients for a digital low-pass Butterworth filter."""
    return butter(order, cutoff, fs=fs, btype="low", analog=False)


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    """Apply the same causal Butterworth low-pass used by the original script."""
    b, a = butter_lowpass(cutoff, fs, order=order)
    return lfilter(b, a, data)


def interpolate_missing_points(points: np.ndarray) -> np.ndarray:
    """Linearly fill NaNs independently in X and Y before temporal filtering."""
    filled = np.array(points, dtype=float, copy=True)
    sample_idx = np.arange(len(filled))

    for axis in range(2):
        values = filled[:, axis]
        valid = np.isfinite(values)

        if not np.any(valid):
            continue

        if np.count_nonzero(valid) == 1:
            values[:] = values[valid][0]
        else:
            values[:] = np.interp(sample_idx, sample_idx[valid], values[valid])

        filled[:, axis] = values

    return filled


def kalman_smooth_positions(measurements: np.ndarray, fps: float) -> np.ndarray:
    """Smooth 2D board positions with the original constant-acceleration model.

    State vector:
        [x, y, vx, vy, ax, ay]

    Measurement vector:
        [x, y]
    """
    if len(measurements) < 2:
        return np.asarray(measurements, dtype=float)

    dt = 1.0 / fps

    transition_matrix = np.array([
        [1, 0, dt, 0, 0.5 * dt * dt, 0],
        [0, 1, 0, dt, 0, 0.5 * dt * dt],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=float)

    observation_matrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
    ], dtype=float)

    initial_state_mean = np.array([
        measurements[0, 0],
        measurements[0, 1],
        0.0,
        0.0,
        0.0,
        0.0,
    ])

    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=initial_state_mean,
    )

    # Preserve the EM-based parameter estimation from the original script.
    # This is intentionally offline smoothing rather than a frame-by-frame
    # causal Kalman update.
    if KALMAN_EM_ITERATIONS > 0 and len(measurements) >= 3:
        kf = kf.em(measurements, n_iter=KALMAN_EM_ITERATIONS)

    smoothed_state_means, _ = kf.smooth(measurements)
    return smoothed_state_means[:, :2]


def savgol_smooth_positions(points: np.ndarray, fps: float) -> np.ndarray:
    """Optional Savitzky-Golay stage adapted from the original script."""
    if len(points) <= SAVGOL_POLYORDER + 1:
        return points

    window = max(SAVGOL_MIN_WINDOW, int(SAVGOL_WINDOW_SECONDS * fps))

    # savgol_filter requires an odd window not larger than the signal.
    if window % 2 == 0:
        window += 1
    if window > len(points):
        window = len(points) if len(points) % 2 == 1 else len(points) - 1
    if window <= SAVGOL_POLYORDER:
        return points

    result = np.array(points, dtype=float, copy=True)
    result[:, 0] = savgol_filter(result[:, 0], window, SAVGOL_POLYORDER)
    result[:, 1] = savgol_filter(result[:, 1], window, SAVGOL_POLYORDER)
    return result


def filter_landmark_series(
    series: Dict[int, np.ndarray],
    fps: float,
    use_kalman: bool = True,
    use_butterworth: bool = True,
    use_savgol: bool = False,
) -> Dict[int, np.ndarray]:
    """Filter one landmark's board-coordinate trajectory.

    Missing frames between the first and last observation are linearly
    interpolated so the filters operate at the video's fixed sample rate.
    """
    if not series:
        return {}

    frames = list(range(min(series), max(series) + 1))
    points = np.full((len(frames), 2), np.nan, dtype=float)

    for i, frame in enumerate(frames):
        point = series.get(frame)
        if point is not None:
            points[i] = np.asarray(point, dtype=float)[:2]

    points = interpolate_missing_points(points)

    # If one axis somehow remains entirely invalid, filtering cannot proceed.
    if not np.all(np.isfinite(points)):
        return dict(series)

    filtered = points

    if use_kalman:
        filtered = kalman_smooth_positions(filtered, fps)

    if use_butterworth:
        # Same order/cutoff as the original code, now applied to both axes.
        filtered = np.array(filtered, dtype=float, copy=True)
        filtered[:, 0] = butter_lowpass_filter(
            filtered[:, 0],
            BUTTERWORTH_CUTOFF_HZ,
            fps,
            BUTTERWORTH_ORDER,
        )
        filtered[:, 1] = butter_lowpass_filter(
            filtered[:, 1],
            BUTTERWORTH_CUTOFF_HZ,
            fps,
            BUTTERWORTH_ORDER,
        )

    if use_savgol:
        filtered = savgol_smooth_positions(filtered, fps)

    # Keep filtered output inside the physical board limits as well.
    filtered[:, 0] = np.clip(filtered[:, 0], BOUND_X[0], BOUND_X[1])
    filtered[:, 1] = np.clip(filtered[:, 1], BOUND_Y[0], BOUND_Y[1])

    return {
        frame: filtered[i].copy()
        for i, frame in enumerate(frames)
    }


def filter_extracted_data(
    extracted: Dict[str, Dict[int, np.ndarray]],
    fps: float,
    use_kalman: bool = True,
    use_butterworth: bool = True,
    use_savgol: bool = False,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Apply the temporal filter pipeline independently to every landmark."""
    filtered: Dict[str, Dict[int, np.ndarray]] = {}

    for name in LANDMARKS_OF_INTEREST:
        filtered[name] = filter_landmark_series(
            extracted.get(name, {}),
            fps,
            use_kalman=use_kalman,
            use_butterworth=use_butterworth,
            use_savgol=use_savgol,
        )

    return filtered


# ---------------------------------------------------------------------------
# Comparison and plotting.
# ---------------------------------------------------------------------------

def zero_mean(values: np.ndarray) -> np.ndarray:
    """Remove the mean from a signal. Ignore NaN values."""
    if np.all(np.isnan(values)):
        return values
    return values - np.nanmean(values)


def build_comparison_arrays(extracted_series: Dict[int, np.ndarray], gt_lookup, gt_keys: List[str], frames: List[int]):
    """Build 4 aligned arrays: the extracted position and the ground truth
    position, for a range of frame numbers.

    :param extracted_series: Dict from frame number to an extracted (x, y) point.
    :param gt_lookup: The ground truth lookup table.
    :param gt_keys: The ground truth keys for this landmark.
    :param frames: The list of frame numbers to align to.
    :return: A tuple (extracted_x, extracted_y, gt_x, gt_y). Each is a
        numpy array of the same length as ``frames``. A missing value is
        NaN.
    """
    extracted_x = np.full(len(frames), np.nan)
    extracted_y = np.full(len(frames), np.nan)
    gt_x = np.full(len(frames), np.nan)
    gt_y = np.full(len(frames), np.nan)

    for i, frame in enumerate(frames):
        point = extracted_series.get(frame)
        if point is not None:
            extracted_x[i] = point[0]
            extracted_y[i] = point[1]

        gt_point = ground_truth_point(gt_lookup, gt_keys, frame)
        if gt_point is not None:
            gt_x[i] = gt_point[GT_AXIS_FOR_BOARD_X]
            gt_y[i] = gt_point[GT_AXIS_FOR_BOARD_Y]

    return extracted_x, extracted_y, gt_x, gt_y


def plot_comparison(
    extracted,
    gt_lookup,
    fps: float,
    grip_counts: Optional[Dict[int, int]] = None,
    use_heuristic: bool = False,
):
    """Plot the extracted landmark positions against the ground truth.

    Each row is 1 landmark. The left column compares the horizontal
    position. The right column compares the height.

    In non-heuristic mode, samples can be color-coded by the number of
    unique grips visible in the corresponding frame:
        red    = 1 grip
        orange = 2 grips
        blue   = 3 grips
        green  = 4 or more grips

    :param extracted: The filtered or raw extracted landmark trajectories.
    :param gt_lookup: The output of ``load_ground_truth``.
    :param fps: The video frame rate. Used to build a time axis.
    :param grip_counts: Dict mapping frame number to number of unique grips.
    :param use_heuristic: If True, disable grip-count coloring because the
        heuristic uses the same solver regardless of grip count.
    """
    names = list(LANDMARKS_OF_INTEREST.keys())
    all_frames = sorted(set().union(*[extracted[name].keys() for name in names]))
    if not all_frames:
        print("WARNING: No frame had a valid extracted position. Nothing to plot.")
        return

    frames = list(range(min(all_frames), max(all_frames) + 1))
    time_axis = np.array(frames) / fps

    color_by_grips = (
        COLOR_BY_GRIP_COUNT
        and not use_heuristic
        and grip_counts is not None
    )

    if color_by_grips:
        grip_count_array = np.array([grip_counts.get(frame, 0) for frame in frames])
    else:
        grip_count_array = None

    fig, axes = plt.subplots(
        len(names),
        2,
        figsize=(12, 3 * len(names)),
        sharex=True,
        squeeze=False,
    )

    for row, name in enumerate(names):
        ex_x, ex_y, gt_x, gt_y = build_comparison_arrays(
            extracted[name], gt_lookup, GROUND_TRUTH_KEYS[name], frames
        )

        ex_x_zero = zero_mean(ex_x)
        ex_y_zero = zero_mean(ex_y)
        gt_x_zero = zero_mean(gt_x)
        gt_y_zero = zero_mean(gt_y)

        ax_x, ax_y = axes[row]

        if color_by_grips:
            # Keep a thin continuous trajectory so the temporal shape remains
            # readable, then overlay each valid frame with the color belonging
            # to the number of grips used for that frame's transform.
            ax_x.plot(
                time_axis,
                ex_x_zero,
                color="0.55",
                linewidth=1.0,
                label="Extracted",
                zorder=1,
            )
            ax_y.plot(
                time_axis,
                ex_y_zero,
                color="0.55",
                linewidth=1.0,
                label="Extracted",
                zorder=1,
            )

            grip_buckets = [
                (1, "1 grip"),
                (2, "2 grips"),
                (3, "3 grips"),
                (4, "4+ grips"),
            ]

            for bucket, label in grip_buckets:
                if bucket < 4:
                    mask = grip_count_array == bucket
                else:
                    mask = grip_count_array >= 4

                # Do not add empty legend entries.
                if not np.any(mask):
                    continue

                color = GRIP_COUNT_COLORS[bucket]
                valid_x = mask & np.isfinite(ex_x_zero)
                valid_y = mask & np.isfinite(ex_y_zero)

                ax_x.scatter(
                    time_axis[valid_x],
                    ex_x_zero[valid_x],
                    s=12,
                    color=color,
                    label=label,
                    zorder=3,
                )
                ax_y.scatter(
                    time_axis[valid_y],
                    ex_y_zero[valid_y],
                    s=12,
                    color=color,
                    label=label,
                    zorder=3,
                )
        else:
            ax_x.plot(time_axis, ex_x_zero, label="Extracted")
            ax_y.plot(time_axis, ex_y_zero, label="Extracted")

        ax_x.plot(
            time_axis,
            gt_x_zero,
            color="black" if color_by_grips else None,
            linewidth=1.2,
            label="Ground truth",
            zorder=2,
        )
        # Fix the scale, simulation error i guess
        gt_y_zero = gt_y_zero*0.5 -1.0

        ax_y.plot(
            time_axis,
            gt_y_zero,
            color="black" if color_by_grips else None,
            linewidth=1.2,
            label="Ground truth",
            zorder=2,
        )

        ax_x.set_ylabel(f"{name}\nhorizontal")
        ax_y.set_ylabel("height")
        ax_x.legend(loc="upper right", fontsize=8)
        ax_y.legend(loc="upper right", fontsize=8)

    axes[-1][0].set_xlabel("Time (s)")
    axes[-1][1].set_xlabel("Time (s)")

    valid = np.isfinite(ex_y_zero) & np.isfinite(gt_y_zero)
    err_y = ex_y_zero[valid] - gt_y_zero[valid]
    print(f"{name}: Y MSE={np.mean(err_y ** 2):.6f}, MAE={np.mean(np.abs(err_y)):.6f}")

    if color_by_grips:
        fig.suptitle(
            "Extracted position vs. ground truth (zero mean) — "
            "marker color = visible grip count"
        )
    else:
        fig.suptitle("Extracted position vs. ground truth (zero mean)")

    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("> CUDA is available. Using the GPU.")
    else:
        print("> CUDA is not available. Using the CPU.")

    try:
        yolo_model = YOLO(_PATH_TO_MODEL)
    except FileNotFoundError:
        print(f"ERROR: Could not load the model file at {_PATH_TO_MODEL}")
        raise SystemExit(1)

    print("> Running the detector and the pose model on the video...")
    raw_extracted_data, video_fps, grip_counts = run_extraction(
        yolo_model,
        _PATH_TO_FOOTAGE,
        use_heuristic=USE_HEURISTIC,
    )

    if USE_FILTERING:
        print(
            "> Filtering extracted board coordinates "
            f"(Kalman={FILTER_USE_KALMAN}, "
            f"Butterworth={FILTER_USE_BUTTERWORTH}, "
            f"Savitzky-Golay={FILTER_USE_SAVGOL})..."
        )
        extracted_data = filter_extracted_data(
            raw_extracted_data,
            video_fps,
            use_kalman=FILTER_USE_KALMAN,
            use_butterworth=FILTER_USE_BUTTERWORTH,
            use_savgol=FILTER_USE_SAVGOL,
        )
    else:
        extracted_data = raw_extracted_data

    print("> Loading the ground truth file...")
    ground_truth = load_ground_truth(_PATH_TO_GROUND_TRUTH)

    print("> Plotting the comparison...")
    plot_comparison(
        extracted_data,
        ground_truth,
        video_fps,
        grip_counts=grip_counts,
        use_heuristic=USE_HEURISTIC,
    )
