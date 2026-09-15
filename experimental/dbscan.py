from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

import mediapipe as mp
# Alias for sublibrary
mp_pose = mp.solutions.pose

from pykalman import KalmanFilter
from ultralytics.engine.results import Results
from ultralytics import YOLO

# Import the ground data for the grips, computed from official refence boards.
from grip_ground_data import *

def pack_into_points(i_xyxy: torch.Tensor, classes: torch.Tensor, pad_zero_z_src: bool = False):
    classes_cpu = classes.cpu().numpy().astype(int)
    if pad_zero_z_src:
        source = np.array(GRIP_VALUES_Z_EQ_ZERO[classes_cpu])
    else:
        source = np.array(GRIP_VALUES[classes_cpu])

    xyxy_cpu = i_xyxy.cpu().numpy()
    # Compute the centroids for each box (mapping them onto the CPU beforehands)
    dst_centroids = (xyxy_cpu[:, :2] + xyxy_cpu[:, 2:]) / 2
    return dst_centroids, source

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def var_n_pnp_solve(i_src: np.ndarray, i_dst: np.ndarray, h_coord, bound_x=None, bound_y=None):
    """ This function returns the computed position of a point over a reference board

    :param i_src:
    :param i_dst:
    :param h_coord:
    :param bound_x:
    :param bound_y:
    :return:
    """
    # N is the number of tracked reference points.
    n = i_src.shape[0]

    if n == 0:
        return None
    x_point = None
    if n >= 4:
        #
        H_inverse, _ = cv2.findHomography(i_dst, i_src, cv2.RANSAC)
        if H_inverse is None:
            # plt.plot(i_src[:, 0], i_src[:, 1], "ro")
            # plt.show()
            # plt.plot(i_dst[:, 0], i_dst[:, 1], "ro")
            # plt.show()
            H_inverse, _ = cv2.findHomography(i_dst, i_src, cv2.LMEDS)
        if H_inverse is None:
            print("NONE AGAIN!")
            # collinearity_mask = remove_collinear_points(i_dst)
            # print(collinearity_mask)
            # i_src = i_src[collinearity_mask == 1]
            # i_dst = i_dst[collinearity_mask == 1]
            return var_n_pnp_solve(i_src[:3], i_dst[:3], h_coord, bound_x, bound_y)
        else:
            x_point_w = np.dot(H_inverse, np.array(h_coord + [1.0]).T)
            x_point = np.array([x_point_w[0] / x_point_w[2], x_point_w[1] / x_point_w[2]])
    elif n == 3:
        # Avoid a P3P algorithm as its very noisy and fails to converge
        H = cv2.getAffineTransform(i_src.astype(np.float32), i_dst.astype(np.float32))
        A = H[:, :2]
        t = H[:, 2:].T
        # Compute the inverse of the affine transformation matrix and subtract the translation
        # vector
        A_inv = np.linalg.inv(A)
        # x_point = A^-1 ( h - t )
        x_point = np.dot(A_inv, (np.array(h_coord) - np.array(t)).T).ravel()
    elif n <= 2:
        val = xyxy.cpu().numpy()
        GRIP_SIZE = 0.35

        if n == 2:
            # Known scale
            s = [(val[0][2] - val[0][0]), (val[0][3] - val[0][1]), (val[1][2] - val[1][0]), (val[1][3] - val[1][1])]
            s = (sum(s) / 4) / GRIP_SIZE

            v1_to_h = -(np.array(h_coord) - i_dst[1]) / s
            v0_to_h = -(np.array(h_coord) - i_dst[0]) / s

            x_point = ((i_src[0] + v0_to_h) + (i_src[1] + v1_to_h)) / 2
        elif n == 1:
            # Known scale
            s = [(val[0][2] - val[0][0]), (val[0][3] - val[0][1])]
            s = (sum(s) / 2) / GRIP_SIZE
            v0_to_h = -(np.array(h_coord) - i_dst[0]) / s
            x_point = (i_src[0] + v0_to_h)

    if x_point is None:
        print("WARNING: NUMERICAL FAILURE")
    if bound_x is not None:
        x_point[0] = np.clip(x_point[0], bound_x[0], bound_x[1])
    if bound_y is not None:
        x_point[1] = np.clip(x_point[1], bound_y[0], bound_y[1])
    return x_point


def var_averaged_solve(i_src: np.ndarray, i_dst: np.ndarray, h_coord, bound_x=None, bound_y=None, use_weighted=False):
    """ A function which takes a set of reference grips and returns the predicted space position
    over the reference board starting from a set of measurements contained into i_dst for the grips.
    This function uses a weighted average of the computed reference grips

    :param i_src: The input (ground truth) array of reference pair points.
    :param i_dst: The array containing observed points through the reference camera.
    :param h_coord: The reference coordinate over which the displacement vector is computed
    :param bound_x: If set, bound the predicted value of x
    :param bound_y: If set, bound the predicted value of y
    :param use_weighted: Whether to employ a weighted criterion, weighted on the relative distance
        by the observed grip
    :return: The observed position point
    """
    # N is the number of tracked reference points. Unlike PnP, this value does not change the computation formula used.
    n = i_src.shape[0]

    if n == 0:
        return None
    cpu_xyxy = xyxy.cpu().numpy()
    weights = []
    x_points = []
    for i in range(len(val)):
        local_observed_width = [(cpu_xyxy[i][2] - cpu_xyxy[i][0]), (cpu_xyxy[i][3] - cpu_xyxy[i][1])]
        rescaled_width = (sum(local_observed_width) / 2) / GRIP_SIZE
        v0_to_h = -(np.array(h_coord) - i_dst[i]) / rescaled_width
        if use_weighted:
            weights.append(1 / ((float(np.linalg.norm(v0_to_h)) ** 2 )/18 + 1))
        x_points.append((i_src[i] + v0_to_h))

    point = None
    if use_weighted:
        weights = softmax(np.array(weights))
        point = np.average(x_points, weights=weights, axis=0)
    else:
        point =  np.average(x_points, axis=0)
    if bound_x and abs(point[0] > bound_x):
        point[0] = bound_x
    if bound_y and abs(point[1] > bound_y):
        point[1] = bound_y
    return point

def compute_mp_pose_com(landmarks, width, height, pose_landmark = -1) -> list:
    lm_lm = landmarks.landmark
    x = y = None
    # A value of -1 corresponds to the centre of mass, we compute it heuristically
    # with an average of the four landmarks: 2 hips and 2 shoulders.
    if pose_landmark == -1:
        x = (lm_lm[mp_pose.PoseLandmark.LEFT_HIP].x + lm_lm[mp_pose.PoseLandmark.RIGHT_HIP].x +
             lm_lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x + lm_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 4
        y = (lm_lm[mp_pose.PoseLandmark.LEFT_HIP].y + lm_lm[mp_pose.PoseLandmark.RIGHT_HIP].y +
             lm_lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y + lm_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 4
    elif pose_landmark == 0:
        x = lm_lm[mp_pose.PoseLandmark.NOSE].x
        y = lm_lm[mp_pose.PoseLandmark.NOSE].y
    elif pose_landmark == 15:
        x = lm_lm[mp_pose.PoseLandmark.LEFT_WRIST].x
        y = lm_lm[mp_pose.PoseLandmark.LEFT_WRIST].y
    elif pose_landmark == 16:
        x = lm_lm[mp_pose.PoseLandmark.RIGHT_WRIST].x
        y = lm_lm[mp_pose.PoseLandmark.RIGHT_WRIST].y
    elif pose_landmark == 27:
        x = lm_lm[mp_pose.PoseLandmark.LEFT_ANKLE].x
        y = lm_lm[mp_pose.PoseLandmark.LEFT_ANKLE].y
    elif pose_landmark == 28:
        x = lm_lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x
        y = lm_lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y
    elif pose_landmark == 25:
        x = lm_lm[mp_pose.PoseLandmark.LEFT_KNEE].x
        y = lm_lm[mp_pose.PoseLandmark.LEFT_KNEE].y
    elif pose_landmark == 26:
        x = lm_lm[mp_pose.PoseLandmark.RIGHT_KNEE].x
        y = lm_lm[mp_pose.PoseLandmark.RIGHT_KNEE].y
    return [x * width, y * height]

def suppress_border_detection(boxes: np.ndarray, border_ratio: float, image_dim: Tuple[int, int]):
    h, w = image_dim
    xmin, ymin, xmax, ymax = boxes.T # We take the transpose to obtain 4 columns instead of 4 rows
    margin_x, margin_y = border_ratio*w, border_ratio*h
    mask = (xmin > margin_x) & (ymin > margin_y) & (xmax < w-margin_x) & (ymax < h-margin_y)
    return mask


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

from random import randint

def random_color():
    """ A trivial function to return a random color. to be used in visualization routines. """
    return randint(0, 255), randint(0, 255), randint(0, 255)


if __name__ == '__main__':

    model = YOLO(
        r"C:\Users\picul\PycharmProjects\ClimberCUDA\runs\detect\yolo_speedclimbing_hyper_tune48\weights\best.pt")
    video_path = r"C:\Users\picul\Documents\Untitled.png"
    #video_path = r"C:\Users\picul\Pictures\CUDA\wetransfer_dataset-speed_2025-10-15_0813\dataset\images\test\video1_19.jpg"
    video_path = r"C:\Users\picul\Downloads\speed-wall.jpeg"
    # video_path = r"C:\Users\picul\Downloads\yt01e2ez.jpeg"
    # video_path = r"C:\Users\picul\Downloads\xPURKWjq02eVNXeo.jpg"
    video_path = r"C:\Users\picul\Downloads\speed-wall.webp"
    # rzmro6dgh8lgy4ko8xid  UX_WEBSITE_IMAGES_SPEEDWALL
    # video_path = r"C:\Users\picul\Downloads\rzmro6dgh8lgy4ko8xid.png"
    # video_path = r"C:\Users\picul\Downloads\UX_WEBSITE_IMAGES_SPEEDWALL.jpg"
    # 52315662630_ad969f8d1e_k
    # video_path = r"C:\Users\picul\Downloads\52315662630_ad969f8d1e_k.jpg"
    # walltopia-speed-walls-377855_1mg
    # video_path = r"C:\Users\picul\Downloads\walltopia-speed-walls-377855_1mg.jpg"
    # 201201-Speed-Climbing-Wall-1
    #video_path = r"C:\Users\picul\Downloads\201201-Speed-Climbing-Wall-1.jpg"

    data: list[Results] = model(video_path)[0]

    repr_img = data.plot()
    r = data.boxes
    xyxy = r.xyxy
    cls = r.cls

    dst, src = pack_into_points(xyxy, cls)
    # Sample 2D data
    from cydpscan import calculate_dbscan_2d
    import numpy as np

    dist = []
    for i in range(0, 20):
        dist.append(np.linalg.norm(np.array(GRIP_VALUES_LIST[i+1]) - np.array(GRIP_VALUES_LIST[i])))
    max_dist = (max(dist))
    GRIP_SIZE = 0.35

    measured_grip_sizes = []
    val = xyxy.cpu().numpy()
    for i in range(len(xyxy)):
        measured_grip_sizes.append((val[i][2] - val[i][0]))
        measured_grip_sizes.append((val[i][3] - val[i][1]))
    s = (sum(measured_grip_sizes) / len(measured_grip_sizes))
    print(measured_grip_sizes)
    print("Perceived grip size:", s)
    # perc : real = eps : max_dist ovvero esp = (perc / real) * max_dist
    eps = max_dist * (s / GRIP_SIZE) * (1.10)
    print("Epsi maybe", eps)
    # Sample 2D data
    # DBSCAN clustering
    labels = calculate_dbscan_2d(dst, eps=eps, min_pts=2)
    # Map all keys to the 0, len range
    labels = {k: val for k, val in zip(range(len(labels.keys())), labels.values())}
    centroids = {key: np.average(labels[key], axis=0) for key in labels}

    print("Labels (DSCAN)", labels)
    max_x = 1.50
    eps_x = (s / GRIP_SIZE)*max_x

    from collections import defaultdict

    def merge_lists_pairwise(lists, func):
        print("LITS", lists)
        n = len(lists.keys())
        print("N", n)
        uf = UnionFind(n)

        done_already = []
        # build connectivity
        for i in lists.keys():
            for j in lists.keys():
                if (i, j) not in done_already and func(i, j):
                    done_already.append((i, j))
                    uf.union(i, j)

        # collect components
        groups = defaultdict(list)
        print(groups)
        for i, lst in lists.items():
            print(i, lst)
            print(groups[uf.find(i)])
            groups[uf.find(i)].extend(lst)

        return groups

    print()

    print(labels)
    with mp_pose.Pose() as pose:
        frame = cv2.imread(video_path)
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print()
        results = pose.process(image_rgb)
        print(results.__dict__)
        if results is not None and results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
            cv2.circle(repr_img, (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * repr_img.shape[1]),
                                   int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * repr_img.shape[0])),
                       8, (0, 0, 255), -1)
            cv2.circle(repr_img, (int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x * repr_img.shape[1]),
                                   int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y * repr_img.shape[0])),
                       8, (150, 255, 0), -1)
        print("LABELS!", labels)
        for key_i in labels.keys():
            color = random_color()
            for label in labels[key_i]:
                print(label)
                cv2.circle(repr_img, (int(label[0]), int(label[1])),
                           6, color, -1)
        for cent in centroids.values():
            color = random_color()
            print(cent)
            cv2.circle(repr_img, (int(cent[0]), int(cent[1])),
                       15, color, -1)
    cv2.imshow("Annotated Image", repr_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Centroids:", centroids)
    merged = merge_lists_pairwise(labels, lambda i1, i2: abs(centroids[i1][0]-centroids[i2][0]) < eps_x)
    print("Merged:", merged)

    merged = dict(merged)
    repr_img = data.plot()
    with mp_pose.Pose() as pose:
        frame = cv2.imread(video_path)
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print()
        results = pose.process(image_rgb)
        print(results.__dict__)
        if results is not None and results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
            cv2.circle(repr_img, (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * repr_img.shape[1]),
                                   int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * repr_img.shape[0])),
                       8, (0, 0, 255), -1)
            cv2.circle(repr_img, (int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x * repr_img.shape[1]),
                                   int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y * repr_img.shape[0])),
                       8, (150, 255, 0), -1)
        for key_i in merged.keys():
            color = random_color()
            for label in merged[key_i]:
                print(label)
                cv2.circle(repr_img, (int(label[0]), int(label[1])),
                           6, color, -1)
                for label2 in merged[key_i]:
                    cv2.line(repr_img, (int(label[0]), int(label[1])), (int(label2[0]), int(label2[1])), color, 2)

    sin_rotation = None
    rotation_matrix = None
    # Rotate the known shapes, rescale the known shapes, then compute the translation
    # vectors which maximizes energy
    tolerance = 1.2 # A relative distance of 0.2 from the real position is still considered perfect.


    cv2.imshow("Annotated Image", repr_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(GRIP_VALUES_LIST)
    line = np.polyfit([v[0] for v in GRIP_VALUES_LIST], [v[1] for v in GRIP_VALUES_LIST], 1)
    print(line)

if __name__ == '__main__' and False:
    model = YOLO(
        r"C:\Users\picul\PycharmProjects\ClimberCUDA\runs\detect\yolo_speedclimbing_hyper_tune48\weights\best.pt")
    video_path = r"C:\Users\picul\Pictures\CUDA\duddu.mp4"
    import cv2
    import json

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5,
                      smooth_landmarks=True)

    cap = cv2.VideoCapture(video_path)
    pose_sequence = {}

    for i in range(0, 40):
        success, frame = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results is not None and results.pose_landmarks:
            landmarks = [
                [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
            ]
            pose_sequence[i] = (landmarks)

    cap.release()
    pose.close()

    with open("pose_data.json", "w") as f:
        json.dump(pose_sequence, f)

# Note: this should go into a config file instead in production
_PATH_TO_MODEL = r"C:\Users\picul\PycharmProjects\ClimberCUDA\runs\detect\yolo_speedclimbing_hyper_tune48\weights\best.pt"
_PATH_TO_FOOTAGE = r"C:\Users\picul\Videos\video\cropped_footage.mp4"

if __name__ == '__main__' and False:
    try:
        model = YOLO(_PATH_TO_MODEL)
    except FileNotFoundError:
        print("ERROR: Failed to import the model. Please ensure that the correct path to the"
              " best.pt file is provided into " + _PATH_TO_MODEL.__name__)
        exit(1)

    ys = []
    ps = []
    confidences = []
    # If you want the model to stop and display individual frames specify them in this list
    display_frames_list = [ 150, 200, 300, 400]
    max_frame: int = 0
    idx = 0
    prev_val = [0, 0]
    import time

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                      static_image_mode=False, smooth_landmarks=True) as pose:

        try:
            cap = cv2.VideoCapture(_PATH_TO_FOOTAGE)
        except FileNotFoundError:
            print("ERROR: Failed to load the example footage. Please ensure the correct path to the footage is "
                  "inside " + _PATH_TO_FOOTAGE.__name__)
        if not cap.isOpened():
            raise IOError(f"ERROR: CV2 Could not open file: {_PATH_TO_FOOTAGE}")

        while cap.isOpened() and (not max_frame or idx < max_frame):
            ret, frame = cap.read()
            t0 = time.perf_counter()

            if not ret:
                break
            idx += 1
            try:
                # CV2 legacy imports images as BGR instead of RGB, so map back.
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except IOError:
                print(f"WARNING: Failed to map frame {idx} into RGB format. Terminating early the footage scan.")
                break
            res: list[Results] = model.track(frame, persist=True, iou=0.40, agnostic_nms=False)
            if not res or len(res) == 0:
                continue
            data = res[0]

            repr_img = None
            results = pose.process(image_rgb)
            if results is not None and results.pose_landmarks is not None and idx in display_frames_list:
                repr_img = data.plot()
                landmarks = results.pose_landmarks.landmark
                cv2.circle(repr_img, (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * repr_img.shape[1]),
                                       int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * repr_img.shape[0])),
                           8, (0, 0, 255), -1)
                cv2.circle(repr_img, (int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x * repr_img.shape[1]),
                                       int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y * repr_img.shape[0])),
                           8, (255, 255, 0), -1)

                cv2.imshow("Annotated Image", repr_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if data is None or not data:
                print(f"WARNING: frame {idx} resulted in no data output")
                continue

            # SWITCH THIS TO CHANGE FROM A MULTI-OBJECT TRACK TO A SINGLE-OBJECT TRACK:
            # R IS A SINGLE OBJECT, DATA IS MULTIPLE OBJECTS
            r = data[0]
            # SIMPLY PUT r.boxes instead of data.boxes to get a SINGLE TRACK TRACKING
            boxes = data.boxes  # Bounding boxes
            xyxy = boxes.xyxy.cpu()  # [[x1, y1, x2, y2], ...]
            cls = boxes.cls.cpu()  # Class indices
            confs = boxes.conf.cpu().numpy()

            w, h = image_rgb.shape[:2]
            border_suppression_mask = suppress_border_detection(xyxy.numpy(), 0.03, (w, h))
            if not np.all(border_suppression_mask):
                xyxy = xyxy[border_suppression_mask]
                cls = cls[border_suppression_mask]
                confs = confs[border_suppression_mask]

            dup_keep_mask = np.array([confs[i] == confs[cls.numpy() == c].max() for i, c in enumerate(cls.numpy())])
            xyxy = xyxy[dup_keep_mask]
            cls = cls[dup_keep_mask]

            avg = (np.average(boxes.conf.cpu().numpy()))
            confidences.append(np.sqrt(avg))
            # Filtering and PnP dispatch...
            dst, src = pack_into_points(xyxy, cls)

            lm = results.pose_landmarks
            if lm is None or not lm:
                ps.append(prev_val)
                # ps.append(np.array([0, 0]))
                ys += [prev_val[1]]
            else:
                c_i_m = compute_mp_pose_com(lm, data[0].orig_shape[1], data[0].orig_shape[0])
                pos = var_n_pnp_solve(src, dst, c_i_m, bound_x=[0, 3.0], bound_y=[0, 15.0])
                if pos is None:
                    ps.append(prev_val)
                    # ps.append(np.array([0, 0]))
                    ys += [prev_val[1]]
                else:
                    ps.append(pos)
                    prev_val = pos
                    ys += [pos[1]]
            dt = time.perf_counter() - t0
            print(f"Elapsed time FOR THE LOOP: {dt*1000:.6f} ms")

    dt = 1 / cap.get(cv2.CAP_PROP_FPS)

    import scipy.fftpack
    # Number of samplepoints
    N = np.size(ys)
    # sample spacing
    T = dt
    yf = scipy.fftpack.fft(ys)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.title("Fourier transform of the signal")
    plt.show()


    plt.plot(dt*np.arange(0, np.size(ys)), ys)

    print(f"> Running Kalman filter with time step {dt}")
    F = np.array([
        [1, 0, dt, 0, 0.5 * dt * dt, 0],
        [0, 1, 0, dt, 0, 0.5 * dt * dt],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    from scipy.signal import butter, lfilter, freqz, savgol_filter

    def butter_lowpass(cutoff, fs, order=5):
        return butter(order, cutoff, fs=fs, btype='low', analog=False)


    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y




    # Filter requirements.
    order = 2
    fs = 1/dt  # sample rate, Hz
    cutoff = 1  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    # b, a = butter_lowpass(cutoff, fs, order)
    kf_pos = KalmanFilter(transition_matrices=F, observation_matrices=[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    # print(ps)
    measurements = np.asarray(ps)  # 3 observations
    kf_pos = kf_pos.em([m for m in measurements if not np.isnan(m[0])], n_iter=140)
    (smoothed_state_means, smoothed_state_covariances) = kf_pos.smooth(measurements)
    yf = butter_lowpass_filter(ys, cutoff, fs, order)


    """
    def butter_lowpass(cutoff, fs, order=2):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a


    def lowpass_filter(signal, cutoff, fs, order=2):
        b, a = butter_lowpass(cutoff, fs, order)
        return filtfilt(b, a, signal)
    """

    # smooth2 = lowpass_filter([val[1] for val in smoothed_state_means], 4, 30)
    # reject_outliers(smooth2)

    # print(smoothed_state_means)
    # plt.plot([val[1] for val in smoothed_state_means])
    #  plt.plot(smooth2)
    # plt.plot([val[5] for val in smoothed_state_means])

    plt.plot(dt*np.arange(0, np.size(yf)), yf)
    plt.legend(["Non filtered", "Filtered"])
    plt.title("Posizione")
    plt.show()

    deriv = np.diff(yf) / dt
    derivf = butter_lowpass_filter(deriv, cutoff, fs, order+1)

    plt.plot(derivf)
    plt.title("Filtered numerical differentiation")
    plt.show()

    vel = [val[3] for val in smoothed_state_means]
    velf = butter_lowpass_filter(vel, cutoff, fs, order+1)
    plt.title("Filtered kalmann output")
    plt.plot(velf)

    # plt.plot(dt*np.arange(0, np.size(yf)-1), deriv)
    plt.show()

    # plt.plot([val[3] for val in smoothed_state_means])
    # plt.legend(["Velocity"])
    # plt.show()
