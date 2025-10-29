import re
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import mediapipe as mp
# Alias for sublibrary
mp_pose = mp.solutions.pose

from pykalman import KalmanFilter
from ultralytics.engine.results import Results
from ultralytics import YOLO


_N_GRIPS = 21
_GRIP_DECL = re.compile(r"(?P<id>\d+)]\s+@(?P<relx>[A-Z][1-2])-SN(?P<sn>\d+)#(?P<rely>\d+)")
# According to https://images.ifsc-climbing.org/ifsc/image/private/t_q_good/prd/urwl7n2hnnyvhiwiq0xg.pdf
_GRIP_LOC = """ Tournament grips by specification: 
                1] @F2-SN2#1    2] @G2-SN2#3    3] @A2-SN2#9
                4] @G1-SN3#4    5] @L1-SN3#10   6] @C2-SN4#2
                7] @L1-SN4#8    8] @C2-SN5#3    9] @E2-SN5#9   
                10] @H1-SN6#2   11] @L1-SN6#7   12] @F1-SN6#9
                13] @M1-SN7#4   14] @G1-SN7#9   15] @L1-SN8#1
                16] @I1-SN8#3   17] @C1-SN8#8   18] @A2-SN9#2
                19] @E2-SN9#7   20] @M1-SN9#10  21] @A2-SN10#10 """

# For the 2D plane embedded in 3d real space, assume y-axis = height, x-axis = width, z axis = 0.
GRIP_VALUES_LIST = [
    # See the documentation and the explanation above
    [(ord(rel_x[0]) - ord('A')) * 0.1363 + (int(rel_x[1]) - 1) * 1.5, (int(sec) - 1) * 1.5 + int(rel_y) * 0.1363]
    for g_id, rel_x, sec, rel_y in (grip.groups() for grip in _GRIP_DECL.finditer(_GRIP_LOC))
]
GRIP_VALUES = np.array(GRIP_VALUES_LIST)
GRIP_VALUES_Z_EQ_ZERO = np.array([val + [0.0] for val in GRIP_VALUES_LIST])


def display_ground_truth_grips(grip_dict: dict) -> None:
    """ Plots the spatial distribution of the grips as specified in the ground truth grips
    :param grip_dict: A dictionary of name: coordinate pairs
    """
    f = plt.figure()
    plt.plot([i[0] for i in grip_dict.values()], [i[1] for i in grip_dict.values()], "ro")
    f.set_figwidth(3)
    f.set_figheight(15)
    plt.xlim([0, 3.0])
    plt.ylim([0, 15.0])
    plt.show()


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

def var_n_pnp_solve(i_src: np.ndarray, i_dst: np.ndarray, h_coord, bound_x=None, bound_y=None):
    """

    :param i_src:
    :param i_dst:
    :param h_coord:
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

def compute_mp_pose_com(landmarks, width, height) -> list:
    lm_lm = landmarks.landmark

    x = (lm_lm[mp_pose.PoseLandmark.LEFT_HIP].x + lm_lm[mp_pose.PoseLandmark.RIGHT_HIP].x +
         lm_lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x + lm_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 4
    y = (lm_lm[mp_pose.PoseLandmark.LEFT_HIP].y + lm_lm[mp_pose.PoseLandmark.RIGHT_HIP].y +
         lm_lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y + lm_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 4

    return [x * width, y * height]

def suppress_border_detection(boxes: np.ndarray, border_ratio: float, image_dim: Tuple[int, int]):
    h, w = image_dim
    xmin, ymin, xmax, ymax = boxes.T # We take the transpose to obtain 4 columns instead of 4 rows
    margin_x, margin_y = border_ratio*w, border_ratio*h
    mask = (xmin > margin_x) & (ymin > margin_y) & (xmax < w-margin_x) & (ymax < h-margin_y)
    return mask

if __name__ == '__main__' and False:

    model = YOLO(
        r"C:\Users\picul\PycharmProjects\ClimberCUDA\runs\detect\yolo_speedclimbing_hyper_tune48\weights\best.pt")
    video_path = r"C:\Users\picul\Documents\Untitled.png"
    #video_path = r"C:\Users\picul\Pictures\CUDA\wetransfer_dataset-speed_2025-10-15_0813\dataset\images\test\video1_19.jpg"
    video_path = r"C:\Users\picul\Downloads\speed-wall.jpeg"
    data: list[Results] = model(video_path)[0]

    repr_img = data.plot()
    r = data.boxes
    xyxy = r.xyxy
    cls = r.cls

    dst, src = pack_into_points(xyxy, cls)
    # Sample 2D data
    from cydpscan import calculate_dbscan_2d
    import numpy as np

    # Sample 2D data
    # DBSCAN clustering
    labels = calculate_dbscan_2d(dst, eps=150, min_pts=3)

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
                       8, (255, 255, 0), -1)
    cv2.imshow("Annotated Image", repr_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
_PATH_TO_FOOTAGE = r"C:\Users\picul\Downloads\doublefoot.mp4"

if __name__ == '__main__':
    try:
        model = YOLO(r"C:\Users\picul\PycharmProjects\ClimberCUDA\runs\detect\yolo_speedclimbing_hyper_tune48\weights\best.pt")
    except FileNotFoundError:
        print("ERROR: Failed to import the model. Please ensure that the correct path to the"
              " best.pt file is provided into " + _PATH_TO_MODEL.__name__)
        exit(1)

    ys = []
    ps = []
    confidences = []
    # If you want the model to stop and display individual frames specify them in this list
    display_frames_list = []
    max_frame: int = 0
    idx = 0
    prev_val = [0, 0]
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

    plt.plot(ys)

    dt = 1 / cap.get(cv2.CAP_PROP_FPS)
    print(f"> Running Kalman filter with time step {dt}")
    F = np.array([
        [1, 0, dt, 0, 0.5 * dt * dt, 0],
        [0, 1, 0, dt, 0, 0.5 * dt * dt],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    kf_pos = KalmanFilter(transition_matrices=F, observation_matrices=[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    print(ps)
    measurements = np.asarray(ps)  # 3 observations
    kf_pos = kf_pos.em([m for m in measurements if not np.isnan(m[0])], n_iter=30)
    (smoothed_state_means, smoothed_state_covariances) = kf_pos.smooth(measurements)

    from scipy.signal import butter, filtfilt


    def butter_lowpass(cutoff, fs, order=2):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a


    def lowpass_filter(signal, cutoff, fs, order=2):
        b, a = butter_lowpass(cutoff, fs, order)
        return filtfilt(b, a, signal)


    smooth2 = lowpass_filter([val[1] for val in smoothed_state_means], 4, 30)
    # reject_outliers(smooth2)

    print(smoothed_state_means)
    plt.plot([val[1] for val in smoothed_state_means])
    plt.plot(smooth2)
    # plt.plot([val[5] for val in smoothed_state_means])
    plt.legend(["Non filtered", "Kalman smoothed pos", "Butter pos"])
    plt.show()
    plt.plot(confidences)
    plt.show()
    plt.plot([val[3] for val in smoothed_state_means])
    plt.legend(["Velocity"])
    plt.show()
