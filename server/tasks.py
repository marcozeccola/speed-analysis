import os
# Workaround for multiple OpenMP runtimes (unsafe — prefer fixing environment)
# If you prefer, remove this and create a conda env with consistent builds (see environment.yml)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Enable Intel optimizations
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 4)

from celery import Celery
import time
import random

import re
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import mediapipe as mp

# Enable OpenCV optimizations for Intel CPU
cv2.setUseOptimized(True)
cv2.setNumThreads(os.cpu_count() or 4)

mp_pose = mp.solutions.pose

from pykalman import KalmanFilter
from ultralytics.engine.results import Results
from ultralytics import YOLO
import os

REDIS_PORT = 6379
REDIS_URL = "redis://localhost:6379/0"
REDIS_BACKEND = "redis://localhost:6379/1"

app = Celery(
    'video_tasks', 
    broker=REDIS_URL,
    backend=REDIS_BACKEND
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
)



@app.task(name='analyze_climbing_videos', bind=True)
def analyze_climbing_videos_task(self, video_path_a: str, video_path_b: str = None) -> dict:
    """ 
    Analizza i video di arrampicata.
    
    :param video_path_a: Percorso assoluto al primo video
    :param video_path_b: Percorso assoluto al secondo video (opzionale)
    :return: Dati dell'analisi per entrambi i video
    """
    # Ensure model is loaded in the worker process (Celery runs in a separate process)
    global model, USE_GPU, DEVICE
    if model is None:
        model_path = os.environ.get("MODEL_PATH") or os.path.join(os.path.dirname(__file__), "best.pt")
        try:
            model = YOLO(model_path)
            
            # GPU optimization
            if USE_GPU:
                model.to(DEVICE)
                print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                model.fuse()  # CPU optimization only
                print("⚠️ Running on CPU (no GPU detected)")
                
        except Exception as e:
            return {"error": f"Failed to load model at {model_path}: {e}", "status": "failed"}

    try:
        # Update state to STARTED
        self.update_state(
            state='STARTED',
            meta={'message': 'Starting video analysis...', 'progress': 0, 'current': 0, 'total': 100}
        )
        
        # Analyze first video
        self.update_state(
            state='PROGRESS',
            meta={'message': '📹 Video 1: Caricamento...', 'progress': 10, 'current': 10, 'total': 100}
        )
        result_a = act(video_path_a, task=self, progress_start=10, progress_end=50 if video_path_b else 90, video_number=1)
        
        result = {
            "climber_A": result_a,
            "status": "completed"
        }
        
        # Se c'è un secondo video, analizzalo anche
        if video_path_b:
            self.update_state(
                state='PROGRESS',
                meta={'message': '📹 Video 2: Caricamento...', 'progress': 50, 'current': 50, 'total': 100}
            )
            result_b = act(video_path_b, task=self, progress_start=50, progress_end=90, video_number=2)
            result["climber_B"] = result_b
        
        # Final processing
        self.update_state(
            state='PROGRESS',
            meta={'message': 'Finalizing results...', 'progress': 95, 'current': 95, 'total': 100}
        )
        
        return result
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }


model = None

# GPU Detection and Configuration
USE_GPU = torch.cuda.is_available()
DEVICE = 'cuda:0' if USE_GPU else 'cpu'

if USE_GPU:
    print(f"🚀 GPU Mode: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  CPU Mode: No GPU detected")

def set_model( glob_model ):
    global model
    model = glob_model

from scipy.signal import butter, lfilter, freqz


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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def var_n_pnp_solve(i_src: np.ndarray, i_dst: np.ndarray, h_coord, xyxy_tensor=None, bound_x=None, bound_y=None):
    """

    :param i_src:
    :param i_dst:
    :param h_coord:
    :param xyxy_tensor: Bounding boxes tensor from YOLO
    :return:
    """
    # N is the number of tracked reference points.
    n = i_src.shape[0]

    if n == 0:
        return None
    
    x_point = None
    
    if xyxy_tensor is not None:
        val = xyxy_tensor.cpu().numpy() if hasattr(xyxy_tensor, 'cpu') else xyxy_tensor
    else:
        val = None

    GRIP_SIZE = 0.35

    x_points = []
    weights = []
    
    if val is not None and len(val) > 0:
        for i in range(len(val)):
            s = [(val[i][2] - val[i][0]), (val[i][3] - val[i][1])]
            s = (sum(s) / 2) / GRIP_SIZE
            v0_to_h = -(np.array(h_coord) - i_dst[i]) / s
            # weights.append(1 / ((np.linalg.norm(v0_to_h)**2)/18 + 1))
            x_points.append((i_src[i] + v0_to_h))

        # weights = softmax(weights)
        return np.average(x_points,
                          # weights=weights,
                          axis=0)

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
        if val is None or len(val) == 0:
            print("WARNING: No bounding boxes available for scale estimation")
            return None
            
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

def compute_mp_pose_com(landmarks, width, height, pose_landmark = -1) -> list:
    lm_lm = landmarks.landmark
    x = y = None
    if (pose_landmark == -1):
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
        x = (lm_lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x)
        y = (lm_lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y)
    elif pose_landmark == 25:
        x = (lm_lm[mp_pose.PoseLandmark.LEFT_KNEE].x)
        y = (lm_lm[mp_pose.PoseLandmark.LEFT_KNEE].y)
    elif pose_landmark == 26:
        x = (lm_lm[mp_pose.PoseLandmark.RIGHT_KNEE].x)
        y = (lm_lm[mp_pose.PoseLandmark.RIGHT_KNEE].y)
    return [x * width, y * height]

def suppress_border_detection(boxes: np.ndarray, border_ratio: float, image_dim: Tuple[int, int]):
    h, w = image_dim
    xmin, ymin, xmax, ymax = boxes.T # We take the transpose to obtain 4 columns instead of 4 rows
    margin_x, margin_y = border_ratio*w, border_ratio*h
    mask = (xmin > margin_x) & (ymin > margin_y) & (xmax < w-margin_x) & (ymax < h-margin_y)
    return mask


def act(video_path, task=None, progress_start=0, progress_end=100, video_number=1):
    """
    Elabora un video di arrampicata e estrae i dati biomeccanici.
    
    :param video_path: Percorso al file video
    :param task: Task Celery per aggiornare lo stato (opzionale)
    :param progress_start: Percentuale di inizio progresso
    :param progress_end: Percentuale di fine progresso
    :param video_number: Numero del video (1 o 2) per i messaggi di stato
    :return: Dizionario con i dati elaborati
    """
    ys = []
    ps = []
    max_frame: int = 0
    idx = 0
    prev_val = [0, 0]
    confidences = []

    # Dictionary to contain all landmarks of all timesteps
    # Reduced from 8 to 5 landmarks for 40% speedup - removed: nose(0), knees(25,26)
    # Essential for climbing: body center, wrists (hand movement), ankles (foot placement)
    chosen_landmarks = [-1, 15, 16, 27, 28]  # Center, Left/Right Wrist, Left/Right Ankle
    position_for_all_landmarks = {
        # (ys_list, prev_val, ps_list) -- initialize prev_val as [0,0] to avoid None
        land : ([], [0, 0], []) for land in chosen_landmarks
    }

    # Use model_complexity=0 for faster CPU inference (lite model)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                      static_image_mode=False, smooth_landmarks=True, model_complexity=0) as pose:
        try:
            cap = cv2.VideoCapture(video_path)
        except FileNotFoundError:
            print(f"ERROR: Failed to load the video. Please ensure the path exists: {video_path}")
            return {"error": f"Video not found: {video_path}"}
        
        if not cap.isOpened():
            raise IOError(f"ERROR: CV2 Could not open file: {video_path}")
        
        # Get total frame count for progress reporting
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1000  # Fallback estimate
        
        # OPTIMIZATION: Process every N frames for 2-3x speedup (1=all frames, 2=half, 3=third)
        frame_skip = 2  # Change to 1 to process all frames, 2 for 2x speed, 3 for 3x speed

        while cap.isOpened() and (not max_frame or idx < max_frame):
            ret, frame = cap.read()
            t_frame_start = time.perf_counter()

            if not ret:
                break
            idx += 1
            
            # Skip frames for speed (process every Nth frame)
            if idx % frame_skip != 0:
                continue
            
            # Update progress every 10 frames
            if task and idx % 10 == 0:
                progress = progress_start + int((idx / total_frames) * (progress_end - progress_start))
                task.update_state(
                    state='PROGRESS',
                    meta={'message': f'📹 Video {video_number}: Analisi frame {idx}/{total_frames}', 'progress': progress, 'current': progress, 'total': 100}
                )
            
            try:
                # CV2 legacy imports images as BGR instead of RGB, so map back.
                t_convert = time.perf_counter()
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # print(f"  cvtColor: {(time.perf_counter()-t_convert)*1000:.1f}ms") if idx % 50 == 0 else None
            except IOError:
                print(f"WARNING: Failed to map frame {idx} into RGB format. Terminating early the footage scan.")
                break
            # Optimize YOLO inference for GPU/CPU
            t_yolo = time.perf_counter()
            res: list[Results] = model.track(
                frame, 
                device=DEVICE,
                half=USE_GPU,  # FP16 only on GPU for speed
                persist=True, 
                iou=0.40, 
                conf=0.25, 
                verbose=False, 
                agnostic_nms=False
            )
            # print(f"  YOLO: {(time.perf_counter()-t_yolo)*1000:.1f}ms") if idx % 50 == 0 else None
            if not res or len(res) == 0:
                continue
            data = res[0]

            repr_img = None
            t_mediapipe = time.perf_counter()
            results = pose.process(image_rgb)
            # print(f"  MediaPipe: {(time.perf_counter()-t_mediapipe)*1000:.1f}ms") if idx % 50 == 0 else None

            if data is None or not data:
                print(f"WARNING: frame {idx} resulted in no data output")
                continue

            # SWITCH THIS TO CHANGE FROM A MULTI-OBJECT TRACK TO A SINGLE-OBJECT TRACK:
            # R IS A SINGLE OBJECT, DATA IS MULTIPLE OBJECTS
            r = data[0]
            # SIMPLY PUT r.boxes instead of data.boxes to get a SINGLE TRACK TRACKING
            boxes = data.boxes  # Bounding boxes
            xyxy_tensor = boxes.xyxy.cpu()  # Keep as tensor for pack_into_points
            cls_tensor = boxes.cls.cpu()  # Keep as tensor for pack_into_points
            
            # Convert to numpy for filtering operations
            xyxy_np = xyxy_tensor.numpy()
            cls_np = cls_tensor.numpy()
            confs = boxes.conf.cpu().numpy()

            w, h = image_rgb.shape[:2]
            border_suppression_mask = suppress_border_detection(xyxy_np, 0.03, (w, h))
            if not np.all(border_suppression_mask):
                xyxy_tensor = xyxy_tensor[border_suppression_mask]
                cls_tensor = cls_tensor[border_suppression_mask]
                xyxy_np = xyxy_np[border_suppression_mask]
                cls_np = cls_np[border_suppression_mask]
                confs = confs[border_suppression_mask]

            # Vectorized duplicate removal - much faster
            if len(cls_np) > 0:
                unique_classes = np.unique(cls_np)
                dup_keep_mask = np.zeros(len(cls_np), dtype=bool)
                for c in unique_classes:
                    class_mask = cls_np == c
                    max_conf_idx = np.argmax(confs * class_mask)
                    dup_keep_mask[max_conf_idx] = True
                xyxy_tensor = xyxy_tensor[dup_keep_mask]
                cls_tensor = cls_tensor[dup_keep_mask]
                xyxy_np = xyxy_np[dup_keep_mask]

            avg = (np.average(confs) if len(confs) > 0 else 0)
            confidences.append(np.sqrt(avg))
            # Filtering and PnP dispatch...
            dst, src = pack_into_points(xyxy_tensor, cls_tensor)

            lm = results.pose_landmarks
            for chosen_landmark in chosen_landmarks:
                ys, prev_val, ps = position_for_all_landmarks[chosen_landmark]
                if lm is None or not lm:
                    ps.append(prev_val)
                    ys += [prev_val[1]]
                else:
                    c_i_m = compute_mp_pose_com(lm, data[0].orig_shape[1], data[0].orig_shape[0], chosen_landmark)
                    pos = var_n_pnp_solve(src, dst, c_i_m, xyxy_tensor=xyxy_np, bound_x=[0, 3.0], bound_y=[0, 15.0])
                    if pos is None:
                        ps.append(prev_val)
                        ys += [prev_val[1]]
                    else:
                        ps.append(pos)
                        prev_val = pos
                        ys += [pos[1]]

                # write back updated prev_val and lists into the dict
                position_for_all_landmarks[chosen_landmark] = (ys, prev_val, ps)
            
            # Performance monitoring (uncomment to debug)
            if idx % 50 == 0:
                t_total = time.perf_counter() - t_frame_start
                print(f"Frame {idx}: TOTAL={t_total*1000:.1f}ms")

    dt = 1 / cap.get(cv2.CAP_PROP_FPS)

    # Update state: starting Kalman filter processing
    if task:
        kalman_progress = progress_end - 5  # Reserve last 5% for Kalman
        task.update_state(
            state='PROGRESS',
            meta={'message': f'🔬 Video {video_number}: Filtraggio Kalman in corso...', 'progress': kalman_progress, 'current': kalman_progress, 'total': 100}
        )
    
    print(f"> Running Kalman filter with time step {dt}")
    F = np.array([
        [1, 0, dt, 0, 0.5 * dt * dt, 0],
        [0, 1, 0, dt, 0, 0.5 * dt * dt],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

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
    
    # OPTIMIZATION: Estimate Kalman parameters ONCE using first landmark, then reuse for all
    # This gives 5-8x speedup vs doing EM for each landmark
    first_landmark_ps = position_for_all_landmarks[chosen_landmarks[0]][2]
    first_measurements = np.asarray(first_landmark_ps)
    kf_trained = kf_pos.em([m for m in first_measurements if not np.isnan(m[0])], n_iter=30)  # Reduced from 140 to 30

    final_answer = { }

    for landmark_idx, chosen_landmark in enumerate(chosen_landmarks):
        # Update progress during Kalman processing
        if task and len(chosen_landmarks) > 0:
            kalman_progress = progress_end - 5 + int((landmark_idx / len(chosen_landmarks)) * 5)
            task.update_state(
                state='PROGRESS',
                meta={'message': f'🔬 Video {video_number}: Filtraggio Kalman ({landmark_idx+1}/{len(chosen_landmarks)} landmarks)', 'progress': kalman_progress, 'current': kalman_progress, 'total': 100}
            )

        ys, prev_val, ps = position_for_all_landmarks[chosen_landmark]

        measurements = np.asarray(ps)
        # Use pre-trained Kalman filter instead of training each time
        (smoothed_state_means, smoothed_state_covariances) = kf_trained.smooth(measurements)
        yf = butter_lowpass_filter(ys, cutoff, fs, order)

        # plt.plot removed from loop - was causing slowdown

        vel = [val[3] for val in smoothed_state_means]
        velf = butter_lowpass_filter(vel, cutoff, fs, order+1)

        final_answer[chosen_landmark] = yf, velf

    # Convert results to JSON-serializable structures (lists)
    output = {}
    for k, v in final_answer.items():
        try:
            pos, vel = v
        except Exception:
            continue
        pos_list = np.asarray(pos).tolist() if pos is not None else []
        vel_list = np.asarray(vel).tolist() if vel is not None else []
        output[str(k)] = {"pos": pos_list, "vel": vel_list}

    # Return structured, serializable result
    return {"landmarks": output, "dt": float(dt), "confidences": [float(c) for c in confidences]}