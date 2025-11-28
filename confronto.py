import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import re
from typing import Tuple, List
 
from ultralytics import YOLO
import mediapipe as mp
mp_pose = mp.solutions.pose
from pykalman import KalmanFilter
from scipy.signal import butter, filtfilt

# --- VARIABILI GLOBALI E FUNZIONI DAL TUO SCRIPT PROTOTYPE (NECESSARIE) ---
# ... (Omessa la definizione delle funzioni e costanti per brevità, sono nel file confronto.py) ...

# RIDEFINIZIONE DELLE COSTANTI E FUNZIONI NECESSARIE DAL CONTESTO DI prototype.py e confronto.py
_N_GRIPS = 21
_GRIP_DECL = re.compile(r"(?P<id>\d+)]\s+@(?P<relx>[A-Z][1-2])-SN(?P<sn>\d+)#(?P<rely>\d+)")
_GRIP_LOC = """ Tournament grips by specification: 
                1] @F2-SN2#1    2] @G2-SN2#3    3] @A2-SN2#9
                4] @G1-SN3#4    5] @L1-SN3#10   6] @C2-SN4#2
                7] @L1-SN4#8    8] @C2-SN5#3    9] @E2-SN5#9   
                10] @H1-SN6#2   11] @L1-SN6#7   12] @F1-SN6#9
                13] @M1-SN7#4   14] @G1-SN7#9   15] @L1-SN8#1
                16] @I1-SN8#3   17] @C1-SN8#8   18] @A2-SN9#2
                19] @E2-SN9#7   20] @M1-SN9#10  21] @A2-SN10#10 """
GRIP_VALUES_LIST = [
    [(ord(rel_x[0]) - ord('A')) * 0.1363 + (int(rel_x[1]) - 1) * 1.5, (int(sec) - 1) * 1.5 + int(rel_y) * 0.1363]
    for g_id, rel_x, sec, rel_y in (grip.groups() for grip in _GRIP_DECL.finditer(_GRIP_LOC))
]
GRIP_VALUES = np.array(GRIP_VALUES_LIST)
GRIP_VALUES_Z_EQ_ZERO = np.array([val + [0.0] for val in GRIP_VALUES_LIST])
def pack_into_points(i_xyxy: torch.Tensor, classes: torch.Tensor, pad_zero_z_src: bool = False):
    classes_cpu = classes.cpu().numpy().astype(int)
    if pad_zero_z_src:
        source = np.array(GRIP_VALUES_Z_EQ_ZERO[classes_cpu])
    else:
        source = np.array(GRIP_VALUES[classes_cpu])
    xyxy_cpu = i_xyxy.cpu().numpy()
    dst_centroids = (xyxy_cpu[:, :2] + xyxy_cpu[:, 2:]) / 2
    return dst_centroids, source
def var_n_pnp_solve(i_src: np.ndarray, i_dst: np.ndarray, h_coord, bound_x=None, bound_y=None):
    n = i_src.shape[0]
    if n == 0: return None
    x_point = None
    if n >= 4:
        H_inverse, _ = cv2.findHomography(i_dst, i_src, cv2.RANSAC)
        if H_inverse is None: H_inverse, _ = cv2.findHomography(i_dst, i_src, cv2.LMEDS)
        if H_inverse is None: return var_n_pnp_solve(i_src[:3], i_dst[:3], h_coord, bound_x, bound_y)
        else:
            x_point_w = np.dot(H_inverse, np.array(h_coord + [1.0]).T)
            x_point = np.array([x_point_w[0] / x_point_w[2], x_point_w[1] / x_point_w[2]])
    elif n == 3:
        H = cv2.getAffineTransform(i_src.astype(np.float32), i_dst.astype(np.float32))
        A = H[:, :2]
        t = H[:, 2:].T
        A_inv = np.linalg.inv(A)
        x_point = np.dot(A_inv, (np.array(h_coord) - np.array(t)).T).ravel()
    elif n <= 2: return None # Fallback più sicuro
    if x_point is None: return None
    if bound_x is not None: x_point[0] = np.clip(x_point[0], bound_x[0], bound_x[1])
    if bound_y is not None: x_point[1] = np.clip(x_point[1], bound_y[0], bound_y[1])
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
    xmin, ymin, xmax, ymax = boxes.T 
    margin_x, margin_y = border_ratio*w, border_ratio*h
    mask = (xmin > margin_x) & (ymin > margin_y) & (xmax < w-margin_x) & (ymax < h-margin_y)
    return mask
def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def lowpass_filter(signal, cutoff, fs, order=2):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, signal)

# --- FUNZIONE PRINCIPALE DI ELABORAZIONE ---

def process_video_and_filter(video_path: str, model: YOLO) -> Tuple[np.ndarray, float]:
    """
    Esegue il tracciamento del CdM, la stima della posizione 3D e il filtraggio.
    Restituisce le medie filtrate con Kalman e il frame rate.
    """
    ps: List[np.ndarray] = []
    prev_val = np.array([0.0, 0.0]) # [x, y]
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                      static_image_mode=False, smooth_landmarks=True) as pose:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERRORE: Impossibile aprire il file video: {video_path}")
            return np.array([]), 0.0

        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            try: image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except: break

            # YOLO Detection and Tracking 
            res = model.track(frame, persist=True, iou=0.40, agnostic_nms=False, verbose=False)
            if not res or len(res) == 0: continue
            
            data = res[0]
            boxes = data.boxes
            xyxy = boxes.xyxy.cpu()
            cls = boxes.cls.cpu()
            confs = boxes.conf.cpu().numpy()

            w, h = image_rgb.shape[:2]
            border_suppression_mask = suppress_border_detection(xyxy.numpy(), 0.03, (w, h))
            if not np.all(border_suppression_mask):
                xyxy = xyxy[border_suppression_mask]
                cls = cls[border_suppression_mask]
                confs = confs[border_suppression_mask]

            # Keep only the highest confidence detection for each class (grip ID)
            dup_keep_mask = np.array([confs[i] == confs[cls.numpy() == c].max() for i, c in enumerate(cls.numpy())])
            xyxy = xyxy[dup_keep_mask]
            cls = cls[dup_keep_mask]

            dst, src = pack_into_points(xyxy, cls)

            # MediaPipe Pose Estimation
            results = pose.process(image_rgb)
            lm = results.pose_landmarks
            
            # PnP/Homography Solve
            if lm is None or not lm: pos = prev_val
            else:
                c_i_m = compute_mp_pose_com(lm, data.orig_shape[1], data.orig_shape[0])
                pos = var_n_pnp_solve(src, dst, c_i_m, bound_x=[0, 3.0], bound_y=[0, 15.0])
            
            if pos is None: ps.append(prev_val)
            else:
                ps.append(pos)
                prev_val = pos
        
        cap.release()

    # --- FILTRO DI KALMAN ---
    dt = 1 / fps
    F = np.array([
        [1, 0, dt, 0, 0.5 * dt * dt, 0],
        [0, 1, 0, dt, 0, 0.5 * dt * dt],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    measurements = np.asarray(ps)
    valid_measurements = [m for m in measurements if not np.isnan(m[0]) and not np.isinf(m[0])]
    
    kf_pos = KalmanFilter(transition_matrices=F, observation_matrices=[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    
    if len(valid_measurements) > 2:
        # Usa il filtro di Kalman adattato per l'FPS
        kf_pos = kf_pos.em(valid_measurements, n_iter=30)
    else:
        print(f"AVVISO: Non abbastanza dati validi ({len(valid_measurements)}) per l'EM di Kalman.")
        return np.array([]), fps
        
    (smoothed_state_means, _) = kf_pos.smooth(measurements)
    return smoothed_state_means, fps

# --- CONFIGURAZIONE E ESECUZIONE (MODIFICATA) ---

if __name__ == '__main__':
    
    # 1. CONFIGURAZIONE DEI PERCORSI E FILTRI
    
    _PATH_TO_MODEL = r"C:\Users\marco\OneDrive\Documenti\speed-analysis\best.pt"
    # SOSTITUISCI CON I PERCORSI DEI TUOI DUE VIDEO
    VIDEO_PATH_CLIMBER_A = r"C:\Users\marco\OneDrive\Documenti\speed-analysis\videos\video11.mp4"
    VIDEO_PATH_CLIMBER_B = r"C:\Users\marco\OneDrive\Documenti\speed-analysis\videos\video4.mp4"

    # --- NUOVE VARIABILI DI CONFIGURAZIONE DEL FILTRO ---
    # Frequenza di Taglio (cutoff) per il filtro Butterworth.
    # Valori tipici per il movimento del climber: 3-5 Hz.
    # Se il Video A è ad alto FPS (es. 120), usa un cutoff più basso (es. 2.5) per filtrare più rumore.
    CUTOFF_FREQ_A = 3
    CUTOFF_FREQ_B = 3
    
    # CARICA IL MODELLO UNA VOLTA
    try:
        model = YOLO(_PATH_TO_MODEL)
    except FileNotFoundError:
        print(f"ERRORE: Modello non trovato. Controlla il percorso: {_PATH_TO_MODEL}")
        exit(1)

    # 2. ELABORAZIONE DEI DUE VIDEO
    print(f"Elaborazione del Video A: {VIDEO_PATH_CLIMBER_A}")
    states_A, fps_A = process_video_and_filter(VIDEO_PATH_CLIMBER_A, model)

    print(f"Elaborazione del Video B: {VIDEO_PATH_CLIMBER_B}")
    states_B, fps_B = process_video_and_filter(VIDEO_PATH_CLIMBER_B, model)

    if states_A.size == 0 or states_B.size == 0:
        print("Impossibile generare i grafici: uno o entrambi i video non hanno prodotto dati validi.")
        exit(1)

    # 3. ESTRAZIONE E FILTRAGGIO DELLE METRICHE
    
    # --- POSIZIONE Y (Componente 1) ---
    # Applica il filtro Low-pass (Butter) utilizzando il cutoff e l'FPS appropriati
    smooth_pos_A = lowpass_filter([val[1] for val in states_A], CUTOFF_FREQ_A, fps_A)
    smooth_pos_B = lowpass_filter([val[1] for val in states_B], CUTOFF_FREQ_B, fps_B)

    # --- VELOCITÀ Y (Componente 3) ---
    # La Velocità Y è già filtrata dal Filtro di Kalman. Non applichiamo un secondo filtro.
    vel_Y_A = [val[3] for val in states_A]
    vel_Y_B = [val[3] for val in states_B]

    # --- ACCELERAZIONE Y (Componente 5) ---
    # L'accelerazione è la più rumorosa, quindi applichiamo il filtro Low-pass anche qui.
    # Utilizziamo lo stesso cutoff e FPS per mantenere la coerenza con la posizione.
    acc_Y_A_raw = [val[5] for val in states_A]
    acc_Y_B_raw = [val[5] for val in states_B]
    
    acc_Y_A_smooth = lowpass_filter(acc_Y_A_raw, CUTOFF_FREQ_A, fps_A)
    acc_Y_B_smooth = lowpass_filter(acc_Y_B_raw, CUTOFF_FREQ_B, fps_B)
    
    # Tempo (asse X) basato sui frame rate
    time_A = np.arange(len(smooth_pos_A)) / fps_A
    time_B = np.arange(len(smooth_pos_B)) / fps_B

    # 4. GENERAZIONE DEI GRAFICI DI CONFRONTO

    # --- GRAFICO 1: POSIZIONE VERTICALE (Y) ---
    plt.figure(figsize=(12, 6))
    plt.plot(time_A, smooth_pos_A, label='Climber A (Posizione Y)', color='blue')
    plt.plot(time_B, smooth_pos_B, label='Climber B (Posizione Y)', color='red', linestyle='--')
    plt.title('Confronto Traiettoria Verticale del Centro di Massa')
    plt.xlabel(f'Tempo (s) - (FPS A: {fps_A:.1f}, FPS B: {fps_B:.1f})')
    plt.ylabel('Altezza (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- GRAFICO 2: VELOCITÀ VERTICALE (Vy) ---
    # Usiamo i dati solo da Kalman (vel_Y_A/B)
    plt.figure(figsize=(12, 6))
    plt.plot(time_A, vel_Y_A, label='Climber A (Velocità Y)', color='blue')
    plt.plot(time_B, vel_Y_B, label='Climber B (Velocità Y)', color='red', linestyle='--')
    plt.title('Confronto Velocità Verticale del Centro di Massa (Kalman)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Velocità (m/s)')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='gray', linestyle='-') # Linea di riferimento per v=0
    plt.show()

    # --- GRAFICO 3: ACCELERAZIONE VERTICALE (Ay) ---
    # Usiamo i dati FILTRATI con Butterworth (acc_Y_A_smooth/B_smooth)
    plt.figure(figsize=(12, 6))
    plt.plot(time_A, acc_Y_A_smooth, label=f'Climber A (Acc Y, Cutoff={CUTOFF_FREQ_A}Hz)', color='blue')
    plt.plot(time_B, acc_Y_B_smooth, label=f'Climber B (Acc Y, Cutoff={CUTOFF_FREQ_B}Hz)', color='red', linestyle='--')
    plt.title('Confronto Accelerazione Verticale del Centro di Massa (Filtro Butterworth)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Accelerazione (m/s²)')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='gray', linestyle='-') # Linea di riferimento per a=0
    plt.show()