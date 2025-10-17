import cv2 
import numpy as np  
import mediapipe as mp  
import matplotlib.pyplot as plt
from collections import defaultdict  
import os  

# Inizializzazione dei moduli MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
 
def extract_keypoints(video_path, visualize=True):
     """
     Processa il video per estrarre i 33 punti chiave del corpo umano (keypoints)
     utilizzando MediaPipe Pose e li salva in coordinate normalizzate (0.0 a 1.0).

          Se visualize=True, mostra il video con lo scheletro disegnato.
     
     Ritorna: 
          - athlete_keypoints (dict): {frame_idx: array NumPy dei landmark [x, y, z, visibilità]}
          - total_frames (int): Numero totale di frame processati
     """
     print(f"-> Fase 1: Inizio estrazione keypoints con MediaPipe da: {video_path}")
     
     cap = cv2.VideoCapture(video_path)
     if not cap.isOpened():
          raise IOError(f"Impossibile aprire il file video: {video_path}")
          
     frame_idx = 0
     athlete_keypoints = defaultdict(list)
     
     # Configurazione del modello MediaPipe
     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
          
          while cap.isOpened():
               ret, frame = cap.read()
               if not ret:
                    # Fine del video
                    break 

               # conversione BGR a RGB
               image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
               image_rgb.flags.writeable = False  
               
               frame_height = image_rgb.shape[0]
               frame_width = image_rgb.shape[1]
               # Esecuzione del modello
               results = pose.process(image_rgb)

               # Archiviazione dei dati
               if results.pose_landmarks:
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                         landmarks.append([
                         landmark.x * frame_width, 
                         frame_height - landmark.y * frame_height,
                         landmark.visibility 
                         ])
 
                    athlete_keypoints[frame_idx] = np.array(landmarks)
                     
                    if visualize:
                         # Riconverte il frame in BGR e lo rende scrivibile per disegnare sopra
                         image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                         image_bgr.flags.writeable = True 

                         # Disegna i punti chiave e il rig
                         '''mp_drawing.draw_landmarks(
                              image_bgr, 
                              results.pose_landmarks, 
                              mp_pose.POSE_CONNECTIONS, 
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                              connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=1)
                         )'''

                         #disegna i punti di polso destro e piede sinistro
                         cv2.circle(image_bgr, (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][0] * image_bgr.shape[1]), 
                                                int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][1] * image_bgr.shape[0])), 
                                                8, (0, 0, 255), -1)
                         cv2.circle(image_bgr, (int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value][0] * image_bgr.shape[1]), 
                                                int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value][1] * image_bgr.shape[0])), 
                                                8, (255, 255, 0), -1) 
                          
                         cv2.imshow('MediaPipe pose visualization (Premi q per uscire)', image_bgr)
                         
                         # Interrompe la visualizzazione se l'utente preme 'q'
                         if cv2.waitKey(1) & 0xFF == ord('q'):
                              visualize = False
               
               frame_idx += 1
               
     cap.release()
     cv2.destroyAllWindows()
     print(f"-> Fase 1 completata. Keypoints estratti per {len(athlete_keypoints)} frame.")
     return athlete_keypoints, frame_idx
 
def calculate_camera_shift(video_path, total_frames, x_exclude_start_norm=0.35, x_exclude_end_norm=0.65):
     """
     Calcola lo spostamento medio verticale (shift Y) della telecamera tra 
     frame consecutivi tracciando le feature statiche sulla parete (Optical Flow).
     
     Parametri:
     - x_exclude_start_norm (float): Inizio (0.0-1.0) della zona da escludere.
     - x_exclude_end_norm (float): Fine (0.0-1.0) della zona da escludere.

     Ritorna:
          - frame_shifts_y (array NumPy): Vettore dello shift Y in pixel per ogni frame.
     """
     print(f"-> Fase 2: Inizio calcolo shift telecamera (Compensazione)")
     
     cap = cv2.VideoCapture(video_path)
     if not cap.isOpened():
          raise IOError(f"Impossibile aprire il file video: {video_path}")
          
     frame_shifts_y = np.zeros(total_frames) 
     
     ret, prev_frame = cap.read()
     if not ret: return frame_shifts_y
 
     #prev_frame = cv2.rotate(prev_frame, cv2.ROTATE_90_CLOCKWISE)
     
     prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
     
     # Ottieni le dimensioni del frame  
     H, W, _ = prev_frame.shape

     # creazione maschera per escludere la zona centrale
     mask = np.ones((H, W), dtype=np.uint8) * 255 # Tutta bianca inizialmente

     x_start_pix = int(W * x_exclude_start_norm)
     x_end_pix = int(W * x_exclude_end_norm)
 
     mask[:, x_start_pix:x_end_pix] = 0
     
     # Parametri per ShiTomasi corner detection e Lucas-Kanade Optical Flow
     feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 20, blockSize = 7)
     lk_params = dict(winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

     # Rilevamento iniziale delle features
     p0 = cv2.goodFeaturesToTrack(prev_frame_gray, mask=mask, **feature_params)
     
     frame_idx = 1
     
     while cap.isOpened():
          ret, frame = cap.read()
          if not ret: break
          if frame_idx >= total_frames: break
 
          #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

          current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          
          if p0 is not None and len(p0) > 0:
               
               p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_frame_gray, current_frame_gray, p0, None, **lk_params)

               if p1 is not None and st is not None and np.any(st):
                    
                    # Filtraggio e Reshape
                    good_new = p1[st.flatten() == 1]
                    good_old = p0[st.flatten() == 1]
                    
                    good_new = good_new.reshape(-1, 2)
                    good_old = good_old.reshape(-1, 2)
                    
                    if len(good_new) > 0:
                         # Calcolo dello Spostamento Medio  
                         diff = good_new - good_old 
                          
                         y_shift = diff[:, 1]
                         
                         mean_shift_y = np.mean(y_shift)
                         frame_shifts_y[frame_idx] = mean_shift_y
                         
                         # Aggiornamento per l'iterazione successiva
                         prev_frame_gray = current_frame_gray.copy()
                         p0 = good_new.reshape(-1, 1, 2) 
                         
                    else:
                         p0 = cv2.goodFeaturesToTrack(current_frame_gray, mask=mask, **feature_params)
               else:
                    p0 = cv2.goodFeaturesToTrack(current_frame_gray, mask=mask, **feature_params)
          else:
               p0 = cv2.goodFeaturesToTrack(current_frame_gray, mask=mask, **feature_params)

          frame_idx += 1
          
     cap.release()
     print("-> Fase 2 completata.")
     return frame_shifts_y
 
def visualize_mask_calibration(video_path, x_exclude_start_norm, x_exclude_end_norm):
    """
    Visualizza il primo frame del video con l'area di esclusione  
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire il file video: {video_path}")
        return

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return
 
    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    H, W, _ = frame.shape

    # Calcola i pixel a partire dalle coordinate  
    x_start_pix = int(W * x_exclude_start_norm)
    x_end_pix = int(W * x_exclude_end_norm)

    # Crea un frame di output
    output_frame = frame.copy()

    # Disegna linee verticali sui bordi della zona esclusa
    # Bordo sinistro  
    cv2.line(output_frame, (x_start_pix, 0), (x_start_pix, H), (0, 0, 255), 5)
    # Bordo destro 
    cv2.line(output_frame, (x_end_pix, 0), (x_end_pix, H), (0, 0, 255), 5)
      
    cv2.imshow('Calibrazione Maschera - Regola le linee ROSSE', output_frame)
    # Aspetta che l'utente prema un tasto
    key = cv2.waitKey(0) 
    cv2.destroyAllWindows()
    cap.release()

    if key & 0xFF == ord('q'):
        return False
    else:
        return True
    
 
def run_speed_climbing_analysis(video_file, x_exclude_start, x_exclude_end):
      
     
     print("--- INIZIO ANALISI VIDEO SPEED CLIMBING ---")
     
     # Verifica file  
     if not os.path.exists(video_file):
          print(f"ERRORE: File video non trovato. Assicurati che il file '{video_file}' esista.")
          return

     #Estrazione dei keypoints
     keypoints_data, total_frames = extract_keypoints(video_file)
     
     # Calcolo dello shift della telecamera  
     camera_shifts = calculate_camera_shift(
          video_file, 
          total_frames,
          x_exclude_start_norm=x_exclude_start,
          x_exclude_end_norm=x_exclude_end
     )
      
     print("\n--- SINTESI DEI DATI ESTRATTI ---")
   
     current_shift = 0
     if len(camera_shifts) > 1:
          print(f"\n2. Shift telecamera calcolati per {len(camera_shifts)} frame.")
          
          # STAMPA TUTTI GLI SHIFT
          print("\n[Shift Y per Frame (Pixel Differenziale)]")

          shifts_rounded = np.round(camera_shifts, 2)
          
          comulative_shift = 0
          frames = []
          hip_positions = []
          hand_positions = []
          foot_positions = []

          for frame, shift in enumerate(camera_shifts):
               
               idx_anca_destra = mp_pose.PoseLandmark.RIGHT_HIP.value 
               idx_right_hand = mp_pose.PoseLandmark.RIGHT_WRIST.value
               idx_left_foot = mp_pose.PoseLandmark.LEFT_HEEL.value

               hip_y_norm = keypoints_data[frame][idx_anca_destra][1]
               right_hand_y_norm = keypoints_data[frame][idx_right_hand][1]
               left_foot_y_norm = keypoints_data[frame][idx_left_foot][1]

               comulative_shift += shift
               real_hip_y = hip_y_norm + comulative_shift
               real_hand_y = right_hand_y_norm + comulative_shift
               real_foot_y = left_foot_y_norm + comulative_shift
               diff = real_hand_y - real_foot_y



               frames.append(frame)
               hip_positions.append(real_hip_y)
               hand_positions.append(real_hand_y)
               foot_positions.append(real_foot_y)

               #print(f"  Frame {frame} right_hand_y_norm: {right_hand_y_norm:.2f} left_foot_y_norm: {left_foot_y_norm:.2f}")
               print(f"Frame {frame}  real_hand_y: {real_hand_y:.2f} real_hip_y: {real_hip_y:.2f} real_foot_y: {real_foot_y:.2f} diff: {diff:.2f}")
          
           

          print(shifts_rounded) 
          wall_in_px = max(hand_positions)-min(foot_positions)
          print(f"low foot: {min(foot_positions)} high hand: {max(hand_positions)}")
          wall_in_m = 15
          px_in_m = wall_in_px/wall_in_m
          m_in_px = 1/px_in_m

          print(f"\n[Calibrazione Parete]")
          print(f"   Altezza parete in pixel: {wall_in_px:.2f} px")
          print(f"   Altezza parete in metri: {wall_in_m} m")
          print(f"   Pixel per metro: {px_in_m:.2f} px/m")
          print(f"   Metri per pixel: {1/px_in_m:.4f} m/px")
          
          hip_positions = np.array(hip_positions)
          hand_positions = np.array(hand_positions)
          foot_positions = np.array(foot_positions)

          real_hip_y_m = hip_positions * m_in_px
          real_hand_y_m = hand_positions * m_in_px     
          real_foot_y_m = foot_positions * m_in_px

          print("\nValori real_hip_y_m:")
          print(real_hip_y_m)
          print("\nValori real_hand_y_m:")
          print(real_hand_y_m)
          print("\nValori real_foot_y_m:")
          print(real_foot_y_m) 

          #plot velocita dell hip
          plt.figure(figsize=(10, 6))
          plt.plot(frames, real_hand_y_m)
          plt.show()
          
          shift_cumulativo_totale = np.sum(camera_shifts)
          
          '''
          print(f"\n[Sintesi Numerica]")
          print(f"   Shift Y cumulativo totale (pixel): {shift_cumulativo_totale:.2f}")

          if total_frames > 20:
               esempio_frame = 20
               shift_y = camera_shifts[esempio_frame]
               shift_cumulativo = np.cumsum(camera_shifts)[esempio_frame] 
               print(f"   Esempio (Frame {esempio_frame}): Shift Y rispetto al frame precedente (pixel): {shift_y:.2f}")
               print(f"   Shift Y cumulativo a Frame {esempio_frame} (pixel): {shift_cumulativo:.2f}")
          '''

if __name__ == "__main__": 
     VIDEO_INPUT = './video/test7.mp4' 
      
     X_EXCLUDE_START = 0.45 
     X_EXCLUDE_END = 1    
 
     #print(f"Parametri attuali: X_EXCLUDE_START={X_EXCLUDE_START}, X_EXCLUDE_END={X_EXCLUDE_END}")
     #print("Premi un tasto (es. Invio) dopo aver visualizzato e confermato il frame per continuare l'analisi.")
     #should_continue = visualize_mask_calibration(VIDEO_INPUT, X_EXCLUDE_START, X_EXCLUDE_END)
     
     #if not should_continue:
        #print("\n[TERMINAZIONE] Analisi interrotta dall'utente (premuto 'q').")
        #exit()  
 
     run_speed_climbing_analysis(VIDEO_INPUT, X_EXCLUDE_START, X_EXCLUDE_END)