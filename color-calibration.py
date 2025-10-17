import cv2
import numpy as np
 
VIDEO_PATH = './video/test5.mp4' 
 
lower_red_1 = np.array([0, 130, 100])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([160, 130, 100])
upper_red_2 = np.array([180, 255, 255])
lower_dark_red = np.array([0, 50, 40])  # Per prese scure/piedi
upper_dark_red = np.array([180, 129, 255])

MIN_HOLD_AREA = 50   # Area minima per filtrare il rumore
MAX_HOLD_AREA = 5000 # Area massima
 
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERRORE: Impossibile aprire il file video: {VIDEO_PATH}")
    exit()
 
def detect_and_map_holds(image):
    # Rilevamento Colore e Maschere
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2) 
     
    final_mask = mask1 + mask2 
     
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel) 
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    # Trova i contorni
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hold_data = [] 
    output_image = image.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > MIN_HOLD_AREA and area < MAX_HOLD_AREA:
            
            # Calcola il Centroide (CX, CY)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Registra i dati per l'ordinamento
                hold_data.append({
                    'center_x': cX,
                    'center_y': cY,
                    'contour': contour
                })
 
    hold_data.sort(key=lambda x: x['center_y'], reverse=True) 

    #  Assegna l'ID  
    holds_centers_with_id = []
    
    for i, data in enumerate(hold_data): 
        hold_id = f"P{i + 1}"
         
        holds_centers_with_id.append({
            'id': hold_id,
            'x': data['center_x'],
            'y': data['center_y']
        })
        
        # Disegno: Contorno e Centro
        cv2.drawContours(output_image, [data['contour']], 0, (0, 255, 255), 2)
        cv2.circle(output_image, (data['center_x'], data['center_y']), 5, (0, 255, 0), -1)
         
        cv2.putText(output_image, hold_id, (data['center_x'] + 10, data['center_y'] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return output_image, final_mask, holds_centers_with_id
 
while cap.isOpened():
     success, image = cap.read()
    
     if not success:
        break

     #image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) 
     processed_image, color_mask, mapped_holds = detect_and_map_holds(image)
      
     # Visualizzazione
     cv2.imshow('Analisi Speed Climbing - Mappatura Prese', processed_image)
     cv2.imshow('Maschera Colore (Verifica)', color_mask)
     
     if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()