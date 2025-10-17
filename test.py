import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose(
    static_image_mode=False,
    model_complexity=2,  
    min_detection_confidence=0.3 )

cap = cv2.VideoCapture('video/test4.mp4')
pTime = 0
while True:
     success, img = cap.read()
     if not success:
        break
    
     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     results = pose.process(imgRGB)
     print(results.pose_landmarks)
     if(results.pose_landmarks):
         mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

     cTime = time.time()
     fps = 1 / (cTime - (pTime - 1))
     pTime = cTime

     cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
     
     cv2.imshow("Image", img)
     if cv2.waitKey(1) & 0xFF == ord('q'):
        break