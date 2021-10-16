import cv2
import mediapipe as mp
import time


mpDraw = mp.solutions.drawing_utils
mpPoseDetection = mp.solutions.pose
poseDetection = mpPoseDetection.Pose()

pTime = 0

cap = cv2.VideoCapture(0)


while 1:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = poseDetection.process(image = imgRGB)
    
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPoseDetection.POSE_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("img", img)
    
    cv2.waitKey(1)

