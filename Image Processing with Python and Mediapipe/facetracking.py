import cv2
import mediapipe as mp
import time

from mediapipe.python import solutions


mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()

pTime = 0

cap = cv2.VideoCapture(0)
(ih, iw, ic) = cap.read()[1].shape

while 1:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = faceDetection.process(image = imgRGB)
    
    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxc = detection.location_data.relative_bounding_box
            bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih), int(bboxc.width * iw), int(bboxc.height * ih)
            cv2.rectangle(img, bbox, (0, 255, 0), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255, 0), 2)
            
            #mpDraw.draw_detection(img, detection)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("img", img)
    
    cv2.waitKey(1)

