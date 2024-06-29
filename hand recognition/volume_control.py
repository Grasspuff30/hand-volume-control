import cv2
import caer
import time
import math
import numpy as np
from modules import hand_tracking_module as htm
from subprocess import call

# 250,30
w = 640
h = 480

cap = cv2.VideoCapture(0)
ptime = 0
cap.set(3, w)
cap.set(4, h)

detector = htm.Hand_Detector(detection_confidence=0.7)

while True:
    ret, frame = cap.read()
    cv2.flip(frame, 1, frame)
    frame = detector.find_hands(frame)
    lm_list = detector.find_position(frame, draw=False)
    if len(lm_list):
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        dist = math.hypot(x2 - x1, y2 - y1)
        cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 5)
        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 10, (255, 0, 0), cv2.FILLED)
        target_volume = np.interp(dist, [30, 250], [0, 100])
        call([f"osascript -e 'set volume output volume {target_volume}'"], shell=True)
        if dist < 40:
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 10, (0, 255, 0), cv2.FILLED)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(frame, f" FPS : {int(fps)}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
