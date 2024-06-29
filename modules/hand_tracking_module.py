import cv2
import mediapipe as mp
import time


class Hand_Detector:
    def __init__(self, mode=False, num_hands=2, complexity=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.results = None
        self.mode = mode
        self.num_hands = num_hands
        self.complexity = complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.num_hands, self.complexity, self.detection_confidence, self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLandMarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLandMarks, self.mpHands.HAND_CONNECTIONS)
        return frame

    def find_position(self, frame, hand_no=0 , draw=True):

        lm_list = []
        if self.results.multi_hand_landmarks:
            hand_land_marks = self.results.multi_hand_landmarks[hand_no]
            h, w, c = frame.shape
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for id, lms in enumerate(hand_land_marks.landmark):
                cx, cy = int(lms.x * w), int(lms.y * h)
                lm_list.append([id, cx, cy])
                x, y = int(lms.x * w), int(lms.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255,0,255), cv2.FILLED)
            if draw:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return lm_list

def main():
    cap = cv2.VideoCapture(0)
    ptime = 0

    detector = Hand_Detector()

    while True:
        ret, frame = cap.read()
        cv2.flip(frame, 1, frame)
        frame = detector.find_hands(frame)
        lm_list = detector.find_position(frame)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, f" FPS : {int(fps)}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)


if __name__ == "__main__":

    main()
