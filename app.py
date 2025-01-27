from flask import Flask, render_template, Response
import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import pygame
from facial_emotion_recognition import EmotionRecognition

app = Flask(__name__)

# Initialize alarm sound settings
pygame.mixer.init()
pygame.mixer.music.load('alarm.wav')

def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize variables
count = 0
alarm_triggered = False
earThresh = 0.3
earFrames = 30
shapePredictor = "shape_predictor_68_face_landmarks.dat"

# Initialize emotion recognition
er = EmotionRecognition(device='cpu')

# Initialize face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)

# Get the coordinates for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def generate_frames():
    global count, alarm_triggered
    cam = cv2.VideoCapture(0)  # Use 0 for default camera
    
    while True:
        _, frame = cam.read()
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eyeAspectRatio(leftEye)
            rightEAR = eyeAspectRatio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

            if ear < earThresh:
                count += 1
                if count >= earFrames:
                    if not alarm_triggered:
                        pygame.mixer.music.play(-1)
                        alarm_triggered = True
                    cv2.putText(frame, "DROWSINESS DETECTED", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                count = 0
                if alarm_triggered:
                    pygame.mixer.music.stop()
                    alarm_triggered = False

            frame = er.recognise_emotion(frame, return_type='BGR')

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)








