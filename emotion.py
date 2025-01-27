import cv2
import numpy as np
from facial_emotion_recognition import EmotionRecognition

# Initialize emotion recognition
er = EmotionRecognition(device='cpu')

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Process frame with emotion recognition
    frame = er.recognise_emotion(frame, return_type='BGR')

    # Display the frame
    resized_img = cv2.resize(frame, (1000, 700))
    cv2.imshow('Facial emotion analysis', resized_img)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()