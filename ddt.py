from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame
from facial_emotion_recognition import EmotionRecognition


pygame.mixer.init()
pygame.mixer.music.load('alarm.wav')

def eyeAspectRatio(eye):
    # Vertical distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Horizontal distance
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize variables
count = 0
alarm_triggered = False
earThresh = 0.3  # Threshold for eye aspect ratio
earFrames = 30  # Number of frames for which eyes need to be below threshold
shapePredictor = "shape_predictor_68_face_landmarks (2).dat"

# Initialize emotion recognition
er = EmotionRecognition(device='cpu')

# Start video capture
cam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)

# Get the coordinates for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    # Read frame from camera
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate the eye aspect ratio (EAR) for both eyes
        leftEAR = eyeAspectRatio(leftEye)
        rightEAR = eyeAspectRatio(rightEye)

        # Average the EAR for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Draw contours around the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

        # Check if the EAR is below the threshold
        if ear < earThresh:
            count += 1

            # If the eyes have been closed for the specified number of frames
            if count >= earFrames:
                if not alarm_triggered:
                    pygame.mixer.music.play(-1)  # Play alarm in loop
                    alarm_triggered = True
                    if key == ord("p"):
                        break

                cv2.putText(frame, "DROWSINESS DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print("DROWSINESS DETECTED")

        else:
            count = 0
            if alarm_triggered:
                pygame.mixer.music.stop()  # Stop the alarm
                alarm_triggered = False
            print("DROWSINESS Not Detected")

        # Perform emotion recognition
        frame = er.recognise_emotion(frame, return_type='BGR')

    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit the loop if 'q' is pressed
    if key == ord("q"):
        break

# Release the video capture and close all windows
cam.release()
cv2.destroyAllWindows()
