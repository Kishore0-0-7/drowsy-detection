import streamlit as st
import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import pygame
from facial_emotion_recognition import EmotionRecognition
import time

# Page configuration
st.set_page_config(
    page_title="Drowsiness Detection System",
    page_icon="🚗",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stAlert > div {
        padding: 0.5rem 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Settings")
earThresh = st.sidebar.slider("EAR Threshold", 0.1, 0.5, 0.3, 0.01)
earFrames = st.sidebar.slider("Number of Frames", 10, 50, 30, 1)
timeThresh = st.sidebar.slider("Time Threshold (seconds)", 1, 10, 3, 1)  # New time threshold slider

def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def main():
    st.title("Drowsiness Detection System 🚗🏎️")
    
    # Initialize session state
    if 'alarm_triggered' not in st.session_state:
        st.session_state.alarm_triggered = False
    if 'count' not in st.session_state:
        st.session_state.count = 0
    if 'drowsy_start_time' not in st.session_state:  # New session state for time tracking
        st.session_state.drowsy_start_time = None

    # Initialize components
    pygame.mixer.init()
    pygame.mixer.music.load('alarm.wav')
    
    # Initialize face detection components
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    er = EmotionRecognition(device='cpu')
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # Create placeholder for video feed and status
    col1, col2 = st.columns([2, 1])
    with col1:
        video_placeholder = st.empty()
    with col2:
        status_placeholder = st.empty()
        stats_container = st.empty()
        
    # Start button
    start = st.button("Start Detection")
    stop = st.button("Stop")

    if start:
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break

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

                # Draw eye contours
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # Drowsiness detection logic
                if ear < earThresh:
                    if st.session_state.drowsy_start_time is None:
                        st.session_state.drowsy_start_time = time.time()
                    
                    drowsy_time = time.time() - st.session_state.drowsy_start_time
                    st.session_state.count += 1
                    
                    if st.session_state.count >= earFrames and drowsy_time >= timeThresh:
                        if not st.session_state.alarm_triggered:
                            pygame.mixer.music.play(-1)
                            st.session_state.alarm_triggered = True
                        
                        cv2.putText(frame, f"DROWSINESS ALERT! Time: {drowsy_time:.1f}s", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        status_placeholder.error(f"⚠️ DROWSINESS DETECTED! Duration: {drowsy_time:.1f}s")
                else:
                    st.session_state.count = 0
                    st.session_state.drowsy_start_time = None
                    if st.session_state.alarm_triggered:
                        pygame.mixer.music.stop()
                        st.session_state.alarm_triggered = False
                    status_placeholder.success("😊 Alert")

                # Display stats
                elapsed_time = time.time() - st.session_state.drowsy_start_time if st.session_state.drowsy_start_time else 0.0
                stats_container.markdown(f"""
                    ### Statistics
                    - Eye Aspect Ratio: {ear:.2f}
                    - Frames Count: {st.session_state.count}
                    - Threshold: {earThresh}
                    - Time Elapsed: {elapsed_time:.1f}s
                """)

            # Perform emotion recognition
            frame = er.recognise_emotion(frame, return_type='BGR')
            
            # Display the frame
            video_placeholder.image(frame, channels="BGR", use_column_width=True)

        cap.release()
        pygame.mixer.music.stop()
        
    if stop:
        st.session_state.alarm_triggered = False
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

if __name__ == "__main__":
    main()
