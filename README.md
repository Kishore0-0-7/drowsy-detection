# Drowsiness Detection System 🚗

A real-time drowsiness detection system that monitors driver alertness using computer vision and machine learning techniques.

## Features 🌟

- Real-time face detection and eye tracking
- Drowsiness detection using Eye Aspect Ratio (EAR)
- Facial emotion recognition
- Customizable alert thresholds
- Audio alerts for drowsiness detection
- Interactive web interface using Streamlit
- Real-time statistics display

## Requirements 📋

- Python 3.7+
- OpenCV
- dlib
- streamlit
- pygame
- numpy
- imutils
- scipy
- facial_emotion_recognition

## Installation 🔧

1. Clone the repository:

```bash
git clone https://github.com/kishore0-0-7/drowsy-detection.git
cd drowsy-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download required models:

- Download `shape_predictor_68_face_landmarks.dat` from dlib
- Ensure `alarm.wav` is in the root directory
- Download emotion recognition models (`fer.json` and `fer.h5`)

## Usage 🚀

1. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

2. Adjust settings in the sidebar:

- EAR Threshold: Controls sensitivity of eye closure detection
- Number of Frames: Consecutive frames for drowsiness detection
- Time Threshold: Duration before triggering alert

3. Click "Start Detection" to begin monitoring
4. Press "Stop" to end the session

## How It Works 🔍

### Drowsiness Detection

- Uses facial landmarks to detect eye positions
- Calculates Eye Aspect Ratio (EAR)
- Triggers alert when EAR falls below threshold for specified duration

### Emotion Recognition

- Detects facial expressions in real-time
- Classifies emotions into categories:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral

## Components 🛠️

1. **Main Application (`streamlit_app.py`)**

   - Web interface
   - Real-time video processing
   - Alert system
   - Statistics display

2. **Emotion Detection (`emotion.py`)**
   - Facial emotion recognition
   - Expression classification
   - Real-time emotion tracking

## Safety Note ⚠️

This system is intended as an auxiliary tool and should not be relied upon as the sole means of preventing drowsy driving. Always ensure proper rest before operating vehicles.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.
