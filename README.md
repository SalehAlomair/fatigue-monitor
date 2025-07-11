# Fatigue Monitor 💤👁️
<img width="384" height="256" alt="Skitch" src="https://github.com/user-attachments/assets/205b36fa-807d-4913-824e-1a07afcb472f" />

A real-time eye-based fatigue and drowsiness detection system using OpenCV and dlib.

## 🧠 What It Does

This desktop application uses webcam video to detect signs of fatigue by analyzing the Eye Aspect Ratio (EAR). If eyes remain closed for too long, the system triggers an alarm to alert the user.

## 🚀 Features

- 🖥️ GUI interface using Tkinter
- 📷 Real-time video processing
- 👁️ Eye detection with dlib facial landmarks
- 📉 Adjustable EAR threshold & consecutive frame settings
- 🔊 Alarm sound when drowsiness is detected
- 📦 Easy to run with Python

## 🛠️ Tech Stack

- Python 3.10+
- OpenCV
- dlib
- imutils
- scipy
- Pillow (PIL)
- Tkinter
- simpleaudio

## 📋 Requirements

- All project dependencies are listed in the [`requirements.txt`](requirements.txt) file.
- The file `shape_predictor_68_face_landmarks.dat` is included in this repository and must be present in the project directory.

## 🖼️ Screenshots
<img width="1202" height="832" alt="imagee" src="https://github.com/user-attachments/assets/a347b581-d082-45c0-ae79-305b6e35e017" />

