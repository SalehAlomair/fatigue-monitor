# Fatigue Monitor 💤👁️

<img width="1080" height="524" alt="Skitch" src="https://github.com/user-attachments/assets/969b83da-d3d4-4233-907d-bd49effba7a4" style="display: block; margin: 0 auto;" />
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
- The file `shape_predictor_68_face_landmarks.dat` is required for this project.  
You can download it from here:  
https://www.kaggle.com/datasets/sajikim/shape-predictor-68-face-landmarks?resource=download  
After downloading, place the file in the project directory.


## 🖼️ Screenshots
<img width="1202" height="832" alt="imagee" src="https://github.com/user-attachments/assets/a347b581-d082-45c0-ae79-305b6e35e017" />

