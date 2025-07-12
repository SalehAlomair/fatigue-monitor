# Fatigue Monitor 💤👁️

<img width="1202" height="832" alt="image1" src="https://github.com/user-attachments/assets/4bfe27f3-35ee-4637-a644-7914f513a479" />

<img width="502" height="632" alt="image2" src="https://github.com/user-attachments/assets/8409b1ce-8b0c-48b0-a64b-e08f6d89f046" />

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
The file `shape_predictor_68_face_landmarks.dat` is required for this project.  
You can download it from [here](https://www.kaggle.com/datasets/sajikim/shape-predictor-68-face-landmarks?resource=download).  
After downloading, place the file in the project directory.

## 🖼️ Explanation

<div style="display: flex; justify-content: center;">
  <img width="768" height="512" alt="Skitch" src="https://github.com/user-attachments/assets/f615d6b6-990e-4485-b86b-48915cdf3d1e" />
</div>
