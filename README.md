# ObjectDetection with AI voice assistance-
import streamlit as st
import torch
import cv2
import subprocess
import os
import time
import numpy as np

# === eSpeak Path ===
ESPEAK_PATH = r"C:\Program Files (x86)\eSpeak\command_line\espeak.exe"
if not os.path.isfile(ESPEAK_PATH):
    raise FileNotFoundError(f"eSpeak not found at: {ESPEAK_PATH}")

# === Load Model (YOLOv5x for higher accuracy) ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('yolov5', 'yolov5x', source='local')  # use yolov5x for better accuracy
model.conf = 0.6
model.iou = 0.45
model.to(device)

# === Voice + Cooldown ===
last_spoken = {}
SPEECH_COOLDOWN = 5  # seconds

def speak(text):
    try:
        subprocess.Popen([ESPEAK_PATH, text])
    except Exception as e:
        st.error(f"[❌] espeak error: {e}")

# === Streamlit Layout ===
st.set_page_config(page_title="Object Detection with Voice", layout="centered")
st.markdown("<h1 style='text-align:center;'> Object Detection with Voice</h1>", unsafe_allow_html=True)
st.markdown("Detects objects via webcam and speaks the most prominent object.")
st.markdown("---")

col1, col2 = st.columns(2)
start = col1.button(" Start Webcam")
stop = col2.button(" Stop Webcam")

FRAME_WINDOW = st.empty()
status_text = st.empty()


if start:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error(" Cannot access webcam.")
    else:
        st.success(" Webcam started. Detecting objects...")
        prev_time = time.time()
    while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(img_rgb, size=640)
            df = results.pandas().xyxy[0]
            df = df[df['confidence'] > model.conf]
            max_area = 0
            dominant_obj = None
            dominant_box = None
            for _, row in df.iterrows():
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    dominant_obj = row['name']
                    dominant_box = (x1, y1, x2, y2, f"{row['name']} {row['confidence']:.2f}")
            current_time = time.time()
            if dominant_obj and dominant_box:
                last_time = last_spoken.get(dominant_obj, 0)
                if current_time - last_time > SPEECH_COOLDOWN:
                    speak(f"{dominant_obj} detected")
                    last_spoken[dominant_obj] = current_time
                    status_text.success(f"🗣️ Speaking: {dominant_obj}")
                x1, y1, x2, y2, label = dominant_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        status_text.info(" Webcam stopped.")
