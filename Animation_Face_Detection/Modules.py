import cv2
import numpy as np
import tkinter as tk
from tkinter import Menu, Label, Toplevel, Button
from PIL import Image, ImageTk


# 얼굴 검출기 초기화
face_cascade = cv2.CascadeClassifier('cascades/haarcascades/haarcascade_frontalface_default.xml')

# Tkinter 애플리케이션 초기화
root = tk.Tk()
root.title("Face Detection in Animation")

# 비디오 파일 경로
video_path = "[FHD] 6. Frozen(겨울왕국) - For The First Time In Forever (Reprise) (영어+한글자막).mp4"

# VideoCapture 객체 생성 (비디오 파일)
cap = cv2.VideoCapture(video_path)


# 스냅샷 저장
def take_snapshot():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detectMultiScale() 함수의 인수를 설명하기 바람
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 캡처된 스냅샷 창에 표시
        show_snapshot(frame)

        filename = "snapshot.png"
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved as {filename}")


def show_snapshot(original_image, processed_image=None):
    # 캡처된 이미지를 다른 창에 표시
    snapshot_window = tk.Toplevel(root)
    snapshot_window.title("Snapshot")

    # 원본 이미지 표시
    original_label = Label(snapshot_window)
    original_label.pack()
    original_img = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    original_imgtk = ImageTk.PhotoImage(image=original_img)
    original_label.imgtk = original_imgtk
    original_label.configure(image=original_imgtk)

    # 추가적인 처리가 필요한 경우 (예: thresholding)
    if processed_image is not None:
        threshold_label = Label(snapshot_window)
        threshold_label.pack()
        threshold_img = Image.fromarray(processed_image)
        threshold_imgtk = ImageTk.PhotoImage(image=threshold_img)
        threshold_label.imgtk = threshold_imgtk
        threshold_label.configure(image=threshold_imgtk)


