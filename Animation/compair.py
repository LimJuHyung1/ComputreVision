import cv2
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# 얼굴 검출기 초기화
haar_cascade_path = 'cascades/haarcascades/haarcascade_frontalface_default.xml'
lbp_cascade_path = 'cascades/lbpcascades/lbpcascade_animeface.xml'

haar_face_cascade = cv2.CascadeClassifier(haar_cascade_path)
lbp_face_cascade = cv2.CascadeClassifier(lbp_cascade_path)

# Tkinter 애플리케이션 초기화
root = tk.Tk()
root.title("Face Detection Comparison")

# 비디오 파일 경로
video_path = "[FHD] 6. Frozen(겨울왕국) - For The First Time In Forever (Reprise) (영어+한글자막).mp4"

# VideoCapture 객체 생성 (비디오 파일)
cap1 = cv2.VideoCapture(video_path)
cap2 = cv2.VideoCapture(video_path)

# 라벨 생성
label1 = Label(root)
label1.pack(side=tk.LEFT)

label2 = Label(root)
label2.pack(side=tk.RIGHT)


def show_frame():
    ret1, frame1 = cap1.read()  # 첫 번째 비디오 캡처
    ret2, frame2 = cap2.read()  # 두 번째 비디오 캡처

    if not ret1 or not ret2:
        root.after(10, show_frame)  # 비디오가 끝났을 때 대기
        return

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # 흑백 이미지로 변환
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)  # 흑백 이미지로 변환

    faces1 = haar_face_cascade.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces2 = lbp_face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 얼굴에 사각형 그리기
    for (x, y, w, h) in faces1:
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for (x, y, w, h) in faces2:
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # OpenCV 이미지 포맷을 PIL 이미지 포맷으로 변환
    cv2image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGBA)
    cv2image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)

    img1 = Image.fromarray(cv2image1)
    img2 = Image.fromarray(cv2image2)

    imgtk1 = ImageTk.PhotoImage(image=img1)
    imgtk2 = ImageTk.PhotoImage(image=img2)

    # Tkinter 라벨에 이미지 업데이트
    label1.imgtk = imgtk1
    label1.configure(image=imgtk1)

    label2.imgtk = imgtk2
    label2.configure(image=imgtk2)

    label1.after(50, show_frame)  # 10밀리초마다 업데이트
    label2.after(50, show_frame)  # 10밀리초마다 업데이트


# 메인 루프
show_frame()
root.mainloop()

# 캡처 객체와 윈도우 해제
cap1.release()
cap2.release()
cv2.destroyAllWindows()
