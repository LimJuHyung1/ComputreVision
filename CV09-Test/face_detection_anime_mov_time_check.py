import cv2
import numpy as np
import time

# 경로 설정
video_path = '../movie/Frozen.mp4'
cascade_default_path = 'cascades/haarcascades/haarcascade_frontalface_default.xml'
anime_cascade_path = 'cascades/lbpcascades/lbpcascade_animeface.xml'


def detect_and_display(frame, cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame


def process_video(video_path, cascade_path):
    video_capture = cv2.VideoCapture(video_path)
    cascade = cv2.CascadeClassifier(cascade_path)

    start_time = time.time()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = detect_and_display(frame, cascade)
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Esc 키를 눌러 종료
            break

    end_time = time.time()
    video_capture.release()
    cv2.destroyAllWindows()

    return end_time - start_time


# 첫 번째 XML 파일로 실행
time_default = process_video(video_path, cascade_default_path)
print(f"Time taken with 'haarcascade_frontalface_default.xml': {time_default:.2f} seconds")

# 두 번째 XML 파일로 실행
time_anime = process_video(video_path, anime_cascade_path)
print(f"Time taken with 'lbpcascade_animeface.xml': {time_anime:.2f} seconds")
