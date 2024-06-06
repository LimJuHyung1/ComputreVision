import cv2
import numpy as np

# Cascade 파일 경로 설정
anime_cascade_path = 'cascades/lbpcascades/lbpcascade_animeface.xml'
cascade_default_path = 'cascades/haarcascades/haarcascade_frontalface_default.xml'

# Cascade Classifier 로드
anime_face_cascade = cv2.CascadeClassifier(anime_cascade_path)
default_face_cascade = cv2.CascadeClassifier(cascade_default_path)

# 비디오 캡처 객체 생성
video_path = 'Images/Frozen.mp4'
video_capture = cv2.VideoCapture(video_path)

def detect_and_display(frame, cascade, color, label):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # 원본 프레임을 복사하여 각각의 Cascade를 사용하여 얼굴 검출
    frame_anime = detect_and_display(frame.copy(), anime_face_cascade, (255, 0, 0), 'Anime')
    frame_default = detect_and_display(frame.copy(), default_face_cascade, (0, 255, 0), 'Default')

    # 두 결과를 나란히 비교하여 표시
    combined_frame = np.hstack((frame_anime, frame_default))

    cv2.imshow('Face Detection Comparison', combined_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc 키를 눌러 종료
        break

video_capture.release()
cv2.destroyAllWindows()
