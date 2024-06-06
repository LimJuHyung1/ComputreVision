import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# import time

# 모델 및 영상 파일 경로
caffe_model_path = "ssd_models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
prototxt_path = "ssd_models/deploy.prototxt"
video_path = '../movie/겨울왕국 OST ❄️ Love Is An Open Door - Kristen Bell, Santino Fontana [영상_가사_해석_발음_한글_자막_lyrics].mp4'

# 네트워크 로드
net = cv.dnn.readNetFromCaffe(prototxt_path, caffe_model_path)

# 비디오 캡처 객체 생성
video_capture = cv.VideoCapture(video_path)


# 얼굴 검출 및 디스플레이 함수
def detect_and_display(frame, confidence_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104., 117., 123.], False, False)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            text = "{:.1f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv.putText(frame, text, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


# 두 개의 서로 다른 confidence_threshold 값을 설정
confidence_threshold_1 = 0.3
confidence_threshold_2 = 0.8

# start_time = time.time()
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # 두 개의 프레임을 복사하여 각각 다른 threshold로 검출
    frame1 = detect_and_display(frame.copy(), confidence_threshold_1)
    frame2 = detect_and_display(frame.copy(), confidence_threshold_2)

    # 두 프레임을 수평으로 결합
    combined_frame = np.hstack((frame1, frame2))

    # 결과 표시
    cv.imshow('Anime Face Detection Comparison', combined_frame)

    if cv.waitKey(1) & 0xFF == 27:  # Esc 키를 눌러 종료
        break

# end_time = time.time()
video_capture.release()
cv.destroyAllWindows()

# 전체 걸린 시간 출력
# print(f'Total time taken: {end_time - start_time:.2f} seconds')
