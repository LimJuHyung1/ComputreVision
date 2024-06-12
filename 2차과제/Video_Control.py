from Modules import Label, Image, ImageTk, np, tk, cv2  # 라이브러리
from Modules import root, cap, face_cascade


# 모델 파일 경로
caffe_model_path = "ssd_models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
prototxt_path = "ssd_models/deploy.prototxt"

# 네트워크 로드
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffe_model_path)

# 상태 변수
is_running = True               # 영상이 재생 중인가
is_running_rect = True          # face detection 사각형 출력 여부
is_showed_confidence = False    # DNN을 통한 신뢰도 출력 여부
confidence_threshold = 0.3      # 이 이상의 신뢰도일 경우만 출력

# 프레임 생성
video_frame = tk.Frame(root, width=800, height=600)
video_frame.pack(side=tk.TOP, padx=10, pady=10)

# 라벨 생성
label = Label(video_frame)
label.pack()

# 화면 출력
def show_frame():
    if not is_running:
        return

    ret, frame = cap.read()  # 비디오 캡처
    if not ret:
        root.after(10, show_frame)  # 비디오가 끝났을 때 대기
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 흑백 이미지로 변환
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 얼굴에 사각형 그리기
    if is_running_rect:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (25, 0, 255), 2)

    if is_showed_confidence:
        frame = detect_and_display(frame.copy(), confidence_threshold)

    # OpenCV 이미지 포맷을 PIL 이미지 포맷으로 변환
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)

    # Tkinter 라벨에 이미지 업데이트
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, show_frame)  # 10밀리초마다 업데이트


def start_video():
    global is_running
    if not is_running:
        is_running = True
        show_frame()


def stop_video():
    global is_running
    is_running = False


def turn_on_detection_rect():
    global is_running_rect
    is_running_rect = True


def turn_off_detection_rect():
    global is_running_rect
    is_running_rect = False


def turn_on_confidence():
    global is_showed_confidence
    is_showed_confidence = True


def turn_off_confidence():
    global is_showed_confidence
    is_showed_confidence = False


# DNN을 통한 신뢰도 출력
def detect_and_display(frame, confidence_threshold=0.7):
    h, w = frame.shape[:2]  # 프레임(영상)의 높이와 너비를 가져옵니다.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104., 117., 123.], False, False)
    net.setInput(blob)
    detections = net.forward()

    # detections.shape[2] - 탐지된 객체의 수
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]     # 현재 객체의 신뢰도
        if confidence > confidence_threshold:
            # 객체 사각형 좌표 출력
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

            # 신뢰도 표시
            text = "{:.1f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame
