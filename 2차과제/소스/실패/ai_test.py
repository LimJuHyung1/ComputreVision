# 03조 (임주형,이세비,최하은)
import cv2
import numpy as np
import tkinter as tk
from tkinter import Menu, Label, Toplevel, Button
from PIL import Image, ImageTk
import os

# 얼굴 검출기 초기화
face_cascade = cv2.CascadeClassifier('cascades/haarcascades/haarcascade_frontalface_default.xml')

# Tkinter 애플리케이션 초기화
root = tk.Tk()
root.title("Face Detection in Animation")

# 비디오 파일 경로
video_path = "[FHD] 6. Frozen(겨울왕국) - For The First Time In Forever (Reprise) (영어+한글자막).mp4"

# VideoCapture 객체 생성 (비디오 파일)
cap = cv2.VideoCapture(video_path)

# 상태 변수
frame_count = 0  # frame_count를 밖으로 옮겨서 매 프레임마다 초기화되지 않도록 함
is_running = True
is_running_rect = True

# 얼굴 검출기 목록
cascade_files = {
    "Default": "cascades/haarcascades/haarcascade_frontalface_default.xml",
    "Animeface": 'cascades/lbpcascades/lbpcascade_animeface.xml'
}

# 프레임 생성
video_frame = tk.Frame(root, width=800, height=600)
video_frame.pack(side=tk.TOP, padx=10, pady=10)

# 라벨 생성
label = Label(video_frame)
label.pack()


def show_frame():
    global frame_count
    output_directory = "frozen_characters/Captures/"

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

            # 얼굴 영역 추출
            face_image = frame[y:y + h, x:x + w]

            # 이미지 저장
            save_path = os.path.join(output_directory, f"frame_{frame_count}_face.jpg")
            cv2.imwrite(save_path, face_image)

            frame_count += 1

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

def take_snapshot():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 캡처된 스냅샷 창에 표시
        show_snapshot(frame)

        filename = "snapshot.png"
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved as {filename}")

def threshold_image(threshold_type):
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if threshold_type == "cv2.threshold":
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        elif threshold_type == "cv2.adaptiveThreshold":
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        elif threshold_type == "threshold_otsu":
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 캡처된 스냅샷 창에 표시
        show_snapshot(frame, thresh)

def show_snapshot(original_image, processed_image=None):
    # 캡처된 이미지를 다른 창에 표시
    snapshot_window = Toplevel(root)
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

# ---------------------------------------------------------

# Geotrans 함수들
def translation():
    ret, frame = cap.read()
    if ret:
        rows, cols, _ = frame.shape
        M = np.float32([[1, 0, 50], [0, 1, 50]])
        translated = cv2.warpAffine(frame, M, (cols, rows))
        show_snapshot(translated)

def rotation():
    ret, frame = cap.read()
    if ret:
        rows, cols, _ = frame.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
        rotated = cv2.warpAffine(frame, M, (cols, rows))
        show_snapshot(rotated)

def affine_transformation():
    ret, frame = cap.read()
    if ret:
        rows, cols, _ = frame.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [230, 80], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)
        transformed = cv2.warpAffine(frame, M, (cols, rows))
        show_snapshot(transformed)

def perspective_transformation():
    ret, frame = cap.read()
    if ret:
        rows, cols, _ = frame.shape
        pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
        pts2 = np.float32([[0, 0], [250, 0], [0, 300], [250, 300]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        transformed = cv2.warpPerspective(frame, M, (cols, rows))
        show_snapshot(transformed)

# ---------------------------------------------------------

# 첫 번째 버튼: Draw Contours
def draw_contours():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Contour 그리기
        contour_image = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 3)

        show_snapshot(contour_image)

# 두 번째 버튼: Contours Centroid
def moments():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Contour 그리기
        contour_image = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 3)

        # 각 Contour의 중심점 계산 및 표시
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(contour_image, (cX, cY), 4, (125, 80, 255), -1)

        show_snapshot(contour_image)

# ---------------------------------------------------------

# K-Means Color Quantization 함수
def color_quantization(image, k):
    num_maxCount = 100  # 최대 반복 횟수
    eps = 0.2  # 수렴 기준
    num_attempts = 10  # 중심 포인트 선택 시도 횟수

    # 이미지를 2D 배열로 변환하고 모든 픽셀을 3차원 벡터로 변환한다.
    data = np.float32(image).reshape((-1, 3))

    # (최대 반복 횟수, 수렴 기준)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, num_maxCount, eps)

    _, label, center = cv2.kmeans(data, k, None, criteria, num_attempts, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    result = center[label.flatten()]

    # 영상과 같은 shape로 만든다.
    result = result.reshape(image.shape)

    return result

# K-Means Color Quantization 버튼 생성 함수
def create_kmeans_button(k):
    def apply_kmeans():
        ret, frame = cap.read()
        if ret:
            quantized_image = color_quantization(frame, k)
            show_snapshot(quantized_image)
    return apply_kmeans

# ---------------------------------------------------------

def change_cascade(cascade):
    global face_cascade, selected_cascade
    selected_cascade = cascade
    face_cascade = cv2.CascadeClassifier(cascade_files[cascade])

# ---------------------------------------------------------

# 메뉴 생성
menu = Menu(root)
root.config(menu=menu)

file_menu = Menu(menu)
menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Exit", command=root.quit)

# 시작 및 중지 버튼 추가
video_control_menu = Menu(menu)
menu.add_cascade(label="Video Control", menu=video_control_menu)
video_control_menu.add_command(label="Start", command=start_video)
video_control_menu.add_command(label="Stop", command=stop_video)
video_control_menu.add_command(label="Visualization Rect", command=turn_on_detection_rect)
video_control_menu.add_command(label="Unvisualization Rect", command=turn_off_detection_rect)

# 얼굴 검출기 선택 메뉴 추가
cascade_menu = Menu(menu)
menu.add_cascade(label="Select Cascade", menu=cascade_menu)

# 스냅샷 메뉴 추가
snapshot_menu = Menu(menu)
menu.add_cascade(label="Threshold", menu=snapshot_menu)
snapshot_menu.add_command(label="Take Snapshot", command=take_snapshot)
snapshot_menu.add_command(label="cv2.threshold", command=lambda: threshold_image("cv2.threshold"))
snapshot_menu.add_command(label="cv2.adaptiveThreshold", command=lambda: threshold_image("cv2.adaptiveThreshold"))
snapshot_menu.add_command(label="threshold_otsu", command=lambda: threshold_image("threshold_otsu"))

for cascade in cascade_files:
    cascade_menu.add_command(label=cascade, command=lambda c=cascade: change_cascade(c))

# Geotrans 메뉴 추가
geotrans_menu = Menu(menu)
menu.add_cascade(label="Geotrans", menu=geotrans_menu)
geotrans_menu.add_command(label="Translation", command=translation)
geotrans_menu.add_command(label="Rotation", command=rotation)
geotrans_menu.add_command(label="Affine Transformation", command=affine_transformation)
geotrans_menu.add_command(label="Perspective Transformation", command=perspective_transformation)

# Contour 메뉴 추가
contour_menu = Menu(menu)
menu.add_cascade(label="Contour", menu=contour_menu)
contour_menu.add_command(label="Draw Contours", command=draw_contours)
contour_menu.add_command(label="Contours Centroid", command=moments)

# K-Means 메뉴 추가
kmeans_menu = Menu(menu)
menu.add_cascade(label="K-Means", menu=kmeans_menu)
# K값에 따른 K-Means Color Quantization 버튼 추가
k_values = [2, 4, 8, 16, 32]
for k in k_values:
    kmeans_menu.add_command(label=f"Apply K-Means (k={k})", command=create_kmeans_button(k))

# 초기화 함수 호출
show_frame()
root.mainloop()
