import cv2
import dlib
import numpy as np

# 이미지 파일 경로
file = "Images/kazuma.png"
file = "Images/kagejitsu.mp4"

full_file_name = file

# 랜드마크 검출 모델 파일 경로
p = "../dlib_shape_predictors/shape_predictor_68_face_landmarks.dat"
# p = "../dlib_shape_predictors/shape_predictor_5_face_landmarks.dat"

# 랜드마크 포인트 정의
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_BRIDGE_POINTS = list(range(27, 31))
LOWER_NOSE_POINTS = list(range(31, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))
ALL_POINTS = list(range(0, 68))

def draw_shape_lines_all(np_shape, image):
    """얼굴 랜드마크를 선으로 연결하여 그리기"""
    draw_shape_lines_range(np_shape, image, JAWLINE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, LEFT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, NOSE_BRIDGE_POINTS)
    draw_shape_lines_range(np_shape, image, LOWER_NOSE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, LEFT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_OUTLINE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_INNER_POINTS, True)

def draw_shape_lines_range(np_shape, image, range_points, is_closed=False):
    """특정 범위의 랜드마크를 선으로 연결하여 그리기"""
    np_shape_display = np_shape[range_points]
    points = np.array(np_shape_display, dtype=np.int32)
    cv2.polylines(image, [points], is_closed, (255, 255, 0), thickness=1, lineType=cv2.LINE_8)

def draw_shape_points_pos_range(np_shape, image, points):
    """특정 범위의 랜드마크 포인트와 그 위치를 그리기"""
    np_shape_display = np_shape[points]
    draw_shape_points_pos(np_shape_display, image)

def draw_shape_points_pos(np_shape, image):
    """랜덤마크 포인트와 그 위치를 그리기"""
    for idx, (x, y) in enumerate(np_shape):
        # 랜드마크 번호 표시
        cv2.putText(image, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
        # 랜드마크 포인트 그리기
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

def draw_shape_points_range(np_shape, image, points):
    """특정 범위의 랜드마크 포인트 그리기"""
    np_shape_display = np_shape[points]
    draw_shape_points(np_shape_display, image)

def draw_shape_points(np_shape, image):
    """랜드마크 포인트 그리기"""
    for (x, y) in np_shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

def shape_to_np(dlib_shape, dtype="int"):
    """dlib의 shape 객체를 numpy 배열로 변환"""
    coordinates = np.zeros((dlib_shape.num_parts, 2), dtype=dtype)
    for i in range(0, dlib_shape.num_parts):
        coordinates[i] = (dlib_shape.part(i).x, dlib_shape.part(i).y)
    return coordinates

# 프로그램 시작
# 얼굴 검출기와 랜드마크 검출기 초기화
detector = dlib.get_frontal_face_detector()     # 얼굴 검출 객체 생성
predictor = dlib.shape_predictor(p)             # 랜드마크 검출 객체 생성. p는 맨 위에 정의함.

# VideoCapture 객체 생성
#video_capture = cv2.VideoCapture(0)            # USB CAM 카메라
video_capture = cv2.VideoCapture(full_file_name)    # 비디오 파일

while True:
    # 프레임 캡처
    ret, frame = video_capture.read()

    # 비디오 스트림이 끝났을 때 종료
    if not ret:
        break

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    rects = detector(gray, 0)

    # 얼굴이 검출되었는지 확인
    if len(rects) > 0:
        # 모든 검출된 얼굴에 대해 랜드마크 검출
        for rect in rects:
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)

            # 랜드마크 그리기
            draw_shape_points(shape, frame)

    # 영상 출력
    cv2.imshow("Landmarks detection using dlib", frame)

    # ESC 키로 종료
    key_in = cv2.waitKey(1) & 0xff
    if key_in == 27:
        break

# 비디오 캡처 해제 및 윈도우 닫기
video_capture.release()
cv2.destroyAllWindows()
