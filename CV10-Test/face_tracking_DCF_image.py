import cv2
import dlib

path = 'Images/'  # 이미지 파일이 존재하는 경로
file = 'alpha.jpg'  # 이미지 파일 이름
full_file_name = path + file

def draw_text_info(frame, tracking_face):
    """텍스트 정보를 그리는 함수"""
    # 텍스트와 메뉴 정보를 그릴 위치 설정
    menu_pos_1 = (10, 20)
    menu_pos_2 = (10, 40)

    # 텍스트 그리기
    cv2.putText(frame, "Press space bar to re-initialize tracking", menu_pos_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255))
    if tracking_face:
        cv2.putText(frame, "Tracking the face", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    else:
        cv2.putText(frame, "Detecting a face to initialize tracking...", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255))

# 이미지 파일 불러오기
frame = cv2.imread(full_file_name)

# dlib의 얼굴 검출기 생성
detector = dlib.get_frontal_face_detector()

# dlib의 상관 추적기 생성
tracker = dlib.correlation_tracker()

# 얼굴 추적 상태를 나타내는 변수
tracking_face = False

# 그레이스케일로 변환하여 얼굴 검출
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Try to detect a face to initialize the tracker:
rects = detector(gray, 0)
# 얼굴을 검출한 경우 첫 번째 얼굴을 추적 대상으로 설정
if len(rects) > 0:
    # Start tracking:
    tracker.start_track(frame, rects[0])
    tracking_face = True

if tracking_face is True:  # 얼굴 추적 중이라면...
    # 추적기를 업데이트하고 신뢰 점수를 출력한 후 위치를 반환받아 해당 위치에 사각형을 그림
    score = tracker.update(frame)
    # print(f"{score:#.1f}")
    # 추적된 객체의 위치 가져오기
    pos = tracker.get_position()
    # 위치를 사각형으로 그리기
    cv2.rectangle(frame, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (0, 255, 0), 3)

# 기본 정보 그리기
draw_text_info(frame, tracking_face)

# 이미지 표시
combined_frame = cv2.hconcat([cv2.imread(full_file_name), frame])
cv2.imshow("Original Image | Face tracking using dlib frontal face detector and correlation filters for tracking", combined_frame)
cv2.waitKey(0)

# 모든 자원 해제
cv2.destroyAllWindows()
