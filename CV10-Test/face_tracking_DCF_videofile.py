import cv2
import dlib

path = 'Images/'  # 동영상이 존재하는 경로
file = 'kagejitsu2.mp4'
full_file_name = path + file

def draw_text_info(frame, tracking_face):
    """Draw text information"""

    # 텍스트와 메뉴 정보를 그릴 위치 설정
    menu_pos_1 = (10, 20)
    menu_pos_2 = (10, 40)

    # 텍스트 그리기
    cv2.putText(frame, "Use space bar to re-initialize tracking", menu_pos_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255))
    if tracking_face:
        cv2.putText(frame, "tracking the face", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    else:
        cv2.putText(frame, "detecting a face to initialize tracking...", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255))

# 비디오 파일 캡처 객체 생성
capture = cv2.VideoCapture(full_file_name)

# dlib의 얼굴 검출기 생성
detector = dlib.get_frontal_face_detector()

# dlib의 상관 추적기 생성
tracker = dlib.correlation_tracker()

# 얼굴 추적 상태를 나타내는 변수
tracking_face = False

# 프레임 간격 설정
frame_skip = 50
frame_count = 0

while True:
    # 비디오로부터 프레임 캡처
    ret, frame = capture.read()

    # 비디오가 끝나면 루프 종료
    if not ret:
        break

    # 기본 정보 그리기
    draw_text_info(frame, tracking_face)

    # 특정 프레임 간격마다 얼굴 추적 상태를 리셋
    if frame_count % frame_skip == 0:
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
        pos = tracker.get_position()    # 추적된 객체의 위치 가져오기
        # 위치를 사각형으로 그리기
        cv2.rectangle(frame, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (0, 255, 0), 3)

    frame_count += 1

    # 키보드 이벤트 캡처
    key = 0xFF & cv2.waitKey(1)

    # 종료하려면 'ESC' 키를 누름
    if key == 0x1b:  # 0x1b=esc key
        break

    # 영상을 출력하고, 파일이라면 잠시(1/30 초) 출력 지연시간 유지
    cv2.imshow("Face tracking using dlib frontal face detector and correlation filters for tracking", frame)
    cv2.waitKey(10)

# 모든 자원 해제
capture.release()
cv2.destroyAllWindows()
