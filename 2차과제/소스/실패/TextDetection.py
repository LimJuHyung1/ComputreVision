# 03조 (임주형,이세비,최하은)
import cv2
import pytesseract

# Tesseract 실행 파일 경로 설정 (윈도우 환경 예시)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 비디오 파일 경로
video_path = '[FHD] 6. Frozen(겨울왕국) - For The First Time In Forever (Reprise) (영어+한글자막).mp4'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 프레임 간격 설정 (예: 5 프레임마다 처리)
frame_skip = 50
frame_count = 0

# 텍스트 인식할 특정 영역 설정 (0, 210)에서 (640, 280)까지
region_x_start, region_y_start = 0, 210
region_x_end, region_y_end = 640, 286

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        # 특정 영역만 잘라내기
        roi = frame[region_y_start:region_y_end, region_x_start:region_x_end]

        # 특정 영역을 사각형으로 시각화
        cv2.rectangle(frame, (region_x_start, region_y_start), (region_x_end, region_y_end), (0, 255, 0), 2)

        # 이미지 전처리 (회색조 변환 및 블러링)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 텍스트 인식
        text_eng = pytesseract.image_to_string(blur, lang='eng')
        print("Detected English Text:", text_eng)
        text_kor = pytesseract.image_to_string(blur, lang='kor')
        print("Detected Korean Text:", text_kor)

    cv2.rectangle(frame, (region_x_start, region_y_start), (region_x_end, region_y_end), (0, 255, 0), 2)

    # 결과 프레임 출력
    cv2.imshow('Video with Text Detection', frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
