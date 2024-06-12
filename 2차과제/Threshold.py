from Modules import cv2
from Modules import cap, face_cascade, show_snapshot


# threshold 예시(cv2.threshold(), cv2.adaptiveThreshold(), threshold_otsu) 출력
def threshold_image(threshold_type):
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Threshold 메뉴의 버튼에 따라 다른 작업이 이루어 짐
        # 각 Threshold 과정을 인지하기를 바람
        if threshold_type == "cv2.threshold":
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        elif threshold_type == "cv2.adaptiveThreshold":
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        elif threshold_type == "threshold_otsu":
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 캡처된 스냅샷 창에 표시
        show_snapshot(frame, thresh)
