from Modules import cv2
from Modules import cap, show_snapshot


# 첫 번째 버튼: Draw Contours
def draw_contours():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 포인트

        # Contour 그리기
        contour_image = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 3)    # 포인트

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
            M = cv2.moments(cnt)        # 포인트
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(contour_image, (cX, cY), 4, (125, 80, 255), -1)

        show_snapshot(contour_image)
