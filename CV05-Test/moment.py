import cv2
import numpy as np

path = 'Images/lastorder.jpg'

# 이미지를 불러오고 그레이스케일로 변환합니다.
image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 이진화를 수행하여 윤곽선을 찾습니다.
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# 각 윤곽선에 대한 모멘트를 계산합니다.
for contour in contours:
    # 윤곽선의 모멘트를 계산합니다.
    M = cv2.moments(contour)

    # 윤곽선의 중심을 계산합니다.
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

    # 서로 다른 색상을 생성합니다.
    color = tuple(map(int, np.random.randint(0, 255, size=3)))
    cv2.drawContours(contour_image, [contour], -1, color, 2)

    # 중심점을 이미지에 표시합니다.
    cv2.circle(contour_image, (cx, cy), 5, (0, 0, 255), -1)  # 빨간색으로 중심점을 표시합니다.

# 결과 이미지를 출력합니다.
cv2.imshow('origin', image)
cv2.imshow('gray', gray)
cv2.imshow('Image with Centers', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
