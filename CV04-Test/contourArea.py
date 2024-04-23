
import cv2
import numpy as np

# 이미지를 불러오고 그레이스케일로 변환합니다.
image = cv2.imread('Images/love_live.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 이진화를 수행하여 윤곽선을 찾습니다.
_, thresh = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 윤곽선을 그릴 빈 이미지를 생성합니다.
contour_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# 윤곽선을 그립니다.
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# 모든 윤곽선에 대해 면적을 계산합니다.
for contour in contours:
    area = cv2.contourArea(contour)
    print("Contour area:", area)

    # 윤곽선의 중심점을 계산합니다.
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 면적을 이미지에 표시합니다.
        cv2.putText(contour_image, f"{area}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 150, 40), 1, cv2.LINE_AA)



cv2.imshow("original image", image)
cv2.imshow("gray image", gray)
cv2.imshow("contours image", contour_image)

cv2.waitKey()
