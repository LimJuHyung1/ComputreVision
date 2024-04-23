import cv2
import numpy as np

path = 'Images/saenai.png'

# 이미지를 불러오고 그레이스케일로 변환합니다.
image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 이진화를 수행하여 윤곽선을 찾습니다.
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# 각 윤곽선의 길이를 계산하여 출력합니다.
for contour in contours:
    arc_length = cv2.arcLength(contour, closed=True)  # 윤곽선이 닫힌 윤곽선인지 여부를 closed 매개변수로 지정합니다.
    print("Contour arc length:", arc_length)

    # 서로 다른 색상을 생성합니다.
    color = tuple(map(int, np.random.randint(0, 255, size=3)))
    # 윤곽선을 그립니다.
    cv2.drawContours(contour_image, [contour], -1, color, 2)

    # 윤곽선의 중심점을 계산합니다.
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 면적을 이미지에 표시합니다.
        cv2.putText(contour_image, f"{arc_length:.2f}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 0, 255), 1, cv2.LINE_AA)

cv2.imshow("origin", image)
cv2.imshow("gray", gray)
cv2.imshow("arcLength", contour_image)
cv2.waitKey()

