"""
개요
    영상에 대해 Affine 변환을 행하는 과정을 보인다.
    1) src 영상에서 임의의 3점을 지정하여 이 점이 dst 영상의 맵핑되는 지점을 지정하고
    2) 이 점들이 서로 매칭하도록 하는 변환 매트릭스를 구한다. => matrix = getAffineTransform(src, dst)
    3) 구해진 변환 매트릭스를 사용하여 warpAffine() 함수를 사용하여 변환한다.
    4) 역변환 과정을 통해 원본영상을 복원해 본다.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

path = 'Images/nanami.png'

# -------------------------------------------------------------------------------------------
# 단계 1: 영상 파일을 읽고 affine 변환 매트릭스를 구하기 위한 src영상에 변환을 위한 위치에 원을 위치를 알린다.
# 변환 후에 이 원이 어느 곳에 이동했는지 보기 위함이다.
# -------------------------------------------------------------------------------------------
img = cv2.imread(path)
cv2.imshow("Original", img)
dst = img.copy()

rows, cols, ch = img.shape
c_color = (255, 0, 0)

# 점을 찍을 지점
pts1 = (83, 90)
pts2 = (447, 90)
pts3 = (83, 230)

cv2.circle(img, pts1, 8, c_color, -1)
cv2.circle(img, pts2, 8, c_color, -1)
cv2.circle(img, pts3, 8, c_color, -1)
cv2.imshow("src with marking", img)

# -------------------------------------------------------------------------------------------
# 단계 2: 아래 함수를 이용하여 affine 변환 매트릭스를 구한다.
#   변환 전 3개의 점(pts_src)과 변환 후 3개의 점(pts_dst),
#   총 2x3개의 점에 대한 변환 매트릭스(matrix)를 생성한다.
#   matrix = cv2.getAffineTransform(pts_src, pts_dst)
# -------------------------------------------------------------------------------------------

pts_src = np.float32([pts1, pts2, pts3])    # 시작점
pts_dst = np.float32([[0, 0], [447, 90], [150, 230]])     # 도착점

cv2.circle(dst, [0, 0], 8, c_color, -1)
cv2.circle(dst, [447, 90], 8, c_color, -1)
cv2.circle(dst, [150, 230], 8, c_color, -1)
cv2.imshow("dst with ", dst)

matrix = cv2.getAffineTransform(pts_src, pts_dst)
print(f'Fig.1: matrix.shape={matrix.shape}, matrix=\n', matrix)

cv2.waitKey()

# -------------------------------------------------------------------------------------------
# 단계 3: affine 함수를 이용하여 affine 변환을 행한다.
#   result = cv2.warpAffine(img, matrix, (cols, rows))
#   matrix를 이용하여 img에 대한 affine 변환을 행한다.
# -------------------------------------------------------------------------------------------

result = cv2.warpAffine(img, matrix, (cols, rows))
cv2.imshow("1) Affine transformation", result)

# -------------------------------------------------------------------------------------------
# 단계 4: 이제 affine 역변환 매트릭스를 구하여 원영상을 복구해 본다.
# 제대로 복구되는 것 같지는 않음
# 단계 3 -> 단계 4 == 원영상으로 복구 - Fig 2 에서 없어진 부분은 되돌릴 수 없음
# -------------------------------------------------------------------------------------------
iM = cv2.invertAffineTransform(matrix)
print(f'\nFig.2: inverse affine matrix iM.shape={iM.shape}, matrix=\n', iM)
output_size = tuple(map(int, (1.1 * cols, 1.1 * rows)))
result2 = cv2.warpAffine(img, iM, output_size)
cv2.imshow("2) Inverse affine transformation with iM", result2)

# -------------------------------------------------------------------------------------------
# 단계 5: affine 역변환 매트릭스를 구하지 않고
# warpAffine() 함수내에서 파라미터 지정을 통해 원영상을 복구할 수도 있다.
#
# WARP_INVERSE_MAP: 정방향 매트릭스를 역방향 변환인 것으로 가정한다.
# 정변환된 영상에 대해 변환을 행하면 원본 영상이 나온다.
# 사실상 내부에서 invertAffineTransform()을 수행해서 역변환 매트릭슬 구해서 수행한다.

# cv2.invertAffineTransform, cv2.wrapAffine 같은 결과 나오게 할 수 있음?
# -------------------------------------------------------------------------------------------
img3 = cv2.warpAffine(result, matrix, (cols, rows), flags=cv2.WARP_INVERSE_MAP)
cv2.imshow("3) Inv affine trans. without iM, same as Fig. 2", img3)

cv2.waitKey()