"""
개요
    간단한 translation 변환에 대해 affine과 perspective 매트릭스가 어떻게 다른지 확인한다.
    영상에 대해 Affine 변환을 행하는 과정을 보인다.
    이동 변위와 dx, dy를 이용해 직접 매트릭스를 선언하는 방법을 취한다.

유의 사항
    affine 변환을 위한 변환행렬은 2x3이고,
    perspective 변환을 위한 변환행렬은 3x3이다.

"""


import numpy as np
import matplotlib.pyplot as plt
import cv2

#"""
# -------------------------------------------------------------------------------------------
# 실습 1:
# -------------------------------------------------------------------------------------------

img = cv2.imread('lenna.jpg')
rows, cols, ch = img.shape
dx = 160
dy = 40

# affine 변환을 위한 변환행렬은 2x3이고,
# perspective 변환을 위한 변환행렬은 3x3이다.

p_matrix = np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1]])
a_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
print(f'perspective matrix: p_matrix.dtype={p_matrix.dtype}')
print(f'perspective matrix: p_matrix.shape={p_matrix.shape}, p_matrix=\n', p_matrix)
print(f'\naffine matrix: a_matrix.dtype={a_matrix.dtype}')
print(f'affine matrix: a_matrix.shape={a_matrix.shape}, a_matrix=\n', a_matrix)

a_result = cv2.warpAffine(img, a_matrix, dsize=(cols, rows))
p_result = cv2.warpPerspective(img, p_matrix, (cols, rows))     # dsize = (넓이, 높이)
cv2.imshow("Original", img)
cv2.imshow("Affine transformation", a_result)
cv2.imshow("Perspective transformation", p_result)
cv2.waitKey()



iM = cv2.invertAffineTransform(a_matrix)    # inverse transform matrix
output_size = tuple(map(int, (1.1* cols , 1.1 * rows)))
result2 = cv2.warpAffine(a_result, iM, output_size)
cv2.imshow("Inverse affine transformation with iM", result2)

# WARP_INVERSE_MAP: 정방향 매트릭스를 역방향 변환인 것으로 가정한다.
# 정변환된 영상에 대해 변환을 행하면 원본 영상이 나온다.
# 사실상 내부에서 invertAffineTransform()을 수행해서 역변환 매트릭슬 구해서 수행한다.
img3 = cv2.warpAffine(a_result, a_matrix, (cols, rows), flags=cv2.WARP_INVERSE_MAP)
cv2.imshow("Inverse affine transformation without iM", img3)

cv2.waitKey()

exit(0)
#"""

