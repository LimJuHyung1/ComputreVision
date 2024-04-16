"""
개요
    영상에 대해 투영 변환을 행한 과정을 보인다.
    1) 임의의 4각형 꼭지점 4개를 src 영상에 지정한다.
    2) dst 영상에 대해 src 영상의 꼭지점에 매칭할 4개의 위치를 지정한다.
    3) getPerspectiveTransform() 함수로 이를 변환한 매트릭스(3x3)를 생성한다.
    4) 위 변환 매트릭스를 이용하여 warpPerspective() 함수를 사용하여 변환한다.


점검 포인트
     1) src 영상에 대해 선정한 4점의 위치가 dst 영상에 지정한 위치에 맵핑되었는지를 살핀다.
     2) Perspective matrix에 대한 inverse를 구하는 함수는 없다.
        dst의 위치와 src의 위치를 바꾸어 변환하면 그 자체가 invere transform이 된다.


    dst = cv.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])
        src: input image.
        dst: output image that has the size dsize and the same type as src .
        M: 3×3 transformation matrix.
        dsize: size of the output image.
        flags: combination of interpolation methods (see InterpolationFlags) and the optional flag WARP_INVERSE_MAP that means that M is the inverse transformation ( dst→src ).
        borderMode: pixel extrapolation method (see BorderTypes); when borderMode=BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to the "outliers" in the source image are not modified by the function.
        borderValue: value used in case of a constant border; by default, it is 0.


"""


import numpy as np
import matplotlib.pyplot as plt
import cv2

# ------------------------------------------------------------------------------------
# Fig 1: 원본 영상을 보인다.
# ------------------------------------------------------------------------------------
img = cv2.imread('lenna.jpg')
rows, cols, ch = img.shape
cv2.imshow("1) Original", img)
dst = img.copy()

# ------------------------------------------------------------------------------------
# Fig 2: src 영상의 위치를 지정하고 이를 영상에 원으로 표기한다.
# cv.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
# ------------------------------------------------------------------------------------
# 평행성이 없어서 4지점을 다 지정해야 한다.
# pts4_src = [(83, 90), (447, 90), (83, 472), (500, 472)]    # src (x, y)
pts4_src = [(0, 0), (cols-1, 0), (0, rows-1), (cols-1, rows-1)]  # src, (x, y), 원본 전 영역

radius = 8
color_out = (255, 0, 0)     # BGR
color_in = (0, 255, 255)

# src의 4개 지점을 번호와 함께 마킹한다.
for i, center in enumerate(pts4_src):
    cv2.circle(img, center, radius, color_out, -1)      # thickness =-1 => FILLED
    cv2.circle(img, center, int(radius/2), color_in, -1)
    cv2.putText(img, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
cv2.imshow("2) src with marking", img)


# ------------------------------------------------------------------------------------
# Fig 3: src 영상의 위치를 지정하고 이를 영상에 원으로 표기한다.
# cv.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
# ------------------------------------------------------------------------------------

pts4_dst = [[0, 0], [447, 90], [150, 472], [420, 320]]     # dst

# dst의 4개 지점을 마킹한다.
color_out = (0, 0, 255)     # BGR
color_in = (255, 255, 0)
for i, center in enumerate(pts4_dst):
    cv2.circle(dst, center, radius, color_out, -1)      # thickness =-1 => FILLED
    cv2.circle(dst, center, int(radius/2), color_in, -1)
    cv2.putText(dst, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
cv2.imshow("3) dst with marking", dst)
cv2.waitKey()

# ------------------------------------------------------------------------------------
# Fig 4: 투영 변환된 영상을 보인다.
# ------------------------------------------------------------------------------------

# 1쌍의 좌표 정보를 바탕으로 getPerspectiveTransform() 함수를 이용하여 투영 변환 매트릭스를 구한다.
pts1 = np.float32(pts4_src)     # src
pts2 = np.float32(pts4_dst)      # dst
matrix = cv2.getPerspectiveTransform(pts1, pts2)
print(f'matrix.shape={matrix.shape} matrix.shape=\n', matrix)

# 취득한 매트릭스로 투영 변환을 행하고 결과를 출력한다.
result = cv2.warpPerspective(img, matrix, (cols, rows))
print('\nsize of input image=', img.shape)
print('size of result image=', result.shape)
cv2.imshow("4) Perspective transformation", result)

# ------------------------------------------------------------------------------------
# Fig 5: 역투영 매트릭스를 구해 원상 복구된 영상을 보인다.
# ------------------------------------------------------------------------------------

# 역 투영변환을 행한다. 위의 투영변환 결과에 대해 역 투영변환을 행한다. => 원본 영상이 출력되어야 한다.
matrix_inv = cv2.getPerspectiveTransform(pts2, pts1)
result = cv2.warpPerspective(result, matrix_inv, (cols, rows))
cv2.imshow("5) Inverse perspective transformation", result)


cv2.waitKey()


