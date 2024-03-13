"""

1. 개요
    단순한 erode(), dilate() 함수의 용법을 적용하여 결과를 관찰한다.



erode(), dilate() 함수의 용법
    erode(), dilate() 함수는 iterations 회수를 지정할 수 있다.
    iterations=1일 때는 생략할 수 있다.
    사례: img1 = cv.erode(img, s, iterations=3)
    사례: img1 = cv.dilate(img, s, iterations=3)
    사례: cv.erode(img, s, iterations=1) = cv.erode(img, s, iterations=3)

"""

import cv2 as cv
import numpy as np

# step1: 아래 그레이 영상 혹은 칼라 영상 중의 하나를 선택하여 사용하시오.
#img = cv.imread('circles_rects.png', cv.IMREAD_GRAYSCALE)
img = cv.imread('circles_rects_noise.png', cv.IMREAD_GRAYSCALE)
#img = cv.imread('circles_rects_2.png', cv.IMREAD_GRAYSCALE)
cv.imshow("step1: original", img)
print(f"shape={img.shape}, dtype={img.dtype}")

# step2: 필요시 이진화 시켜 활용
ret, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
cv.imshow('step2: input for morphology operation', img)
print(f"shape={img.shape}, dtype={img.dtype}")
key=cv.waitKey()

# 여러 종류의 StructuringElement중 하나를 정의하여 사용한다.
s3= cv.getStructuringElement(cv.MORPH_RECT, (3,3))
s5= cv.getStructuringElement(cv.MORPH_RECT, (5,5))
#s= cv.getStructuringElement(cv.MORPH_CROSS, (7,7))
#s= cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))

# step3: iterations=1일 때의 침식과 팽창 결과를 한 화면에 보인다.
img1 = cv.erode(img, s3)     # iterations=1
img2 = cv.dilate(img, s3)     # iterations=1
cv.imshow('step3: Erosion(3x3, iterations=1)|Dilation(3x3, iterations=1)', cv.hconcat((img1, img2)))
cv.waitKey()

# step4: 아래 2개의 처리는 결과가 유사하다.
# 같을 수도 있으나 같으란 보장은 없다.
img1 = cv.erode(img, s3, iterations=3)
img2 = cv.erode(img, s5, iterations=1)
cv.imshow('step4 Erosion: (3x3, iterations=3) | (5x5, iterations=1))', cv.hconcat((img1, img2)))
cv.waitKey()

# step5: dialtion에 대해서도 결과가 유사한지 살펴본다.
# 같을 수도 있으나 같으란 보장은 없다.
img1 = cv.dilate(img, s3, iterations=3)
img2 = cv.dilate(img, s5, iterations=1)
cv.imshow('step5 Dilation: (3x3, iterations=3) | (5x5, iterations=1)', cv.hconcat((img1, img2)))
cv.waitKey()
