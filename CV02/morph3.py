"""
다양한 모폴러지 연산을 한개의 함수로 지정할 수 있는 morphologyEx() 함수
    img1 = cv.morphologyEx(img, cv.MORPH_ERODE, np.ones((3, 3)))
    img2 = cv.morphologyEx(img, cv.MORPH_DILATE, np.ones((3, 3)))
    img3 = cv.morphologyEx(img, cv.MORPH_OPEN, np.ones((3, 3)))
    img4 = cv.morphologyEx(img, cv.MORPH_CLOSE, np.ones((3, 3)))

관찰 결과
OPEN
    foreground object 안에 있는 검은 배경을 유지.
    background에 있는 잡음성 foreground는 제거하는 경향을 보인다.

CLOSE
    foreground object 안에 있는 검은 잡음을 제거.
    background에 있는 foreground object는 유지하는 경향을 보인다.

"""
import cv2 as cv
import numpy as np

# 아래 그레이 영상 혹은 칼라 영상 중의 하나를 선택하여 사용하시오.
#img = cv.imread('circles_rects.png', cv.IMREAD_GRAYSCALE)
img = cv.imread('circles_rects_noise2.png', cv.IMREAD_GRAYSCALE)
#img = cv.imread('circles_rects_2.png', cv.IMREAD_GRAYSCALE)
cv.imshow("original", img)
print(f"shape={img.shape}, dtype={img.dtype}")

# 필요시 이진화 시켜 활용
ret, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
cv.imshow('input for morphology operation', img)
print(f"shape={img.shape}, dtype={img.dtype}")
key=cv.waitKey()

# 기본 함수에 대한 실험
img1 = cv.morphologyEx(img, cv.MORPH_ERODE, np.ones((3, 3)))
cv.imshow('cv.MORPH_ERODE', img1)

img2 = cv.morphologyEx(img, cv.MORPH_DILATE, np.ones((3, 3)))
cv.imshow('cv.MORPH_DILATE', img2)
cv.waitKey()

# 변형 함수에 대한 실험
img1 = cv.morphologyEx(img, cv.MORPH_OPEN, np.ones((3, 3)))
cv.imshow('cv.MORPH_OPEN', img1)

img1 = cv.morphologyEx(img, cv.MORPH_CLOSE, np.ones((3, 3)))
cv.imshow('cv.MORPH_CLOSE', img1)
cv.waitKey()


