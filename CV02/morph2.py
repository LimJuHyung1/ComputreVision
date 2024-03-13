"""

1. 개요
    입력 영상을 이진화하여 키입력에 따라 침식과 팽창을 무한 반복한다.
    e 혹은 E: Erode
    d 혹은 d: Dilate
    esc: 프로그램 종료

2. 주요 관찰 포인트
    e를 반복하면 고립된 잡음이 점점 사라지지만 대신 전경객체가 줄어든다.
    d는 사라진 전경 객체를 복원시키지만 그 한계가 있다.
    작은 커널로 모폴로지 연산을 2회하는 것은 큰 커널로 1회하는 것과 효과가 유사하다.

"""


import cv2 as cv
import numpy as np

# 아래 그레이 영상 혹은 칼라 영상 중의 하나를 선택하여 사용하시오.
#img = cv.imread('circles_rects.png', cv.IMREAD_GRAYSCALE)
img = cv.imread('circles_rects_noise.png', cv.IMREAD_GRAYSCALE)
#img = cv.imread('circles_rects_2.png', cv.IMREAD_GRAYSCALE)
cv.imshow("original", img)
print(f"shape={img.shape}, dtype={img.dtype}")

# 필요시 이진화 시켜 활용 ==> 주의!! 이진화를 했지만 영상 구조체는 bool type이 아니라, uint8이다.
ret, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
cv.imshow('input for morphology operation', img)
print(f"shape={img.shape}, dtype={img.dtype}")  # shape=(219, 544), dtype=uint8
#key=cv.waitKey()

# 여러 종류의 StructuringElement중 하나를 정의하여 사용한다.
s= cv.getStructuringElement(cv.MORPH_RECT, (3,3))
#s= cv.getStructuringElement(cv.MORPH_CROSS, (7,7))
#s= cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))

img1 = img.copy()
print("\nPress e/E for erosion or d/D for Dilation or esc key to quit!")
key = 0
while key != 0x1b:      # esc key code
    if key == ord('e') or key == ord('E'):     # ord(문자) : 문자의 ASCII 코드를 반환한다. 반대: chr(0x41) => 'A"
        img1 = cv.erode(img1, s)
    elif key == ord('d') or key == ord('D'):       # 0x64는 d의 ascii. 파이썬에는 &&, || 논리연산자 지원 안됨.
        img1 = cv.dilate(img1, s)
    cv.imshow('Press e/E, d/D or esc key.', img1)
    key = cv.waitKey()
    print(f"Pressed key: ASCII={key:x}, chr='{chr(key)}, Press esc to quit'")

