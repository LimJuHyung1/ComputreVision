"""
모멘트 - 객체(윤곽선, 영상, 점들의 집합)의 특징 기술자

    계조 영상 모멘트의 중심을 찾아 마킹한다.
    모멘트 연산할 때 적용하는 옵션, binaryImage에 따라 어떻게 다른 결과가 나오는지 검토한다.
    이 예제 프로그램의 다음 phase II를 참조한다. -> moments2_imageAndcontour_based.py

# Phase I: 계조 영상과 이진 영상에 영상 모멘트 추출 실험. 영상의 중심에 빨간색 점을 찍는다.
# 1 단계: 컬러 영상을 읽고 gray로 변환한다.
# 2 단계: 그레이 계조 영상에 대해 2종의 옵션을 적용하여 모멘트를 구한다.
# 3 단계: 모멘트를 추출한 각각의 개념상의 영상에 센터를 표현한다.


retval = cv.moments(array[, binaryImage=false])
    Calculates all of the moments up to the third order of a polygon or rasterized shape.
    The function computes moments, up to the 3rd order, of a vector shape or a rasterized shape.
    array : Raster image (single-channel, 8-bit or floating-point 2D array)
        or an array ( 1×N or N×1 ) of 2D points (Point or Point2f ). - 1채널 영상이거나 연속적인 좌표(x,y)의 점들의 집합
    binaryImage: If it is true, all non-zero image pixels are treated as 1's.
        The parameter is used for images only.
    retval : returned in the structure, cv::Moments.

"""

import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Path = '../Images/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
#Path = 'd:/work/StudyImages/Images/'
#Path = ''
Path = '../data/'

#Name = 'CONTOUR_TEST_ORG.jpg'       # 바깥 테두리가 검은 색
#Name = 'bullseye.jpg'
#Name = 'rects.png'
#Name = 'shapes_r.png'
#Name = 'circles.jpg'
#Name = '10.jpg'
#Name = 'drawing10.png'
#Name = 'drawing2.png'
#Name = '4regions.png'
Name = '3regions.png'
#Name = 'drawing1.png'       # 입력영상의 이진화 여부가 중심점의 위치에 조금 영향을 준다.
#Name = "smooth.jpg"        # 입력영상의 이진화 여부가 중심점의 위치를 크게 가른다.

# ==================================================================================================
# Phase I: 계조 영상에 대한 영상 모멘트 추출 실험. 영상의 중심에 빨간색 점을 찍는다.
# 1 단계: 컬러 영상을 읽고 gray로 변환한다.
# 2 단계: 그레이 계조 영상에 대해 2종의 옵션을 적용하여 모멘트를 구한다.
# 3 단계: 모멘트를 추출한 각각의 개념상의 영상에 센터를 표현한다.
# ==================================================================================================

# ------------ 1 단계: 컬러 영상을 읽고 gray로 변환한다.
FullName = Path + Name
img = cv.imread(FullName)
assert img is not None, "Failed to load image file:"
cv2.imshow("step1: input, " + Name, img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv2.imshow("step1: gray, " + Name, gray)

# ------------ 2 단계: flag = True - 바이너리 영상으로 간주하여 무게 중심을 구한 경우
# 모멘트를 추출한 영상에 센터를 표현한다.
# 이때는 1 이상이면 모두 1의 화소 값을 갖는 것으로 간주한다.
# 단, 화면 출력을 위해 255로 표현하기로 한다.
#_, imgBin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)  # 0보다 크면 255로 바꾼다.
otsu_thr, imgBin = cv2.threshold(gray,  # otsu_thr = 연산된 임계치. , otsu_mask = 바이너리 영상
                                    -1,     # 원래 임계값을 지정하는 것인데  오츠는 자동 연산하기 때문에 활용되지 않는다.
                                    255,    # 임계값 보다 클 경우 배정될 값. 나머지는 0이된다.
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
flag = True
m = cv2.moments(imgBin, binaryImage=flag)
x = m['m10'] / m['m00']
y = m['m01'] / m['m00']
center_img = (int(x+0.5), int(y+0.5))
print(f"flag={flag}: image's center=", center_img)    # 영상 전체의 중심점

img2 = cv.cvtColor(imgBin, cv.COLOR_GRAY2BGR)
radius = 7; color=(0, 0, 255); thickness = -1 # fill
cv2.circle(img2, center_img, radius, color, thickness)
cv2.imshow(f"step2: Binary(flag=True), center={center_img}:", img2)


# ------------ 3 단계: flag = False, default - 그레이 영상으로 간주하여 무게 중심을 구한 경우
# 모멘트를 추출한 영상에 센터를 표현한다.
flag = False
m = cv2.moments(gray, binaryImage=flag)
x = m['m10'] / m['m00']
y = m['m01'] / m['m00']
center_img = (int(x+0.5), int(y+0.5))
print(f"flag={flag}: image's center=", center_img)    # 영상 전체의 중심점

img2 = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
radius = 7; color=(0, 0, 255); thickness = -1 # fill
cv2.circle(img2, center_img, radius, color, thickness)
cv2.imshow(f"step3: Gray(flag=False), center={center_img}", img2)

cv.waitKey(0)

