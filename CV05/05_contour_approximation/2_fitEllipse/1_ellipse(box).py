"""
1. 개요
    타원형을 그리는 함수, ellipse()는 아래 2가지 방법으로 호출하여 타원형을 그린다.
    1) cv.ellipse(img, center, axes, angle,
        startAngle, endAngle, color[, thickness[, lineType[, shift]]]) ->	img
    2) cv.ellipse(img, box, color[, thickness[, lineType]]	) ->img
    여기서는 그중 box를 제공하는 함수 활용법이다.


2. 함수 호출방법

    img	= cv.ellipse(img, box, color[, thickness[, lineType]]
    box: Alternative ellipse representation via RotatedRect.
    This means that the function draws an ellipse inscribed in the rotated rectangle.

   box 데이터는 3개의 원소로 이루어진 튜플(중심점, 가로x세로, 각도)이다.
   이 데이터는 부동소수를 지원한다.
   center = (x,y) # 타원의 중심
   size = (width, height) # 타원의 크기. 반지름 성격이 아님. 전체길이. 가로x세로.
   angle = box(타원)가 x좌표에 대해 얼마나(degree) 기울었는가?

   box_ellipse = (center, size, angle)
   box_ellipse = ((x,y), (width, height), angle)
   box_ellipse 정보를 ellipse() 함수에 제공한다.

   img = cv.ellipse(img, box_ellipse, color[, thickness[, lineType]]

"""

import cv2
import cv2 as cv
import numpy as np
import random

# --------------------------------------------------------------------------------
# Rotated Rectangle형의 box 데이터형으로 타원형 정보를 넘긴다.
# 90도씩 회전하면 axes의 장단축 길이 정보를 뒤집은 것과 다름 없음을 보인다.
# --------------------------------------------------------------------------------

w=640; h=480;   # 가로, 세로
Center = (int(w / 2), int(h / 2))
hg = 90
Axes=[50*2, 240*2]  # 여기서는 전체길이 정보를 다 주어야 한다.
img = np.full((h, w, 3), (255, 255, 255), np.uint8)
for Angle in range(0, 360, 90):
    box_ellipse = (Center, Axes, Angle)
    img = cv.ellipse(img, box=box_ellipse,
        color=((255+hg) & 0xff, 0, hg),    # 타원의 색깔
        thickness=1)
    cv.putText(img, f"axes={Axes}, angle={Angle}", (20, hg), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    hg += 90
    tmp = Axes[0]
    Axes[0] = Axes[1]
    Axes[1] = tmp
    cv.imshow('ellipse - varying axes & axes', img)
    cv.waitKey()
print("모양이 변하지 않습니다....")
exit()






#"""
#=================================================================================================================
# 연습-타원 그리기:
# cv.cv.ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]] )
# axis = (장축_반지름, 단축_반지름)
# angle : 타원이 기준축 x에 대해 기울어진 각도
#=================================================================================================================
w=640; h=480;   # 가로, 세로
img = np.full((h, w, 3), (255, 255, 255), np.uint8)
cv.ellipse(img, (int(w/4), int(h/6)), (100, 50), 0, 0, 360, (255, 0, 0), -1)   # 0=기울기. 0,360=0~360. -1=채우기
cv.ellipse(img, (int(w/2), int(h/6)), (50, 25), 30, 0, 360, (255, 0, 0), -1)   # 30=기울기. 0,360=0~360. -1=채우기
cv.ellipse(img, (int(3*w/4)+50, int(h/6)), (100, 50), 30, 45, 180, (255, 0, 0), -1) # 0=기울기. 45,180=45~180. -1=채우기
cv.ellipse(img, (int(w/2),int(h/2)), (100,50), 45, 0,270, (0, 255, 0), 2)  # 45=기울기. 0,270=0~270. 2=선두께
cv.ellipse(img, (int(w/2)-150,int(h/2)+150), (50, 100), 0, 0,360, (255, 0, 255), 3) # 0=기울기. 0,360=0~360. 3=선두께
cv.ellipse(img, (int(w/2)-150,int(h/2)+150), (50, 100), 0, 0,90, (255, 0, 0), -1) # 0=기울기. 0,90=0~90. -1=채우기
cv.imshow('ellipse', img)
cv.waitKey()
exit(0)
#"""

