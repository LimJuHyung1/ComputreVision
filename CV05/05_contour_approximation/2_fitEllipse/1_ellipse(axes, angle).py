"""
1. 개요
    타원형을 그리는 함수, ellipse()는 아래 2가지 방법으로 호출하여 타원형을 그린다.

    1) cv.ellipse(img, center, axes, angle # 타원을 몇도로 돌렸냐,
        # 타원을 그리다 말 수도 있다 - startAngle, endAngle, color[, thickness[, lineType[, shift]]]) ->	img
    2) cv.ellipse(img, box, color[, thickness[, lineType]]	) ->img
    여기서는 그중 axes, anle를 제공하는 함수 활용법이다.


2. 함수 호출방법

    https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69
    img	= cv.ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]]	)
    Draws a simple or thick elliptic arc or fills an ellipse sector.

    img – image
    center – 타원의 중심
    axes – 중심에서 가장 큰 거리와 작은 거리(반지름 개념)
    angle – 타원의 기울기 각
    startAngle – 타원의 시작 각도
    endAngle – 타원이 끝나는 각도
    color – 타원의 색
    thickness – 선 두께 -1이면 안쪽을 채움
    사례: img = cv2.ellipse(img, (256,256), (100,50), 0, 0, 180, 255, -1)


3. 참고사항: -> 다른 예제로 독립시킴
img	= cv.ellipse(img, box, color[, thickness[, lineType]]
    box:	Alternative ellipse representation via RotatedRect.
    This means that the function draws an ellipse inscribed in the rotated rectangle.


"""
import cv2
import cv2 as cv
import numpy as np
import random

"""
# --------------------------------------------------------------------------------
# 실습 1: angle의 역할 - 타원을 시계 방향으로 회전시키는 각도를 제어한다.
# 아무 키나 입력하면 타원이 30도씩 시계 방향으로 회전한다.
# --------------------------------------------------------------------------------
w=640; h=480;   # 가로, 세로
Center = (int(w / 2), int(h / 2))
hg = 30
img = np.full((h, w, 3), (255, 255, 255), np.uint8)
for Angle in range(0, 360, 30):
    img = cv.ellipse(img, center=Center,
                     axes=(50, 240),        # (x축 길이, y축 길이)
                     angle=Angle,              # 타원의 회전각도, x축 기준 시계방향
                     startAngle=0, endAngle=360,    # 타원의 선 긋기 각도. angle=0을 기준
                     color=((255+hg) & 0xff, 0, hg),    # 타원의 색깔
                     thickness=1)      #  [[, lineType[, shift]]]
    cv.putText(img, f"center={Center}, angle={Angle}", (20, hg), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    hg += 30
    cv.imshow('ellipse - varying Angles', img)
    cv.waitKey()



# --------------------------------------------------------------------------------
# 실습 2: startAngle의 역할 - 타원을 그리는 선의 시작 각도를 의미한다.
# 아무 키나 입력하면 타원이 일정 각도 만큼 시계 방향으로 시작점이 회전한다.
# --------------------------------------------------------------------------------
w=640; h=480;   # 가로, 세로
Center = (int(w / 2), int(h / 2))
for startAngle in range(0, 720+1, 60):
    img = np.full((h, w, 3), (255, 255, 255), np.uint8)
    img = cv.ellipse(img, center=Center,
                     axes=(240, 100),        # (x축 길이, y축 길이)
                     angle=0,              # 타원의 회전각도, x축 기준 시계방향
                     startAngle=startAngle, endAngle=360,    # 타원의 선 긋기 각도. angle=0을 기준
                     color=(255, 0, 0),    # 타원의 색깔
                     thickness=1)      #  [[, lineType[, shift]]]
    cv2.putText(img, f"center={Center}, startAngle={startAngle}", (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv.imshow('ellipse - varying startAngle', img)
    cv.waitKey()
exit()
"""

# --------------------------------------------------------------------------------
# 실습 3: angle과 axes의 역할 - 실습3: X축을 90도 시계 방향 회전시키면 y축이 된다.
# angle을 90증가시키고, axes를 바꾸면 모양이 유지된다.
# 아무 키나 입력하면 타원이 90도씩 시계 방향으로 회전하면서,
# 동시에 axes의 (x, y) -> (y, x)로 자리를 바꾼다.
# 눈에 보이는 타원형의 모양은 바뀌지 않는다.
# --------------------------------------------------------------------------------
w=640; h=480;   # 가로, 세로
Center = (int(w / 2), int(h / 2))
hg = 90
Axes=[50, 240]
img = np.full((h, w, 3), (255, 255, 255), np.uint8)
for Angle in range(0, 360, 90):
    img = cv.ellipse(img, center=Center,
                     axes=Axes,        # (x축 길이, y축 길이)
                     angle=Angle,              # 타원의 회전각도, x축 기준 시계방향
                     startAngle=0, endAngle=360,    # 타원의 선 긋기 각도. angle=0을 기준
                     color=((255+hg) & 0xff, 0, hg),    # 타원의 색깔
                     thickness=1)      #  [[, lineType[, shift]]]
    cv.putText(img, f"axes={Axes}, angle={Angle}", (20, hg), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # 축 정보 교환 과정
    hg += 90
    tmp = Axes[0]
    Axes[0] = Axes[1]
    Axes[1] = tmp
    cv.imshow('ellipse - varying axes & axes', img)
    cv.waitKey()
print("모양이 변하지 않습니다....")
exit()



# --------------------------------------------------------------------------------
# 실습 4: Rotated Rectangle형의 box 데이터형으로 타원형 정보를 넘긴다.
#   box 데이터는 3개의 원소로 이루어진 튜플(중심점, 가로x세로, 각도)이다.
#   이 데이터는 부동소수를 지원한다.
#   center = (x,y) # 타원의 중심
#   size = (width, height) # 타원의 크기. 반지름 성격이 아님. 가로x세로.
#   angle = box(타원)가 x좌표에 대해 얼마나(degree) 기울었는가?
#
#   box_ellipse = (center, size, angle)
#   box_ellipse = ((x,y), (width, height), angle)
#   box_ellipse 정보를 ellipse() 함수에 제공한다.
#
#   img = cv.ellipse(img, box_ellipse, color[, thickness[, lineType]]
# --------------------------------------------------------------------------------
w=640; h=480;   # 가로, 세로
Center = (int(w / 2), int(h / 2))
hg = 90
Axes=[50, 240]
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

