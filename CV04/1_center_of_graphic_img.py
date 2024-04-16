"""
모멘트 - 객체(윤곽선, 영상, 점들의 집합)의 특징 기술자

모멘트 연산에 활용할 임의 영상을 생성한다.
그 영상을 바탕으로 모멘트의 기본적인 값을 분석해 본다.
-> 모멘트를 이용해 영상의 중심점 좌표를 얻어내는 함수 centroid()를 제작하였다.

간단한 지식:
1) OpenCV에서 제공하는 모멘트(m) 계산 함수
    m = cv2.moments(영상이나 컨투어 자료)
        m은 3종의 모멘트 값이 모두 연산된 사전형 자료이다.
2) OpenCV 반환 값, 사전형 자료 m에서 원하는 모멘트 찾아내기
    방법 1: m[Key]를 사용, 예: m['m01']
    방법 2: for loop에 두고 찾을 때는 m.items() 사용.
            for key, val in m.items():  # 사전형 데이터임.
                print(f"{key}: {val:#18.6f}")


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
Path = '../'
#Name = 'lenna.bmp'
#Name = 'monarch.bmp'
#Name = 'fruits.jpg'
#Name = 'bone.jpg'
#Name = 'woman_in_scarf(c).jpg'
#Name = 'man_woman.jpg'

Name = 'CONTOUR_TEST_ORG.jpg'       # 바깥 테두리가 검은 색
#Name = 'bullseye.jpg'
Name = 'rects.png'
#Name = 'shapes_r.png'
#Name = 'circles.jpg'
Name = '10.jpg'
Name = 'drawing1.png'
#Name = 'drawing10.png'
#Name = 'drawing2.png'
#Name = '3regions.png'
#Name = '4regions.png'



def centroid(moments):
    # 사전형 자료에서 moments에서 원하는 모멘트 찾기: 딕셔너리_변수_이름[Key]
    x_centroid = round(moments['m10'] / moments['m00'])
    y_centroid = round(moments['m01'] / moments['m00'])
    # center = (int(x+0.5), int(y+0.5)) 이것도 같은 결과
    return x_centroid, y_centroid


def print_moments(m, limit=3):  # 기본 3개만 출력. m00, m10, m01
    """ 일부 모멘트의 값을 출력한다. 맨 앞(0 번째)에서부터 (limit-1)번까지 """
    #
    # items 메서드는 Key와 Value의 쌍을 튜플로 묶은 값을 dict_items 객체로 리턴한다.
    # >>> a.items()
    # dict_items([('name', 'pey'), ('phone', '010-9999-1234'), ('birth', '1118')])
    i = 0
    for key, val in m.items():  # 사전형 데이터임.
        if i < limit:
            print(f"{key}: {val:#18.6f}")
        i += 1


# ------------------------------------------------------------------------
# 실험 1) 영상을 파일로부터 입력받지 않고, 인조 영상을 만들어 사용한다. 
# 단순 4각형 이진 영상을 만들어 0차, 1차 모멘트 만이라도 직접 계산해 본다.
# 이것을 함수, moments()로부터 구한 멘트와 비교해 본다.
# ------------------------------------------------------------------------
"""
# 이진 영상 정의
size = (w, h) = (11, 5)
imgBin = np.ones((h, w), np.uint8)

# m00 계산
print(f'rectangle dimension(w x h)={size}')
print(f'm00(전체 계조값의 합), area={w} x {h}: {w*h}') # 11 x 5: 55

# m10 계산 -원점(0,y)에 대해 x축을 지렛대로 토크를 구한다고 생각하자.
sum_x = np.sum(range(w))    # 0+1+2+...+(w-1) -> 0+1+2+...+10 = 55
print(f'\n1개의 x축(가로 축)에 따른 좌표 값의 합={sum_x}')
print(f'가로선의 수={h}: m10={sum_x * h}')   # 가로선의 수=5: m10=275

# m10 계산-원점(x,0)에 대해 y축을 지렛대로 토크를 구한다고 생각하자.
sum_y = np.sum(range(h))    # 0+1+2+...+(h-1) -> 0+1+2+3+4 =10
print(f'\n1개의 y축(세로 축)에 따른 좌표 값의 합={sum_y}')    # 10
print(f'세로선의 수={w}: m01={sum_y * w}')   # 세로선의 수=11: m01=110

# 이제 OpenCV 함수, moments()로 구해보자.
m = cv2.moments(imgBin)
print('\ntype(m)', type(m))       # type(m) <class 'dict'>
print_moments(m, 3)     # 일부만 출력해 본다.
#print_moments(m, 200)          # 모두 프린트 하고 싶으면...

center = centroid(m)    # 중심좌표를 구하는 함수
print('center=', center)    # center= (5, 2)
exit(0)
"""


# ------------------------------------------------------------------------
# 실험 2) 영상을 파일로부터 입력받지 않고, 인조 영상을 만들어 사용한다.
# - 타원형 인조 영상을 만든다.
# - 이를 기반으로 cv.moments() 함수로 영상모멘트를 구한다.
# - 반환되는 모멘트 값 중에서 중심점을 구하는데 필요한 값들을 구해 타원형의 중심점을 손쉽게 구한다.
# - 해 본다.
# ------------------------------------------------------------------------

size = (w, h) = (640, 480)

print(f'rectangle dimension(w x h)={size}')

# for image A ---------
imgBin = np.zeros((h, w), np.uint8)
# 타원 그리는 함수: cv.ellipse(img, center, axes, angle, startAngle, endAngle, color)
cv2.ellipse(imgBin, (420, 240), (200, 100), 0, 0, 360, 255, -1)
m = cv2.moments(imgBin)     # binaryImage=false, default
center_A = centroid(m)
print('A: center=', center_A)
imgBin3 = cv2.cvtColor(imgBin, cv2.COLOR_GRAY2BGR)
radius = 7; color = (0, 0, 255); thickness = -1 # fill
cv2.circle(imgBin3, center_A, radius, color, thickness)
cv2.imshow('image A', imgBin3)
cv2.waitKey(0)

# for image B ---------
imgBin = np.zeros((h, w), np.uint8)
cv2.ellipse(imgBin, (110, 240), (100, 200), 0, 0, 360, 255, -1)
m = cv2.moments(imgBin)
center_B = centroid(m)
print('B: center=', center_B)
imgBin3 = cv2.cvtColor(imgBin, cv2.COLOR_GRAY2BGR)
radius = 7; color=(0, 0, 255); thickness = -1 # fill
cv2.circle(imgBin3, center_B, radius, color, thickness)
cv2.imshow('image B', imgBin3)
cv2.waitKey(0)

x_avg = center_A[0] + center_B[0]
y_avg = center_A[1] + center_B[1]
center_AB = (round(x_avg/2), round(y_avg/2))
print('(A+B)/2: center=', center_AB)


# for image C=A+B ---------
imgBin = np.zeros((h, w), np.uint8)
cv2.ellipse(imgBin, (420, 240), (200, 100), 0, 0, 360, 255, -1)
cv2.ellipse(imgBin, (110, 240), (100, 200), 0, 0, 360, 255, -1)
m = cv2.moments(imgBin)
center_C = centroid(m)
print('C: center=', center_C)

imgBin3 = cv2.cvtColor(imgBin, cv2.COLOR_GRAY2BGR)
radius = 7; color=(0, 0, 255); thickness = -1 # fill
cv2.circle(imgBin3, center_C, radius, color, thickness)
cv2.imshow('image C', imgBin3)
cv2.waitKey(0)
exit(0)

