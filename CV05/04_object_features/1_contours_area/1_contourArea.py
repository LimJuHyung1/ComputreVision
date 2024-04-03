"""
retval = cv.contourArea(contour[, oriented=false])
    contour: Input vector of 2D points (contour vertices), stored in std::vector or Mat.
    oriented: Oriented area flag.
If it is true, the function returns a signed area value, depending on the contour orientation
True=clockwise or False=counter-clockwise
Default: false= 절대값을 반환
객체의 면적을 반환
"""

import cv2, random
import cv2 as cv
import numpy as np

r = 240
c = 320
img = np.zeros((r,c), np.uint8)


"""
# ========================================================================================================
# 실험 1: 사각형
# 4각형을 정의하여 이것으로 컨투어 데이터를 만들어 올바른 면적을 반환하는지 살핀다.
# 더불어 반환하는 부호를 살펴거 주어진 컨투어가 시계방향으로 되었는지 반시계 방향으로 되었는지를 정확히 진단하는지 확인한다.
#   가상 컨투어를 만들어 제공하였으므로 시계 방향 반시계 방향을 모두 만들어 제공하여 그 결과를 살필 수 있다.
# ========================================================================================================
img2 = img.copy()
x1 = 50; y1 = 50          # 좌측 상단의 모서리
x2 = 150; y2 = 150        # 우측 하단의 모서리
upperleft = (x1, y1)
lowerright = (x2, y2)

# 1) 도형(4각형)을 정의하고, 내부를 흰색으로 채운다.
cv.rectangle(img2, upperleft, lowerright, (255, 255, 255), -1 )

# 2) 추정되는 면적을 미리 계산한다. 4각형이므로 우리는 그 면적을 계산으로 알 수 있다.
print('\nComputational area:')
area_exp = abs(x2-x1) * abs(y2-y1)
print('\tupperleft =', upperleft)
print('\tlowerright =', lowerright)
print(f'\tarea ={area_exp:#.2f}')

# 3) 가상의 영상에 대해 컨투어 검출을 실시한다.
contours, hierarchy = cv2.findContours(image=img2, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
print(f"hierarchy : {hierarchy}")
# 4) contourArea() 함수를 호출하여 컨투어의 면적을 구한다.
contour = contours[0]
print('\ncontour points\n', contour)

# print('-------------------------------------------------------')
area_func = cv2.contourArea(contour, False)     # oriented=False default - 반시계방향 - 절대값 변환
                                                        # True 시계방향 - 부호 있음
print(f'Function, contourArea() based result={area_func:#.2f}')
# print('-------------------------------------------------------')

# 5) 계산상의 면적와 실제 구한 면적의 차이를 확인한다.
error = area_exp - area_func
print(f'Error: expectaion-contourArea={error:#.2f}, percentage={error*100/area_exp:#.2f}')

# 6) 부호를 검토해 보자. 부호가 있는 면적은 oriented 파리미터를 True로 지정해야 구할 수 있다.
area = cv2.contourArea(contour, True)
print(f'\nFor this contour:')
print(f'area with sign={area:.2f}')

if area > 0:
    print('\torientation: clockwise')
else:
    print('\torientation: counter-clockwise.')


# 7) 콘투어를 역방향으로 재 배열하고 어떤 부호가 검출되는지 검토한다.
contour2 = contour[::-1]
print('\n\ncontour2 points\n', contour2)
area = cv2.contourArea(contour2, True)
print(f'\nFor this reverse contour:')
print(f'area with sign={area:.2f}')
if area > 0:
    print('\torientation: clockwise')
else:
    print('\torientation: counter-clockwise.')

cv.imshow('rectangle', img2)
cv.waitKey()
exit(0)
"""


#"""
# ========================================================================================================
# 실험 2: 원
# ========================================================================================================
# 1) 도형(원)을 정의하고, 내부를 흰색으로 채운다.
img2 = img.copy()
center = (int(c/2+0.5), int(r/2+0.5))
radius = 100
cv.circle(img2, center, radius, (255, 255, 255), -1)


# 2) 추정되는 면적을 미리 계산한다.
print('\n2) Computational area:')
area_exp = np.pi * radius **2
print(f'\tarea_exp={area_exp:#.2f}')

# 3) 가상의 영상에 대해 컨투어 검출을 실시한다.
contours, hierarchy = cv2.findContours(image=img2, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
contour = contours[0]

# 4) contourArea() 함수를 호출하여 컨투어의 면적을 구한다.
contour = contours[0]
#print('\ncontour points\n', contour)
area_func = cv2.contourArea(contour)     # oriented=False default
print(f'4) cv2.contourArea(contour) 함수로 반환받은 면적\n'
      f'oriented=False. default\n'
      f'contourArea()={area_func:#.2f}')

# 5) 계산상의 면적와 실제 구한 면적의 차이를 확인한다.
error = area_exp - area_func
print(f'5) Error: expectaion-contourArea={error:#.2f},\n\tpercentage={error*100/area_exp:#.2f}%')

# 6) 부호를 검토해 보자. 부호가 있는 면적은 oriented 파리미터를 True로 지정해야 구할 수 있다.
print(f'\n6) 부호있는 면적을 반환: contourArea(contour, oriented=True)')
area = cv2.contourArea(contour, True)
print(f'area with sign={area:.2f}')

if area > 0:
    print('\torientation: clockwise')
else:
    print('\torientation: counter-clockwise.')


# 7) 콘투어를 역방향으로 재 배열하고 어떤 부호가 검출되는지 검토한다.
contour2 = contour[::-1]
#print('\n\ncontour2 points\n', contour2)
area = cv2.contourArea(contour2, True)
print(f'\n7) 취득한 컨투어를 역방향으로 제공합니다.\n'
      f'부호있는 면적, contourArea(contour[::-1], oriented=True)')
print(f'area with sign={area:.2f}')
if area > 0:
    print('\torientation: clockwise')
else:
    print('\torientation: counter-clockwise.')

cv.imshow('rectangle', img2)
cv.waitKey()


# 8) default 설정이 반시계 방향이라고 하는데 그것을 확인하기 위해 화면에 색상으로 표시하여 보인다.
# 시작점 -> 1/3인 지점 -> 2/3인 지점에 각각 적색, 노란 색, 초록색의 원을 그려 보인다.
imgC = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

x, y = contour[0][0]    # 시작점
point = (x,y)
print(point)
cv.circle(imgC, point, 10, (0, 0, 255), -1)     # starting location

x, y = contour[int(len(contour)/3)][0]  # 컨투어의 1/3 되는 지점의 좌표
point = (x, y)
print(point)
cv.circle(imgC, point, 5, (0, 255, 255), -1)    # 1/3 location

x,y = contour[int(len(contour)*2/3)][0]
point = (x, y)
print(point)
cv.circle(imgC, point, 3, (0, 255, 0), -1)      # 2/3 location
cv.imshow('imgC', imgC)
cv.waitKey()
exit(0)
#"""


# ========================================================================================================
# 실험 3: 1/4 타원
# ========================================================================================================

# 1) 도형(원, 혹은 타원...)을 정의하고, 내부를 흰색으로 채운다.
img2 = img.copy()
center = (int(c/2+0.5), int(r/2+0.5))
radius = 100
axes = (radius, radius)
angle =0
#cv2.ellipse(img2, center, axes, angle, 0, 360, (255, 255, 255), -1)
cv2.ellipse(img2, center, axes, angle, 0, 90, (255, 255, 255), -1)

# 2) 타원형 면적 계산은 생략합니다....

# 3) 가상의 영상에 대해 컨투어 검출을 실시한다.
contours, hierarchy = cv2.findContours(image=img2, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

# 이하 생략합니다.....