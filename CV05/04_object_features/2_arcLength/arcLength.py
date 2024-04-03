"""
실험의 개요
    사각형, 원, 1/4 원, 선 총 4개 도형에 대해 윤곽선의 길이를 arcLength() 함수를 이용해 게산한다.
    편의상 4개의 각 화면에 화면마다 1개의 도형만 그려 이것의 바깥 윤곽선을 검출하기로 한다.
    인조영상이기때문에 주변 둘레를 예측(answer)할 수 있는데 이것과 arcLength() 함수를 이용한 것과의 차이를 비교해 볼 수 있다.
    이때 윤곽선 추출 방법을 4가지로 바꾸어 추출하면서 알고리즘에 따라 추출되는 점의 개수 및 둘레 길이를 추출한다.
    - 프로그램의 선택 사항
            open contour까지 출력할 것인지는 함수에서 결정할 수 있다.
            입력 영상을 어떤 것을 사용할지는 소스의 주석문 변경을 통해 약간 바꿀 수 있다.

실험 결과
    close loop는 폐곡선의 길이를 잴 때 사용하고, open loop는 개곡선의 길이를 잴 때 사용한다.
    폐곡선인데 open loop를 적용하거나, 개곡선인데 close loop 파라미터 적용하면 올바를 결과를 반환받을 수 없는 것으로 실험결과 알 수 있었다.
    실제 개곡선의 사례는 4번의 'line'의 경우이다. 라인의 길이를 알고자 할 때는 open loop가 설득력이 있어 보인다.
    'line'에 대해 폐곡선을 적용하면 선을 둘러싸는 가상선에 대하여 둘레길이가 반환되기 때문에 실제 길의의 2배가 되는 것을 확인할 수 있다.



사용한 함수

retval=cv.arcLength(curve, closed)
    Parameters
        curve: Input vector of 2D points, stored in std::vector or Mat.
        closed:	bool. Flag indicating whether the curve is closed or not.
    Calculates a contour perimeter or a curve length.
    The function computes a curve length or a closed contour perimeter.

"""

import cv2, random
import cv2 as cv
import numpy as np

# input 1 channel gray image -> findContour -> arcLength -> draw contour
# 영상을 입력받아 외형 윤곽선을 추출하고 그것의 둘레 길이를 arcLength로 추출한다.
# 이때 윤곽선 추출 방법을 4가지로 바꾸어 추출하면서 알고리즘에 따라 추출되는 점의 개수 및 둘레 길이를 추출한다.
# 또한, open contour의 길이를 flag=False 선택으로 출력하게 하였다.
def draw_arc_contour(img, title, flag=True):
    #methods = ['CHAIN_APPROX_NONE', 'CHAIN_APPROX_SIMPLE', 'CHAIN_APPROX_TC89_L1', 'CHAIN_APPROX_TC89_KCOS']
    methods = ['APPROX_NONE', 'APPROX_SIMPLE', 'APPROX_TC89_L1', 'APPROX_TC89_KCOS']
    for mtd in methods:
        contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_EXTERNAL, method=eval('cv2.CHAIN_' + mtd))
        cl_len = cv2.arcLength(contours[0], True)           # closed contour
        print(f'\t{mtd}: pts={len(contours[0])}, Length-c={cl_len:#.2f}')
        if flag == False:   # open contour 조건이면 한번 더 구해본다.
            op_len = cv2.arcLength(contours[0], False)      # open contour
            print(f'\t{mtd}: pts={len(contours[0])}, Length-o={op_len:#.2f}')
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)     # draw the contour of the last method
    cv.imshow(title, img)
    cv.waitKey()





r = 240
c = 320
image = np.zeros((r,c), np.uint8)

# ----------------------------------------------------------------------------------------------------
# 실험 1: 사각형
# ----------------------------------------------------------------------------------------------------
print('\n1. For a square ...')

# 1) make the image
img = image.copy()
s_x, s_y = 100, 100     # start point = (x, y). 좌측 상단 꼭지점
e_x, e_y = 200, 200     # end point = (x, y). 우측 하단 꼭지점
w, h = e_x-s_x, e_y-s_y
cv.rectangle(img, (s_x, s_y), (e_x, e_y), (255, 255, 255), -1)  # 시작점, 대각 꼭지점, 색상

# 2) print expected Length
print(f'width={w}, height={h}: Length={2*w + 2*h}')

# 3) extract contour & compute its length
draw_arc_contour(img, 'square')



# ----------------------------------------------------------------------------------------------------
# 실험 2: 원
# ----------------------------------------------------------------------------------------------------
print('\n2. For a circle ...')

# 1) make the image
img = image.copy()
center = (int(c/2+0.5), int(r/2+0.5))
radius = 100
cv.circle(img, center, radius, (255, 255, 255), -1)

# 2) print expected Length
print(f'radius={radius}: Length={2 * np.pi * radius:#.2f}')

# 3) extract contour & compute its length
draw_arc_contour(img, 'circle')




# ----------------------------------------------------------------------------------------------------
# 실험 3: full or 1/4 타원
# ----------------------------------------------------------------------------------------------------
print('\n3. For a ellipse ...')

# 1) make the image
img = image.copy()
a = radius      # long
b = int(radius/2+0.5)   # short
axes = (a, b)
angle = 0
cv2.ellipse(img, center, axes, angle, 0, 360, (255, 255, 255), -1)         # 1) full 타원
#cv2.ellipse(img, center, axes, angle, 0, 90, (255, 255, 255), -1)         # 2) 1/4 타원

# 2) print expected Length
perimeter = 2 * np.pi * ((1/2) * (a**2 +b**2))**(1/2)       # 1) 타원(full)의 전체 둘레. 근사화 공식이라 함.
#perimeter = perimeter/4 + (a + b)      # 2) 1/4 타원의 둘레=원주/4 + (장축 반지름 + 단축 반지름)
print(f'a={a}, b={b}: Length={perimeter:#.2f}')

# 3) extract contour & compute its length
draw_arc_contour(img, 'ellipse')




# ----------------------------------------------------------------------------------------------------
# 실험 4: 선
# ----------------------------------------------------------------------------------------------------
print('\n4. For a line ...')

# 1) make the image.
img = image.copy()
#s_x, s_y = 100, 100     # start point = (x, y)
#e_x, e_y = 100, 200     # end point = (x, y_

s_x, s_y = 100, 0; e_x, e_y = 100, r-1          # vertical line

cv2.line(img, (s_x, s_y), (e_x, e_y), (255, 255, 255), 1 )

# 2) print expected Length
d = np.sqrt( (s_x - e_x)**2 + (s_y - e_y)**2 )
print(f'Length={d:#.2f}')

# 3) extract contour & compute its length
draw_arc_contour(img, 'line', flag=False)   # show closed & open results

"""
# ----------------------------------------------------------------------------------------------------
# 실험 5: white wall - 번외...
# ----------------------------------------------------------------------------------------------------
print('\nFor a wall ...')

# 1) make the image.
r = 240
c = 320
image = np.zeros((r,c), np.uint8)
d =100
image[:, 0:d] = 255

# 2) print expected Length
print(f'Length={d:#.2f}')

# 3) extract contour & compute its length
draw_arc_contour(image, 'wall', flag=False)   # show closed & open results
"""