"""
개요
    산재한 데이터를 타원 모델에 맞추어 피팅(modeling)한다.
    피팅된 타원 정보는 2가지 방법으로 타원 도형으로 그릴 수 있다.

수행내용
    점으로 직선과 유사한 점들의 집합을 잡음을 첨가하여 만든다.
    잡음이 첨가된 점들을 이용하여 이를 타원 혹은 직선으로 피팅하여 점들의 집합을 선으로 모델링한다.
    모델링한 결과를 2가지 방법으로 보이고,
    원래의 형상 파라미터와 얼마가 유사한지 검토한다.

"""
import cv2
import cv2 as cv
import numpy as np
import random



#"""
#=================================================================================================================
# 단계(타원) 1: 임의 특징(축, 각도)을 가진 타원을 화면의 중앙에 배치하고 그 타원의 근처에 산재하는 화소들을 정의한다.
#=================================================================================================================

# 1) 배경영상을 정의한다.
img = np.full((512, 512, 3), 255, np.uint8)

# 2) 타원의 특징을 결정하는 3개 요소 중 중심은 고정하고 랜덤으로 나머지 2개 요소(축의_길이, 각도)를 정의한다.
center = (256, 256)     # 고정
axes = (int(256*random.uniform(0, 1)), int(256*random.uniform(0, 1)))   # (가로, 세로)
angle = int(180*random.uniform(0, 1))   # 각도 0~180
print('*** original: center=', center, '| axes=', axes, '| angle=', angle)
center_org = center
axes_org = axes
angle_org = angle

# 3) 주어진 축, 각도, 중심점 정보를 기반으로 polyline으로 타원을 근사화한 점의 위치 정보를 반환받는다.
# polyline은 곡선을 다수의 직선으로 모델링하는 선분들이다.
# ellipse2Poly() 함수는 선분을 구성하기 위한 다수의 점(좌표)의 정보를 ndarray로 반환한다.
pts = cv2.ellipse2Poly(center, axes, angle, 0, 360, 1)   # 0,360, 1 = arcStart, arcEnd, delta
#print('type(pts)=', type(pts))      # type(pts)= <class 'numpy.ndarray'>
#print('pts.shape=', pts.shape)    # =(점들의 수, 2). 2인 이유는 (x, y) 때문임.

# 4) pts.shape와 같은 차원의 -noise~+noise의 잡음을 생성하여 pts에 더해 준다. 즉, 위치에 잡음을 첨가한다.
noise = 15      # 위에서 얻은 정보에 첨가할 노이즈의 크기. 값이 작으면 모델링이 잘된다.
pts += np.random.uniform(-noise, +noise, pts.shape).astype(np.int32)

# 5) 위에서 정의한 타원을 초록색으로 그려 보인다. - 참고용임.
cv2.ellipse(img, center, axes, angle, 0, 360, (0, 255, 0), 1)

# 6) 산재한 점을 붉은 색 작은 원으로 그려 보인다. 이것이 타원을 예측할 때 사용되는 데이터이다.
for pt in pts:
    cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)   # (int(pt[0]), int(pt[1]) = (x,y). 3=반지름

cv2.imshow(f'St1: source, ellipse with noise={noise}', img)
cv2.waitKey()
#cv2.destroyAllWindows()

#=================================================================================================================
# 단계(타원) 2: fitEllipse() 함수는 산재한 점을 바탕으로 그것을 둘러싸는 rotated rectangle 형의 box 데이터를 반환한다.
# rotated rectangle 형의 box 데이터는 타원형의 특징 요소(중심점, 축, 각도)로 추정(fitting)해서 반환한다.
# 반환 값은 Rotated Rectangle형의 box 데이터인데 다음 3개의 요소로 이루어진 튜플이다.
#   - center coordinates: 타원형의 중심(x, y) 좌표
#   - width (major axis length), height (minor axis length): (w, h) 혹은 (h, w)
#   - rotation angle: 타원형을 둘러싸는 박스를 가정한 박스의 회전 각도
#
# 반환된 값으로 타원을 그리는 방법으로는 다음 2가지가 있다.
#
# 1. 반환된 데이터로 ellipse() 함수를 이용해 바로 타원형을 그릴 수도 있다.
#   box 데이터는 3개의 원소로 이루어진 튜플(중심점, 가로x세로, 각도)이다.
#   이 데이터는 부동소수를 지원한다.
#   ret_box = fitEllipse(영상)
#   img = cv.ellipse(img, ret_box, color[, thickness[, lineType]]
#
# 2. 반환 값을 3개의 변수로 unpacking 시켜 이를 기반으로  그린다.
#   center, axis, angle = fitEllipse(영상)  # 3개의 변수로 unpacking
#   반환된 값은 약간의 가공을 필요로 한다.
#   1) center = (int(x), int(y)). : 정수형 변환 필요. mass center of rectangle. 튜플 => 타원의 중심.
#       혹은 center = {tuple(map(int, center)) center의 각 원소에 int 함수를 취한 것을 튜플로 반환
#       다른 방법으로는 np.intp(center)로 정수형으로 만들수도 있다. 대신 tuple형이 ndarray로 바뀐다. OK!!
#   2) axis = (int(w/2), int(h/2)). # 추정 반환된 데이터는 타원형처럼 반지름이 아니다. 유의 요망.
#   3) angle = 부동소수 그대로 써도 됨. box(타원)가 x좌표에 대해 얼마나(degree) 기울었는가?
#   img = cv.ellipse(img, center, axes, angle, startAngle, endAngle,
#       color[, thickness[, lineType[, shift]]]	)
#=================================================================================================================
# 1) 주어진 점에 맞도록 타원을 모델링한다. 모델링된 정보는 ellipse에 반환된다.
ellipse = cv2.fitEllipse(pts)       # rotated rectangle 형의 box 데이터를 반환한다.

# 2) 특징요소 정보를 화면에 출력한다.
print('ellipse=', ellipse)     # 부동소수점 데이터를 출력하여 다음과 같이 바꾸었다.
print(f'type(ellipse)={type(ellipse)}, len(ellipse)={len(ellipse)}')  # <class 'tuple'>
center, axes, angle = ellipse
#print(f'type(center)={type(center)}, type(axis)={type(axis)}, type(angle)={type(angle)}')
#rgb = tuple(map(int, a))    # a의 각 요소에 int() 함수를 씌운 후 tuple로 변환한다.
#print(f'unpacked values: center={center}, axes={axes}, angle={angle}')

print(f'unpacked values: center={tuple(map(int, center))}, axes={tuple(map(int, axes))}, angle={int(angle)}')
#center = (int(center[0]), int(center[1]))
center = np.intp(center)        # 이게 더 편리...
print(f"type(center)={type(center)}")

#axes = (int(axes[0]/2), int(axes[1]/2))     # 반환받은 정보는 반지름이 아니라 지름 성분이다.
axes = np.intp(np.array(axes)/2)     # 일단 ndarray로 만들어 2로 나누고 다시 정수회를 한다.

print(f'*** original: center={center_org}, axes={axes_org}, angle={angle_org}')
print(f'*** fitting: center={center}, axes={axes}, angle={angle:#.2f}')


# 3) 피팅된 타원을 화면에 그린다.
# Rotated Rectangle형의 box 정보를 unpacking하여 그리기
img2 = img.copy()
cv.ellipse(img2, center, axes, angle, 0, 360, (0, 0, 0), 3)     # 원본 그림에 검은색으로 덧칠해 그린다.
cv2.imshow(f'St2 output1: Fit ellipse(1) with noise={noise}', img2)

# Rotated Rectangle형의 box 정보를 그대로 이용하여 그리기
img2 = img.copy()
cv2.ellipse(img2, ellipse, (255, 0, 0), 3)     # 원본 그림에 푸른색으로 덧칠해 그린다.
cv2.imshow(f'St2 output2: Fit ellipse(2) with noise={noise}', img2)

cv2.waitKey()
cv2.destroyAllWindows()
exit()
#"""


