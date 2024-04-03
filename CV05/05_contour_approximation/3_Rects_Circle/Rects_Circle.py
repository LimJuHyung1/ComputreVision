"""
개요:
  윤곽선을 추출하여 객체를 둘러싼 4각형, 원, 기울어진 4각형으로 근사화한 도형으로 만들어 도시한다.

수행과정:
  단계 0 : 입력 영상을 읽어들인다.
  	  실험 영상은 이진화가 용이한 것을 선택하는 것이 분석에 도움이 된다.

  단계 1 : 입력 영상의 이진화 작업.
	  편의상 OTSU 등의 이진화 알고리즘을 활용한다. 임계값을 이용한 이진화가 용이한 영상에 적용.

  단계 2 : 객체의 외곽 윤곽선을 찾는다.

  단계 3 : 영상에 윤곽선을 도형으로 근사화하여 그린다.
  	 다음 3개의 방법으로 근사화한다.

	  1) boundingRect()로 근사화 => rectangle() 함수로 그린다.
	  2) minEnclosingCircle() 로 단순화 => circle() 함수로 그린다.
      3) minAreaRect()로 단순화 =>
		boxPoints() 함수로 4개의 점으로 변환
		=> contour 형태의 데이터로 만들어
		=> drawContours 함수로 그린다.


"""

import cv2
import cv2 as cv
import numpy as np


Path = '../../data/'

#Name = 'hammer.jpg'              # 이 영상은 지나치게 커서 img = cv2.pyrDown(img) 줄일 필요가 있다.
Name = 'CONTOUR_TEST_ORG.jpg'       # 바깥 테두리가 검은 색
#Name = 'bullseye.jpg'
#Name = 'rects.png'
Name = 'shapes_r.png'
#Name = 'circles.jpg'
#Name = '10.jpg'
Name = 'drawing1.png'
#Name = 'drawing2.png'
#Name = 'drawing3.png'
#Name = 'drawing4.png'
Name = 'drawing5.png'
#Name = 'drawing6.png'
#Name = 'drawing7.png'
#Name = 'BnW.png'                  # Path = '../data/'
#Name = 'lightening.png'
#Name = 'lightening2.png'


#=======================================================================================================
# 단계 0 : 입력 영상을 읽어들인다.
# 실험 영상은 이진화가 용이한 것을 선택하는 것이 분석에 도움이 된다.
#=======================================================================================================
FullName = Path + Name
img = cv.imread(FullName)   # 편의상 모노 영상도 3채널로 읽은 것으로 가정한다.
assert img is not None, "Failed to load image file:"
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv2.imshow("input:"+str(Name), img)
cv2.imshow("gray", gray)
cv2.waitKey()
#=======================================================================================================
# 단계 1 : 입력 영상의 이진화 작업.
# 편의상 OTSU 등의 이진화 알고리즘을 활용한다. 임계값을 이용한 이진화가 용이한 영상에 적용.
#=======================================================================================================

otsu_thr, imgBin = cv2.threshold(gray,  # otsu_thr = 연산된 임계치. , otsu_mask = 바이너리 영상
                                    -1,     # 원래 임계값을 지정하는 것인데  오츠는 자동 연산하기 때문에 활용되지 않는다.
                                    255,    # 임계값 보다 클 경우 배정될 값. 나머지는 0이된다.
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow("imgBin by thresholding", imgBin)
cv2.waitKey()

#=======================================================================================================
# 단계 2 : 객체의 외곽 윤곽선을 찾는다.
# image, contours, hierarchy=findContours(image, mode, method[, contours[, hierarchy[, offset]]])
#=======================================================================================================
contours, hier = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('\nNumber of total contours = ', len(contours) )

#=======================================================================================================
# 단계 3 : 영상에 윤곽선을 도형으로 근사화하여 그린다.
# 3개의 방법으로 근사화한다.
#=======================================================================================================
# 윤곽선을 그릴 base 영상을 선택한다.
base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)      # 윤곽선을 칼라로 그리기 위해 3채널로 확장한다.
#base = img


# 있는 그대로 그린다: drawContours로 그린다.
#img2 = base.copy()
cv2.drawContours(base, contours, -1, (0, 255, 255), 2)    # -1=모든 윤곽선 다 그린다.
cv2.imshow('drawContours', base); cv2.waitKey()


# 1) boundingRect()로 근사화 => rectangle() 함수로 그린다.
print('\n1) boundingRect()')
print("객체별로 4개의 점을 반환하는데 x, y, width, height로 해석한다.")
img2 = base.copy()
i = 0
for c in contours:
  # 윤곽선 c를 둘러싼 4각형의 좌측 상단 좌표(x, y)와 폭(w), 높이(h)를 반환한다.
  ret = cv2.boundingRect(c)
  print(f'{i}: ret=', ret)
  x, y, w, h = ret
  cv2.rectangle(img2, (x,y), (x+w, y+h), (0, 255, 0), 2)
  cv2.putText(img2, str(i), (x+int(w/2), y+int(h/2)),
              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  i += 1

cv2.imshow("1) boundingRect", img2)
cv2.waitKey()


# 2) minEnclosingCircle() 로 단순화 => circle() 함수로 그린다.
print('\n2) minEnclosingCircle()')
print("2개의 원소를 가진 튜플 반환: ((중심점 x, 중심점 y), 반지름))")
img2 = base.copy()

i = 0
for c in contours:
  # 윤곽선 C를 둘러싸는 원의 중심(x,y)과 반지름을 반환한다.
  ret = cv2.minEnclosingCircle(c)
  print(f'{i}: ret=', ret)
  (x, y), radius = ret
  center = (int(x), int(y))
  radius = int(radius)
  cv2.circle(img2, center, radius, (255, 255, 0), 2)
  cv2.putText(img2, str(i), center, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  i += 1

cv2.imshow("2) minEnclosingCircle", img2)
cv2.waitKey()


# 3) minAreaRect()로 단순화 =>
# boxPoints() 함수로 4개의 점으로 변환
# => contour 형태의 데이터로 만들어
# => drawContours 함수로 그린다.

def pprint_rr(i, ret):   # print rotated rectangle
  (x, y), size, angle = ret
  print(f'\nShape {i}:\ncenter(x,y)=({int(x):d}, {int(y):d})')
  print(f'size(w,h)=({int(w):d}, {int(h):d})')
  print(f'angle={angle:.1f}')

print('\n3) minAreaRect()')
print("무게 중심(x, y), 크기(w, h), 각도 -> 이른바 rotated rectangle 정보를 반환한다.")
img2 = base.copy()

for i, c in enumerate(contours):
  # minAreaRect - calculates and returns the minimum-area bounding rectangle (possibly rotated) for a specified point set.
  # 반환 값 = 무게 중심(x, y), 크기(w, h), 각도 -> 이른바 rotated rectangle 정보
  ret = cv2.minAreaRect(c)
  pprint_rr(i, ret)    # print rotated rectangle
  (x, y), size, angle = ret

  # Calculates the up-right bounding rectangle of a point set or non-zero pixels of gray-scale image.
  # 4각형의 4개의 꼭지점 좌표(x,y)를 ndarray 타입으로 반환한다. 부동소수형
  box = cv2.boxPoints(ret)
  #print('type(box)=', type(box))      # <class 'numpy.ndarray'>
  #print('box.shape=', box.shape)      # box.shape= (4, 2)

  # change coordinates to integers
  #box1 = np.int0(box)            # 정수로 변환. DeprecationWarning: `np.int0` is a deprecated alias for `np.intp`.  (Deprecated NumPy 1.24)
  #print('box=', box)            # 기울어진 사각형에 대한 4개 꼭지점 좌표. 부동소수가 있을 수 있다. 객체가 많으면 주석문 처리.
  box1 = np.intp(box)   # box의 각 요소 값을 정수로 바꾼다.
  print('box1=', box1)
  #print(type(box))    # <class 'numpy.ndarray'>

  cv2.drawContours(img2, [box1], 0, (0, 0, 255), 2)
  cv2.putText(img2, str(i), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("3) minAreaRect", img2)
cv2.waitKey()

cv2.destroyAllWindows()


