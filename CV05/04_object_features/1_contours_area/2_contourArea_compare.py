"""
윤곽선을 추출하여 객체를 둘러싼 4각형, 원, 기울어진 4각형을 영상에 도시한다.


"""

import cv2
import cv2 as cv
import numpy as np


Path = '../../../CV04/'
#Name = 'lenna.bmp'
#Name = 'monarch.bmp'
#Name = 'fruits.jpg'
#Name = 'bone.jpg'
#Name = 'woman_in_scarf(c).jpg'
#Name = 'man_woman.jpg'

#Name = 'hammer.jpg'              # 이 영상은 지나치게 커서 img = cv2.pyrDown(img) 줄일 필요가 있다.
Name = 'CONTOUR_TEST_ORG.jpg'       # 바깥 테두리가 검은 색
#Name = 'bullseye.jpg'
#Name = 'rects.png'
Name = 'shapes_r.png'
#Name = 'circles.jpg'
#Name = '10.jpg'
Name = 'drawing1.png'
Name = 'drawing2.png'
#Name = 'drawing3.png'
#Name = 'drawing4.png'
#Name = 'drawing5.png'
Name = 'drawing6.png'
#Name = 'BnW.png'                  # Path = '../data/'
#Name = 'bw.png'                    # Path = '../data/'


def centroid(moments):
    """Returns centroid based on moments"""
    x_correction = - 30     # 출력 폰트의 추정 크기 만큼 보정한다.
    y_correction = + 10
    x_centroid = round(moments['m10'] / moments['m00']) + x_correction
    y_centroid = round(moments['m01'] / moments['m00']) + y_correction
    return x_centroid, y_centroid


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
#cv2.waitKey()
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

contours, _ = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('\nNumber of total contours = ', len(contours) )



#=======================================================================================================
# 단계 2 : 최대, 최소 객체를 따로 표기한다.
#=======================================================================================================

area = []
for i in range(len(contours)):
    a = cv2.contourArea(contours[i])            # 지정된 contour의 면적을 반환한다.
    print(f'area of contour[{i}]:{a}')
    area.append(a)
i_min = np.argmin(area)
i_max = np.argmax(area)
print('index of min=', i_min)
print('index of max=', i_max)


img2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)      # 윤곽선을 칼라로 그리기 위해 3채널로 확장한다.

# 1) 가장 면적이 작은 컨투어의 중심에 Min으로 표기한다.
cv2.drawContours(img2, contours, i_min, (255, 0, 0), -1)    # 특정 콘투어 지정
M = cv2.moments(contours[i_min])    # 영상 모멘트를 계산한다.
x, y = centroid(M)                  # 중점
cv2.putText(img2, 'Min', (x,y), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)


# 2) 가장 면적이 큰 컨투어의 중심에 Max로 표기한다.
#cv2.drawContours(img2, contours, i_max, (0, 0, 255), -1)
# 특정 콘투어를 list로 담아야 contours 자료가 된다.
cv2.drawContours(img2, [contours[i_max]], -1, (0, 0, 255), -1)
M = cv2.moments(contours[i_max])
x, y = centroid(M)
cv2.putText(img2, 'Max', (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

cv2.imshow("Contours" + Name, img2)
cv2.waitKey()

# 3) 있는 그대로 그린다: drawContours로 그린다.
# 윤곽선을 그릴 base 영상을 선택한다.
#cv2.drawContours(img2, contours, -1, (0, 255, 0), 2)    # -1=모든 윤곽선 다 그린다.

for i in range(len(contours)):
    #cv2.drawContours(img2, contours, contourIdx=i, color=(0, 0, 255), thickness=2)
    cv2.drawContours(img2, contours, i, (0, 255, 255), 1)
    pts = contours[i]; x, y = pts[0][0]; point = (x,y)  # 첫번 째 점에 컨투어 번호를 적는다.
    cv.putText(img2, str(i), point, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
cv2.imshow("Contours" + Name, img2)
cv2.waitKey()

cv2.destroyAllWindows()


