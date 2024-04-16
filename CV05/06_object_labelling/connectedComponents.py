"""
개요: connectedComponents()는 바이너리 영상에서 개별 객체를 찾아 이것을 레이블링한다.
    이 함수를 이용해 이진 영상의 객체를 분리해서 같은 라벨을 가진 객체들의 랜덤칼라로 칠한다.
    검출된 객체의 중심을 표시하고, 객체의 번호를 적어 넣는다.

Connected Component Labelling 방법에 대한 설명 자료
    http://aishack.in/tutorials/labelling-connected-components-example/
    http://aishack.in/tutorials/connected-component-labelling/

"""

import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Path = '../data/'
#Path = '../data/'
#Name = 'lenna.bmp'
#Name = 'monarch.bmp'
#Name = 'fruits.jpg'
#Name = 'bone.jpg'
#Name = 'woman_in_scarf(c).jpg'
#Name = 'man_woman.jpg'

#Name = 'hammer.jpg'              # 이 영상은 지나치게 커서 img = cv2.pyrDown(img) 줄일 필요가 있다.
Name = 'CONTOUR_TEST_ORG.jpg'       # 바깥 테두리가 검은 색
#Name = 'bullseye.jpg'
Name = 'rects.png'
#Name = 'shapes_r.png'
#Name = 'circles.jpg'
#Name = '10.jpg'
#Name = 'drawing1.png'
Name = 'drawing1.png'
Name = 'drawing3.png'

#=======================================================================================================
# 단계 0 : 입력 영상을 읽어들인다.
# 실험 영상은 이진화가 용이한 것을 선택하는 것이 분석에 도움이 된다.
#=======================================================================================================
FullName = Path + Name
img = cv.imread(FullName)
assert img is not None, "Failed to load image file:"
cv2.imshow("input:"+str(Name), img)
cv2.waitKey()
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)



"""
# 현재 오동작 중...
#=======================================================================================================
# 실험 1 : connectedComponents() 함수를 이용하여 이진 영상의 레이블링을 시행한 결과를 보인다.
#=======================================================================================================

# 단계 1 : 입력 영상의 이진화 작업.
otsu_thr, imgBin = cv2.threshold(gray,  # otsu_thr = 연산된 임계치. , otsu_mask = 바이너리 영상
                                    -1,     # 원래 임계값을 지정하는 것인데  오츠는 자동 연산하기 때문에 활용되지 않는다.
                                    255,    # 임계값 보다 클 경우 배정될 값. 나머지는 0이된다.
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow("imgBin by thresholding", imgBin)
cv2.waitKey()

# 단계 2 : 레이블링을 시행한다.
connectivity = 8
num_labels, labelmap = cv2.connectedComponents(imgBin, connectivity, cv2.CV_32S)
print(num_labels)

# 단계 3: 원본 영상과 레이블링된 영상을 보인다.
# 레이블링 결과는 0~255의 값을 라벨의 수(num_labels - 1))로 나누어 그 단계 만큼의 계조 변화로 보인다.
# 라벨 0는 배경이기 때문에 -1 연산이 필요하다.
img = np.hstack((img, labelmap.astype(np.float32)/(num_labels - 1)))
cv2.imshow('Connected components', img)
cv2.waitKey()
cv2.destroyAllWindows()
exit()
"""

#=======================================================================================================
# 실험 2:
# connectedComponentsWithStats() 함수를 이용하여 라벨이 다른 객체에 대해 다른 색상으로 표현하고 그 중심점을 표시한다.
# computes the connected components labeled image of boolean image and also produces a statistics output for each label
# https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gaedef8c7340499ca391d459122e51bef5
# stats[label, COLUMN] where available columns are defined below.
#  0: cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
#  1: cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
#  2: cv2.CC_STAT_WIDTH The horizontal size of the bounding box
#  3: cv2.CC_STAT_HEIGHT The vertical size of the bounding box
#  4: cv2.CC_STAT_AREA The total area (in pixels) of the connected component
# 위 정보로 해당 라벨 객체를 둘러싸는 bounding box를 그릴 수 있다. -> C++ 사례 참조: https://webnautes.tistory.com/823
#=======================================================================================================

#print(cv2.CC_STAT_LEFT); print(cv2.CC_STAT_TOP); print(cv2.CC_STAT_WIDTH); print(cv2.CC_STAT_HEIGHT); print(cv2.CC_STAT_AREA); exit(0)


# 단계 1: 이진화한다.
otsu_thr, otsu_mask = cv2.threshold(gray, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 단계 2: 분리된 객체에 대해 라벨을 부여한다. 또한 그 중심 좌표를 반환받는다.
connectivity = 8
output = cv2.connectedComponentsWithStats(otsu_mask, connectivity, cv2.CV_32S)
ret_val, labelmap, stats, centers = output
print('ret_val=', ret_val)                  # number of labels. 배경(0번) 포함.
print(f'labelmap.shape={labelmap.shape}')   # 1채널 영상의 크기
print(f'labelmap.dtype={labelmap.dtype}')   # int32. label을 지정하는 길이.
print(f'stats.shape={stats.shape}')

#centers = np.int0(centers)      # 부동소수를 정수로 변환. 사라질 함수.
centers = np.intp(centers)      # 부동소수를 정수로 변환. 추천함수
print("centers=\n", centers)      # 라벨 별로 (x, y)의 값을 출력한다. 라벨 0은 배경이다.
print("stats=\n",stats)        # stats[label, COLUMN]. 라벨별로 0~5개의 COLUMN이 존재한다.

# 단계 3: 레이블링 결과를 칼라로 표현한다.   읽을 때 1차원으로 읽었음.
#colored = np.full((img.shape[0], img.shape[1], 3), 0, np.uint8)
colored = np.full(img.shape, 0, np.uint8)

for i in range(1, ret_val): # 0번은 배경 색이므로 제외한다.
    #if stats[i][4] > 200:
    if stats[i][cv2.CC_STAT_AREA] > 200:
        # 특정 라벨로 배정된 화소에 라벨에 따라 다른 색상을 배정한다.
        rdm_c = (0, 255*i/ret_val, 255*ret_val/i)     # i에 의해 만들어지는 색
        colored[labelmap == i] = rdm_c
        centroid = (int(centers[i][0]), int(centers[i][1]))     # (x, y)
        #cv2.circle(colored, centroid, 8,(255, 0, 0), cv2.FILLED)
        cv2.circle(colored, centroid, 4, (255, 255, 255), cv2.FILLED)
        cv.putText(colored, str(i), centroid, cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2);

cv2.imshow('colored', colored)
cv2.waitKey()
cv2.destroyAllWindows()





