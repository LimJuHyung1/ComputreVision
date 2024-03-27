"""
모멘트 - 객체(윤곽선, 영상, 점들의 집합)의 특징 기술자

이진 영상 모멘트 vs 컨투어 영상에 대한 모멘트 추출 실험. ==> 중심점을 비교해 본다.

아래에서 3 단계까지는 이진 영상에 대한 영상 모멘트를 구하는 과정이다.
    1 단계: 컬러 영상을 읽고 gray로 변환한다.
    2 단계: 계조 영상에 대해 Binary 변환을 행한다.
    3 단계: 변환된 이진 영상에 대해 'binaryImage=True' 옵션을 적용하여 영상에 대해 모멘트를 구하여
    영상의 중심점(RED)을 표시한다.

4 단계부터는 contour moments를 구하여 각 윤곽선들의 중심점을 마킹을 실시한다.
   4 단계: 윤곽선을 추출하고 윤곽선(Green)을 그린 후, 윤곽선의 번호(Magenta)를 표기한다.
   5 단계: 윤곽선을 입력으로 한 모멘트기반의 중심점에 점(Red)과 번호(Yellow)를 표기한다.

검토 사항 - 'drawing1.png' 사례
    1번 소스에서는 그림에서 각 객체의 중심점을 찾아 평균한 값이
    두 객체를 한 그림에 넣고 객체를 찾은 위치가 같았다.
    그러나 여기서는 여러 객체가 모인 그림의 중심점이
    각각의 객체를 윤곽선을 찾아 그 중심점들을 평균한 값과 같지 않았다.
    ==> 검토 요망
"""

import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Path = '../Images/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
#Path = 'd:/work/StudyImages/Images/'
#Path = ''
Path = ''

Name = 'CONTOUR_TEST_ORG.jpg'       # 바깥 테두리가 검은 색
#Name = 'bullseye.jpg'
Name = 'rects.png'
#Name = 'shapes_r.png'
#Name = 'circles.jpg'
Name = '10.jpg'
#Name = 'drawing10.png'
#Name = 'drawing2.png'
#Name = '4regions.png'
Name = 'drawing1.png'
#Name = '3regions.png'
#Name = 'obj_with_no_hole.jpg'
#Name = 'objs_with_no_holes.jpg'
#Name = 'obj_with_hole.jpg'
#Name = 'objs_with_holes.jpg'

def centroid(moments):
    """Returns centroid based on moments"""

    x_centroid = round(moments['m10'] / moments['m00'])
    y_centroid = round(moments['m01'] / moments['m00'])
    return x_centroid, y_centroid


# ==================================================================================================
# 이진 영상 모멘트 vs 컨투어 영상에 대한 모멘트 추출 실험.
# 아래에서 3 단계까지는 이진 영상에 대한 영상 모멘트를 구하는 과정이다.
# 4 단계부터는 contour moments를 구하는 과정이다.
#   1 단계: 컬러 영상을 읽고 gray로 변환한다.
#   2 단계: 계조 영상에 대해 Binary 변환을 행한다.
#   3 단계: 변환된 이진 영상에 대해 'binaryImage=True' 옵션을 적용하여 영상에 대해 모멘트를 구하여 영상의 중심점(RED)을 표시한다.
#   4 단계: 윤곽선을 추출하고 윤곽선(Green)을 그린 후, 윤곽선의 번호(Magenta)를 표기한다.
#          윤곽선을 입력으로 한 모멘트기반의 중심점에 점(Red)과 번호(Yellow)를 표기한다.
# ==================================================================================================

# ------------ 1 단계: 컬러 영상을 읽고 gray로 변환한다.
FullName = Path + Name
img = cv.imread(FullName)
assert img is not None, "Failed to load image file:"
cv2.imshow("St1: input:"+str(Name), img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)      ## 그레이로 변환
cv2.imshow("St1: gray:"+str(Name), gray)


# ------------ 2 단계: 계조 영상에 대해 Binary 변환을 행한다.

print('\nSt2: For binary image')
## 바이너리를 오츠로 - 밝은 객체가 많으면 검은 것은 배경으로 인식함
otsu_thr, imgBin = cv2.threshold(gray,  # otsu_thr = 연산된 임계치. , otsu_mask = 바이너리 영상
                                    -1,     # 원래 임계값을 지정하는 것인데  오츠는 자동 연산하기 때문에 활용되지 않는다.
                                    255,    # 임계값 보다 클 경우 배정될 값. 나머지는 0이된다.
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("St2: Binary image, imgBin", imgBin)
print('type(imgBin.dtype)=', imgBin.dtype)  # 주의!!: threshold() 함수는 이진 영상을 uint8로 반환한다.



# ------------ 3 단계: 이진 영상으로 간주하는 옵션을 적용하여 영상 모멘트를 구해 영상의 중심점을 구한다.

#  바이너리 영상에 대한 모멘트를 구할 것인지, 그레이 영상에 대한 모멘트를 구할 것인지 결정한다.
flag = False    # gray 영상 사용
flag = True     # binary 영상 사용 <--- 추천
if flag == False:
    input_image = gray_image.copy()
    ttle = 'gray'
else:
    input_image = imgBin.copy()
    ttle = 'binary'


# 영상의 중심점은 RED로 표현한다.
m = cv2.moments(input_image, binaryImage=flag)
center_img = centroid(m)
print(f"\nSt3: flag={flag}: {ttle} image's center=", center_img)    # 영상 전체의 중심점

img2 = cv.cvtColor(input_image, cv.COLOR_GRAY2BGR)
radius = 7; color=(0, 0, 255); thickness = -1   # fill

cv2.imshow(f"St3: {ttle} image moment's center={center_img}", img2)
cv2.waitKey()

print(f'\ntype(m)={type(m)}')
for key, value in m.items():
    print(f'{key}: {value:#.3f}')



# ------------ 4 단계: 윤곽선을 추출하고 윤곽선(Green)을 그린다.
# 각 오브젝트의 평균 센터를 컬러 동심원(노란 바탕에 빨간색)으로 표기해 보자.
# 윤곽선을 입력으로 한 모멘트기반의 중심점에 점(Red)과 번호(Yellow)를 표기한다.

MODE = cv2.RETR_TREE
MODE = cv2.RETR_CCOMP
#MODE = cv2.RETR_LIST
MODE = cv2.RETR_EXTERNAL
contours, hierarchy = cv2.findContours(image=imgBin, mode=MODE, method=cv2.CHAIN_APPROX_NONE)

# 검출된 객체의 총 수를 출력한다.
print('\nSt4: Number of total contours = ', len(contours) )     ## 컨투어의 수를 출력

gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)     # 1채널 모노 그레이 영상을 3채널 모노 그레이 영상으로 바꾼다.
                                                      ## 3채널은 컬러 가능
# image=drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]])
# contourIdx Parameter indicating a contour to draw. If it is negative, all the contours are drawn.
img2 = cv2.drawContours(image=gray_3ch,     # 윤곽선을 그릴 화면. destination image
                       contours=contours, # 윤곽선 정보, 점들의 집합으로 구성. 윤곽선의 총 개수 = len(contours)
                       contourIdx=-1,     # 윤곽선의 번호. 음수이면 모든 윤곽선을 그린다.
                       color=(0, 255, 0),     # 윤곽선의 색상. = (B, G, R)
                       thickness=2)         # 윤곽선의 두께.

for i in range(len(contours)):      # 윤곽선에 번호를 표시한다.
    ## pts - 중심점을 잡아냄
    ## contours 0번은 contour, pts[0][0] == n * m * 2?, point - 튜플로 모음
    pts = contours[i]; x, y = pts[0][0]; point = (x, y)  # 첫번 째 점에 컨투어 번호를 적는다.

    ## putText - 튜플로 들어와야 함
    cv.putText(img2, str(i), point, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

#cv2.imshow("Step 4: contours", gray_3ch)
#cv2.waitKey()

# 윤곽선을 입력으로 한 모멘트기반의 중심점에 점(Red)과 번호(Yellow)를 표기한다.
i = 0
np_center = np.array([0, 0])    # ndarray 형 좌표. 좌표 값 누적 연산을 위해 선언..
for c in contours:      # 윤곽선 별로 모멘트를 계산한다.
    m = cv2.moments(c)
    center = centroid(m)
    np_center += np.array(center )  # 각 컨투어들의 센터를 구해서 더함?
    print('cn=', i, ' center=', center) # 현재 센터를 출력
    radius = 5; color = (0, 0, 255); thickness = -1   ## fill - 빨간색 점을 칠함
    cv2.circle(gray_3ch, center, radius, color, thickness)
    cv.putText(img2, str(i), center, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    i += 1

# 센터들의 평균을 구함
avg_centr = np_center/i

xx = int(avg_centr[0])
yy = int(avg_centr[1])
center_tuple = (xx, yy)

# 영상 모멘트의 중심을 컬러 동심원(노란 바탕에 빨간색)으로 표기해 보자.
cv2.circle(gray_3ch, center_tuple, 10, (0, 0, 255), -1)       # 큰 원 - 빨강
cv2.circle(gray_3ch, center_tuple, 8, (0, 255, 255), -1)      # 중간원 - 노랑
cv2.circle(gray_3ch, center_tuple, 4, (0, 0, 255), -1)        # 작은 원 - 빨강

print("contour moments's centers=\n", avg_centr)
print(f"image moment center using {ttle} image  ={center_img}")
cv2.imshow("St4: contours moments's centers on gray", gray_3ch)
cv2.waitKey()


# 검토사항
# 각 객체의 중심점(영상 모멘트)의 평균을 구해 보았는데 이진 영상의 중심점(컨투어 중심점)과 일치하지 않는다.
# 이것은 1_center_of_graphic_img.py의 결론과 다르다. 이유는? =>
# 추정:
# Hole이 없는 객체가 1개이면 일치한다.
# 그러나 객체가 많아지면 고려할 점이 생길 것으로 예상된다.
# (어쩌면 Hole이 있어서 비슷한 문제가 생길 것이다.)
# 1) 객체가 많아지면 각 객체의 중심이 똑같은 가중치를 가지면 안될 것으로 본다.
#   면적의 비율만큼 가중치를 가져야 한다.
#   예를 들어 극단적으로 작은 크기의 객체(예를 들어 점의 크기)가 한 쪽 구석에 있다고 가정해 보자.
#   이것 때문의 중심점의 위치가 크게 변하게 되겠는가?
# 2) 또한, hole이 있으면 hole을 뺀 무게 중심을 구하는 방안이 검토되어야 한다.
# 3) 그럼 왜 예제 1은 통과되었을까?
#   ans: 객체가 1개만 존재하는 영상을 모멘트를 구하고, 이런 작업을 2회에 걸쳐 반복해 평균했기 때문입니다.
# 4) 여러 객체가 있을 때 그 중심 모멘트는 영상 모멘트로 구하는 것이 맞다고 봅니다.
#    이런 목적으로 사용된 컨투어 모멘트는 잘 못 적용된 것입니다.
# 미션: 위의 추정 내용을 바탕으로 여러분이 직접 코딩을 통해 검토해 보세요..


