"""
개요
    영상을 읽어들여, 이진화하고 이를 기반으로 윤곽선을 검출한다.
    검출할 때는 윤곽선의 계층구조(MODE)와 윤곽선의 표현 방법(METHOD)를 지정할 수 있는데
    이들에 따라 달라지는 출력 결과를 관찰한다.

"""


import cv2, random
import cv2 as cv
import numpy as np

Path = 'd:/work/StudyImages/Images/'
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 data 폴더.


Name = 'CONTOUR_TEST_ORG.jpg'       # 바깥 테두리가 검은 색. 이진 영상과 구분이 안됨
#Name = 'CONTOUR_TEST2.jpg'       # 바깥 테두리가 검은 색. 이진화가 꼭 필요
#Name = 'drawing6.png'

#Name = 'bullseye.jpg'
#Name = 'rects.png'
#Name = 'shapes_r.png'
#Name = 'circles.jpg'
#Name = '10.jpg'                   # 방법 4의 출력이 도움이 되는 사례.
#Name = 'drawing1.png'                   # 방법 4의 출력이 도움이 되는 사례.
#Name = 'drawing1.png'
#Name = 'drawing2.png'
#Name = 'drawing3.png'
#Name = 'drawing4.png'
#Name = 'drawing5.png'
#Name = 'drawing7.png'
#Name = 'drawing8.png'
#Name = 'drawing9.png'



#=======================================================================================================
# 단계 0 : 입력 영상을 읽어들인다.
# 실험 영상은 이진화가 용이한 것을 선택하는 것이 분석에 도움이 된다.
#=======================================================================================================
FullName = Path + Name
img = cv.imread(FullName)   # 편의상 모노 영상도 3채널로 읽은 것으로 가정한다.
assert img is not None, "Failed to load image file:"
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)   # 컬러 그림을 사용할 경우 대비하여 변환
cv2.imshow("input: "+Name, img)
cv2.imshow("gray", gray)
cv2.waitKey()
cv2.destroyAllWindows()

#=======================================================================================================
# 윤곽선을 찾으려면 이진영상을 구해야 한다.
# 단계 1 : 입력 영상의 이진화 작업. a 혹은 b 방법 중의 하나를 선택하여 적용한다.
#=======================================================================================================

# 방법 (a) : OTSU 등의 이진화 알고리즘을 활용한다. 임계값을 이용한 이진화가 용이한 영상에 적용.
# 검토 요망 - adaptiveThreshold

#"""
otsu_thr, imgBin = cv2.threshold(gray,  # otsu_thr = 연산된 임계치. , otsu_mask = 바이너리 영상
                                    -1,     # 원래 임계값을 지정하는 것인데  오츠는 자동 연산하기 때문에 활용되지 않는다.
                                    255,    # 임계값 보다 클 경우 배정될 값. 나머지는 0이된다.
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("step1: imgBin by thresholding", imgBin)
cv2.waitKey()
#"""

# 방법 (b) : 캐니 에지를 사용하여 이진화한다. 임계값을 이용한 이진화 적용이 용이하지 않을 때 적용.
# 윤곽선이 에지를 중심으로 2회 검출될 수 있으니 비추전합니다.!!!
#imgBin = cv2.Canny(gray, 60, 180)
#cv2.imshow("imgBin by Canny edge", imgBin)
#cv2.waitKey()


#=======================================================================================================
# 단계 2 : 객체의 외곽 윤곽선을 찾고, 이를 그림으로 보이고, 계층구조 정보를 출력한다.
#=======================================================================================================
print('\nStep 2 -----')
# 1) MODE와 METHOD를 지정한다.
MODE = cv2.RETR_TREE        # =3.  모든 윤곽선을 수직적 관계로 계층 구조화
#MODE = cv2.RETR_CCOMP       # =2. 2개 레벨로 구조화: contour/hole
#MODE = cv2.RETR_LIST       # =1. 계층구조는 없고 모두 추출
#MODE = cv2.RETR_EXTERNAL   # =0. 맨 바깥의 윤곽선만 추출

METHOD = cv2.CHAIN_APPROX_NONE          # =1
METHOD = cv2.CHAIN_APPROX_SIMPLE        # =0
METHOD = cv2.CHAIN_APPROX_TC89_L1
METHOD = cv2.CHAIN_APPROX_TC89_KCOS     # =4.
method = METHOD

print(f"MODE={MODE}, METHOD={METHOD}")
# OpenCV 3에서는 3개를 반환
# https://stackoverflow.com/questions/47547221/valueerror-too-many-values-to-unpack-expected-2-cv2
#im2, contours, hierarchy = cv2.findContours(image=imgBin, mode=MODE, method=METHOD)

# 2) 컨투어를 검출하고, 이를 그림으로 출력한다.
contours, hierarchy = cv2.findContours(image=imgBin, mode=MODE, method=METHOD)
#img2 = imgBin.copy()   # 1채널이라 그림을 그릴 수 없음
imgBin_3ch = cv2.cvtColor(imgBin, cv2.COLOR_GRAY2BGR)
img2 = imgBin_3ch.copy()
cv2.drawContours(image=img2,          # 윤곽선을 그릴 화면. 동시에 입력영상이면서 반환 받을 영상임.
                 contours=contours, # 윤곽선 정보, 점들의 집합으로 구성. 윤곽선의 총 개수 = len(contours)
                 contourIdx=-1,     # 윤곽선의 번호. 음수이면 모든 윤곽선을 그린다.
                 color=(0, 255, 0),     # 윤곽선의 색상. = (B, G, R)
                 thickness=2)         # 윤곽선의 두께.
cv2.imshow("step2: contours on binary of " + Name, img2)

# 3) hierarchy 반환 값의 기본 정보를 출력한다.
print('hierarchy=\n', hierarchy)                  # contour 수가 적으면 출력해 볼만 함.
print('type(hierarchy)=', type(hierarchy))      # <class 'numpy.ndarray'>
print('hierarchy.shape=', hierarchy.shape)
print('number of total hierarchy: len(hierarchy)=', len(hierarchy))
# 사례('CONTOUR_TEST_ORG.jpg', MODE=2, METHOD=4): hierarchy.shape= (1, 9, 4) len(hierarchy)= 1

print('type(hierarchy[0])=', type(hierarchy[0]))    # <class 'numpy.ndarray'>
print('hierarchy[0].shape', hierarchy[0].shape, 'len(hierarchy[0])=', len(hierarchy[0]))    #
# 사례('CONTOUR_TEST_ORG.jpg', MODE=2, METHOD=4): hierarchy[0].shape (9, 4) len(hierarchy[0])= 9
print('hierarchy[0][0])=', hierarchy[0][0])     # hierarchy[0, 0] numpy은 이런 식으로 써도 됨.
#print('hierarchy[0][1])=', hierarchy[0][1])

# 4) 각 contour 별로 내부의 hierarchy 정보를 출력한다.
cn=0    # contour number
print('contour 별 계층구조의 값들을 출력한다.')
print("contour num: [Next, Previous, First_Child, Parent]")
for i in range(len(hierarchy[0])):
    print(f"      {cn}: hierarchy[0][{i}]")     # [0][i] = [0,i]
    cn += 1

cn=0
for hier in hierarchy[0]:   # 내부는 4개로 정보로 구성된다.
    #print('cntr={0:2d}: Next={1:2d}, Previous={2:2d}, First_Child={3:2d}, Parent={4:2d}'
    #      .format(cn, hier[0], hier[1], hier[2], hier[3]) )
    print(f'cntr={cn:2d}: Next={hier[0]:2d}, Previous={hier[1]:2d}, First_Child={hier[2]:2d}, Parent={hier[3]:2d}')
    cn += 1

cv2.waitKey()
cv2.destroyAllWindows()


#=======================================================================================================
# 단계 3 : 원본 영상(mono gray)에 윤곽선 윤곽선 번호와 윤곽선을 키를 입력할 때마다 그린다.
#     이때, 아래처럼 [contour(i)]를 사용하고 contourIdx=-1로 모두 그리게 하는 방법도 있다.
#     cv2.drawContours(img2, [contours[i]], -1, color=(255, 0, 0), thickness=2)
#=======================================================================================================
gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
img2 = gray_3ch.copy()
# 일단 노란색으로 모든 컨투어를 바탕에 그려 넣고 시작한다.
cv2.drawContours(img2, contours, -1, (0, 255, 255), 2)

# contourIdx를 윤곽선 번호를 지정하여 키 입력을 받을 때마다 컨투어를 magenta 색으로 하나씩 그린다.
# 이때 컨투어의 번호를 각 컨투어의 첫 번째 점의 위치에 출력한다.
for i in range(len(contours)):
    cv2.drawContours(img2, contours, i, (255, 0, 255), 2)
    pts = contours[i]; x, y = pts[0][0]; point = (x,y)  # 첫번 째 점에 컨투어 번호를 적는다.
    cv.putText(img2, str(i), point, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv.putText(img2, str(i), point, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("step3: contours of " + Name, img2)
    cv2.waitKey()

