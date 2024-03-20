"""
개요
    영상을 읽어들여, 이진화하고 이를 기반으로 윤곽선을 검출한다.
    검출할 때는 윤곽선의 계층구조(MODE)와 윤곽선의 표현 방법(METHOD)를 지정할 수 있는데
    이중 윤곽선을 이루는 점의 배열이 어떤 순서에 따라 배열되는지를 도시하여 관찰하여 본다.

"""


import cv2, random
import cv2 as cv
import numpy as np

Path = 'd:/work/StudyImages/Images/'
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 data 폴더.


Name = 'CONTOUR_TEST_ORG.jpg'       # 바깥 테두리가 검은 색. 이진 영상과 구분이 안됨
#Name = 'CONTOUR_TEST2.jpg'       # 바깥 테두리가 검은 색. 이진화가 꼭 필요
Name = 'drawing6.png'

#Name = 'bullseye.jpg'
#Name = 'rects.png'
#Name = 'shapes_r.png'
#Name = 'circles.jpg'
#Name = '10.jpg'
#Name = 'drawing1.png'
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
cv2.imshow("step0: input="+Name, img)
cv2.imshow("step0: gray", gray)
#cv2.waitKey()


#=======================================================================================================
# 윤곽선을 찾으려면 이진영상을 구해야 한다.
# 단계 1 : 입력 영상의 이진화 작업. a 혹은 b 방법 중의 하나를 선택하여 적용한다.
# 방법 (a) : OTSU 등의 이진화 알고리즘을 활용한다. 임계값을 이용한 이진화가 용이한 영상에 적용.
# 검토 요망 - adaptiveThreshold
#=======================================================================================================
#"""
otsu_thr, imgBin = cv2.threshold(gray,  # otsu_thr = 연산된 임계치. , otsu_mask = 바이너리 영상
                                    -1,     # 원래 임계값을 지정하는 것인데  오츠는 자동 연산하기 때문에 활용되지 않는다.
                                    255,    # 임계값 보다 클 경우 배정될 값. 나머지는 0이된다.
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("step1: imgBin by thresholding", imgBin)

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
#MODE = cv2.RETR_TREE        # =3.  모든 윤곽선을 수직적 관계로 계층 구조화
MODE = cv2.RETR_CCOMP       # =2. 2개 레벨로 구조화: contour/hole
#MODE = cv2.RETR_LIST       # =1. 계층구조는 없고 모두 추출
MODE = cv2.RETR_EXTERNAL   # =0. 맨 바깥의 윤곽선만 추출

METHOD = cv2.CHAIN_APPROX_NONE          # =1
#METHOD = cv2.CHAIN_APPROX_SIMPLE        # =0
#METHOD = cv2.CHAIN_APPROX_TC89_L1
#METHOD = cv2.CHAIN_APPROX_TC89_KCOS     # =4.
method = METHOD

print(f"MODE={MODE}, METHOD={METHOD}")

# 컨투어를 검출한다.
contours, hierarchy = cv2.findContours(image=imgBin, mode=MODE, method=METHOD)

#=======================================================================================================
# 단계 3 : 원본 영상(mono gray)에 윤곽선 윤곽선 번호와 윤곽선을 그린다.
#     이때, 아래처럼 [contour(i)]를 사용하고 contourIdx=-1로 모두 그리게 하는 방법도 있다.
#     cv2.drawContours(img2, [contours[i]], -1, color=(255, 0, 0), thickness=2)
#=======================================================================================================
gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
img2 = gray_3ch.copy()
# 일단 노란색으로 모든 컨투어를 바탕에 그려 넣고 시작한다.
cv2.drawContours(img2, contours, -1, (0, 255, 255), 2)

# contourIdx를 윤곽선 번호를 지정하여 컨투어를 magenta 색으로 하나씩 그린다.
# 이때 컨투어의 번호를 각 컨투어의 첫 번째 점의 위치에 출력한다.
for i in range(len(contours)):
    cv2.drawContours(img2, contours, i, (255, 0, 255), 2)
    pts = contours[i]; x, y = pts[0][0]; point = (x,y)  # 첫번 째 점에 컨투어 번호를 적는다.
    cv.putText(img2, str(i), point, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv.putText(img2, str(i), point, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
cv2.imshow("step3: contours of " + Name, img2)
#cv2.waitKey()

#=======================================================================================================
# Mission:
#   이진 영상에서 컨투어를 검출하고, 해당 컨투어의 첫 번째 점에 컨투어의 번호를 표기한다. 
#   그리고, 각 컨투어의 시작점과 종료점을 색상의 변화(빨간색 --> 노란색)로 표기하시오.
# 결과:
#   contour는 점의 위치정보가 반시계 방향으로 배열되고,
#   hole은 점의 위치정보가 시계 방향으로 배열되는 것이 관찰되었다.
#=======================================================================================================
print('\nStep 4 -----')
# contours 형의 자료형과 길이를 확인한다.
print('type(contours)=', type(contours))        # <class 'list'>
n_contours = len(contours)
print('Number of total contours = ', n_contours)

mark_num = 20  # 최대 이 개수 미만으로 윤곽선의 점을 표현된다.
print(f"각 윤곽선은 최대 'mark_num = {mark_num}'개 미만으로 표시됩니다.")
print("빨간색 점으로 시작해서 노란색 점으로 끝납니다. 그 중간의 점들은 두 색의 혼합색으로 표현됩니다.")
# 모든 윤곽선에 대해 특정 모서리점(최대 mark_num개 이하)의 좌표를 출력하고 작은 원으로 표시한다.
# 예를 들어 윤곽선 n=3일 경우, 시작점, 1/n 지점, 2/n 지점)
# 이때 시작점에 윤곽선의 번호를 출력한다.

# 윤곽선상의 n개의 대표되는 점을 찍고 그 중 시작점에 윤곽선 번호를 출력한다.
img2 = gray_3ch.copy()
for i in range(n_contours):
    print('For contour {}...'.format(i))
    print('\ttype(contour[{}])={}'.format(i, type(contours[i])))    # <class 'numpy.ndarray'>. i번째 윤곽선의 타입
    print('\tcontours[{}].shape={}'.format(i, contours[i].shape))   # (len(contours[0]),  1, 2). 여기서 2는 (x,y)를 의미.
    print('\tlen(contours[{}])={}'.format(i, len(contours[i])))     # i번째 윤곽선을 구성하기 위한 모서리 점들의 개수.
    pts = contours[i]
    print('\tpts = contours[{}]: type(pts)={}'.format(i, type(pts)) )                     # <class 'numpy.ndarray'>
    print('\tnumber of points: len(pts)=', len(pts))        # i번째 윤곽선을 구성하기 위한 모서리 점들의 개수.

    if len(pts) <= mark_num:
        samples = range(len(pts))
    else:
        samples = np.linspace(start=0, stop=len(pts), num=mark_num)  # mark_num개의 등간격으로 나눈다. stop까지 포함.
        samples = list(map(int, samples))
        samples = samples[0:-1]     # stop까지 포함되어기 때문에 맨 마지막 지점을 빼야 됨
        print(samples)

    for k in samples:  # 총 n개의 점에 대한 정보를 출력하고자 한다. range(초기값, 종료값, 증분)
        #print('\t\tpts[{}][0]={}'.format(k, pts[k][0]))
        x, y = pts[k][0]; point = (x,y)
        #print('(x,y)=', point)
        # (B,G,R) 중 G의 값이 K가 증가함에 따라 커져서 최대 255까지 된다. 그러면 R=255이므로 노란 색이 만들어 질 것이다.
        # 색상을 관찰해 보면 적색에서 노란색으로 컨투어가 구성됨을 알 수 있다. -> counter-clockwise
        cv2.circle(img2, center=point, radius=5, color=(0, int(255 * k / (len(pts)-1) + 0.5), 255), thickness=-1)
        if k == 0:
            cv.putText(img2, str(i), point, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("step4: "+ Name + f", MODE={MODE}, METHOD={METHOD}, mark_num={mark_num}: RtoY, ", img2)
cv2.waitKey()
