import cv2, random
import cv2 as cv
import numpy as np

path = 'Images/toka.png'

#=======================================================================================================
# 단계 0 : 입력 영상을 읽어들인다.
# 실험 영상은 이진화가 용이한 것을 선택하는 것이 분석에 도움이 된다.
#=======================================================================================================
img = cv.imread(path)   # 편의상 모노 영상도 3채널로 읽은 것으로 가정한다.
assert img is not None, "Failed to load image file:"
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)   # 컬러 그림을 사용할 경우 대비하여 변환
cv2.imshow("input: toka", img)
cv2.imshow("gray", gray)
cv2.waitKey()
cv2.destroyAllWindows()

#=======================================================================================================
# 윤곽선을 찾으려면 이진영상을 구해야 한다.
# 단계 1 : 입력 영상의 이진화 작업. a 혹은 b 방법 중의 하나를 선택하여 적용한다.
#=======================================================================================================

# 원래는 임계값을 따로 설정해 주어야 하지만 오츠 알고리즘을 통해 자동으로 설정되어
# -1로 설정하였다
otsu_thr, imgBin = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow("step1: imgBin by thresholding", imgBin)
cv2.waitKey()

#=======================================================================================================
# 단계 2 : 객체의 외곽 윤곽선을 찾고, 이를 그림으로 보이고, 계층구조 정보를 출력한다.
#=======================================================================================================

MODE = cv2.RETR_TREE        # =3.  모든 윤곽선을 수직적 관계로 계층 구조화
#MODE = cv2.RETR_CCOMP       # =2. 2개 레벨로 구조화: contour/hole
#MODE = cv2.RETR_LIST       # =1. 계층구조는 없고 모두 추출
#MODE = cv2.RETR_EXTERNAL   # =0. 맨 바깥의 윤곽선만 추출

METHOD = cv2.CHAIN_APPROX_NONE          # =1
METHOD = cv2.CHAIN_APPROX_SIMPLE        # =0
METHOD = cv2.CHAIN_APPROX_TC89_L1
METHOD = cv2.CHAIN_APPROX_TC89_KCOS     # =4.
method = METHOD

# 컨투어를 검출하고, 이를 그림으로 출력한다.
contours, hierarchy = cv2. findContours(image=imgBin, mode=MODE, method=METHOD)
imgBin_3ch = cv2.cvtColor(imgBin, cv2.COLOR_GRAY2BGR)
img2 = imgBin_3ch.copy()
cv2.drawContours(image=img2
                 , contours=contours    # 윤곽선의 정보
                 , contourIdx=-1        # 윤곽선의 번호, 음수면 모든 윤곽선을 그린다
                 , color=(200, 30, 200)
                 , thickness=2)
cv2.imshow("step2: contours on binary of toka", img2)
cv2.waitKey()
cv2.destroyAllWindows()
