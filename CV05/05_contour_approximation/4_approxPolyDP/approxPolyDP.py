"""

개요: approxPolyDP() 함수를 이용해 컨투어를 근사화한다.
    이진 영상으로 만들어 컨투어를 구한다.
    트랙바를 2개 설치한다.
        입실론 제어: 근사화를 얼마나 많이 할지. 많이 할수록 점의 수가 줄어든다.
        컨투어 선택: 근사화하고자 하는 컨투어를 선택한다.
    2개의 트랙바를 제어하면서 선택한 결과에 따라 출력되는 결과를 보인다.




approxCurve	=cv.approxPolyDP(curve, epsilon, closed[, approxCurve]	)
Input Parameters
    curve: Input vector of a 2D point stored in std::vector or Mat
    epsilon: Parameter specifying the approximation accuracy.
        This is the maximum distance between the original curve and its approximation.
    closed: If true, the approximated curve is closed (its first and last vertices are connected).
        Otherwise, it is not closed.
Output Parameters
    approxCurve:Result of the approximation. The type should match the type of the input curve.
function
    Approximates a polygonal curve(s) with the specified precision.
    The function cv::approxPolyDP approximates a curve or a polygon with another curve/polygon
    with less vertices so that the distance between them is less or equal to the specified precision.
    It uses the Douglas-Peucker algorithm http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm


retval = cv.contourArea(contour[, oriented])
   contour: Input vector of 2D points (contour vertices), stored in std::vector or Mat.
   oriented: Oriented area flag. If it is true, the function returns a signed area value,
       depending on the contour orientation (clockwise or counter-clockwise).
   Using this feature you can determine orientation of a contour by taking the sign of an area.
   By default, the parameter is false, which means that the absolute value is returned.


"""


import cv2, random
import cv2 as cv
import numpy as np


Path = '../../data/'
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
#Name = 'shapes_r.png'
#Name = 'circles.jpg'
Name = '10.jpg'                   # 방법 4의 출력이 도움이 되는 사례.
#Name = 'drawing1.png'
#Name = 'drawing2.png'
#Name = 'drawing3.png'
#Name = 'drawing4.png'          # 추천 2
Name = 'drawing5.png'           # 추천 1
#Name = 'lightening.png'

#Name = 'BnW.png'                  # Path = '../data/'
#Name = 'bw.png'                    # Path = '../data/'


#=======================================================================================================
# 단계 0 : 입력 영상을 읽어들인다.
# 실험 영상은 이진화가 용이한 것을 선택하는 것이 분석에 도움이 된다.
#=======================================================================================================
FullName = Path + Name
img = cv.imread(FullName)   # 편의상 모노 영상도 3채널로 읽은 것으로 가정한다.
assert img is not None, "Failed to load image file:"
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv2.imshow("input:"+str(Name), img)
#cv2.imshow("gray", gray)
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
# 함수 선언부
# 트랙바로 epsilon의 변화를 입력받아 이에 따라 다각형 근사화를 수행한다.
# epsilon의 값이 커짐에 따라 허용되는 근사화의 범위가 커지므로 모서리 점의 수는 줄어든다.
# 대신 도형은 점점 원본과 오류가 많아진다.
#=======================================================================================================
def trackbar_callback(epsilon):
    global img2
    # print('epsilon=', epsilon)
    approx = cv2.approxPolyDP(contour, epsilon, True)  # False
    print('type(approx)=', type(approx) )       # <class 'numpy.ndarray'>
    print('approx.shape=', approx.shape)        # (모서리의 수, 1, 2)
    print('len(approx)=', len(approx))  # 단순화된 윤곽선을 표기하기 위한 모서리(vertex)점의 개수
    img2 = np.copy(color)

    # approx는 모서리 점들의 ndarray이다.
    # contour 함수로 그림을 그리려면 list형으로 만들어야 한다.
    # approx를 contours처럼 윤곽선 데이터형(list)로 만들어야 contour 함수로 그림을 그릴 수 있다.
    #   cntr = [approx]  # contour like. list type
    #   print('type(cnt2)=', type(cntr))  # <class 'list'>
    #   cv2.drawContours(img2, cntr, -1, (255, 0, 255), 2)
    cv2.drawContours(img2, [approx], -1, (255,0,255), 2)

    # 화면의 좌측 상단에 vertex(모서리)의 개수를 출력한다.
    cv.putText(img2, str(len(approx)), (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    for i in range(len(contours)):
        # cv2.drawContours(img2, contours, contourIdx=i, color=(0, 0, 255), thickness=2)
        cv2.drawContours(img2, contours, i, (0, 0, 255), 2)
        pts = contours[i]
        x, y = pts[0][0]
        point = (x, y)  # 첫번 째 점에 컨투어 번호를 적는다.
        cv.putText(img2, str(i), point, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

def contour_selection_callback(num):
    global contour, contours, img2
    contour = contours[num]
    print(f'\nNew contour[{num}]: len(contour)={len(contour)}')
    for i in range(len(contours)):
        # cv2.drawContours(img2, contours, contourIdx=i, color=(0, 0, 255), thickness=2)
        cv2.drawContours(img2, contours, i, (0, 0, 255), 2)
        pts = contours[i]
        x, y = pts[0][0]
        point = (x, y)  # 첫번 째 점에 컨투어 번호를 적는다.
        cv.putText(img2, str(i), point, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        #cv2.imshow("step4(a): contours[i] on " + Name, img2)
        #cv2.waitKey()


#=======================================================================================================
# 단계 2 :
# 객체의 외곽 윤곽선을 찾고 contours[0]의 모서리 개수를 출력한다.
# 또한 윤곽선을 color 화면에 그린다.
#=======================================================================================================

# 편의상 외곽 윤곽선만 고려 대상으로 한다.
contours, hierarchy = cv2.findContours(imgBin, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
print('Number of total contours = ', len(contours) )# 현재의 프로그램은 여러개가 있다고 해도 그중 0번 윤곽선 1개만 지원함.
contour = contours[0]
print('len(contour)=', len(contour))        # 윤곽선을 표기하기 위한 모서리(vertex)점의 개수
color = cv2.cvtColor(imgBin, cv2.COLOR_GRAY2BGR)
cv2.drawContours(color, contours, -1, (0, 255, 0), 3)        # contour은 초록색으로 표현

#=======================================================================================================
# 단계 3 :
# contour가 녹색으로 표현된 영상(img2)을 화면에 출력한다.
# 트랙바를 설치하고 exc 키를 입력할 때까지 무한 루프를 수행한다.
#=======================================================================================================

cv2.namedWindow('TrackBar GUI')
img2 = np.copy(color)

# epsilon 값은 0~150까지 설정가능하며 초기값은 0으로 설정한다.
#cv2.createTrackbar('Epsilon', 'TrackBar GUI', 0, 150, lambda epsilon: trackbar_callback(epsilon))
cv2.createTrackbar('Epsilon', 'TrackBar GUI', 0, 150, trackbar_callback)
cv2.createTrackbar('Contour', 'TrackBar GUI', 0, len(contours), contour_selection_callback)
while True:
    cv2.imshow('TrackBar GUI', img2)
    key = cv2.waitKey(3)
    if key == 27:   # esc. key
        break

cv2.destroyAllWindows()

