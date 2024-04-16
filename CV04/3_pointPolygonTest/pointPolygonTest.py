"""
프로그램 동작 설명
    1. 창을 열어 원본 영상을 보인다.
    2. 창을 열어 이진 영상을 보인다.
    3. 트랙바 창을 열어 트랙 바로 결정한 콘투어 번호의 객체를 초록색으로 선택한다.
       1) 이때 임의 위치에 마우스 왼쪽 버튼을 클릭하면 해당 위치에 채워진 작은 원이 그려지며,
       그 점과 선택된 컨투어와의 거리가 화면에 출력된다.
       2) +이면 객체 내부의 점(초록색으로 표기)을 클릭한 것이고(초록색),
       -이면 객체 외부의 점(붉은색으로 표기)을 클릭한 것이다.
       3) 트랙 바로 새로운 객체를 다시 선택할 수 있다.
       4) esc 키를 입력하면 종료한다.


retval=cv.pointPolygonTest(contour, pt, measureDist)
이 함수는 다음 둘 중의 하나를 반환한다.
    1) 지정한 점이 윤곽선과의 가장 가까운 거리를 반환한다.
    measureDist =True
    pt와 가장 가까운 contour와의 부호가 붙은 거리를 반환한다. (안에 있으면 +, 밖에 있으면 -)

    2) 지정된 점이 윤곽선 안에 있는지, 밖에 있는지, 윤곽선상에 존재하는지 반환한다.
    measureDist = False
    pt가 contour안에 있으면 +1, 밖에 있으면 -1, 선상에 위치하면 0을 반환한다.


Function
    Performs a point-in-contour test.
    The function determines whether the point is inside a contour, outside, or lies on an edge
    (or coincides with a vertex).
    When measureDist=True, the return value is a signed distance between the point and the nearest contour edge.
    Otherwise, the return value is +1, -1, and 0, respectively.

Input Parameters
    contour: Input contour.
    pt:	Point tested against the contour.
    measureDist:	If true, the function estimates the signed distance from the point
        to the nearest contour edge.
        Otherwise, the function only checks if the point is inside a contour or not.

Return Value
    When measureDist=false, the return value is +1, -1, and 0, respectively.
    It returns positive (inside), negative (outside), or zero (on an edge) value, correspondingly.
    hen measureDist=True, it returns a signed distance between the point and the nearest contour edge.

미션
    아래 링크에 있는 샘플 영상을 생성할 수 있는 프로그램을 작성하시오.
        https://docs.opencv.org/4.1.0/d3/dc0/group__imgproc__shape.html#ga1a539e8db2135af2566103705d7a5722
    방법 예시
        원본 영상과 같은 별도의 창을 생성하여 그 창의 모든 점에 대하여
        트랙바로 선택한 컨투어에 대한 최단 거리를 색상의 강도로 표현한다.
        matplotlib의 jetcolormap이 적용 가능할 것으로 생각됨.

"""


import cv2, random
import cv2 as cv
import numpy as np


Path = '../../data/'                           # 현재 폴더에 그림이 있는 경우 이것을 사용.
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
#Name = '10.jpg'                   # 방법 4의 출력이 도움이 되는 사례.
Name = 'drawing1.png'
Name = 'drawing2.png'
Name = 'drawing3.png'
Name = 'drawing4.png'
Name = 'drawing5.png'
Name = 'drawing6.png'
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
# 단계 2 :
# 또한 윤곽선을 color 화면에 그린다.
#=======================================================================================================

contours, hierarchy = cv2.findContours(imgBin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print('len(contours)=', len(contours))
color = cv2.cvtColor(imgBin, cv2.COLOR_GRAY2BGR)
#cv2.drawContours(color, contours, -1, (0, 255, 0), 3)
TITLE='contours: '+Name +" : 'm' to toggle measure, esc. to quit"
cv2.imshow(TITLE, color)
#cv2.waitKey()
#cv2.destroyAllWindows()

# In[17]:


#contour = contours[0]
image_to_show = np.copy(color)
measure = True
print(image_to_show.shape[1])

def mouse_callback(event, x, y, flags, param):
    global contours, image_to_show, Num

    if event == cv2.EVENT_LBUTTONUP:
        distance = cv2.pointPolygonTest(contours[Num], (x, y), measure)
        image_to_show = np.copy(color)
        cv2.drawContours(image_to_show, contours, Num, (0, 255, 0), 3)
        if distance > 0:
            pt_color = (0, 255, 0)      # 테스트하는 점이 다각형 안에 있음을 의미. 초록색
        elif distance < 0:
            pt_color = (0, 0, 255)      # 테스트하는 점이 다각형 밖에 있음을 의미. 붉은색
        else:
            pt_color = (128, 0, 128)    # 테스트하는 점이 다각형 선상에 있음을 의미. 자홍색(magenta)
        cv2.circle(image_to_show, (x, y), 5, pt_color, -1)
        cv2.putText(image_to_show, '%.2f' % distance,
                    (0, image_to_show.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)


def trackbar_callback(num):
    global contours, image_to_show, Num
    image_to_show = np.copy(color)
    Num = num
    cv2.drawContours(image_to_show, contours, Num, (0, 255, 0), 3)

Num = 0
cv2.namedWindow(TITLE)
cv2.setMouseCallback(TITLE, mouse_callback)
cv2.drawContours(image_to_show, contours, 0, (0, 255, 0), 3)        # 일단 0번 윤곽선만 그린다.
cv2.createTrackbar('number', TITLE, 0, len(contours)-1, lambda num: trackbar_callback(num))

while (True):
    cv2.imshow(TITLE, image_to_show)
    k = cv2.waitKey(1)

    if k == ord('m'):
        measure = not measure       # meassure 방식을 토글링한다.
    elif k == 27:
        break

cv2.destroyAllWindows()




