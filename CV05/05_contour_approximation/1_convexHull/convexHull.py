"""

1. 개요
    영상을 그레이 변환 -> 이진화 -> 컨투어 추출 -> convex hull 변환 과정을 보여 준다.
    convexHull() 함수를 이용해 컨투어를 convex hull로 변환하는 작업을 수행한다.
    isContourConvex() 함수를 이용해 컨투어가 convex hull인지를 판별한다.

2. 절차
    단계 0 : 입력 영상을 읽어들인다. => 화면 1에 출력.
    단계 1 : 입력 영상의 이진화 작업. OTSU 등의 이진화 알고리즘을 활용한다. => 화면 2에 출력
    단계 2 : 객체의 외곽 윤곽선을 찾아 그리고, 객체 번호를 출력하며, convexHull 여부를 판단한다. => 화면 3에 출력
        컨투어의 윤곽선을 녹색으로 그린다.
        윤곽선이 convex Hull이면 그 중점을 찾아 "CX"로 마킹하고, convex Hull의 외곽은 붉은색으로 덧대어 그린다.
    단계 3 : 객체의 외곽 윤곽선을 convexHull로 바꾸고, 객체 번호와 convexHull 여부를 판단한다. => 화면 4에 출력
        검출된 모든 컨투어를 녹색으로 표현한다.
        각 컨투어의 첫 번째 점의 위치에 윤곽선의 번호를 기재한다.
        검출된 모든 컨투어를 convex hull로 변환한다. 단계 4와 다른 점: 이때 returnPoints=True로 수행한다. default.
        각 컨투어가 convex Hull 인지 판단한다.(확인용: 당연히 모두 convex Hull이다.)
        convex Hull이면 그 중점을 찾아 "CX"로 마킹하고,
        convex Hull의 외곽을 빨간색으로 덧대어 그린다.

    단계 4: 단계 3과의 차이점
        검출된 모든 컨투어를 convex hull로 변환할 때 returnPoints=False 옵션을 사용한다.
        hull=cv.convexHull(	points[, hull[, clockwise=False[, returnPoints=True]]]	)
        위 함수에서 returnPoints=False로 설정하여 함수를 호출하면 반환되는 hull의 정보는 입력 points의 index 번호가 된다.
        본 예제는 이렇게 반환 받은 hull 정보로 윤곽선을 그려보는 연습 프로그램이다.
        좀 까다롭습니다.


4. 주요 함수 2개 소개

1) hull=cv.convexHull(	points[, hull[, clockwise[, returnPoints]]]	)
Input Parameters
    points:	Input 2D point set, stored in std::vector or Mat.
    hull: Output convex hull.
        It is either an integer vector of indices or vector of points.
        In the first case, the hull elements are 0-based indices of the convex hull points in the original array
        (since the set of convex hull points is a subset of the original point set).
        In the second case, hull elements are the convex hull points themselves.
    clockwise: Orientation flag. If it is true, the output convex hull is oriented clockwise.
        Otherwise, it is oriented counter-clockwise. The assumed coordinate system has its X axis pointing to the right, and its Y axis pointing upwards.
    returnPoints: Operation flag. In case of a matrix, when the flag is true,
        the function returns convex hull points.
        Otherwise, it returns indices of the convex hull points.
        When the output array is std::vector, the flag is ignored,
        and the output depends on the type of the vector:
        std::vector<int> implies returnPoints=false, std::vector<Point> implies returnPoints=true.
Return Value
        hull: Output convex hull. 입력에 사용되었던 파라미터.
Function
    Finds the convex hull of a point set.
    The function cv::convexHull finds the convex hull of a 2D point set
    using the Sklansky's algorithm [193]
    that has O(N logN) complexity in the current implementation.


2) retval	=	cv.isContourConvex(	contour	)
Input Parameters
    contour: Input vector of 2D points, stored in std::vector<> or Mat
Return Value
    True or False
Function
    Tests a contour convexity.
    The function tests whether the input contour is convex or not.
    The contour must be simple, that is, without self-intersections.
    Otherwise, the function output is undefined.



"""
import cv2, random
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


Path = '../../../CV04/'

#Name = 'hammer.jpg'              # 이 영상은 지나치게 커서 img = cv2.pyrDown(img) 줄일 필요가 있다.
#Name = 'CONTOUR_TEST_ORG.jpg'       # 바깥 테두리가 검은 색
#Name = 'bullseye.jpg'
#Name = 'rects.png'
#Name = 'shapes_r.png'
#Name = 'circles.jpg'
#Name = '10.jpg'
Name = 'drawing1.png'        # convex hull: none
Name = 'drawing2.png'       # convex hull: none
Name = 'drawing3.png'        # convex hull: 2
#Name = 'drawing7.png'
#Name = 'drawing8.png'
#Name = 'BnW.png'                  # Path = '../data/'
#Name = 'bw.png'                    # Path = '../data/'
#Name = 'shapes_sizes.png'
#Name = 'world_map.png'

Name = 'drawing4.png'       # 미션용의 그림. convex hull: none

Name = 'drawing6.png'      # 사례1 contours 자체에 2개의 convex hull 도형 존재
#Name = 'drawing5.png'       # 사례2 contours 자체에는 convex hull 없음
#Name = 'drawing9.png'       # 사례3

def centroid(moments):
    """Returns centroid based on moments"""

    x_centroid = round(moments['m10'] / moments['m00'])
    y_centroid = round(moments['m01'] / moments['m00'])
    return x_centroid, y_centroid

def get_position_to_draw(text, point, font_face, font_scale, thickness):
    """Gives the coordinates to draw centered"""

    text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
    text_x = point[0] - text_size[0] / 2
    text_y = point[1] + text_size[1] / 2
    return round(text_x), round(text_y)

def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


#=======================================================================================================
# 단계 0 : 입력 영상을 읽어들인다. => 화면 1에 출력.
# 실험 영상은 이진화가 용이한 것을 선택하는 것이 분석에 도움이 된다.
#=======================================================================================================
FullName = Path + Name
img = cv.imread(FullName)   # 편의상 모노 영상도 3채널로 읽은 것으로 가정한다.
assert img is not None, "Failed to load image file:"

fig = plt.figure(figsize=(8, 7))
fig.patch.set_facecolor('silver')
show_img_with_matplotlib(img, f"Step0: input image", 1)

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv2.imshow("input:"+str(Name), img)
#cv2.imshow("gray", gray)
#cv2.waitKey()



# ------------------------------------------------------------------------------------------------------
# 윤곽선을 찾으려면 이진영상을 구해야 한다.
# 단계 1 : 입력 영상의 이진화 작업. OTSU 등의 이진화 알고리즘을 활용한다. => 화면 2에 출력
# ------------------------------------------------------------------------------------------------------
otsu_thr, imgBin = cv2.threshold(gray,  # otsu_thr = 연산된 임계치. , otsu_mask = 바이너리 영상
                                    -1,     # 원래 임계값을 지정하는 것인데  오츠는 자동 연산하기 때문에 활용되지 않는다.
                                    255,    # 임계값 보다 클 경우 배정될 값. 나머지는 0이된다.
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(f"Step1: Otsu Threhold value={int(otsu_thr)}")

canvas = cv2.cvtColor(imgBin, cv2.COLOR_GRAY2BGR)
show_img_with_matplotlib(canvas, f"Step1: imgBin by Otsu thresholding={otsu_thr}", 2)
canvas_cntr = canvas.copy()     # contour을 표기하기 위한 영상
canvas_hull = canvas.copy()     # convex hull을 표기하기 위한 영상

# ------------------------------------------------------------------------------------------------------
# 단계 2 : 객체의 외곽 윤곽선을 찾아 그리고, 객체 번호를 출력하며, convexHull 여부를 판단한다. => 화면 3에 출력
#   검출된 모든 컨투어를 녹색으로 표현한다.
#   각 컨투어의 첫 번째 점의 위치에 윤곽선의 번호를 기재한다.
#   컨투어가 convex Hull 인지 판단한다.
#   convex Hull이면 그 중점을 찾아 "CNVX"로 마킹한다.
#   convex Hull의 외곽은 붉은색으로 덧대어 그린다.
# ------------------------------------------------------------------------------------------------------
# 정해진 MODE의 hierarchy를 갖는 윤곽선을 추출한다.
str_MODE = "cv2.RETR_TREE"
#str_MODE = "cv2.RETR_CCOMP"
#str_MODE = "cv2.RETR_LIST"
#str_MODE = "cv2.RETR_EXTERNAL"
MODE = eval(str_MODE)

str_mthd = "cv2.CHAIN_APPROX_NONE"
#str_mthd = "cv2.CHAIN_APPROX_SIMPLE"
#str_mthd = "cv2.CHAIN_APPROX_TC89_L1"
str_mthd = "cv2.CHAIN_APPROX_TC89_KCOS"
approx_mthd = eval(str_mthd)

print(f"\nStep2: Contour detection scheme:\n{str_mthd}, {str_MODE}")

plt.suptitle(f"{Name}: Contours & Convex Hull\nApprox.={str_mthd}, "
             f"MODE={str_MODE}", fontsize=10, fontweight='bold')


contours, hierarchy = cv2.findContours(image=imgBin, mode=MODE, method=approx_mthd)
print(f'Number of detected contours={len(contours)}')

cnvx_cnt = 0    # convex hull의 총 개수
for i, contour in enumerate(contours):
    cv2.drawContours(canvas_cntr, [contour], -1, (0, 255, 0), 5)  # 모든 컨투어는 두꺼운 녹색으로 표현
    x, y = contour[0][0]  # 첫 번째 점에 컨투어 번호를 적는다.
    cv2.putText(canvas_cntr, str(i), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cnvx = cv2.isContourConvex(contour)  # 컨투어가 컨벡스가 아니므로 False를 반환할 수도 있다.
    if cnvx == True:    # 그 중 콘투어가 convex hull이면 내부를 cyan 색으로 채운다.
        cnvx_cnt += 1
        print(f'{i}: CONVEX(O), contour.shape={contour.shape}') # This contour is convexHull.
        #cv2.drawContours(canvas_cntr, [contour], -1, (255, 255, 128), -1)
        cv2.drawContours(canvas_cntr, [contour], -1, (0, 0, 255), 2)
        m = cv2.moments(contour)
        center = centroid(m)
        center = get_position_to_draw("CX", center, font_face=cv.FONT_HERSHEY_SIMPLEX, font_scale=0.5, thickness=1)
        cv2.putText(canvas_cntr, "CX", center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        print(f'{i}: CONVEX(X), contour.shape={contour.shape}') # This contour is not convexHull

show_img_with_matplotlib(canvas_cntr, f"Step2: Contours={len(contours)}, cnvxHull={cnvx_cnt}", 3)


#"""
# ------------------------------------------------------------------------------------------------------
# 단계 3 : 객체의 외곽 윤곽선을 convexHull로 바꾸고, 객체 번호와 convexHull 여부를 판단한다. => 화면 4에 출력
#   검출된 모든 컨투어를 녹색으로 표현한다.
#   각 컨투어의 첫 번째 점의 위치에 윤곽선의 번호를 기재한다.
#   검출된 모든 컨투어를 convex hull로 변환한다. 실험 2와 다른 점: 이때 returnPoints=True로 수행한다. default.
#   각 컨투어가 convex Hull 인지 판단한다.(확인용: 당연히 모두 convex Hull이다.)
#   convex Hull이면 convex Hull의 그 중점을 찾아 "CNVX"로 마킹하고,
#       convex Hull의 외곽을 빨간색으로 그린다.
# ------------------------------------------------------------------------------------------------------
print('\nStep3: All contours are converted to Convex Hulls.')
# 모든 contour을 convexHull로 변환시킨 후 컨벡스헐인지 검사한다.
cnvx_cnt = 0    # convex hull의 총 개수
for i, contour in enumerate(contours):
    cv2.drawContours(canvas_hull, [contour], -1, (0, 255, 0), 5)  # 모든 컨투어는 두꺼운 녹색으로 표현
    x, y = contour[0][0]  # 첫 번째 점에 컨투어 번호를 적는다.
    cv2.putText(canvas_hull, str(i), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cnvx_hull = cv2.convexHull(contour)  # convex hull로 변환한다. returnPoints=True로 수행한다. default.
    cnvx = cv2.isContourConvex(cnvx_hull)  # 컨투어가 컨벡스가 아니므로 False를 반환할 수도 있다.
    if cnvx == True:    # 그 중 콘투어가 convex hull이면 내부를 cyan 색으로 채운다.
        cnvx_cnt += 1
        print(f'{i}: CONVEX(O), cnvx_hull.shape={cnvx_hull.shape}') # This converted contour is found to be convexHull.
        #cv2.drawContours(canvas_hull, [contour], -1, (255, 255, 128), -1)
        cv2.drawContours(canvas_hull, [cnvx_hull], -1, (0, 0, 255), 2)
        m = cv2.moments(cnvx_hull)
        center = centroid(m)
        center = get_position_to_draw("CX", center, font_face=cv.FONT_HERSHEY_SIMPLEX, font_scale=0.7, thickness=1)
        cv2.putText(canvas_hull, "CX", center, cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    else:
        print(f'{i}: CONVEX(X), contour.shape={contour.shape}') # This converted contour is found not to be convexHull

show_img_with_matplotlib(canvas_hull, f"Step3: Cvted Contours={len(contours)}, cnvxHull={cnvx_cnt}", 4)

plt.show()




# ------------------------------------------------------------------------------------------------------
# 단계 4: 도전 주제
# 아래 소스에 @@로 마킹한 것처럼 콘벡스 변환을 할 때
# returnPoints=False로 정하여 반환받는 값을 컨투어의 인덱스가 되게 한다.
# 컨투어의 특정 포인트를 액세스하는 연습을 하는 것이 본 예제의 목표이다.
# 컨투어 정보를 완격히 다루어야 다른 함수를 대할 때 부담이 없다.
# 도전적인 주제이므로 다른 부분을 마스터하고 학습하기를 권장한다.
# ------------------------------------------------------------------------------------------------------
"""
print("\nStep4: Contour to convexHull by indexing")
print("The indexed convexHull is transformed to contour data type. -> new_cnt")
num_ch2 = 0  # total number of convex hull made of hull_index

for contour, i in zip(contours, range(len(contours))):
    cv2.drawContours(canvas_hull, [contour], -1, (0, 255, 0), 5)  # 모든 컨투어는 두꺼운 녹색으로 표현
    x, y = contour[0][0]  # 첫 번째 점에 컨투어 번호를 적는다.
    cv2.putText(canvas, str(i), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    hull_index = cv2.convexHull(contour, returnPoints=False)  # @@ 이 부분이 실험 2의 핵심.
    #cnvx_cnt = cv2.isContourConvex(contour)     # 컨투어가 컨벡스인가?

    # 인덱스로 컨투어를 만든다. 그래야 그림도 그릴 수 있고,
    # isContourConvex()로 컨벡스인지 확인해 볼 수 있다.
    idx = np.squeeze(hull_index) # (N,1) -> (N)
    print(f"{i}: idx.shape={idx.shape}, idx={idx}")
    pts_lst = []
    for k in range(len(idx)):
        m = idx[k]      # m = index for contour
        pts_lst.append(contour[m])
        #print(pts_lst[k])
    # contour 데이터 형으로 변환
    new_cnt = np.array(pts_lst)       # 컨투어 완성
    #print(f"new_cnt.shape={new_cnt.shape}")
    cnvx_cnt2 = cv2.isContourConvex(new_cnt)  # 컨투어가 컨벡스인가?

    if cnvx_cnt2 == True:
        num_ch2 += 1
        print(f'Convex: new_cnt.shape={new_cnt.shape}')
        # 그 중 콘투어가 convex hull이면 내부를 cyan 색으로 표현
        cv2.drawContours(canvas, [contour], -1, (255, 255, 128), -1)
        # convexHull의 외곽을 그린다.
        cv2.drawContours(canvas, [new_cnt], -1, (0, 0, 255), 2)
    else:
        print(f'Not Convex: contour.shape={contour.shape} => hull.shape=', hull_index.shape)

    #x, y = contours[i][0][0]       # 첫 번째 점에 컨투어 번호를 적는다.
    #cv2.putText(canvas, str(i), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

cv2.imshow(Name + f', Step4: ConvexHull={num_ch2} EA', canvas)
cv2.waitKey()

"""