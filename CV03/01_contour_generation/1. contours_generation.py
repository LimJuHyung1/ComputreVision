"""
*** 같은 내용이 말이 중복 기술되어 있는 경우도 있습니다.
*** 강조하다보니 그건 것이니, 감안하고 보아주기 바랍니다.

1. 프로그램의 수행 내용
    1. 삼각형, 사각형 각 1개의 윤곽선을 ndarray로 정의하여 이것으로 contours 자료형을 만든다.
       두 개의 윤곽선을 리스트로 만들면 됨.
    2. 정의된 윤곽선 정보로 3가지 관점에서 각각의 화면에 그 내용을 출력한다.
        1) 삼각형을 표현 하는 3개의 점 + 삼각형을 표현 하는 4개의 점= 총 7개의 점을 해당 지점에 점으로 도시한다.
        2) 해당 지점을 서로 선으로 연결하여 두 물체의 윤곽선을 표현한다.
        3) 점과 선을 모두 함께 한 화면에 보인다.
           이때 점과 선의 색상은 랜덤으로 선택하기 때문에 잘 안 보일 수 있다.
2. 개요
    1. 물체(object)의 contour(윤곽선)은 여러 개의 꼭지점들의 집합으로 정의할 수 있다.
        꼭지점과 꼭지점 사이는 직선으로 연결한다는 전제이다.
        삼각형은 3개의 점으로, 4각형은 4개의 점으로 묘사가능하여 그 이상의 다면체는 그 만큼의 꼭지점 수가 필요하다.
        원형의 경우는 더욱 많은 꼭지점으로 구성하면 될 것이다.
    2. 파이썬 방식으로 표현한다면 contour 1 개를 구성하기 위한 점들은 ndarray 구조체로 구성된다.
        contour.shape=(점의_개수, 1, 2). <= 1 은 고정(사용하지 않는 차원. 2는 (x좌표, y좌표) 2개차원
        삼각형 윤곽선의 shape=(3, 1, 2). 사각형 윤곽선의 shape=(4, 1, 2).
    3. 한 영상에서 검출한 물체의 윤곽선들은 한 개의 list 자료 구조로 표현된다.
        contours = [contour1, contour2, ...]
        예: 화면에 삼각형 1개와 사각형 1개 있는 contours=[contour_삼각형, contour_사각형]

* 주의 사항:
    OpenCV 윤곽선 관련 함수는 리스트형으로 정의된 contours 형태여야 한다.
    예) contour 1개의 윤곽선을 그리는 함수를 호출할 때는 [contour] 형태로 호출해야 한다.

3. 정리
 OpenCV에서는 물체(object)의 형체를 나타낼 때 다음 구조체를 사용한다.
 1. contours(윤곽선)은 여러 개의 윤곽선(contour)로 이루어진 list 구조체로 구성한다 .
    즉, contours = [contour, contour, ...]
 2.contour는 여러 점들의 ndarray 구조체로 구성된다.
   보통 점은 (x, y), 2축 좌표면 충분한데 여기서는 용도 불명의 좌표축 1개를 더 추가하여 (1, 2)로 구성한다.
   즉, contour = np.array([x, y], [x, y], [x, y],...])로 정의해도 될 것을
   차원을 1개 늘려서 contour = np.array([[x, y], [x, y], [x, y],...]]) 표시한다는 것이다.
   이때문에 (x, y) 좌표 정보를 액세스 하기 위해서는 다음 기법들이 사용된다.
   1) contours[0][점의_번호, 0]
   2) squeeze = np.squeeze(contour)   # (점의_번호, 1, 2) => (점의_번호, 2).
        squeeze()은 요소의 갯수가 1인 차원을 제거한다.

4. 미션!!:
    squeeze() 함수를 사용하지 말고,
    contours[i][j, 0] 혹은 contour[j, 0] 방식으로 단계2 를 다시 작성하시오.

참고: 아래 프로그램과 거의 유사한 기능을 수행함.
   contours_introduction_2.py

"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import random


# random color generation: 간단하긴 한데 검은색에 가려 안보일 가능성을 배제하지는 못한다.
def random_color():
    r = random.randint(0, 256)  # [0, 256) : 0~255 중의 정수
    g = random.randint(0, 256)
    b = random.randint(0, 256)
    rgb = [r, g, b]
    return rgb

# Converts array to tuple
def array_to_tuple(arr):
    #print(f'before converting=', arr, type(arr))
    # 원래 reshape(n, m)은 (n, m) 2차원 array를 반환한다.
    # reshape(1, -1)은 (1, 끝) 크기의 2차원 array를 반환한다.
    # arr.reshape(1, -1): 1 x 점의 갯수, 2 dimension. numpy ndarry class.
    # arr.reshape(1, -1)[0]: 점의 갯수, 1 dimension, numpy ndarry class.
    #print(f"arr.reshape(1, -1)[0]={arr.reshape(1, -1)[0]}, {arr.reshape(1, -1)[0].shape}")
    # 1차원으로 끝까지 해라(?)
    # arr.reshape(1, -1) - 2차원
    # arr.reshape(1, -1)[0] - 1차원
    arr = tuple(arr.reshape(1, -1)[0]) # 1차원 어레이를 python sequence(열거형) 자료로 반환한다.
    #print(f'after converting=', arr, type(arr))     # <class 'tuple'>
    return arr


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    """ OpenCV  BGR 이다 """
    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    # low : 1, column : 3
    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# ==========================================================================================================
# 메인 프로그램의 시작
# ==========================================================================================================

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 5))
plt.suptitle("Contours introduction", fontsize=14, fontweight='bold')       # 대제목
fig.patch.set_facecolor('silver')       # 여러가지 색깔이 지원됨

# Create the canvas :
#canvas = np.zeros((640, 640, 3), dtype="uint8")        # black image with three channels

# 인조 영상을 만듬
# row : 640, column : 640, channel : 3인 인조 영상
canvas = 50 * np.ones((640, 640, 3), dtype="uint8")    # white image with three channels

# ----------------------------------------------------------------------------------------------------
# 단계 1: 삼각형과 사각형의 꼭지점을 OpenCV 방식으로 정의한다.
# ----------------------------------------------------------------------------------------------------

# 1) 거의 원에 가까운 다각형 1개로 이루어진 contours - 고생하셨습니다
# contours = [np.array(
#    [[[600, 320]], [[563, 460]], [[460, 562]], [[320, 600]], [[180, 563]], [[78, 460]], [[40, 320]], [[77, 180]],
#     [[179, 78]], [[319, 40]], [[459, 77]], [[562, 179]]], dtype=np.int32)]

# 2) 2개의 object로 이루어진 contours
contour1 = np.array([[[300, 150]], [[400, 300]], [[150, 450]]], dtype=np.int32)             # 삼각형. 꼭지점의 위치(x, y)
contour2 = np.array([[[100, 50]], [[500, 150]], [[500, 400]], [[30, 550]]], dtype=np.int32) # 사각형. 꼭지점의 위치(x, y)
contours = [contour1, contour2]
print(f"contour1.shape :{contour1.shape}")  # (3, 1, 2) - 가운데 인수가 1인 이유는 호환성을 위해서 - [[400, 300]]
print(f"contour2.shape :{contour2.shape}")  # (4, 1, 2)
#   [triangle, rectangle]
print("\n1.1 여러 개의 contour가 모인 contours의 관찰합니다.")
print(f"\tcontours: type(contours)={type(contours)}")   # contours은 list 자료로 구성되어 있다. contours = {contour,..}
print(f"\tcontours: len(contours)={len(contours)}")     # contour이 몇 개인지 알아본다.

print("\n1.2 각 contour를 이루는 점의 좌표를 관찰합니다.=> contours 변수로 액세스 합니다.")
for i in range(len(contours)):
    print(f"\n\tcontour num {i}: type={type(contours[i])}, shape={contours[i].shape}")
    num = contours[i].shape[0]      # i번째 contour의 point의 개수
    print(f"\tnumber of points in contours[{i}]={num}")
    for j in range(len(contours[i])):
        print(f"\tpoint {j}:")
        print(f"\t2차원 어레이 반환: contours[{i}][{j}]={contours[i][j]}, "
              f"{contours[i][j].shape}, ndim={contours[i][j].ndim}")
        print(f"\t1차원 어레이 반환: contours[{i}][{j}, 0]={contours[i][j, 0]}, "
              f"{contours[i][j, 0].shape}, ndim={contours[i][j, 0].ndim}")
        print(f"\t\t(x, y) = ", end="")     # (x, y)에 접근하려면 또 한번의 loop가 필요하다.
        for k in range(2):      # access to x, y
            print(contours[i][j, 0, k], end=" ")
        print()

print("\n1.3 각 contour를 이루는 점의 좌표를 관찰합니다.=> contours 내부의 변수로 액세스 합니다.")
for i, contour in enumerate(contours):
    print(f"\n\tcontour num {i}: type={type(contour)}, shape={contour.shape}")
    num = contour.shape[0]      # i번째 contour의 point의 개수
    print(f"\tnumber of points in {i}th contour={num}")
    for j, pt in enumerate(contour):
        print(f"\tpoint {j}:")
        print(f"\t2차원 어레이 반환: {pt}, {type(pt)}, {pt.shape}, ndim={pt.ndim}")
        print(f"\t1차원 어레이 반환: {pt[0]}, {type(pt[0])}, {pt[0].shape}, ndim={pt[0].ndim}")
        #print(f"\t{contours[i][j]}", end="")
    print()



# ----------------------------------------------------------------------------------------------------
# 단계 2: sub fig.1 : 캔버스(검은 바탕화면)에 각 컨투어 꼭지점 좌표에 circle() 함수로 소형 원(점)을 그린다.
# 아직 화면에 출력은 안하고 img 파일에 그린 결과를 저장해 둔다.
# 윤곽선 내의 좌표를 액세스하는데 불편이 따르는데 여기서는 간단히 squeeze() 함수로 처리하였다.
#
# 미션!!: squeeze() 함수를 사용하지 말고,
# contours[i][j, 0] 혹은 contour[j, 0] 방식으로 단계2를 다시 작성하시오.
# ----------------------------------------------------------------------------------------------------
img_points = canvas.copy()  # 첫번째 화면

i = 1       # contour 번호
for contour in contours:
    squeeze = np.squeeze(contour)   # (3, 1, 2) => (3, 2). squeeze()은 요소의 갯수가 1인 차원을 제거한다.
    color = random_color()
    for p in squeeze:
        p = array_to_tuple(p)   # ndarray를 tuple 데이터로 바꾼다. array([x, y]) => (x, y).
        # img_points = (640, 640, 3)
        cv2.circle(img_points, p, 10, color, -1)    # circle 함수에는 list, tuple만 들어가야 함, nparray는 안 됨
    i += 1


# ----------------------------------------------------------------------------------------------------
# 단계 3: sub fig.3 : 캔버스(검은 바탕화면)에 각 컨투어 꼭지점 좌표에 직선으로 연결한다.
# 아직 화면에 출력은 안하고 img 파일에 그린 결과를 저장해 둔다.
# ----------------------------------------------------------------------------------------------------
img_outline = canvas.copy()

# -1: 모든 컨투어들을 한꺼번에 다 그린다. 색상이 같다.
#cv2.drawContours(img_outline, contours, -1, color, thickness=5)
# 1: 1번 컨투어만 그린다.
#cv2.drawContours(img_outline, contours, 1, color, thickness=5)

"""
# 컨투어마다 색상이 다르다.
for cnt in contours:
    #print([cnt])    # 어레이를 list에 넣어 넘겨 주는 것. 즉, 1개만 넘긴다.
    #print(cnt)      # 어레이를 넘겨준다.
    color = random_color()
    cv2.drawContours(img_outline, [cnt], 0, color, thickness=5)     # contouridx : 0 - 하나밖에 없어서 0이라 써도 됨
    # contouridx 를 증가시키는 방식으로 위의 for문을 변경시킬 수 있다.
    # [cnt]: 1개씩 교대로 그리려면 각 컨투어를 어레이에 담아 넘겨주어야 한다.
    # drawContours() 함수가 list자료의 컨투어를 요구하기 때문이다.
"""

# 실습해 본 내용
for j in range(len(contours)):
    color = random_color()
    cv2.drawContours(img_outline, contours, j, color, thickness=5)  # contouridx : 0 - 하나밖에 없어서 0이라 써도 됨

# ----------------------------------------------------------------------------------------------------
# 단계 4: sub fig.3 : 위의 그림 2개를 더해서 점과 선을 한 화면에 그린다.
# 아직 화면에 출력은 안하고 img 파일에 그린 결과를 저장해 둔다.
# ----------------------------------------------------------------------------------------------------
img_points_outline = img_points | img_outline



# ----------------------------------------------------------------------------------------------------
# 단계 5: img 파일의 내용을 화면에 출력한다.
# ----------------------------------------------------------------------------------------------------
show_img_with_matplotlib(img_points, "contour points", 1)
show_img_with_matplotlib(img_outline, "contour outline", 2)
show_img_with_matplotlib(img_points_outline, "contour outline and points", 3)

plt.show()
