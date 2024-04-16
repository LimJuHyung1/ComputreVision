"""
Matching contours using cv2.matchShapes() against a selected template shape

개요
    이진화 과정을 거친 영상에 대해 검출한 컨투어 객체에 대해 랜덤 함수로 선택한 컨투어를 번호를
    기준 (template)으로 할 때 각 객체와의 Hu moment의 매칭 score를 3가지 관점(I1~I3)으로 보였다.
    그림에서 좌측 상단은 검출한 컨투어와 그 번호를 보였고, 임의 선택된 객체는 분홍색으로 표시하였다.
    나머지 화면은 해당 객체의 매칭 score를 3가지 지표(I1~I3)로 보인 것이다.



미션
    100% 모양이 같은 shape 객체를 한 화면에 10개의 같은 shape를 복사해 넣는다
    방법: 1. 그림판 혹은 유사도구를 사용한 그림 파일 생성(기본 점수), 2. 프로그램 구현(가점 점수)
    크기 조정, 회전 각도 조정, 위치 이동
    매칭 지표별: 특정한 변동에 대해 자기자신에 가장 가까운 매칭 스코어를 보이는 shape를 다음과 같이 순차적으로 나열하고 통계를 구해 보자.
       매칭지표1, 크기변동, 제1 shape(자기 자신), 제2 shape, ...... 평균, 표준편차
       매칭지표2, 크기변동, 제1 shape(자기 자신), 제2 shape, ...... 평균, 표준편차
       매칭지표3, 크기변동, 제1 shape(자기 자신), 제2 shape, ...... 평균, 표준편차

       매칭지표1, 회전변동, 제1 shape(자기 자신), 제2 shape, ...... 평균, 표준편차
       매칭지표2, 회전변동, 제1 shape(자기 자신), 제2 shape, ...... 평균, 표준편차
       매칭지표3, 회전변동, 제1 shape(자기 자신), 제2 shape, ...... 평균, 표준편차

       매칭지표1, 위치변동, 제1 shape(자기 자신), 제2 shape, ...... 평균, 표준편차
       매칭지표2, 위치변동, 제1 shape(자기 자신), 제2 shape, ...... 평균, 표준편차
       매칭지표3, 위치변동, 제1 shape(자기 자신), 제2 shape, ...... 평균, 표준편차

    위 자료를 바탕으로 분석자료를 도출하시오.
    1. 가장 크기 변동에 견실한 특징을 보이는 매칭 지표는?
    2. 가장 회전 변동에 견실한 특징을 보이는 매칭 지표는?
    3. 가장 위치 변동에 견실한 특징을 보이는 매칭 지표는?


"""

# Import required packages:

Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 data 폴더.

#Name = 'drawing1.png'
Name = 'drawing2.png'   # 검토 대상. 같은 모양이 회전 상태로 배열..
#Name = 'drawing3.png'
#Name = 'drawing4.png'
#Name = 'drawing5.png'
#Name = 'drawing7.png'
#Name = 'lightening2.png'
#Name = 'match_shapes.png'       # 객체가 검은 선이므로 아래와 같은 다른 이진화 기법이 필요함.
#ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY_INV)
#Name = 'match_shapes2.png'     # 주의: 말풍선 객체가 여러 조각으로 분리되어 버렸음.
Name = 'match_shapes3.png'      # 추천 영상



import numpy as np
import cv2
from matplotlib import pyplot as plt

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

FullName = Path + Name
image = cv2.imread(FullName)   # 편의상 모노 영상도 3채널로 읽은 것으로 가정한다.
assert image is not None, "Failed to load image file:"
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply cv2.threshold() to get binary images:
#ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY_INV)
otsu_thr, thresh = cv2.threshold(gray_image,  # otsu_thr = 연산된 임계치. , otsu_mask = 바이너리 영상
                                    -1,     # 원래 임계값을 지정하는 것인데  오츠는 자동 연산하기 때문에 활용되지 않는다.
                                    255,    # 임계값 보다 클 경우 배정될 값. 나머지는 0이된다.
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Find contours using the thresholded images:
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# template contour num. 찾고자 하는 shape의 contour number
# 매칭 시켜보고자 하는 컨투어의 번호를 랜덤하게 선택하게 하였다.
import random
num_t = random.randint(0, len(contours))  # [0, 256) : 0~255 중의 정수


# 빈 공간에 컨투어를 그리고 컨투어의 번호를 적어 넣는다.
img = np.full_like(image, 255)

for i, c in enumerate(contours):      # 윤곽선 별로 모멘트를 계산한다.
    if i == num_t:
        c_color = (255, 0, 255)     # magenta
        line_thick = 5      # 윤곽선의 두께.
    else:
        c_color = (70, 70, 70)
        line_thick = 2
    cv2.drawContours(image=img,  # 윤곽선을 그릴 화면. 동시에 입력영상이면서 반환 받을 영상임.
                     contours=[c],  # 윤곽선 정보, 점들의 집합으로 구성. 윤곽선의 총 개수 = len(contours)
                     contourIdx=-1,  # 윤곽선의 번호. 음수이면 모든 윤곽선을 그린다.
                     color=c_color,  # 윤곽선의 색상. = (B, G, R)
                     thickness=line_thick)  # 윤곽선의 두께. -1이면 채우기.
    m = cv2.moments(c)
    center = centroid(m)
    #print('cn=', i, ' center=', center)
    radius = 5; color = (0, 0, 255); thickness = -1   # fill
    cv2.circle(img, center, radius, color, thickness)
    cv2.putText(img, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

# Copy three images to show the results:
img2 = np.full_like(image, 255)
cv2.drawContours(image=img2,          # 윤곽선을 그릴 화면. 동시에 입력영상이면서 반환 받을 영상임.
                 contours=contours, # 윤곽선 정보, 점들의 집합으로 구성. 윤곽선의 총 개수 = len(contours)
                 contourIdx=-1,     # 윤곽선의 번호. 음수이면 모든 윤곽선을 그린다.
                 color=(70, 70, 70),     # 윤곽선의 색상. = (B, G, R)
                 thickness=2)         # 윤곽선의 두께. -1이면 채우기.

result_1 = img2.copy()
result_2 = img2.copy()
result_3 = img2.copy()

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(8, 7))
plt.suptitle(f"Total number of shapes found={len(contours)}\n"
             f"Contour number to be matched={num_t}", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# At this point we compare all the detected contours with the circle contour to get the similarity of the match
for contour in contours:
    # Compute the moment of contour:
    M = cv2.moments(contour)

    # The center or centroid can be calculated as follows:
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    # We match each contour against the circle contour using the three matching modes:
    ret_1 = cv2.matchShapes(contours[num_t], contour, cv2.CONTOURS_MATCH_I1, 0.0)
    ret_2 = cv2.matchShapes(contours[num_t], contour, cv2.CONTOURS_MATCH_I2, 0.0)
    ret_3 = cv2.matchShapes(contours[num_t], contour, cv2.CONTOURS_MATCH_I3, 0.0)

    # Get the positions to draw:
    (x_1, y_1) = get_position_to_draw(str(round(ret_1, 3)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    (x_2, y_2) = get_position_to_draw(str(round(ret_2, 3)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    (x_3, y_3) = get_position_to_draw(str(round(ret_3, 3)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)

    # Write the obtainted scores in the result images:
    cv2.putText(result_1, str(round(ret_1, 3)), (x_1, y_1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    cv2.putText(result_2, str(round(ret_2, 3)), (x_2, y_2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
    cv2.putText(result_3, str(round(ret_3, 3)), (x_3, y_3), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

# Plot the images:
show_img_with_matplotlib(img, f"template contour number={num_t}", 1)
show_img_with_matplotlib(result_1, "CONTOURS_MATCH_I1", 2)
show_img_with_matplotlib(result_2, "CONTOURS_MATCH_I2", 3)
show_img_with_matplotlib(result_3, "CONTOURS_MATCH_I3", 4)

# Show the Figure:
plt.show()
