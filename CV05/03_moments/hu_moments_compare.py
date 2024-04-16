"""
Hu moments calculation
영상을 입력받아 이를 회전시키고 원본 영상과 회전 영상의 Hu moments를 비교해 본다.

미션
    'drawing1.png' 영상을 사용하면 회전 후 센터가 일치 하지 않는 것으로 보인다.
    1) 이유를 설명하고 이를 바로 잡을 수 있도록 프로그램을 수정하시오. = > hu_moments_compare_solution.py
    2) 아래 소스에서 case 3 실험 @@@@ 마킹의 조건을 사용하시오.
    3) (센터를 일치하도록) 바로 잡았을 경우의 matchShapes의 결과를 CONTOURS_MATCH_I1을 기준으로 제시하시오.
        현재의 matching rate=0.333240...   0에 가까운 결과가 나와야 올바른 풀이라고 할 수 있다. 예: 0.0004562056684843663.
    4) 구현 방법
        영상 A에 있는 콘투어의 번호와 영상 B에 있는 컨투어의 번호를 모두 일치 시킨다.
        CONTOURS_MATCH_I1을 기준으로 기준으로 각 컨투어들의 매칭 레이트를 계산하여 가장 가깝다고 생각되는 것끼리 콘투어의 번호를 매칭 시킨다.
        정답 제시 방법 예시
                        영상 A        영상 B          CONTOURS_MATCH_I1
        contour 번호        0            5            0.000758216
        contour 번호        1            4            0.002254516
        contour 번호        2            2            0.0020458216
        contour 번호        3            1            0.00097216
        contour 번호        4            2            0.001163265
        contour 번호        5            3            0.00056726

"""
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 data 폴더.


Name = 'CONTOUR_TEST_ORG.jpg'       # 바깥 테두리가 검은 색
Name = 'drawing6.png'
#Name = 'bullseye.jpg'
#Name = 'rects.png'
#Name = 'shapes_r.png'
#Name = 'circles.jpg'
#Name = '10.jpg'
#Name = 'drawing1.png'
#Name = 'drawing2.png'
#Name = 'drawing3.png'
#Name = 'drawing4.png'
#Name = 'drawing5.png'
#Name = 'drawing7.png'
#Name = 'drawing8.png'
#Name = 'drawing9.png'
#Name = 'drawing10.png'
#Name = 'shape_features.png'
#Name = 'shape_features_shift_with_background_value.png'
#Name = 'shape_features_shift_back_ground=1.png'
#Name = 'shape_features_shift.png'
#Name = '3regions.png'
#Name = '4regions.png'
Name1 = 'char_A.jpg'; Name2 = 'char_90_A.jpg'
Name1 = 'char_M.jpg'; Name2 = 'char_big_M.jpg'
#Name = 'drawing1.png'           # 미션: 회전 후 센터가 일치 하지 않는 것으로 보이는 영상. 바로 잡으시오.

# Import required packages:
import cv2
from matplotlib import pyplot as plt


def centroid(moments):
    """Returns centroid based on moments"""

    x_centroid = round(moments['m10'] / moments['m00'])
    y_centroid = round(moments['m01'] / moments['m00'])
    return x_centroid, y_centroid


def draw_contour_outline(img, cnts, color, thickness=1):
    """Draws contours outlines of each contour"""

    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)


# BGR color로 입력받는다. 내부에서 RGB color로 바꾸어 출력함.
def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(3, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=14)
    plt.axis('off')


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 5))
plt.suptitle("Hu moments Comparision", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')


# --------------------------------------------------------------------------------------------------------
# 단계 1. 영상파일을 읽고 읽어 들인 영상을 회전한다.
#    각 영상은 A, B로 칭한다. 화면을 출력한다.
# --------------------------------------------------------------------------------------------------------

# 1) Load the image and convert it to grayscale => grayA:
FullName = Path + Name1
img = cv2.imread(FullName)   # 편의상 모노 영상도 3채널로 읽은 것으로 가정한다.
print(f'\ninput file={Name1}')
print(f'input image A: Width x Height ={img.shape[1]} x {img.shape[0]}')

assert img is not None, "Failed to load image file:"
grayA = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayA_3ch = cv2.cvtColor(grayA, cv2.COLOR_GRAY2BGR)
show_img_with_matplotlib(grayA_3ch, 'inputA', 1)


#"""
# 영상 B를 파일로 부터 읽어 들인다.
# 2) 파일로부터 다른 영상을 하나 더 읽어 들인다.  => grayB
FullName = Path + Name2
img = cv2.imread(FullName)   # 편의상 모노 영상도 3채널로 읽은 것으로 가정한다.
print(f'\ninput file={Name2}')
print(f'input image B: Width x Height ={img.shape[1]} x {img.shape[0]}')

assert img is not None, "Failed to load image file:"
grayB = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayB_3ch = cv2.cvtColor(grayB, cv2.COLOR_GRAY2BGR)
show_img_with_matplotlib(grayB_3ch, 'inputB', 2)
#"""


"""
# A로 부터 영상을 회전 시킬 때는 아래 루틴을 사용한다.
# 2) 입력 영상 gray 영상을 회전한다. => grayB
w, h = img.shape[1], img.shape[0]

# case 1 실험
center = tuple(map(int, (w/2, h/2)))       # center = (int(w/2), int(h/2)). 회전 중심
rot_angle = 30  # case 1 in degrees
scale = 1    # case 1
output_size = (2*w, 2*h)

# case 2 실험
#center = tuple(map(int, (w, h)))       # center = (int(w/2), int(h/2)). 회전 중심
#rot_angle = 90  # case 1 in degrees
#scale = 1    # case 1
#output_size = (2*w, 2*h)

# case 3 실험 - 미션의 조건 @@@@
#center = tuple(map(int, (w, h)))
#rot_angle = 90  # case 2in degrees
#scale = 1.5    # case 2
#output_size = (3*w, 3*h)

r_matrix = cv2.getRotationMatrix2D(center, rot_angle, scale)
grayB = cv2.warpAffine(grayA, r_matrix, output_size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0x0, 0x0, 0x0))
print(f'comparison image B: Width x Height ={grayB.shape[1]} x {grayB.shape[0]}')
grayB_3ch = cv2.cvtColor(grayB, cv2.COLOR_GRAY2BGR)
show_img_with_matplotlib(grayB_3ch, 'inputB', 2)
"""


# --------------------------------------------------------------------------------------------------------
# 단계 2. 계조 영상에 대한 이진 영상들을 만든다.
# --------------------------------------------------------------------------------------------------------

# Apply cv2.threshold() to get a binary image:
#ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)
otsu_thr, imgBinA = cv2.threshold(grayA,  # otsu_thr = 연산된 임계치. , otsu_mask = 바이너리 영상
                                    -1,     # 원래 임계값을 지정하는 것인데  오츠는 자동 연산하기 때문에 활용되지 않는다.
                                    255,    # 임계값 보다 클 경우 배정될 값. 나머지는 0이된다.
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
otsu_thr, imgBinB = cv2.threshold(grayB, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

tmpA = cv2.cvtColor(imgBinA, cv2.COLOR_GRAY2BGR)
tmpB = cv2.cvtColor(imgBinB, cv2.COLOR_GRAY2BGR)
show_img_with_matplotlib(tmpA, 'Binary A', 3)
show_img_with_matplotlib(tmpB, 'Binary B', 4)


# --------------------------------------------------------------------------------------------------------
# 단계 3. 외곽 컨투어를 구하고 그것의 Hu invariants를 출력하고, 센터를 비교한다.
# --------------------------------------------------------------------------------------------------------

# Find contours in the thresholded image:
# Note: cv2.findContours() has been changed to return only the contours and the hierarchy
mtd = cv2.CHAIN_APPROX_TC89_L1
mtd = cv2.CHAIN_APPROX_SIMPLE
contoursA, hierarchy = cv2.findContours(imgBinA, cv2.RETR_EXTERNAL, method=mtd)
contoursB, hierarchy = cv2.findContours(imgBinB, cv2.RETR_EXTERNAL, method=mtd)

# Compute moments:
mA = cv2.moments(contoursA[0])
mB = cv2.moments(contoursB[0])
print(f"A: contour moments: {mA}")
print(f"B: contour moments: {mB}")

# Calculate the centroid of the contour based on moments:
x, y = centroid(mA)
centerA = (x, y)
HuA = cv2.HuMoments(mA)
print(f"A: Hu invariants:\n{HuA}")


# Calculate the centroid of the contour based on moments:
x, y = centroid(mB)
centerB = (x, y)
HuB = cv2.HuMoments(mB)
print(f"B: Hu invariants:\n{HuB}")

result = cv2.matchShapes(contoursA[0], contoursB[0], cv2.CONTOURS_MATCH_I1, 0.0)
print(f'I1 matching rate={result}')

result = cv2.matchShapes(contoursA[0], contoursB[0], cv2.CONTOURS_MATCH_I2, 0.0)
print(f'I2 matching rate={result}')

result = cv2.matchShapes(contoursA[0], contoursB[0], cv2.CONTOURS_MATCH_I3, 0.0)
print(f'I3 matching rate={result}')

draw_contour_outline(grayA_3ch, contoursA[0], (0, 255, 255), 4)       # BGR color 사용. Y 색임. image는 3채널.
draw_contour_outline(grayB_3ch, contoursB[0], (0, 255, 255), 4)       # BGR color 사용. Y 색임. image는 3채널.

cv2.circle(grayA_3ch, centerA, 10, (0, 0, 255), -1)            # 빨간 색, 작은 원이 0번 콘투어의 hu moments 중심.
cv2.circle(grayB_3ch, centerB, 10, (0, 0, 255), -1)            # 빨간 색, 작은 원이 0번 콘투어의 hu moments 중심.

# Plot the images:
show_img_with_matplotlib(grayA_3ch, "contour(yellow), hu center(red) ", 5)     # 내부에서 RGB color로 바꾸어 출력함.
show_img_with_matplotlib(grayB_3ch, "contour(yellow), hu center(red) ", 6)     # 내부에서 RGB color로 바꾸어 출력함.

plt.show()
