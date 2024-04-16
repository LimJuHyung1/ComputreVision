"""
Hu moments calculation
    휴모멘트의 연산 기법을 보인다.
    1개의 오브젝트에 대해 영상 모멘트와 컨투어 모멘트를 구해 휴 모멘트가 얼마나 달라지는지 관찰해 본다.
"""
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 data 폴더.


Name = 'CONTOUR_TEST_ORG.jpg'       # 바깥 테두리가 검은 색
Name = 'drawing6.png'
#Name = 'bullseye.jpg'
#Name = 'rects.png'
#Name = 'shapes_r.png'
#Name = 'circles.jpg'
#Name = '10.jpg'
Name = 'drawing1.png'
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
Name = 'obj_with_no_hole.jpg'           # 홀 없는 객체 1개
#Name = 'objs_with_no_holes.jpg'
#Name = 'obj_with_hole.jpg'
#Name = 'objs_with_holes.jpg'

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

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=14)
    plt.axis('off')


# Create the dimensions of the figure and set title:
#fig = plt.figure(figsize=(12, 5))
#plt.suptitle("Hu moments", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')

# Load the image and convert it to grayscale:
FullName = Path + Name
image = cv2.imread(FullName)   # 편의상 모노 영상도 3채널로 읽은 것으로 가정한다.
assert image is not None, "Failed to load image file:"
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply cv2.threshold() to get a binary image:
#ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)
otsu_thr, imgBin = cv2.threshold(gray_image,  # otsu_thr = 연산된 임계치. , otsu_mask = 바이너리 영상
                                    -1,     # 원래 임계값을 지정하는 것인데  오츠는 자동 연산하기 때문에 활용되지 않는다.
                                    255,    # 임계값 보다 클 경우 배정될 값. 나머지는 0이된다.
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# --------------------------------------------------------------------------------------------------------
# step1. 영상 모멘트를 구한 후 휴 모멘트를 계산해 본다.
# 입력은 flag에 따라 다음 방법 중에 따라 gray 혹은 binary로 택한다.
# --------------------------------------------------------------------------------------------------------

# 바이너리 영상에 대한 모멘트를 구할 것인지, 그레이 영상에 대한 모멘트를 구할 것인지 결정한다.
flag = False    # gray 영상 사용
flag = True     # binary 영상 사용
if flag == False:   # gray 영상을 써야 한다.
    input_image = gray_image.copy()
    ttle = 'gray'
else:               # flag가 True이면 binary 영상을 써야 한다.
    input_image = imgBin.copy()
    ttle = 'binary'

M = cv2.moments(input_image, flag)

# 사전형의 모멘트를 출력한다. 이것은 영상 모멘트다. 콘투어 모멘트가 아니다.
print(f"step1: {ttle} image moments: \n{M}")
# Calculate the centroid of the contour based on moments:
x, y = centroid(M)      # 영상 모멘트를 계산한다.
#mono_3ch = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)    # 출력을 위해 3채널 모노 영상으로 변환.
#show_img_with_matplotlib(mono_3ch, f"image moments(gray_image, {flag})", 1)

# Compute Hu moments:
HuM = cv2.HuMoments(M)      # HuMoments() 함수는 모멘트를 입력받아 휴모멘트를 연산한다.
print(f"\n1) {ttle} image moments based Hu moments:\n{HuM}")


# --------------------------------------------------------------------------------------------------------
# step2. 컨투어 모멘트를 구한 후 휴 모멘트를 구한다.
# 문제를 단순화하기 위해 객체는 Hole이 없는 사례만 관찰한다.
# 그래서 외곽 컨투어만 구해도 되는 사례입니다.
# Hole이 있는 객체의 관찰에 대해서는 강의노트 검토사항에 확인 바랍니다.
# --------------------------------------------------------------------------------------------------------

# Find contours in the thresholded image:
# Note: cv2.findContours() has been changed to return only the contours and the hierarchy
# 혹시 있을 수 있는 컨투어 근사화에서 발생할 수 있는 오오류를 방지하기 위해
# 근사화는 택하지 않습니다. CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Compute moments:
M2 = cv2.moments(contours[0])
print(f"\nStep2: external contour moments of 0th contour: \n{M2}")

# Calculate the centroid of the contour based on moments:
x2, y2 = centroid(M2)

# Compute Hu moments:
HuM2 = cv2.HuMoments(M2)
print(f"\n2) external contour based Hu moments:\n{HuM2}")

"""
# --------------------------------------------------------------------------------------------------------
# Hu Moments에는 무게 중심 정보가 없어 관찰을 생략한다.....
# Step3. 결과를 화면에 출력하여 관찰한다. 특히 중심, centroid를...
# --------------------------------------------------------------------------------------------------------
# Draw the outline of the detected contour:
draw_contour_outline(image, contours, (255, 0, 0), 2)       # BGR color 사용. 파란 색임. image는 3채널.

# 혹시 영상에 객체가 다수이면 EXTERNAL이라 해도 여러 개의 컨투어를 검출할 수도 있다.
# 그래서 이중 0번만 다른 색으로 표현한다. (x2, y2)는 이 객체의 중심이다.
draw_contour_outline(image, contours[0], (0, 255, 255), 3)  # 노란 색이 관심 객체

print(f'\ninput file={Name}')
print(f'Width x Height ={image.shape[1]} x {image.shape[0]}')

# Draw the centroids (it should be the same point):
# (make it big to see the difference)
cv2.circle(image, (x, y), 15, (0, 0, 255), -1)              # BGR color 사용. 빨간색, 큰 원이 영상 모멘트의 중심
print(f"image moments center: (x, y)=({x},{y})")

cv2.circle(image, (x2, y2), 7, (255, 0, 0), -1)            # 파란색, 작은 원이 0번 콘투어의 hu moments 중심.
print(f"contour[0] moments center: (x, y)=({x2},{y2})")

# Plot the images:
show_img_with_matplotlib(image, "centroids: \nimage(red), moments(blue) ", 1)     # 내부에서 RGB color로 바꾸어 출력함.

# (x, y)와 (x2, y2)가 같으리란 법이 없다.
# 객체가 1개일 때는 같아야 한다.
# Show the Figure:
plt.show()
"""