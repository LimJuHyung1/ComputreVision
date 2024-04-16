"""
Approximation methods in OpenCV contours
입력 영상에 대해 4개의 근사화 추출 기법을 적용했을 때 검출되어 나오는 컨투어를 이루는 점들의 개수를 비교해 본다.
이를 통해 어떤 알고리즘이 가장 많이 근사화를 이루는지 관찰 할 수 있다.

"""

# Import required packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt

Path = 'd:/work/StudyImages/Images/'
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 data 폴더.


Name = 'CONTOUR_TEST_ORG.jpg'       # 바깥 테두리가 검은 색
Name = 'drawing6.png'
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


def array_to_tuple(arr):
    """Converts array to tuple"""

    return tuple(arr.reshape(1, -1)[0])


def draw_contour_points(img, cnts, color, radius=3):
    """Draw all points from a list of contours"""

    for cnt in cnts:
        #print(cnt.shape)
        squeeze = np.squeeze(cnt)
        #print(squeeze.shape)

        for p in squeeze:
            pp = array_to_tuple(p)
            cv2.circle(img, pp, radius, color, -1)

    num_contour = len(cnts)
    #print('num_contour=', num_contour)
    num_points_list = []
    for i in range(num_contour):
        how_many = cnts[i].shape[0]  # 각 contours에는 몇 개의 points가 있는지? (contours[i].shape)[0]
        #print(f"how many in contours[{i}]={how_many}")
        num_points_list.append(how_many)

    return img, num_points_list


def build_sample_image():
    """Builds a sample image to search for contours"""

    # Create a 500x500 gray image (70 intensity) with a rectangle and a circle inside:
    img = np.ones((500, 500, 3), dtype="uint8") * 70
    cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 255), -1)
    cv2.circle(img, (400, 400), 100, (255, 255, 0), -1)

    return img


def build_sample_image_2():
    """Builds a sample image to search for contours"""

    # Create a 500x500 gray image (70 intensity) with a rectangle and a circle inside:
    img = np.ones((500, 500, 3), dtype="uint8") * 70
    cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 255), -1)
    cv2.rectangle(img, (150, 150), (250, 250), (70, 70, 70), -1)
    cv2.circle(img, (400, 400), 100, (255, 255, 0), -1)
    cv2.circle(img, (400, 400), 50, (70, 70, 70), -1)

    return img


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


#=======================================================================================================
# 입력 영상을 다음 둘 중 한가지 방법으로 읽어 들인다.
#=======================================================================================================

# (a) 가상 영상: Load the image and convert it to grayscale:
image = build_sample_image_2()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply cv2.threshold() to get a ginary image:
ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)


"""
# (b) 파일에서 불러와서 이진화하여 사용한다.
FullName = Path + Name
image = cv2.imread(FullName)   # 편의상 모노 영상도 3채널로 읽은 것으로 가정한다.
assert image is not None, "Failed to load image file:"
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("input: " + Name, img); cv2.imshow("gray", gray); cv2.waitKey()

otsu_thr, thresh = cv2.threshold(gray_image,  # otsu_thr = 연산된 임계치. , otsu_mask = 바이너리 영상
                                    -1,     # 원래 임계값을 지정하는 것인데  오츠는 자동 연산하기 때문에 활용되지 않는다.
                                    255,    # 임계값 보다 클 경우 배정될 값. 나머지는 0이된다.
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#cv2.imshow("step1: image by thresholding", thresh); cv2.waitKey()
"""


#=======================================================================================================
# 화면을 셋팅하고 4개의 결과 출력용 영상 버퍼에 원본 영상을 복사해둔다.
# 컨투어의 표시는 아래 1), 2) 중의 하나를 선택한다.
#=======================================================================================================

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 8))
plt.suptitle("Contours approximation method", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# 1) 원본 위에 컨투어를 표시해 본다.
image_approx_none = image.copy()
image_approx_simple = image.copy()
image_approx_tc89_l1 = image.copy()
image_approx_tc89_kcos = image.copy()
line_color = (255, 255, 255)

"""
# 2) 이진 영상을 2채널로 만들어 이곳에 컨투어를 표시해 본다.
gray_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
image_approx_none = gray_image.copy()
image_approx_simple = gray_image.copy()
image_approx_tc89_l1 = gray_image.copy()
image_approx_tc89_kcos = gray_image.copy()
line_color = (0, 0, 255)    # 점의 색상
"""

#=======================================================================================================
# 4가지 방법으로 윤곽선 근사화 기법을 지정하여 윤곽선을 검출하고 결과를 화면에 출력한다.
#=======================================================================================================
radius = 7      # 점의 반지름

# Find contours using different methods:
# Note: cv2.findContours() has been changed to return only the contours and the hierarchy
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours3, hierarchy3 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
contours4, hierarchy4 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

# Draw the contours in the previously created images:
_, num_points_list = draw_contour_points(image_approx_none, contours, line_color, radius)
print(f'CHAIN_APPROX_NONE: points list', num_points_list)

_, num_points_list = draw_contour_points(image_approx_simple, contours2, line_color, radius)
print(f'CHAIN_APPROX_SIMPLE: points list', num_points_list)

_, num_points_list = draw_contour_points(image_approx_tc89_l1, contours3, line_color, radius)
print(f'CHAIN_APPROX_TC89_L1: points list', num_points_list)

_, num_points_list = draw_contour_points(image_approx_tc89_kcos, contours4, line_color, radius)
print(f'CHAIN_APPROX_TC89_KCOS: points list', num_points_list)

# Plot all the figures:
show_img_with_matplotlib(image, "image", 1)
show_img_with_matplotlib(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), "threshold = 100", 2)
show_img_with_matplotlib(image_approx_none, "contours (APPROX_NONE)", 3)
show_img_with_matplotlib(image_approx_simple, "contours (CHAIN_APPROX_SIMPLE)", 4)
show_img_with_matplotlib(image_approx_tc89_l1, "contours (APPROX_TC89_L1)", 5)
show_img_with_matplotlib(image_approx_tc89_kcos, "contours (APPROX_TC89_KCOS)", 6)

# Show the Figure:
plt.show()
