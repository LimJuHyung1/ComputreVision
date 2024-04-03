"""
Ordering contours based on the size
    영상 이진화를 보통은 Otsu 알고리즘을 사용했는데
    이 알고리즘으로는 일부 영상은 객체 검출이 않되어서 Sauvola's 이진화 알고리즘을
    사용할 수 있도록 했습니다. 대략. 10여초~30여초 소요됩니다.
    'shapes_sizes.png' 영상은 푸른색 객체가 검출되지 않아 Sauvola 알고리즘응 사용했습니다.
"""

# Import required packages:
import cv2
from matplotlib import pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------------------
# 파이썬 sorting 연습
# 파이썬 내장 함수: sorted()
# 리스트의 메서드: sort()
# ----------------------------------------------------------------------------------------

"""
a = [6, 1, 7, 4, 5]
print('a=', a)
b = sorted(a)
print('b=', b)
b2 = sorted(a, reverse=True)
print('b2=', b2)
a.sort(reverse=True)    # 리스트만 사용가능
print('a=', a)

# sorted() 함수는 사전형 자료에도 적용 가능하다.
c = {'a': 123, 'vf': 89, 'v': 638, '1': 45, 'B': 'B'}
#c = {1: 'X', 5: 'C', 3: 'Q', 9: 'E', 4: 'A'}
print('\nc=', c)
cc = sorted(c)  # key를 바탕으로 소팅한다.
print('c=', cc)
print('3rd value=', c[cc[2]])
"""

"""
# 본 메인 예제의 함수, sort_contours_size(contours)에서 사용한 기법
#   여러 개의 자료를 묶어서 그중 한 자료를 중심으로 소팅할 때...
#   함수에서 활용 예: (cnts_sizes, cnts) = zip(*sorted(zip(cnts_sizes, contours)))
#       이 루틴은 (cnts_sizes, cnts) 자료를 cnts_sizes 중심으로 소팅하여 반환한다.
#
# 간단한 사례로 치환하여 해석해 본다.
# 자료 a, b가 각 순서대로 쌍으로 의미를 갖는다고 하자.
a = [2, 1, 9, 8, 3]
b = ['two', 'one', 'nine', 'eight', 'three']

# 이를 zip으로 묶어 소팅한다.
c = sorted(zip(a, b))   # zip은 a, b의 원소를 각각 1개로 묶어 5개의 tuple로 구성된 iterable 객체를 만든다.
print(f'type({type(c)}): c=', c)        # b는 a 자료의 순서에 따라 소팅된다.

print('*c=', *c)    # 리스트 배열내의 자료를 접근한다.
d, e = zip(*c)
print('d=', d)
print('e=', e)
exit(0)
"""





def get_position_to_draw(text, point, font_face, font_scale, thickness):
    """Gives the coordinates to draw centered"""

    text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
    text_x = point[0] - text_size[0] / 2
    text_y = point[1] + text_size[1] / 2
    return round(text_x), round(text_y)


# 입력되는 contours를 전달받아 면적을 계산하여 면적의 크기순으로 소팅된
# cnt_sizes(면적 리스트)와 면적 크기로 재배열된 cnts를 반환한다.

def sort_contours_size(contours):
    # cnts_sizes는 면적으로 이루어진 list. 아직 소팅은 안됨.
    cnts_sizes = [cv2.contourArea(contour) for contour in contours]
    # (면적, 컨투어들) 자료를 cnts_sizes 중심으로 소팅하여 반환한다.         => 소스 상단의 파이썬 연습 참조
    (cnts_sizes, cnts) = zip(*sorted(zip(cnts_sizes, contours)))
    return cnts_sizes, cnts


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

# Sauvola's 이진화 알고리즘 - 대략. 10여초~30여초 소요됩니다.
def sauvola_threshold(img, window_size=15, k=0.2, R=128):
    # Convert image to grayscale if it's not already
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get image dimensions
    rows, cols = img.shape

    # Initialize output image
    dst = np.zeros_like(img)

    # Pad image
    pad = window_size // 2
    padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

    # Apply Sauvola thresholding
    for i in range(pad, rows + pad):
        for j in range(pad, cols + pad):
            # Calculate local mean
            local_mean = np.mean(padded_img[i - pad:i + pad + 1, j - pad:j + pad + 1])

            # Calculate local standard deviation
            local_std = np.std(padded_img[i - pad:i + pad + 1, j - pad:j + pad + 1])

            # Calculate threshold
            threshold = local_mean * (1 + k * ((local_std / R) - 1))

            # Apply threshold
            if padded_img[i, j] > threshold:
                dst[i - pad, j - pad] = 255
            else:
                dst[i - pad, j - pad] = 0

    return dst


#Path = '../Images/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
#Path = 'd:/work/StudyImages/Images/'
#Path = ''
Path = '../../../CV04/'
#Name = 'lenna.bmp'
#Name = 'monarch.bmp'
#Name = 'fruits.jpg'
#Name = 'bone.jpg'
#Name = 'woman_in_scarf(c).jpg'
#Name = 'man_woman.jpg'

Name = 'CONTOUR_TEST_ORG.jpg'       # 바깥 테두리가 검은 색
#Name = 'bullseye.jpg'
Name = 'rects.png'
#Name = 'shapes_r.png'
#Name = 'circles.jpg'
Name = '10.jpg'
#Name = 'drawing10.png'
#Name = 'drawing2.png'
#Name = '4regions.png'
Name = 'drawing1.png'
#Name = '3regions.png'
Name = 'shapes_sizes.png'   # otsu 알고리즘으로는 푸른 원 객체 하나를 배경으로 간주한다.



# Load the image and convert it to grayscale:
# image = build_sample_image_2()
FullName = Path + Name
image = cv2.imread(FullName)
assert image is not None, "Failed to load image file:"
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(9, 9))
plt.suptitle("Sort contours by size", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the image:
show_img_with_matplotlib(image, "image", 1)

# Apply cv2.threshold() to get a binary image
ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)     # 단순한 트레숄링에 제격.
#thresh = sauvola_threshold(gray_image)  # 시간이 올래 걸리니 주의하세요. 광범위한 경우의 트레숄딩에 대처 가능.
#otsu_thr, thresh = cv2.threshold(gray_image,  # otsu_thr = 연산된 임계치. , otsu_mask = 바이너리 영상
#                                    -1,     # 원래 임계값을 지정하는 것인데  오츠는 자동 연산하기 때문에 활용되지 않는다.
#                                    255,    # 임계값 보다 클 경우 배정될 값. 나머지는 0이된다.
#                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Find contours using the thresholded image:
# Note: cv2.findContours() has been changed to return only the contours and the hierarchy
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.imshow('bin', thresh)
cv2.waitKey(0)

# Show the number of detected contours:
print("detected contours: '{}' ".format(len(contours)))

# Sort the contours based on the size:
(contour_sizes, contours) = sort_contours_size(contours)

for i, (size, contour) in enumerate(zip(contour_sizes, contours)):
    # Compute the moment of contour:
    M = cv2.moments(contour)

    # The center or centroid can be calculated as follows:
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    # Get the position to draw:
    (x, y) = get_position_to_draw(str(i + 1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, 5)

    # Write the ordering of the shape on the center of shapes
    cv2.putText(image, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

# Plot the image
show_img_with_matplotlib(image, "result", 2)

# Show the Figure:
plt.show()
