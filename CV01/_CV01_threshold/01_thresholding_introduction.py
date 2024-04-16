"""
Introduction to thresholding techniques

Thresholding은 전경(foreground)과 배경(background)를 나누는 간단하지만 제법 효과적인 기법이다.
가장 단순한 image segmentation 기술의 한 사례라고 할 수 있다.
입력 영상을 처리하기에 편한 단순한 영상으로 만드는데 활용된다.

"""

# Import required packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt


def build_sample_image():
    """Builds a sample image with 50x50 regions of different tones of gray"""

    # Define the different tones.
    # The end of interval is not included
    tones = np.arange(start=50, stop=300, step=50)
    print(tones)

    # Initialize result with the first 50x50 region with 0-intensity level
    result = np.zeros((50, 50, 3), dtype="uint8")

    # Build the image concatenating horizontally the regions:
    for tone in tones:
        img = np.ones((50, 50, 3), dtype="uint8") * tone
        #result = np.concatenate((result, img), axis=1)     # np 함수로 구현. axis=1은 x축으로 연결을 의미.
        result = cv2.hconcat((result, img))     # cv2 함수로 구현. 인자가 1개임에 유의..

    return result


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(7, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create the dimensions of the figure and set title and color:
fig = plt.figure(figsize=(6, 9))
# fig = plt.figure()
plt.suptitle("Thresholding introduction", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load the image and convert it to grayscale:
image = build_sample_image()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Plot the grayscale images and the histograms:
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR),
                         "img with tones of gray - left to right: (0,50,100,150,200,250)", 1)

# Apply cv2.threshold() with different thresholding values:
# https://docs.opencv.org/4.9.0/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
# cv.threshold(	src, thresh, maxval, type[, dst]) ->	retval, dst
# type = THRESH_BINARY일 경우
#   화솟값이 thresh 보다 크면 maxval을, 작거나 같으면 0을 반환한다.
ret1, thresh1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
ret3, thresh3 = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
ret4, thresh4 = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
ret5, thresh5 = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
ret6, thresh6 = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)
print(f"thresh4.dtype={thresh4.dtype}")     # binary data type이 아니다.

# Plot the images:
show_img_with_matplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "threshold = 0", 2)
show_img_with_matplotlib(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "threshold = 50", 3)
show_img_with_matplotlib(cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR), "threshold = 100", 4)
show_img_with_matplotlib(cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR), "threshold = 150", 5)
show_img_with_matplotlib(cv2.cvtColor(thresh5, cv2.COLOR_GRAY2BGR), "threshold = 200", 6)
show_img_with_matplotlib(cv2.cvtColor(thresh6, cv2.COLOR_GRAY2BGR), "threshold = 250", 7)

# Show the Figure:
plt.show()
