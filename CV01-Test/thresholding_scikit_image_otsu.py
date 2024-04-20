"""
Thresholding example (Otsu's binarization algorithm) using scikit-image
scikit-image를 install 해야 합니다.

"""

# Import required packages:
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage import img_as_ubyte

path = 'Images/kyon.png'

def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_hist_with_matplotlib_gray(hist, title, pos, color, t=-1):
    """Shows the histogram using matplotlib capabilities"""

    ax = plt.subplot(2, 2, pos)
    # plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.axvline(x=t, color='m', linestyle='--')
    plt.plot(hist, color=color)

# Create the dimensions of the figure and set title and color:
fig = plt.figure(figsize=(8, 8))
plt.suptitle("Thresholding scikit-image (Otsu's binarization example)", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load the image and convert it to grayscale:
image = cv2.imread(path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate histogram (only for visualization):
# (이미지, 채널 - (그레이라서 0), None - 전체 이미지가 계산할 영역, 256개의 바이너리 사용, 0~255 픽셀값 나타냄
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Trying Otsu's scikit-image algorithm:
# Return threshold value based on Otsu's method
thresh = threshold_otsu(gray_image)
# Build the binary image (bool array):
# 임계값을 초과하는 픽셀은 True, 이하인 픽셀은 False
binary = gray_image > thresh
print("binary image 'dtype': '{}'".format(binary.dtype)) # bool로 출력됨
# Convert to uint8 data type:
# bool 형식의 이미지를 uint8로 변경 - 이미지 픽셀이 0 ~ 255 사이의 값을 가지게 됨
binary = img_as_ubyte(binary)
print("binary image 'dtype': '{}'".format(binary.dtype)) # uint8로 출력됨

# Plot all the images:
show_img_with_matplotlib(image, "image", 1)
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img", 2)
show_hist_with_matplotlib_gray(hist, "grayscale histogram", 3, 'm', thresh)
show_img_with_matplotlib(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "Otsu's binarization (scikit-image)", 4)

# Show the Figure:
plt.show()
