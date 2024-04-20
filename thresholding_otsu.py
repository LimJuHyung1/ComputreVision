"""
Otsu's binarization algorithm
"""

# Import required packages:
import cv2
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

# matplotlib 기능을 사용하여 히스토그램을 표시 - 선 그래프로 표시
def show_hist_with_matplotlib_gray(hist, title, pos, color, t=-1):
    ax = plt.subplot(2, 2, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])  # x축의 범위 설정

    # t에 해당하는 위치에 수직 선을 추가
    # 마젠타 색, 점선
    plt.axvline(x=t, color='m', linestyle='--')
    plt.plot(hist, color=color)


# Create the dimensions of the figure and set title and color:
fig = plt.figure(figsize=(10, 10))
plt.suptitle("Otsu's binarization algorithm", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load the image and convert it to grayscale:
image = cv2.imread('Images/nagato.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 히스토그램 계산 (시각화를 위해)
# (이미지, 채널 - (그레이라서 0), None - 전체 이미지가 계산할 영역, 256개의 바이너리 사용, 0~255 픽셀값 나타냄
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Threshold the image aplying Otsu's algorithm:
# THRESH_TRUNC - 픽셀 값이 임계값을 초과하면 임계값으로 설정
# THRESH_OTSU - 최적의 임계값을 자동으로 선택
# ret1 - 임계값, th1 - 이진화된 이미지
ret1, th1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)

# Plot all the images:
show_img_with_matplotlib(image, "image", 1)
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img", 2)
show_hist_with_matplotlib_gray(hist, "grayscale histogram", 3, 'm', ret1)
show_img_with_matplotlib(cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR), "Otsu's binarization", 4)

# Show the Figure:
plt.show()
