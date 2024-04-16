"""
K-means clustering algorithm applied to color quantization and calculates also the color distribution image
컬러 분포도를 그림으로 함께 표현하는 기능을 추가하였다.
각 k에 대하여 화소의 수량이 많은 순으로 레이블을 재배치 하여 출력하였다.

"""

# Import required packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt
import collections          # 내장 라이브러리

file_name = '../data/landscape_2.jpg'

def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def color_quantization(image, k):
    """Performs color quantization using K-means clustering algorithm"""

    # Transform image into 'data':
    data = np.float32(image).reshape((-1, 3))
    # print(data.shape)

    # Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
    # In this case the maximum number of iterations is set to 20 and epsilon = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    # Apply K-means clustering algorithm:
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # At this point we can make the image with k colors
    # Convert center to uint8:
    center = np.uint8(center)
    # Replace pixel values with their center value:
    result = center[label.flatten()]
    result = result.reshape(img.shape)

    # Build the 'color_distribution' legend.
    # collection.Counter(): 중복된 데이터가 저장된 배열을 인자로 넘기면 각 원소의 빈도 순서대로 사전형 자료처럼 저장된 객체를 반환한다.
    # We will use the number of pixels assigned to each center value:
    counter = collections.Counter(label.flatten())  # 각 레이블의 갯수가 몇 개씩 존재하는지 알 수 있는 객체를 반환
    print(f'k={k}: counter={counter}')

    # Calculate the total number of pixels of the input image:
    total = img.shape[0] * img.shape[1]

    # Assign width and height to the color_distribution image:
    desired_width = img.shape[1]
    # The difference between 'desired_height' and 'desired_height_colors'
    # will be the separation between the images
    desired_height = 70
    desired_height_colors = 50  # (desired_height - desired_height_colors) 이만큼 빈 공백이 자리잡는다.

    # Initialize the color_distribution image:
    color_distribution = np.ones((desired_height, desired_width, 3), dtype="uint8") * 255
    # Initialize start:
    start = 0

    for key, value in counter.items():
        # Calculate the normalized value:
        value_normalized = value / total * desired_width

        # Move end to the right position:
        end = start + value_normalized

        # Draw rectangle corresponding to the current color:
        cv2.rectangle(color_distribution, (int(start), 0), (int(end), desired_height_colors), center[key].tolist(), -1)
        # Update start:
        start = end

    return np.vstack((color_distribution, result))


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(15, 7))
plt.suptitle("Color quantization using K-means clustering algorithm", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')
#fig.patch.set_alpha(0.2)

# Load BGR image:
img = cv2.imread(file_name)

# Apply color quantization:
color_3 = color_quantization(img, 3)
color_5 = color_quantization(img, 5)
color_10 = color_quantization(img, 10)
color_20 = color_quantization(img, 20)
color_40 = color_quantization(img, 40)

# Plot the images:
show_img_with_matplotlib(img, "original image", 1)
show_img_with_matplotlib(color_3, "color quantization (k = 3)", 2)
show_img_with_matplotlib(color_5, "color quantization (k = 5)", 3)
show_img_with_matplotlib(color_10, "color quantization (k = 10)", 4)
show_img_with_matplotlib(color_20, "color quantization (k = 20)", 5)
show_img_with_matplotlib(color_40, "color quantization (k = 40)", 6)

# Show the Figure:
plt.show()
