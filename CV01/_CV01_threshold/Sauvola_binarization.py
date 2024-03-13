"""

Sauvola's thresholding algorithm
    출처: Adaptive document image binarization (2000)
    global thresholding이 아닌, adaptive thresholding 기법입니다.

참고: adaptive thresholding이란?
    입계값이 영상 영역에 따라 계속 바뀌는 처리 기법입니다.

1. 다음은 ChatGPT에게 문의한 알고리즘의 질문입니다.

질의문:
    Sauvola의 영상 이진화 알고리즘에 대해 설명해 줄래?

대답:
    Sauvola의 이진화 알고리즘은 이미지 처리 분야에서 사용되는 효율적인 이진화 기술 중 하나입니다.
    이 알고리즘은 문서 이미지 처리와 관련이 깊으며, 특히 문서의 텍스트를 추출하고 인식하는 데 사용됩니다.
    Sauvola의 이진화 알고리즘은 주로 흑백 이미지를 대상으로 합니다.
    이미지 내의 각 픽셀을 이진화하는 과정은 해당 픽셀의 주변 화소 값과 함께 이미지의 통계적인 특성을 사용합니다.
    이 알고리즘의 핵심 아이디어는 이미지의 각 픽셀을 이진화하기 위해 해당 픽셀의 주변 영역의 밝기와 대비를 고려하는 것입니다.
    주변 영역의 평균과 표준 편차를 사용하여 각 픽셀을 이진화하는 임계값을 도출합니다.
    ... 공식이 있는데 생략합니다...

2. 다음은 Sauvila의 이진화 알고리즘을 ChatGPT에게 의뢰하여 직접 코딩하게 한 것입니다.
    함수를 이용하지 않게 지시하였습니다.

지시문:
    Sauvola의 영상 이진화 알고리즘을 구현하는 함수를 python으로 코딩해 줘.
    입력영상은 opencv의 imread() 함수로 읽었다고 가정하고, 이진화된 출력 영상은  dst라 정해줘.

평가:
    정확하게 수행한 듯 합니다.
    단지 scikit-image보다 수행 시간이 오래 걸리는 듯 합니다.
    chatGPT를 코딩에 처음 활용해 봤는데 학습에 매우 요긴하다고 평가됩니다.
    강력한 개인교사를 하나 두 었다고 생각하고 잘 활용합시다.~~~

"""

import cv2
import numpy as np

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

# Example usage:
# Read input image
input_image = cv2.imread('sudoku.png', cv2.IMREAD_GRAYSCALE)

# Apply Sauvola thresholding
output_image = sauvola_threshold(input_image)

# Display the result
cv2.imshow('Original Image', input_image)
cv2.imshow('Sauvola Thresholded Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
