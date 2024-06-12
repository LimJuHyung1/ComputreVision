from Modules import cv2, np
from Modules import cap, show_snapshot


# K-Means Color Quantization 함수
def color_quantization(image, k):
    num_maxCount = 100  # 최대 반복 횟수
    eps = 0.2  # 수렴 기준
    num_attempts = 10  # 중심 포인트 선택 시도 횟수

    # 이미지를 2D 배열로 변환하고 모든 픽셀을 3차원 벡터로 변환한다.
    data = np.float32(image).reshape((-1, 3))

    # (최대 반복 횟수, 수렴 기준)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, num_maxCount, eps)      # 포인트

    _, label, center = cv2.kmeans(data, k, None, criteria, num_attempts, cv2.KMEANS_RANDOM_CENTERS) # 포인트

    center = np.uint8(center)
    result = center[label.flatten()]

    # 영상과 같은 shape로 만든다.
    result = result.reshape(image.shape)

    return result


# K-Means Color Quantization 버튼 생성 함수
def create_kmeans_button(k):
    def apply_kmeans():
        ret, frame = cap.read()
        if ret:
            quantized_image = color_quantization(frame, k)
            show_snapshot(quantized_image)
    return apply_kmeans


