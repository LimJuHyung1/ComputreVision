import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

path = 'Images/mikoto.png'

def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    #ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

def color_quantization(image, k):
    global num_maxCount, eps, num_attempts

    # 이미지를 2D 배열로 변환하고 모든 픽셀을 3차원 벡터로 변환한다.
    data = np.float32(image).reshape((-1, 3))
    print(f'\nk={k}: data.shape={data.shape}')       # data.shape=(400000, 3). 영상크기(가로x세로) x 3채널

    # (몰라, 알고리즘이 반복하는 횟수, 수렴 기준)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, num_maxCount, eps)

    s_time = time.time()

    ret, label, center = cv2.kmeans(data        # 군집 대상을 찾을 데이터
                                    , k         # 군집 수
                                    , None      # 랜덤으로 초기 중심 포인트를 찾겠다
                                    , criteria  # 알고리즘 수렴 기준
                                    , num_attempts  # 중심 포인트 선택 시도 횟수
                                    , cv2.KMEANS_RANDOM_CENTERS)    # 초기 중심 포인트를 무작위로 찾겠다
    print(f"label: type={type(label)}, shape={label.shape}, len={len(label)}")
    print(f"center: shape={center.shape}, dtype={center.dtype}")

    e_time = time.time() - s_time

    center = np.uint8(center)   # k개의 센터를 정수형으로 바꿈. - 컬러값이 256이라 8로 잡았다고 하심
    result = center[label.flatten()]

    # 영상과 같은 shape로 만든다.
    result = result.reshape(img.shape)

    # 원본과의 유사도를 나타내는 PSNR 품질을 확인한다.
    psnr = cv2.PSNR(image, result)      # 30 이상이면 원본과 거의 차이가 없음
    print(f'k={k}: time={e_time:#.2f}, PSNR={psnr:#.1f}')
    return result, e_time, psnr

# 프로그램의 시작 --------------------------------------------------------------------------

# 전역변수 - 함수에서도 그때로 쓰일...
num_maxCount = 20       # criteria 지정
eps = 1.0               # criteria 지정
num_attempts = 10       # kmeans() 함수 호출시 전달


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 6))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.85)   # 서브 플롯의 여백 설정


plt.suptitle(f"K-means clustering: maxCount={num_maxCount}, epsilon={eps}, attempts={num_attempts}", fontsize=14, fontweight='bold')

# Load BGR image:
img = cv2.imread(path)
show_img_with_matplotlib(img, "original image", 1)


# 화면 구성이 2x3이므로 원본 영상을 sub 화면 1번으로 하면 최대 5개의 K값을 리스트 자료에 설정할 수 있습니다.
for i, k in enumerate([2, 4, 8, 16, 32]):
    # 영상, 소요시간, psnr 값
    color_2, e_time, psnr = color_quantization(img, k)      # 이 부분에서 k means clustering 발생
    show_img_with_matplotlib(color_2, f"k={k}, time={e_time:#.3f}, PSNR={psnr:#.1f}", i+2)


# Show the Figure:
plt.show()


