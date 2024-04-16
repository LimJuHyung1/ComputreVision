"""
K-means clustering algorithm applied to color quantization

개요
    k-means clustering은 지정된 K개의 그룹으로 나누기 때문에 24비트 칼라를 K개로 그룹핑할 수 있다.
    예를들어 k=256이면, 24비트를 8비트로 줄인 효과를 얻을 수 있다.

문제의 올바른 접근을 위한 고찰:
    1) RGB 3채널별로 K개의 칼라 군집화를 하는 것이 맞을까? -> X
    2) 아니면 화면 전체의 색상에 대해 K 군집화를 하는 것이 맞을까? -> O
    ->
    1)은 사실상 Kx3개로 클러스터링이 이루어지는 것이다. 클러스터의 개수가 3배로 늘어난다.
    또한 채널별로 클러스터링하면 때문에 색상의 부조화가 일어날 수 있다.
    마치 컬러 히스토그램 평활화에서 채널별로 평활화를 하면 안되는 것처럼..
    2)의 방법이 옳다. 가로x세로의 화소에 대해 3개 채널의 중심점을 K개 찾아내는 것이 맞다.


"""

# Import required packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

# path = '../../CV04/'
path = '../data/'
f_name = 'landscape_2.jpg'
f_name = 'landscape_2.jpg'
#f_name = 'rooster.png'
f_name = 'rooster2.png'
f_name = 'img.png'
# f_name = 'bts3.jpg'
# full_name = path + f_name
full_name = f_name

def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    #ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# ----------------------------------------------------------------------------------
# 24비트로 이루어진 영상을 K개의 색상만으로 만들어 반환하는 함수
#   image: 원본 영상, k: K means clustering을 위한 클러스터의 개수.
# 반환 값:
#   K개의 색상으로 표현된 영상.
#   e_time: k-means clustering에 소요된 연산시간
#   PSNR 값(원본과의 유사성 비교 값- 40이상이면 원본과 구분이 어려움.)
# 주의: 만들어진 영상은 K의 색상을 사용하는데 색상 종류가 K라는 것이지,
#   K비트로 표현된다는 것은 아니다. 반환되는 영상은 24비트를 모두 사용한다.
# ----------------------------------------------------------------------------------
def color_quantization(image, k):
    global num_maxCount, eps, num_attempts
    # 영상 데이터를 (가로x세로, 3)의 2차원 배열을 가진 데이터 형태로 나열하여 입력한다.
    # 마치 (x,y) 좌표로 주어진 샘플들의 군집화를 행하는 것처럼
    # 샘플 갯수 x 3채널 상의 데이터 군집화 문제를 해결하는 것이다.
    data = np.float32(image).reshape((-1, 3))   # 맨 뒤자리 차원 3은 고정하고 나머지는 정리하여 화소수x3채널 자료로 만든다.
    print(f'\nk={k}: data.shape={data.shape}')       # data.shape=(400000, 3). 영상크기(가로x세로) x 3채널

    # Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
    # In this case the maximum number of iterations is set to 20 and epsilon = 1.0
    #num_maxCount = 20
    #eps = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, num_maxCount, eps)

    s_time = time.time()
    # Apply K-means clustering algorithm:
    # kmeans() 함수를 통해 군집화를 행하고 다음 반환값을 받는다.
    # 1) center.shape=(k, 3)의 배열을 반환받는다. K개의 대표 컬러값(BGR)이 들어있다.
    # 2) labels.shape=(가로x세로, 1)의 배열을 반환받는다.
    #   각 원소에는 센터의 인덱스 번호(컬러값)가 int32 자료가 담겨있다.
    # => 이후 할 일
    # 이것들을 이용해 (가로x세로, 1)의 배열로 만들어진 영상정보를 인덱스 지정하는 center의 색상으로 교체한다.
    #num_attempts = 10
    ret, label, center = cv2.kmeans(data, k, None, criteria, num_attempts, cv2.KMEANS_RANDOM_CENTERS)
    print(f"label: type={type(label)}, shape={label.shape}, len={len(label)}")
    # label: type=<class 'numpy.ndarray'>, shape=(400000, 1), len=400000
    print(f"center: shape={center.shape}, dtype={center.dtype}")
    # K일 때 center: shape=(K, 3), dtype=float32

    e_time = time.time() - s_time

    # At this point we can make the image with k colors
    # 센터에는 K개의 군집을 대표하는 3채널 RGB 값이 들어 있다.
    # 센터의 자료형이 dtype=float32이므로 컬러값으로 쓰기 위해 uint8형의 바꾼다.
    center = np.uint8(center)   # k개의 센터를 정수형으로 바꿈. - 컬러값이 256이라 8로 잡았다고 하심

    # Replace pixel values with their center value:
    #print(f"label.flatten().shape={label.flatten().shape}") # label.flatten().shape=(가로X세로,). 예: shape=(221520,)
    result = center[label.flatten()]
    #print(f"result.shape={result.shape}")  # result.shape=(가로X세로, 3). 예: (221520, 3)

    # 영상과 같은 shape로 만든다.
    result = result.reshape(img.shape)

    # 원본과의 유사도를 나타내는 PSNR 품질을 확인한다.
    psnr = cv2.PSNR(image, result)
    print(f'k={k}: time={e_time:#.2f}, PSNR={psnr:#.1f}')
    return result, e_time, psnr


# 프로그램의 시작 --------------------------------------------------------------------------

# 전역변수 - 함수에서도 그때로 쓰일...
num_maxCount = 20       # criteria 지정
eps = 1.0               # criteria 지정
num_attempts = 10       # kmeans() 함수 호출시 전달


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 6))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.85)


plt.suptitle(f"K-means clustering: maxCount={num_maxCount}, epsilon={eps}, attempts={num_attempts}", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')
# fig.patch.set_alpha(0.2)

# Load BGR image:
img = cv2.imread(full_name)
show_img_with_matplotlib(img, "original image", 1)

# Apply color quantization:


# 화면 구성이 2x3이므로 원본 영상을 sub 화면 1번으로 하면 최대 5개의 K값을 리스트 자료에 설정할 수 있습니다.
for i, k in enumerate([3, 5]):          # 고속. 확인용으로 적절
#for i, k in enumerate([2, 3, 4, 5, 6]):
#for i, k in enumerate([4, 8, 16, 32, 64]):  # 최대 5개
#for i, k in enumerate([8, 32, 64, 128, 256]):  #
#for i, k in enumerate([3, 3, 3, 3, 3]):
    # 영상, 소요시간, psnr 값
    color_2, e_time, psnr = color_quantization(img, k)      # 이 부분에서 k means clustering 발생
    show_img_with_matplotlib(color_2, f"k={k}, time={e_time:#.3f}, PSNR={psnr:#.1f}", i+2)



# Show the Figure:
plt.show()



"""
기록 보관용
@@@@@@@@@@@@@@@@  과년도 과제     @@@@@@@@@@@@@@@@@@@@@@@@@@@@
1. 본 예제 프로그램에 대하여 제공되는 파일 Figure_bts2.png, Figure_sku1.png와 같은 내용이 출력되도록 프로그램을 수정하시오.
    1) 타이틀에는 k 값과 PSNR이 출력되어야 한다.
    2) 다음과 같은 정보가 파이썬 콘솔창에 출력되어야 한다.(k와 PSNR 정보가 어느 정도 유사하게 일치해야 할 것임. 시간은 다르겠지만...)
        k=2 | PSNR=16.12[dB] | time=  0.58[sec.]
        k=4 | PSNR=20.86[dB] | time=  1.01[sec.]
        k=8 | PSNR=24.31[dB] | time=  2.87[sec.]
        k=16 | PSNR=27.17[dB] | time=  5.46[sec.]
        k=32 | PSNR=29.76[dB] | time= 10.14[sec.]
        k=64 | PSNR=31.89[dB] | time= 19.62[sec.]
        k=128 | PSNR=33.57[dB] | time= 40.72[sec.]

2. K=256로 그룹핑한다면 모든 24비트 칼라를 8비트 칼라로 군집화하는 것이다.
   이 때는 (R, G, B)로 이루어진  256개의 중심 값을 cv2.kmeans() 함수가 centers 변수로 반환한다.
   PNG, BMP 파일은 파일 내부의 팔레트(palette) 헤더에 이 정보를 수록하고,
   화소값은 8비트만 기록하여 파일의 크기를 1/3로 줄일 수 있는 기능을 제공한다.
   1) 이 기능을 이용하여 K개로 줄인 색상을 PNG 파일 형식으로 저장하고,
   2) 다시 파일을 읽어 들어 24비트 원본 영상과 8비트로 줄인 영상의 화질을 PSNR 지표를 사용하여 비교하는 프로그램을 작성하시오.
   - 방법 제공되는 파일(Figure_bts2.png, Figure_sku1.png)의 내용과 같이 화면에 출력하면 됨.
   - 본 과정의 순조로운 프로그램을 위하여 몇 개의 선행 학습용 프로그램을 제공합니다..


"""