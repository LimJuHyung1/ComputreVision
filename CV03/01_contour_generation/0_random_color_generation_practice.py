import numpy as np
import cv2
from matplotlib import pyplot as plt



import random


# random color generation: 밝은 색상을 보장 받기 위해 채도와 밝기를 128 이상으로 정하였다.
# 방법: 1 point/3채널 hsv 영상을 만들고 이를 rgb 영상으로 만든 후 이를 (r, g, b) 튜플로 변환한다.
# 이상하게 실제로 해보면 그다지 효과를 느낄 수 없었다.
# 심지어 검은 색에 묻혀 보이지 않는 색상의 원이 RGB 기반의 접근처럼 관측되었다.
def random_color():
    h = np.array([[random.randint(0, 255)]], dtype=np.uint8)    # hue
    s = np.array([[random.randint(80, 255)]], dtype=np.uint8)  # saturation
    v = np.array([[random.randint(200, 255)]], dtype=np.uint8)  # value
    #print('hue:', type(h), h.shape)     # hue: <class 'numpy.ndarray'> (1, 1)
    hsv = cv2.merge((h, s, v))          # hsv 좌표계로 표현된 1점의 영상 생성
    #print('hsv:', type(hsv), hsv.shape) # hsv: <class 'numpy.ndarray'> (1, 1, 3)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)      # RGB 영상구조체로 변환
    #print('rgb:', type(rgb), rgb.shape, rgb.dtype)  # rgb: <class 'numpy.ndarray'> (1, 1, 3) uint8

    # 1) 1점의 RGB 영상을 color tuple로 변환하는 방법 - 각 채널 값을 지정하여 접근.
    rgb = (int(rgb[0, 0, 0]), int(rgb[0, 0, 1]), int(rgb[0, 0, 2]))  # int() 처리 반드시 해야함.

    # 2) reshape()로 2차원(1x?)으로 나열한 후 첫번째 요소를 액세스하여 1차원 ndarray로 만들고 이를 tuple로 변환한다.
    #a = rgb.reshape(1, -1)[0]
    #print('type(a)', type(a), 'a.shape=', a.shape)  # type(a) <class 'numpy.ndarray'> a.shape= (3,)
    #rgb = tuple(map(int, a))    # a의 각 요소에 int() 함수를 씌운 후 tuple로 변환한다.

    return rgb


# random color generation: 간단하긴 한데 가끔 어둡고 약한 색상이 나와서 만족스럽지 않다.
def random_color2():
    r = random.randint(0, 256)  # [0, 256) : 0~255 중의 정수
    g = random.randint(0, 256)
    b = random.randint(0, 256)
    rgb = [r, g, b]
    return rgb


# Create the canvas (black image with three channels):
w = 640
h = 480
img1 = np.zeros((h, w, 3), dtype="uint8")    # row x column
img2 = np.zeros((h, w, 3), dtype="uint8")    # row x column

for i in range(100):
    p = (random.randint(1, w), random.randint(1, h))    # width x height
    cv2.circle(img1, p, 10, random_color(), -1)         # HSV based
    cv2.circle(img2, p, 10, random_color2(), -1)        # RGB based

plt.subplot(121)
plt.imshow(img1)
plt.title('HSV based random')
plt.axis('off')

plt.subplot(122)
plt.imshow(img2)
plt.title('RGB based random')
plt.axis('off')

plt.show()
exit(0)
