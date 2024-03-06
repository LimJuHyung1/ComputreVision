"""
개요
    matplotlib를 활용하여 영상을 출력한다.
    실습 1: matplotlib의 non-interactive mode 동작

I. 프로그램의 기능
    OpenCV 함수로 읽은 영상을 matplotlib.pyplot 함수로 영상의 색상 구성 성분인 RGB 정보를 해당 색상으로 표현한다.
    본 예제에는 matplotlib.pyplot 함수에서 제공하는 ineteractive mode를 on 혹은 off하여
    그 차이를 경험할 수 있는 2개(실습 1, 2)과 모노 영상을 matplotlib.pyplot 화면에서 보이는 예제를 담고 있다.
        실습 1 : non-interactive mode
        실습 2 - matplotlib의 interactive mode 동작
        실습 3 : 칼라 영상을 모노 영상 변환하여 이를 다양한 칼라맵을 사용하여 출력하여 본다.

II. 프로그램의 목표
    영상 정보에서 특성 색상 정보를 추출하여 표현할 수 있다.
    더하여, python coding 기술을 익히는 목적도 겸하고 있다...

III. 점검 포인트
    # img의 i번째 채널의 모든 데이터를 img2의 해당 채널에 복사한다.
    img2[:, :, i] = img[:, :, i] 혹은 img2[..., i] = img[..., i]

IV. 사례와 함께 살펴 본 주목해야 할 함수
    b, g, r = cv.split(image)
        : 3차원 영상어레이(image)를 3개의 2차원 어레이(b, g, r)로 분할한다.
    img = cv.merge([r,g,b])
        : 2차원 영상 어레이 3개(r, g, b)를 list 자료로 만들어 이를 입력으로 하여 3차원의 영상 어레이 img를 만들어 낸다.

    * 위 2개의 함수를 순차적으로 수행하면 cv로 읽은 BGR 영상을 matplotlib에 출력하기 위해 BGR로 바꾸는데 활용할 수 있다.
        아니면 간단히 다음과 같은 슬라이싱 한 줄로 같은 목적을 달성할 수 있다.
        img = image[..., ::-1]      # image는 BGR 배열, img는 RGB 배열로 새로운 객체로 정의된다.

    imgBlank = np.zeros(img.shape, dtype='uint8')
        : 데이터가 uint8로 구성된 img.shape에 지정한 차원의 내부요소가 0의로 채워진 어레이(imgBlank)를 반환한다.
    imgR = imgBlank.copy()
        : imgBlank와 똑 같은 영상 어레이 imgR을 복사하여 만들어 낸다. 이들은 자료를 서로 공유하지 않는다.
    plt.figure(num='window title')
        : 'window title'로 이름 붙여진 새로운 창을 하나 생성해 낸다.
    plt.subplot(221) => 221의 일반화 => nmk. (n, m, k)로도 표기할 수 있다.
        : 창을 n개의 row, m개의 column으로 나눈 수 k번에 창을 지정한다. 이후 imshow(), plot() 등으로 그림을 그린다.
    plt.imshow(img)
        : 이것만으로는 영상이 출력되지 않는다.
            non-interactive mode에서는 plt.show()를 수행해야 영상이 출혁된다.
        :   interactive mode - plt.waitforbuttonpress()를 수행해야 영상이 출력된다. 마우스 클릭 혹은 키보드를 입력하면 다음 줄로 넘어간다.
    plt.show()
        : non-interactive mode에서는 plt.imshow()를 수행할 때 화면에 출력된다.
        : 창을 닫지 않으면 닫을 때까지 기다린다. 창을 닫아야 다음 줄로 넘어간다.
    plt.title('Original')
        : 출력한 그림 위에 타이틀을 출력한다.
    plt.axis('off')
        : 가로, 세로의 눈금과 대표 값들을 표시하지 않는다.
    plt.ion()
        : interactive mode로 설정한다. plt.show() 없이 plt.imshow()만으로 출력된다.
    plt.waitforbuttonpress()
        : interactive mode 사용시 키 혹은 버튼 입력을 기다린다. openCV의 waitKey()와 유사한 기능이다.

V. 공통 주의 사항
    OpenCV는 영상이 BGR 순으로 배열되어 있다.
    matplot는 영상 채널 배열이 RGB로 구성되어 있다.
    따라서, matplotlib를 이용해 영상을 화면에 출력하려면 사전에 BGR 배열을 RGB 배열로 바꾸는 작업이 필요하다.

VI. 미션 - 1차 레포트로 활용.
    cv.imread() 함수로 읽어 낸 3채널 영상을 칼라와 모노로 다음 조건 하에 matplotlib 화면의 2x2 서브창의  4개 화면에 출력한다.

    조건
    matplot 화면은 interactive 모드로 작동해야 하며 화면에 마우스를 클릭하거나 키보드를 누를 때마다 다음 루틴으로 넘어간다.
    1) 서브 화면 1 - 원본 컬러 영상, title="Original Color Image"
    2) 서브 화면 2 - 모노 영상, title="Mono Image"
    3) 서브 화면 3 - 원본 컬러 영상에서 각 컬러 채널에  1.5를 곱하여 밝게 만든 영상, title="Original * 1.5"
    4) 서브 화면 4 - 원본 컬러 영상에서 R 컬러 채널만  2로 곱하여 R 색상만 높게 반영한 영상, title="Original[..., 0] * 2"
    소스 영상 파일의 이름은 맨 앞에 3줄을 아래 보인 대로 복사하여 작성해 주세요.
        Path = '../data/'
        Name = 'RGBColors.JPG'
        FullName = Path + Name

    힌트: 예제 CV_02_ColorImage_plt.py 소스를 참조. 4번 영상은 붉은 색상이 많이 강화된 영상이 될 것입니다.
    혹, 일부만 완성했으면 일부 완성된 내용이라도 제출하기 바랍니다. 단, 완성 수준은 보고서 앞 부분에 명시해야 합니다.

    제출물 - 아래 3개의 파일을 구글 클래스 룸 1차 레포트 제출코너에, zip으로 묶거나 압축하지 말고, 별개로 올려주세요.
    1) 소스 프로그램(그림은 제외)
    2) PDF 보고서 - 작성 방법은 강의 계획서를 참조 바랍니다. PDF 변환본으로 올려주세요.
    3) 시연 동영상 - 3분이내로 동작을 보이는 시연 동영상을 만들었으면 합니다. 스크린과 함께 음성 구술이 함께 들어가야 합니다.

"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# ===============================================================================
# 단계 0 :  영상이 존재하는 폴더와 파일 이름을 지정하기.
# ===============================================================================
Path = "d:\Work\StudyImages\Images\\"  # \\ 오류 발생 방지. \만 쓰면 오류.
Path = "d:/Work/StudyImages/Images/"  # \\ 오류 발생 방지. \만 쓰면 오류.
# Path = '../../Images/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
Path = "../data/"
Name = "RGBColors.JPG"
Name = "colorbar_chart.jpg"
Name = "RGBColors.JPG"

# Name = 'lenna.bmp'
# Name = 'monarch.bmp'
FullName = Path + Name

# 3개 실험 공통 영역
image = cv.imread(FullName)  # read as 3 channel. default.
assert (
    image is not None
), f"No image file: [{FullName}]....! "  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.
cv.imshow(f"input original: shape={image.shape} - Press any key to continue", image)
cv.waitKey()
cv.destroyAllWindows()

# """
# ========================================================================================================
print("\n실습 1: matplotlib의 non-interactive mode 동작")
# matplotlib 함수를 이용하여 한 화면에 4 개의 영상을 출력한다.
# OpenCV로 읽은 영상 어레이는 RGB 순으로 배열되어 있으나, matplotlib 함수로 출력할 때는 RGB 순으로 배열되어 있어야 한다.
#   이렇게 배열을 바꾸는 방법을 제시하였다.
# 원본 영상과 해당 영상의 RGB 성분의 영상을 해당 색상으로 출력한다.
# 3채널의 R, G, B 3개 영상 어레이를 정의하여 두고 이들을 matplotlib 함수를 이용하여 화면에 출력한다.
# ========================================================================================================

# ImreadMode: 영상 데이터의 반환 모드를 결정
#   IMREAD_COLOR = 1            # default. 모노 영상도 3채널(RGB) 영상이 된다.
#   IMREAD_GRAYSCALE = 0        # 칼라 영상도 모노로 변환하여 연다. 1채널 영상이 됨.
#   IMREAD_UNCHANGED = -1       # 있는 그대로 열기.

# BGR 배열의 영상을 RGB 순서로 재배열한다. matplotlib는 RGB 배열을 전제로 한다.
# 방법 1) BGR->RGB: spilt & merge
# b, g, r = cv.split(image)   # cv2는 bgr 배열을 사용한다.
# print('b.shape = ', b.shape, ': g.shape = ', g.shape, ': r.shape = ', r.shape)
# img = cv.merge([r, g, b])    # matplotlib에서는 rgb를 사용한다. 영상을 RGB 순으로 배열한다.

# 방법 2) BGR->RGB: slicing
img = image[..., ::-1]  # = image[:, :, ::-1]. 같은 표현
# -1::-1 맨 끝에서 맨 앞으로 -1 씩 증분시키면서 복사한다. 혹은 단순히 ::-1 (증분만 -1로 제시). 2개는 같은 표현이다.

print("img.shape=", img.shape, "img.data=", img.data)

# 2) 여러 목적으로 활용될 입력 영상과 같은 크기의 빈 영상을 정의한다.
imgBlank = np.zeros(img.shape, dtype="uint8")  # 원영상과 같은 크기의 빈 영상을 준비한다.
print("imgBlank.shape=", imgBlank.shape)
print("imgBlank.data=", imgBlank.data)

# 3) 입력 영상의 RGB 성분을 각각 해당 영상 어레이에 정의한다.
# 입력 영상의 해당 되는 채널의 화면 정보를 화면 출력을 위한 어레이에 복사한다.
imgR = imgBlank.copy()
imgR[:, :, 0] = img[:, :, 0]  # r
imgG = imgBlank.copy()
imgG[:, :, 1] = img[:, :, 1]  # g
imgB = imgBlank.copy()
imgB[:, :, 2] = img[:, :, 2]  # b

# 4) 창을 열고 화면을 2x2로 나눈 다음 첫 번째 화면에 입력 영상을 출력한다.
plt.figure(
    num="Lab1: Original Image & its color components - close this window to proceed..."
)
plt.suptitle(
    "Lab1: non-interactive mode, color images, close this window to proceed...",
    fontsize=10,
    fontweight="bold",
)
plt.subplot(221)
plt.imshow(img)  # 이것만으로는 영상이 출력되지 않는다. plt.imshow()를 수행할 때야 화면에 출력된다.
plt.title("Original")
plt.axis("off")  # plt.xticks([]), plt.yticks([]). 이것은 테두리가 남아있음.

# 5) 나머지 창에 R, G, B 영상 성분을 해당 색상으로 출력한다.
ii = 0
for color, array in [["Red", imgR], ["Green", imgG], ["Blue", imgB]]:
    plt.subplot(220 + ii + 2)
    plt.imshow(array)
    # 현재 상태로는 x, y 축에 값과 눈금을 그린다. 이것을 없애고 싶으면 아래 둘 중의 한 가지를 실행한다.
    plt.axis("off")  # x, y축에 눈금 값 출력을 하지 않는다.
    # plt.xticks([]), plt.yticks([])     # 이렇게 하면 최소한 그림의 테두리는 그린다.
    plt.title(color)
    ii += 1
print("창을 닫지 않으면 닫을 때까지 기다립니다. 창을 닫아야 다음 줄로 넘어갑니다.")
plt.show()  # 창을 닫지 않으면 닫을 때까지 기다린다. 창을 닫아야 다음 줄로 넘어간다.
# exit(0)
# """
# =============================================================================================
