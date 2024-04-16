"""

1. 개요
    입력된 컬러 영상의 R, G, B 컬러 채널의 영상을 해당 색상으로 화면에 출력한다.

2. 코딩상의 유의사항
    컬러 영상 img는 img.shape=(높이, 넓이, 3). 3차원 어레이이다. => 3 channel
    OpenCV 함수에서는 Blue(ch=0), Green(ch=1), Red(ch=2)의 순서로 배열되는 것으로 간주하여 처리한다.
    ==> 결론적으로; Blue=img[:, :, 0], Green=img[:, :, 1], Red=img[:, :, 2]
    [:, :, 0]에서 :의 의미는 for all이다. 따라서 '첫 번째 :'는 '모든 row에 대해',
        '두 번째 :'는 '모든 colulmn에 대해'를 뜻한다.
    [:, :, 0]는 편의상 [..., 0]으로 기술할 수 있다. '...,'은 '기술한 인덱스외 나머지는 모두'의 의미를 가진다고 할 수 있다.
    matplotlib, PIL 등의 함수들은 RGB 순서로 배열된다. 따라서 이들 모듈을 혼용할 때는 주의를 요한다.

3. 동작
    B, G, R 컬러 채널의 영상을 해당 색상으로 화면에 출력하기
        0으로 초기화된 빈 영상 어레이를 선언한다. 입력영상과 같은 크기에 3채널 빈 어레이가 필요하다.
        이곳에 특정 영상의 컬러 채널을 복사해서 해당 색상으로 그 채널 영상을 보인다.

"""
import cv2  as cv
import numpy as np


#===============================================================================
# 영상이 존재하는 폴더와 파일 이름을 지정하기.
# 질문: 영상 데이터의 여러 속성을 제시하고 그 의미를 설명하시오.
#===============================================================================
Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
#Path = '../../Images/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
Path = '../data/'
Name = 'RGBColors.JPG'
#Name= 'colorbar_chart.jpg'
#Name = 'lenna.bmp'
#Name = 'monarch.bmp'
FullName = Path + Name
print(FullName)

# 주어진 메세지(message)와 함께 스트링 문자열(string)로 주어진 어레이 변수(string)의 속성을 출력한다.
def printImgAtt (message, string):
    print("\n", message, string)
    data = eval(string)
    print(' type :', type(data))           # imge type =  <class 'numpy.ndarray'>
    print(' shape = ', data.shape)      # 영상 어레이의 크기 알아내기. image shape =  (512, 768, 3). (행, 열, 채널)
    print(' data type = ', data.dtype) # 어레이 요소의 데이터 타입 알아내기. uint8 = unsigned 8비트.
    if len(data.shape) >=2:
        print(' row = ', data.shape[0])  # shape는 tuple이다. 원소는 []로 지정. 리스트도 []로 지정한다.
        print(' column = ', data.shape[1])  # 열의 수.
        if len(data.shape) ==3:
            print(' channel = ', data.shape[2])  # 채널의 수. 컬러 영상이면 3. 모도 영상이면 1.


# ========================================================================================================
print("\n단계 1: 영상 파일을 열어 화면에 출력한다.")
# ========================================================================================================
# ImreadMode: 영상 데이터의 반환 모드를 결정
#   IMREAD_COLOR = 1            # default. 모노 영상도 3채널(RGB) 영상이 된다.
#   IMREAD_GRAYSCALE = 0        # 컬러 영상도 모노로 변환하여 연다. 1채널 영상이 됨.
#   IMREAD_UNCHANGED = -1       # 있는 그대로 열기.
image = cv.imread(FullName)     # read as 3 channel. default.

# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert image is not None, f'No image file: [{FullName}]....! '  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.
cv.imshow('input image', image)

#print(f'입력 영상: image.dtype={image.dtype}: min={np.min(image)}, max={np.max(image)}')
printImgAtt('입력 영상 변수:', 'image')
cv.waitKey()
#cv.destroyWindow('input image')


#"""
# ========================================================================================================
print("\n단계 2: R, G, B 컬러 채널의 영상을 해당 색상으로 화면에 출력한다.")
#            0으로 초기화된 빈 영상 어레이를 선언한다. 입력영상과 같은 크기에 3채널 빈 어레이가 필요하다.
#            이곳에 특정 영상의 컬러 채널을 복사해서 해당 색상으로 그 채널 영상을 보인다.
# ========================================================================================================

# Yellow = Green + Red
# Magenta = Red + Blue
# Cyan = Blue + Green
# 질문 1: 아래 문장이 의미하는 바를 설명하시오.
#           img = np.zeros(image.shape, dtype=np.uint8)
#           img[:, :, 0] = image[:, :, 0]
#           img[:, :, 2] = 0
# 질문 2: 아래 방법 3)이 불가한 이유를 설명하시오. 이를 copy()를 활용하여 해결하시오.
# --------------------------------------------------------------------------------------------------------

# 입력 영상과 같은 크기의 모든 요소의 값이 0으로 초기화된 영상 어레이 img를 선언한다.
#img = np.zeros(image.shape, dtype='uint8')     # 방법 1) 이것도 됨.
img = np.zeros(image.shape, dtype=np.uint8)    # 방법 2) 이것도 됨.
#img = image; img[:,:,:] = 0                     # 방법 3) 질문: 이것은 안되는 이유가 무엇인가? -> image도 같이 0이 된다.

printImgAtt('새로 생성한 변수:', 'img')
cv.imshow('img - at the beginning ', img)
cv.waitKey()

# Blue 채널 영상을 만든다.
img[..., 0] = image[..., 0]       # 원본 영상에서 B 채널만 채워 넣는다.
cv.imshow('B', img)
cv.waitKey()

# Green 채널 영상을 만든다.
img[:, :, 0] = 0                  # clear B channel
img[:, :, 1] = image[...,1]       # copy G channel
cv.imshow('G', img)
cv.waitKey()

# Red 채널 영상을 만든다.
img[:, :, 1] = 0                  # clear G channel
img[..., 2] = image[:, :, 2]       # copy R channel
cv.imshow('R', img)
cv.waitKey()
cv.destroyAllWindows()   # 안 해도 됨.
#exit()
#"""


