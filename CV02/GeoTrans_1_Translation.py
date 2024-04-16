"""
개요 - 이동 변환을 이용하여 기하학적 변환의 실행 사례를 보인다.
    1. translation 변환 매트릭스는 어떻게 생성하는지.
    2. 이 매트릭스는 affine 변환 함수를 통해 이동한 영상을 구할 수 있다.
    3.
    2. 변환 매트릭스를 선언하여 warpAffine() 함수를 사용하여 변환한다.
    invertAffineTransform() 함수로 역변환 매트릭스를 취하여 원본 영상을 복구하는 작업을 시도한다.
    변환 결과의 배경 색을 바꾸어 보임으로써 변환 결과를 좀더 명확하게 분석하고자 하였다.

유의사항
    1. warpAffine() 함수의 파라미터 중 dsize는 변환된 영상을 어떤 공간에 그려 넣을 것인가로 이해하는 것이 좋겠다.
    그림 3번과 6번쌍을 보면 3번에서 경계면 칼라를 초록색으로 설정하였고,
    6번에서는 그 결과를 다시 역변환하였더니 원본 그림이 만들어지는데 나머지 배경색은 다시 검은 색으로 채워졌다.
    borderMode=BORDER_CONSTANT default. borderValue를 지정하지 않으면 검은 색이다.
    2. 기하학적 변환된 결과의 온전한 영상을 관찰하려면 충분히 큰 dsize 지정이 필요하다.


참조 함수:
    dst = cv.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]
        src: input image.
        dst: output image that has the size dsize and the same type as src .
        M: 2×3 transformation matrix.
        dsize: size of the output image.
        flags: combination of interpolation methods (see InterpolationFlags) and the optional flag WARP_INVERSE_MAP that means that M is the inverse transformation ( dst→src ).
        borderMode: pixel extrapolation method (see BorderTypes); when borderMode=BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to the "outliers" in the source image are not modified by the function.
        borderValue: value used in case of a constant border; by default, it is 0.


"""


import numpy as np
import matplotlib.pyplot as plt
import cv2


# With jupyter notebook uncomment below line
# %matplotlib inline
# This plots figures inside the notebook


# 서브 플롯 창의 변호를 지정하여 영상을 출력하는 함수
def plot_cv_img(input_image, fig_num, title_str):
    plt.subplot(2, 3, fig_num)
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))    # BGR -> RGB
    plt.title(title_str)
    plt.axis('off')


def main():
    # read an image & show it
    img = cv2.imread('lenna.jpg')
    w, h = img.shape[1], img.shape[0]
    output_size = (w, h)        # 넓이, 높이

    # Fig. 1 -----------------------------------------------------
    # 1. 입력 영상을 있는 그대로 보인다.
    plot_cv_img(img, 1, f'1) {output_size}\nOriginal') # 영상의 크기를 타이틀에 보인다.


    # Fig. 2, Fig. 5 ----------------------------------------------
    # 그림 2. 영상에 대해 translation 변환을 행한 결과를 보인다.
    # 그림 5. Translation의 역변환 매트릭스를 구해 그 결과를 적용하여 원래의 위치에 영상을 복원한다.

    # 단계 1): translation matrix를 정의한다.
    x = 160; y = 40     # translation value - x축으로 160, y축으로 40 이동
    # 다음 2개의 방법(a, b) 중의 한 방법으로 정의한다.

    # '''pdf 2.2 Translation'''
    # 1a. create transformation matrix
    # t_matrix = np.float32([[1, 0, x], [0, 1, y]])
    # print(f'Fig.2: 3) t_matrix={t_matrix.shape}\n{t_matrix}')

    # """
    # 1b. create transformation matrix.
    #   numpy 연습삼아 identity matrix와 translation vector를 결합해 본다.
    identity = np.identity(2) # 대각선 요소가 1인 2 by 2인 매트리스 생성
    print(f'Fig.2: 1) identity.shape={identity.shape}', '\n', identity)
    translation = (np.array([(x, y)])).T        # np.transpose(B)와 같다.
    print('Fig.2: 2) translation.shape=', translation.shape, '\n', translation)
    # 위 2개의 매트릭스를 묶는다. np.hstack()는 2개 매트릭스의 자료형이 같지 않아도 되는 장점이 있다.
    t_matrix = np.hstack ([identity, translation]) # identity 매트리스와 translation 매트리스를 서로 묶음(?)

    # 반면 CV에서 제공하는 hconcat()는 3개의 자료형이 같아야 묶을 수 있다.
    #print(f'\nidentity.dtype={identity.dtype}, translation.dtype={translation.dtype}')
    #translation = translation.astype(np.float64)
    #t_matrix = cv2.hconcat((identity, translation))
    
    print(f'Fig.2: 3) translation_matrix={t_matrix.shape}\n{t_matrix}')



    # 단계 2): affine transformation을 행하고, 그 결과를 그림 2에 보인다.
    transformed2 = cv2.warpAffine(img, t_matrix, dsize=output_size)
    plot_cv_img(transformed2, 2, f'2) {output_size}\nTranslation') # Fig. 2: 이동변환

    # 단계 3) Translation의 역변환 매트릭스를 구해
    # 그 결과를 이동 변환된 영상에 적용하여 원래의 위치에 영상을 복원하여 그림 5에 보인다.
    iM = cv2.invertAffineTransform(t_matrix)    # inverse transform matrix
    print(f'Fig.5: inverse translation_matrix={iM.shape}\n{iM}')
    transformed = cv2.warpAffine(transformed2, iM, output_size)
    plot_cv_img(transformed, 5, f'5) {output_size}\nInv Trans')     # Fig. 5: 이동역변환


    # Fig. 3, Fig. 6 -----------------------------------------------------------
    # 원본 영상을 이동할 때 그림이 손상되지 않도록 타겟 이동 영상의 공간을 2배로 확장하여 그림 3에 출력한다.
    # 이때 원본 영상이 옮겨지지 않은 공간은 초록색으로 구분해 출력한다.
    # 그림 3의 영상에레이에 대해 역변환 이동 매트릭스를 이용해 원본 영상의 위치를 복구하여 그림 6에 출력한다.
    # 그림 6은 영상의 손괴 부분이 없다.
    output_size = (w*2, h*2)
    transformed3 = cv2.warpAffine(img, t_matrix, dsize=output_size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0x0, 0xff, 0x00))
    plot_cv_img(transformed3, 3, f'3) {output_size}\nTranslation')  # 그림 3: 큰 어레이에 이동 영상 출력
    transformed = cv2.warpAffine(transformed3, iM, output_size)
    plot_cv_img(transformed, 6, f'6) {output_size}\nInv Trans')     # 그림 6: 원본 영상 복구에서 잘린 부분이 없다.
    print(f'Fig.6: inverse translation_matrix={iM.shape}\n{iM}')

    # Fig. 4 --------------------------------------------------------------------
    # 타겟 영상의 크기를 1/2로 줄여서 그곳에 이동변환을 시켜 출력해 본다.
    # 2배로 커진 것처럼 보이는 이유는 축소한 영상을 그림 1의 원본 영상의 크기에 맞춰 출력했기 때문이다.
    # 이 때문에 이동 거리도 2배로 보인다.
    output_size = tuple(map(int, (w/2, h/2)))       # output_size = (int(w/2), int(h/2))
    #output_size = (w*2, h*2)
    transformed = cv2.warpAffine(img, t_matrix, output_size)
    plot_cv_img(transformed, 4, f'4) {output_size}\nTranslation')



    plt.show()
if __name__ == '__main__':
    main()