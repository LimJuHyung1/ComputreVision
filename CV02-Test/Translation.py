"""
개요 - 이동 변환을 이용하여 기하학적 변환의 실행 사례를 보인다.
    1. translation 변환 매트릭스는 어떻게 생성하는지.
    2. 이 매트릭스는 affine 변환 함수를 통해 이동한 영상을 구할 수 있다.
    3.
    2. 변환 매트릭스를 선언하여 warpAffine() 함수를 사용하여 변환한다.
    invertAffineTransform() 함수로 역변환 매트릭스를 취하여 원본 영상을 복구하는 작업을 시도한다.
    변환 결과의 배경 색을 바꾸어 보임으로써 변환 결과를 좀더 명확하게 분석하고자 하였다.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

path = 'Images/sorata.png'

# 서브 플롯 창의 변호를 지정하여 영상을 출력하는 함수
def plot_cv_img(input_image, fig_num, title_str):
    plt.subplot(2, 3, fig_num)
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))    # BGR -> RGB
    plt.title(title_str)
    plt.axis('off')

def main():
    img = cv2.imread(path)
    w, h = img.shape[1], img.shape[0]
    output_size = (w, h)

    # 1. 입력 영상을 그대로 보인다.
    plot_cv_img(img, 1, f'1) {output_size}\nOriginal')  # 영상의 크기를 타이틀에 보인다.

    # 그림 2 - 영상에 대해 translation 변환을 행한 결과를 보인다.
    # 그림 5 - Translation의 역변환 매트릭스를 구해 적용하여 원래의 위치에 영상을 복원한다.

    x = 80; y = 20 # 움직일 거리

    # 매트릭스 생성
    identity = np.identity(2)  # 대각선 요소가 1인 2 by 2인 매트리스 생성
    print(f'Fig.2: 1) identity.shape={identity.shape}', '\n', identity)
    translation = (np.array([(x, y)])).T  # np.transpose(B)와 같다.
    print('Fig.2: 2) translation.shape=', translation.shape, '\n', translation)
    t_matrix = np.hstack([identity, translation]) # 두 개의 배열을 수평으로 결합 - 두 배열을 곱합

    # 결과
    print(f'Fig.2: 3) translation_matrix={t_matrix.shape}\n{t_matrix}')

    # 2. affine transformation을 행하고, 그 결과를 그림 2에 보인다.

    # img 이미지에 t_matrix를 적용시켜 이동, 회전, 크기 조정하여 output_size에 맞춰 출력된다.
    transformed2 = cv2.warpAffine(img, t_matrix, dsize=output_size)
    plot_cv_img(transformed2, 2, f'2) {output_size}\nTranslation')  # Fig. 2: 이동변환

    # 3. Translation의 역변환 매트릭스를 구해
    # 그 결과를 변환된 영상에 적용하여 원래의 위치에 영상을 복원
    iM = cv2.invertAffineTransform(t_matrix)
    print(f'Fig.5: inverse translation_matrix={iM.shape}\n{iM}')
    transformed = cv2.warpAffine(transformed2, iM, output_size)
    plot_cv_img(transformed2, 5, f'5) {output_size}\nInv Trans')     # Fig. 5: 이동역변환

    # 원본 영상을 이동할 때 그림이 손상되지 않도록 타겟 이동 영상의 공간을 2배로 확장하여 그림 3에 출력한다.
    # 이때 원본 영상이 옮겨지지 않은 공간은 초록색으로 구분해 출력한다.
    output_size = (w*2, h*2)
    transformed3 = cv2.warpAffine(img, t_matrix, dsize=output_size)
    plot_cv_img(transformed3, 3, f'3) {output_size}\nTranslation')  # 그림 3: 큰 어레이에 이동 영상 출력
    transformed = cv2.warpAffine(transformed3, iM, output_size)
    plot_cv_img(transformed, 6, f'6) {output_size}\nInv Trans')  # 그림 6: 원본 영상 복구에서 잘린 부분이 없다.
    print(f'Fig.6: inverse translation_matrix={iM.shape}\n{iM}')

    # 4. 대상 이미지의 크기를 1/2로 줄여 이동 후 출력
    output_size = tuple(map(int, (w/2, h/2)))
    transformed = cv2.warpAffine(img, t_matrix, dsize=output_size)
    plot_cv_img(transformed, 4, f'4) {output_size}\nTranslation')

    plt.show()
if __name__ == '__main__':
    main()