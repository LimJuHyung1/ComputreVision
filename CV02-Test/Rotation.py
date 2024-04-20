"""
개요
    영상에 대해 rotation 변환을 행한 결과를 보인다.
    변환 매트릭스를 선언하여 warpAffine() 함수를 사용하여 변환한다.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

path = 'Images/mashiro.png'

def plot_cv_img(input_image, fig_num, title_str):
    plt.subplot(2, 3, fig_num)
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))    # BGR -> RGB
    plt.title(title_str)
    #plt.axis('off')
    plt.xticks([]), plt.yticks([])

def print_pram(num, center, rot_angle, scale, output_size):
    print(f'\nFig. {num}: center={center} rot_angle={rot_angle:#d} scale={scale:#.1f} output_size={output_size}')

# 중심 지점 설정하기
def mark_center(image, center, radius):
    img = image.copy()
    cv2.circle(img, center, radius, (0, 0, 255), -1) # bgr, -1 => thickness Filled
    cv2.circle(img, center, int(radius / 2), (255, 255, 0), -1)
    return img

def main():
    img = cv2.imread(path)
    image = img.copy()
    w, h = image.shape[1], image.shape[0]
    output_size = (w, h)

    # 사진 1 - 원본 영상
    num = 1 # 그림 번호
    plot_cv_img(image, num, f'1) size={output_size}\nOriginal') # 영상의 크기를 타이틀에 보인다.

    # 사진 2 - 회전 영상
    num = 2

    # cv2.getRotationMatrix2D(center, angle, scale) → M     : rotation matrix를 반환받는 함수
    #   물체를 평면상의 지정한 center를 중심으로 𝜃 만큼 회전하는 변환.
    #   center: 중심좌표(가로, 세로)
    #   angle: 회전 각도. 양의 각도는 시계반대방향 회전.
    #   scale: 출력되는 영상의 크기

    # 중심은 사진 크기의 절반 지점
    center = tuple(map(int, (w/2, h/2)))
    rot_angle = 30
    scale = 0.5

    # 회전 매트릭스
    r_matrix = cv2.getRotationMatrix2D(center, rot_angle, scale)
    img = mark_center(image, center, 32) # 중심축에 마크표시

    # 이미지 외부 영역 처리 방법 - 테두리 영역에 상수값 적용하여 확장(constant)
    # 테두리 영역 색장 지정 - 하늘색
    transformed = cv2.warpAffine(img, r_matrix, output_size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0xe0, 0xef, 0xff))
    plot_cv_img(transformed, num, f'{num}) size={output_size}\n'+r'$\theta$'+
                f'={rot_angle:#d}, s={scale:#.1f}, c={center}')

    print_pram(num, center, rot_angle, scale, output_size)

    # 사진 3 - 사진 2에서 스케일링
    num = 3

    center = tuple(map(int, (w/2, h/2)))
    rot_angle = 30
    scale = 1
    img = mark_center(image, center, 16)
    r_matrix = cv2.getRotationMatrix2D(center, rot_angle, scale)

    output_size = (w, h)
    transformed = cv2.warpAffine(img, r_matrix, output_size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0xe0, 0xff, 0xff))
    plot_cv_img(transformed, num, f'{num}) size={output_size}\n' + r'$\theta$' +
                f'={rot_angle:#d}, s={scale:#.1f}, c={center}')
    print_pram(num, center, rot_angle, scale, output_size)

    # 그림 4 - 그림 3에서 center를 w, h로 지정
    num = 4
    center = tuple(map(int, (w, h)))
    r_matrix = cv2.getRotationMatrix2D(center, rot_angle, scale)
    img = mark_center(image, center, 64)
    transformed = cv2.warpAffine(img, r_matrix, output_size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0xe0, 0xff, 0xff))
    plot_cv_img(transformed, num, f'{num}) size={output_size}\n' + r'$\theta$' +
                f'={rot_angle:#d}, s={scale:#.1f}, c={center}')

    print_pram(num, center, rot_angle, scale, output_size)

    # 그림 5 - 그림 4와 동일((w, 0)좌표가 center)
    num = 5

    # rotation matrix를 정의한다.
    center = tuple(map(int, (w, 0)))
    rot_angle = 30  # in degrees
    scale = 1  # keep the size same
    img = mark_center(image, center, 64)  # 회전 중심에 마킹을 한다.
    r_matrix = cv2.getRotationMatrix2D(center, rot_angle, scale)
    output_size = (w, h)
    transformed = cv2.warpAffine(img, r_matrix, output_size, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0xe0, 0xff, 0xff))
    plot_cv_img(transformed, num, f'{num}) size={output_size}\n' + r'$\theta$' +
                f'={rot_angle:#d}, s={scale:#.1f}, c={center}')
    print_pram(num, center, rot_angle, scale, output_size)

    # 그림 6 - 그림 5 동일
    num = 6
    # rotation_matrix는 그림 5의 조건을 그대로 반영
    output_size = tuple(map(int, (1.5 * w, 1.5 * h)))
    transformed = cv2.warpAffine(img, r_matrix, output_size, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0xe0, 0xff, 0xff))
    plot_cv_img(transformed, num, f'{num}) size={output_size}\n' + r'$\theta$' +
                f'={rot_angle:#d}, s={scale:#.1f}, c={center}')
    print_pram(num, center, rot_angle, scale, output_size)

    plt.show()

if __name__ == '__main__':
    main()

