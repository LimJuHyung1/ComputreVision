"""
개요
    영상에 대해 rotation 변환을 행한 결과를 보인다.
    변환 매트릭스를 선언하여 warpAffine() 함수를 사용하여 변환한다.

유의사항
    getRotationMatrix2D(center, angle, scale) 함수를 통해 변환 매트릭스를 반환받는다.
    1. center, scale의 역할에 대하여 이해하자.
        center
            회전 변환의 중심점을 지정해 주어야 한다. 지정된 점을 중심으로 반시계 방향의 각도로 회전한다.
            중심점을 확인하기 위해 원본 영상에 mark_center() 함수를 이용하여 동심원을 그려 넣었다.
        scale
            입력 영상을 변환하여 다른 좌표계로 맵핑시킬 때의 스케일링을 말한다.
            영상 정보는 없지만 작게 정하면 변위가 작아지게 변환 매트릭스가 만들어 진다.
            예를 들어 S를 적게하면 x-> x'의 변화량을 작게 만드는 매트릭스를 생성한다.
            따라서 영상도 작아진다.
    2. dsize
        기하학적 변환된 결과의 온전한 영상을 관찰하려면 충분히 큰 dsize 지정이 필요하다.
        그렇다고 dsize를 크게 정한다고 원본 영상의 모든 부분을 관찰할 수 있는 것은 아니다.
        회전했을 때 dsize 영역 바깥으로 나간 부분은 dsize를 크게 정한다고 해서 볼 수 있는 것은 아니다.


미션:
    invertAffineTransform() 함수로 역변환 매트릭스를 취하여 원본 영상을 복구하는 작업을 시도해 보자



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
    #plt.axis('off')
    plt.xticks([]), plt.yticks([])


def print_pram(num, center, rot_angle, scale, output_size):
    print(f'\nFig. {num}: center={center} rot_angle={rot_angle:#d} scale={scale:#.1f} output_size={output_size}')

# 중심 지점 설정하기
def mark_center(image, center, radius):
    img = image.copy()
    cv2.circle(img, center, radius, (0, 0, 255), -1)      # thickness =-1 => FILLED. still BGR
    #cv2.circle(img, center, int(radius/2), (0, 255, 255), -1)
    cv2.circle(img, center, int(radius / 2), (255, 255, 0), -1)
    return img



def main():
    # read an image & show it
    img = cv2.imread('lenna.jpg')
    image = img.copy()
    w, h = img.shape[1], img.shape[0]
    output_size = (w, h)        # 넓이, 높이

    # Fig. 1 --
    num = 1     # 그림번호 1: 원본영상
    plot_cv_img(img, num, f'1) size={output_size}\nOriginal') # 영상의 크기를 타이틀에 보인다.


    # Fig. 2 --------
    num = 2     # 그림번호 2: 회전영상
    # center는 영상의 가로와 세로의 1/2 지점으로 정하고, 30도 회전하고, 크기는 0.5 배로 줄인 영상
    # getRotationMatrix2D(center, angle, scale) → M     : rotation matrix를 반환받는 함수
    #   물체를 평면상의 지정한 center를 중심으로 𝜃 만큼 회전하는 변환.
    #   center: 중심좌표(가로, 세로)
    #   angle: 회전 각도. 양의 각도는 시계반대방향 회전.
    #   scale: 출력되는 영상의 크기

    center = tuple(map(int, (w/2, h/2)))       # center = (int(w/2), int(h/2))
    rot_angle = 30  # in degrees
    scale = 0.5
    r_matrix = cv2.getRotationMatrix2D(center, rot_angle, scale)
    img = mark_center(image, center, 32)    # 회전 중심에 마킹을 한다.
    # apply rotation using warpAffine
    transformed = cv2.warpAffine(img, r_matrix, output_size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0xe0, 0xef, 0xff))
    #                                                            세타 문자 표시 방법 - matplotlib에 포함됨
    plot_cv_img(transformed, num, f'{num}) size={output_size}\n'+r'$\theta$'+
                f'={rot_angle:#d}, s={scale:#.1f}, c={center}')

    print_pram(num, center, rot_angle, scale, output_size)


    # Fig. 3 --------
    num = 3
    # rotation matrix를 정의한다.
    center = tuple(map(int, (w/2, h/2)))       # center = (int(w/2), int(h/2))
    rot_angle = 30  # in degrees
    scale = 1
    img = mark_center(image, center, 16)  # 회전 중심에 마킹을 한다.
    r_matrix = cv2.getRotationMatrix2D(center, rot_angle, scale)

    #output_size = tuple(map(int, (2.5 * w, 2.5 * h)))
    output_size = (w, h)
    transformed = cv2.warpAffine(img, r_matrix, output_size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0xe0, 0xff, 0xff))
    plot_cv_img(transformed, num, f'{num}) size={output_size}\n'+r'$\theta$'+
                f'={rot_angle:#d}, s={scale:#.1f}, c={center}')
    print_pram(num, center, rot_angle, scale, output_size)



    # Fig. 4 --------
    num = 4
    center = tuple(map(int, (w, h)))
    r_matrix = cv2.getRotationMatrix2D(center, rot_angle, scale)
    img = mark_center(image, center, 64)  # 회전 중심에 마킹을 한다.
    transformed = cv2.warpAffine(img, r_matrix, output_size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0xe0, 0xff, 0xff))
    plot_cv_img(transformed, num, f'{num}) size={output_size}\n'+r'$\theta$'+
                f'={rot_angle:#d}, s={scale:#.1f}, c={center}')

    print_pram(num, center, rot_angle, scale, output_size)




    # Fig. 5 --------
    num = 5
    # rotation matrix를 정의한다.
    center = tuple(map(int, (w, 0)))
    rot_angle = 30  # in degrees
    scale = 1  # keep the size same
    img = mark_center(image, center, 64)  # 회전 중심에 마킹을 한다.
    r_matrix = cv2.getRotationMatrix2D(center, rot_angle, scale)
    output_size = (w, h)
    transformed = cv2.warpAffine(img, r_matrix, output_size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0xe0, 0xff, 0xff))
    plot_cv_img(transformed, num, f'{num}) size={output_size}\n'+r'$\theta$'+
                f'={rot_angle:#d}, s={scale:#.1f}, c={center}')
    print_pram(num, center, rot_angle, scale, output_size)


    # Fig. 6 --------
    num = 6
    # rotation_matrix는 그림 5의 조건을 그대로 반영
    output_size = tuple(map(int, (1.5* w , 1.5 * h)))
    transformed = cv2.warpAffine(img, r_matrix, output_size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0xe0, 0xff, 0xff))
    plot_cv_img(transformed, num, f'{num}) size={output_size}\n'+r'$\theta$'+
                f'={rot_angle:#d}, s={scale:#.1f}, c={center}')
    print_pram(num, center, rot_angle, scale, output_size)




    plt.show()

if __name__ == '__main__':
    main()

exit(0)






"""
not completed....
# 아래는 아직 개발 중이고 기록용으로 남겨둔 것이니 학생들은 참고하지 마세요.
# 혼선만 빚을 뿐입니다.
# cos, sin에 의한 회전은 중심점이 (0,0)인 것을 가정하여 만들어진 공식입니다.
# 따라서 원하는 지점으로 지정할 수 있도록 수식의 변환이 필요합니다.
# https://darkpgmr.tistory.com/79

# Fig. 6 --------
num = 6
# a. create transformation(rotation & translation) matrix.
theta = 30    # rotation angle
rot_angle = theta
scale = 0.5

rt_list = [(np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))]
rot = scale * np.array(rt_list)
print('rot=', rot.shape, '\n', rot)
x = 256
y = 256  # translation value
translation = (np.array([(x, y)])).T  # np.transpose(B)와 같다.
print('translation=', translation.shape, '\n', translation)
r_matrix = np.hstack((rot, translation))  # hstack는 인자를 tuple 형대로 입력받음.
print(f'translation_matrix={r_matrix.shape}\n{r_matrix}')

center = tuple(map(int, (0, 0)))       # center = (int(w/2), int(h/2))
rot_angle = -30  # in degrees
output_size = (w, h)
transformed = cv2.warpAffine(img, r_matrix, output_size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0xe0, 0xff, 0xff))
plot_cv_img(transformed, num, f'{output_size}\n{rot_angle:#d}, s={scale:#.1f}, {center}')
print_pram(num, center, rot_angle, scale, output_size)
"""
