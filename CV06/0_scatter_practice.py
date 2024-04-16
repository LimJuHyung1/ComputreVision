"""
scatter 함수 사용 연습
    scatter() 함수는 matplotlib에서 제공하는 지정한 위치에 점을 그리는 함수이다.
    cv.circle()은 한 개의 위치에 점을 그릴 수 있다. 여러 개의 점을 그리려면 반복문이 반복문이 필요하다.
    np.scatter()는 어레이의 좌표에 어레이에 지정된 형태의 다양한 형태의 점을 반복문없이 그릴 수 있다.

하는 일

단계 0: (x, y) 좌표 어레이를 num_points(=30) 만큼 생성한다.
단계 1: 1번 화면에는 1세트(x, y)의 고정된 색상(str_clr)의 데이터 분포를 출력한다.
단계 2: 2번에는 2세트의 데이터 분포를 출력한다.
    마커의 모양도 바꾸고,각각의 컬러를 어레이로 지정하여 해당 컬러맵의 색상을 출력하도록 만든다.


함수 간략 설명

    matplotlib.pyplot.scatter(
    x, y,  # float or array-like, shape (n, ), (x, y) 좌표, (n,1) shape의 어레이도 된다.
    s=None,  # The marker size. float or array-like, shape (n, ), optional.
    c=None,  # The marker colors. array-like or list of colors or color, optional.
        A scalar or
        sequence of n numbers to be mapped to colors using cmap and norm.
            'viridis' 맵이 기본으로 사용된다. 데이터의 몇 개(num_points)이건 full 영역의 color가 사용된다.
        A single color format string; 'b': blue, 'g': green, 'r': red, 'c': cyan, 'm': magenta, 'y': yellow, 'k': black, 'w': white
    marker=None,    # marker style. default: 'o'.
        maker 모양과 color 참고 링크: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    cmap=None,      # colormap. default: 'viridis'
    norm=None, vmin=None, vmax=None, alpha=None,
    linewidths=None, *, edgecolors=None, plotnonfinite=False, data=None, **kwargs)

"""


# Import required packages:
import numpy as np
from matplotlib import pyplot as plt


# 단계 0: (x, y) 좌표 어레이를 num_points 만큼 생성한다.
num_points = 30     # 데이터의 수. 참고: 점의 숫자를 바꾸어도 컬러맵을 사용하는 2번 화면은 색의 범위는 전체 영역을 다 활용한다.
x = np.array([[i] for i in range(num_points)])      # x축 좌표: (30x1). 0~29
print('type(x)=', type(x), x.shape)
y = 5 * np.ones((num_points, 1))                    # y축 좌표: (30x1). 모두 5
print('type(y)=', type(y), y.shape)
data2 = np.float32(np.hstack((x, y)))       # (x, y) 좌표
print('type(data2)=', type(data2), data2.shape)

# 화면을 생성한다.
fig = plt.figure(figsize=(12, 8))
plt.suptitle(f"{num_points} points data drawn by scatter()", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('green')    # 'silver'
fig.patch.set_alpha(0.1)    # opacity of background, 배경의 투명도 설정.

# 단계 1: 1번 화면에는 1세트(x, y)의 고정된 색상(str_clr)의 데이터 분포를 출력한다.
plt.subplot(211)
marker_size = 10        # marker size in in points**2
str_clr = 'c'  # The marker colors. 'b': blue, 'g': green, 'r': red, 'c': cyan, 'm': magenta, 'y': yellow, 'k': black, 'w': white
plt.scatter(data2[:, 0], data2[:, 1], # data2 어레이에 있는 (x, y) 좌표
            s=marker_size,   # marker size,
            marker='s', # marker style. default 'o', circle. 's'=square, '+'=plus
            c=str_clr)  # marker color
plt.title(f"marker: string color={str_clr}, size={marker_size}, square")

# 단계 2: 2번에는 2세트의 데이터 분포를 출력한다.
#   마커의 모양도 바꾸고,각각의 컬러를 어레이로 지정하여 해당 컬러맵의 색상을 출력하도록 만든다.
plt.subplot(212)
#plt.scatter(data2[:, 0], data2[:, 1], s=20, marker='o', c='b')
clr_array1 = np.array([range(num_points)])

# 세트 1
plt.scatter(data2[:, 0], data2[:, 1], s=20, marker='o', c=clr_array1)   # 컬러맵: default='viridis'
# 세트 2
plt.scatter(data2[:, 0], data2[:, 1]*1.5, s=30, marker='+', c=clr_array1, cmap='jet') # 컬러맵: jet
plt.title(f"marker: array_color={clr_array1.shape}, 2 color maps")
# Show the Figure:
plt.show()

