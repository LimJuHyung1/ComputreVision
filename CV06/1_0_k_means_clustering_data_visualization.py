"""
K-means clustering data visualization (introduction to k-means clustering algorithm)
개요
    30개의 랜덤 좌표 x 3세트,총 150개의 랜덤 좌표에 magenta 색상으로 표시한다.
    각 세트는 지정된 범위의 랜덤 좌표(x, y)로 구성되어 있다.
    랜덤 좌표 정보는 randint() 함수로, 점을 찍는 함수는 scatter()로 구현하였다.

함수 간략 설명
    지정한 위치에 점을 그리는 함수 -scatter()
    cv.circle()은 한 개의 위치에 점을 그릴 수 있다.
    np.scatter()는 어레이의 좌표에 어레이에 지정된 형태의 다량한 점을 그릴 수 있다.

    matplotlib.pyplot.scatter(x, y,  # float or array-like, shape (n, ), (x, y) 좌표, (n,1) shape의 어레이도 된다.
    s=None,  # The marker size. float or array-like, shape (n, ), optional.
    c=None,  # The marker colors. array-like or list of colors or color, optional.
    marker=None,    # marker style. default: 'o'. 참고: https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
    cmap=None,      # colormap. default: 'viridis'
    norm=None, vmin=None, vmax=None, alpha=None,
    linewidths=None, *, edgecolors=None, plotnonfinite=False, data=None, **kwargs)

"""


# Import required packages:
import numpy as np
from matplotlib import pyplot as plt


# numpy.random.randint() - 특정 범위의 정수로 이루어진  특정 차원의 어레이 생성하기
# numpy.random.randint(low, high=None, size=None, dtype='l')
#   Return random integers from low (inclusive) to high (exclusive).
#   Return random integers from the “discrete uniform” distribution of the specified dtype in the “half-open” interval [low, high).
#   If high is None (the default), then results are from [0, low).

# Create data (three different 'clusters' of points (it should be of np.float32 data type):
# (x, y) 좌표를 의미하는 2D 데이터 포인트를 50개씩 선언한 것을 column 방향으로 묶는다.
# 첫 번째 스택 분석 사례: np.random.randint(0, 40, (50, 2))
#   위치값이 [0, 40)까지의 좌표 값이 존재하는 50개의 2차원 좌표 어레이를 생성한다.
#data = np.float32(np.vstack(
#    (np.random.randint(0, 40, (50, 2)), np.random.randint(30, 70, (50, 2)), np.random.randint(60, 100, (50, 2)))))

a = np.random.randint(0, 40, (50, 2))       # a의 값의 범위는 0~39. 이런 값들도 이루어진 좌표 데이터 50개.
#print('type(a)=', type(a), a.shape)        # type(a)= <class 'numpy.ndarray'> (50, 2)
b = np.random.randint(30, 70, (50, 2))      # b의 값의 범위는 30~69. 이런 값들도 이루어진 좌표 데이터 50개
c = np.random.randint(60, 100, (50, 2))     # c의 값의 범위는 60~100. 이런 값들도 이루어진 좌표 데이터 50개
print('type(a)=', type(a), a.shape)
print('type(b)=', type(b), b.shape)
print('type(c)=', type(c), c.shape)

data = np.float32(np.vstack((a, b, c)))     # a. b, c의 좌표로 150개의 좌표 데이터
print('type(data)=', type(data), data.shape) # type(data)= <class 'numpy.ndarray'> (150, 2)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(6, 6))
plt.suptitle("K-means clustering algorithm", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')
#fig.patch.set_alpha(0.2)

# Plot the 'original' data:
# scatter(x, y, s=None, c=None,..)
# x, y : array_like, shape (n, )
#
ax = plt.subplot(1, 1, 1)
plt.scatter(data[:, 0], data[:, 1], c='c')      # (x,y) 순으로 지정한다. m=magenta, c=cyan
#plt.scatter(data2[:, 0], data2[:, 1], c='y')      # (x,y) 순으로 지정한다. m=magenta, c=cyan
plt.title("data to be clustered")

# Show the Figure:
plt.show()
