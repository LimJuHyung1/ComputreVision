import numpy as np
from matplotlib import pyplot as plt

a = np.random.randint(0, 40, (50, 2))       # a의 값의 범위는 0~39. 이런 값들도 이루어진 좌표 데이터 50개.
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

ax = plt.subplot(1, 1, 1)
plt.scatter(data[:, 0], data[:, 1], c='m')
#plt.scatter(data[:, 0], data[:, 1], c='c')      # (x,y) 순으로 지정한다. m=magenta, c=cyan
#plt.scatter(data2[:, 0], data2[:, 1], c='y')      # (x,y) 순으로 지정한다. m=magenta, c=cyan
plt.title("data to be clustered")

# Show the Figure:
plt.show()
