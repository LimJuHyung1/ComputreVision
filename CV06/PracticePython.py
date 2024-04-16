"""
Handwritten digits recognition using KNN and raw pixels as features
"""

# Import required packages:
import cv2
import numpy as np

# -----------------------------------------------------------------------------------
# split() method 연습
# -----------------------------------------------------------------------------------

# 1. 리스트를 내부 요소가 ndarray인 자료로 나눈다.
x = np.arange(18) #.reshape(3, 6)
print(type(x))
print(x)

# 리스트에 있는 18개의 원소를 3개의 ndarray로 나누어 이를 원소로 하는 리스트를 생성한다..
y = np.split(x, 3)
print('\n(x, 3): type(y)=', type(y), '\ny=', y)
print('type(y[0])=', type(y[0]))

# 리스트에 있는 18개의 원소를 6개의 ndarray로 나누어 이를 원소로 하는 리스트를 생성한다..
y = np.split(x, 6)
print('\n(x, 6): type(y)=', type(y), '\ny=', y)

#y = np.hsplit(x, 7)     # 오류 발생. 18을 7개로 나눌 수 없어서 오류가 발생한다.

y = np.split(x, [7])   # index=7부터 새로 나눈다. 2개의 ndarray를 갖는 list 생성
print('\n(x, [7]): type(y)=', type(y), '\ny=', y)

y = np.split(x, 9)
print('\n(x, 9): type(y)=', type(y), '\ny=', y)


y = np.split(x, [3, 7])    # index=3, 7부터 새로 나눈다. 2개의 ndarray를 갖는 list 생성
print('\n(x, [3, 7]): type(y)=', type(y), '\ny=', y)


# 2. ndarray도 나눌 수 있다.
x = np.arange(18).reshape(2, 9)
y = np.hsplit(x, 3)     # 2x9 행렬을 3개의 행렬로 나눈다.
print('\nx=', x)
print('(x, 3): type(y)=', type(y), '\ny=', y)


x = np.arange(18).reshape(3, 6)
y = np.split(x, 3)     # 3x6 행렬을 3개의 행렬로 나눈다. 참고: hsplit()와는 다른 결과가 나온다.
print('\nx=', x)
print('(x, 3): type(y)=', type(y), '\ny=', y)




"""
print(np.repeat(3, 4))      # 3을 4회 반복. array([3, 3, 3, 3])
x = np.array([[1, 2], [3, 4]])
print(np.repeat(x, 2))    # array([1, 1, 2, 2, 3, 3, 4, 4])

img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 10, 11]])
print('img.shape=', img.shape)
a = img.flatten()
print('a.shape=', a.shape, 'a=', a)
img2 = np.zeros((4, 3))
a = [img, img2]



# 2. Shuffle data
# RandomState.permutation(x)
#   Randomly permute a sequence, or return a permuted range.
#   >>> np.random.permutation(10)
#   array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6]) # random
#   >>> np.random.permutation([1, 4, 9, 12, 15])
#   array([15,  1,  9,  4, 12]) # random

# numpy.squeeze()
#   Remove single-dimensional entries from the shape of an array.
#   >>> x.shape
#   (1, 3, 1)
#   >>> np.squeeze(x).shape
#   (3,)

b = np.squeeze(a)
print('type(b)=', type(b), '| b.shape=', b.shape)

"""


