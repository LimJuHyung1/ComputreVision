"""
아래 연습 프로그램은 본론에 해당하는 프로그램을 작성/분석을 위한 파이썬 프로그래밍에 관련된 연습 프로그램입니다.
contours_introduction_simple.py를 분석하기 위한 연습들입니다.

"""

import numpy as np
import cv2

"""
# ---------------------------------------------------------------------------------------
# Numpy에서는 row vector, column vector 모두 2차원이다.
# 1 dimension, Python 용어로는 sequence. list/tuple 처럼 나열형 자료 중 하나. len()함수가 통한다.
# ---------------------------------------------------------------------------------------
print("연습 0: row vector, column vector vs sequence")
row = np.array([[1, 2, 3]])
print(f'row: row.shape={row.shape}')
col = np.array([[1], [2], [3]])
print(f'col: col.shape={col.shape}')
r = row.flatten()
print(f'r: r.shape={r.shape} <- This is sequence data')
c = col.flatten()
print(f'c: c.shape={c.shape} <- This is sequence data')
print(f'len(r)={len(r)}, len(c)={len(c)}')

# row vector, column vector의 index=0을 지정해 sequence 자료를 얻을 수 있다.
print(row[0], col[0])
print(len(row[0]), len(col[0]))
exit(0)
"""


"""
# ---------------------------------------------------------------------------------------
print("연습 1: reshape() 메서드")
# reshape() 함수는 지정한 크기로 재배열하는 함수이다.
# 여기서 파라미터가 -1로 지정될 때가 있는데 이것의 의미는 다음과 같다.
# One shape dimension can be -1.
# In this case, the value is inferred from the length of the array and remaining dimensions.
# 즉, 남은 원소로 나머지 배열을 이루라는 뜻이다.
# ---------------------------------------------------------------------------------------
r = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
for i in [1, 2, 3, 4, 6]:
    rr = r.reshape(i, -1)
    rr0 = r.reshape(i, -1)[0]
    print(f'\nrr: r.reshape({i}, -1): ', rr.shape)
    print(f'\n', rr)
    print(f'\nrr0: r.reshape({i}, -1)[0]: ', rr0.shape)
    print(f'\n', rr0)

print("\n\n연습 2: reshape() 메서드와 squeeze()함수")
a1 = np.array([[1, 2, 3], [4, 5, 6]])
a2 = np.array([[[1, 2, 3], [4, 5, 6]]])
a3 = np.array([[[[1, 2, 3], [4, 5, 6]]]])

for a in [a1, a2, a3]:
    print('\na=', a)
    print('a.shape=', a.shape)
    squeeze = np.squeeze(a)
    print('a after squeezing: squeeze(a).shape=', squeeze.shape)
    s = squeeze.reshape(2, -1)
    print('After reshaping, s=', s)
    ss = np.squeeze(s)
    print('After squeezing, ss=', ss)
exit(0)
"""


"""
# ---------------------------------------------------------------------------------------
# 연습 3: squeeze() 함수
# expand_dims(), squeeze()는 서로 반대의 기능을 하는 함수이다.
# expand_dims()는 차원을 늘리는 함수이며, squeeze()는 차원을 줄이는 함수이다.
# ---------------------------------------------------------------------------------------

a = np.array([1, 2, 3])
print('a=', a.shape)        # a= (3,)

b = np.expand_dims(a, axis=0)
print('b:', b)
print('b=', b.shape)        # b= (1, 3)
s_b = np.squeeze(b)
print('s_b=', s_b.shape)    # s_b= (3,)


c = np.expand_dims(a, axis=1)
print('c=', c.shape)        # c= (3, 1)
s_c = np.squeeze(c)
print('s_c=', s_c.shape)    # s_c= (3,)

cc = np.expand_dims(c, axis=0)
print('cc=', cc.shape)      # cc= (1, 3, 1)
s_cc = np.squeeze(cc)
print('s_cc=', s_cc.shape)  # s_cc= (3,)

exit(0)
"""


# ---------------------------------------------------------------------------------------
# 연습 4: flatten() method
# 다차원 배열(array)을 1차원 배열로 나열한다.
#   NumPy의 ravel() 함수도 비슷한 기능을 수행한다.
#   ravel()은 나열하는 방식을 행, 열 등의 순서로 지정할 수 있다. => 검토 생략
# ---------------------------------------------------------------------------------------

a = np.array([[[1, 2, 3], [4, 5, 6]]])
print('a.shape=', a.shape, '\na=', a)
f = a.flatten()
print('f.shape=', f.shape, '\nf=', f)

r = a.ravel()
print('r.shape=', r.shape, '\nr=', r)

exit(0)

