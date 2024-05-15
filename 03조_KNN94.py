# 03조 (임주형,이세비,최하은)

"""
9.4 - knn이 짝수일 때 주변 판세에 영향을 받는다고 잠정 가정
=> 그림을 통해 결론을 도출
=> 교수님 의도 파악 중...
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

N = 16      # train 좌표
T = 4       # 테스트 케이스
k = 3       # knn의 k

np.random.seed(370001)  # 고정 시드 - 실험 1, 2
# np.random.seed(370601)  # 고정 시드 = 실험 3
data = np.random.randint(0, 100, (N, 2)).astype(np.float32)     # 0 ~ 100 범위, N 개의 2차원 좌표
labels = np.random.randint(0, 2, (N, 1)).astype(np.float32)     # 0, 1 Label

sample_arr = []
ret_arr = []
# results_arr = []
# neighbours_arr =[]
dist_arr = []

knn = cv2.ml.KNearest_create()
knn.train(data, cv2.ml.ROW_SAMPLE, labels)

for _ in range(T):
    sample = np.random.randint(0, 100, (1, 2)).astype(np.float32)   # 0 ~ 100 범위, 1개의 테스트 케이스 좌표
    sample_arr.append(sample)

    ret, result, neighbors, dist = knn.findNearest(sample, k)

    ret_arr.append(ret)
    # results_arr.append(result)
    # neighbours_arr.append(neighbors)
    dist_arr.append(dist)


fig = plt.figure(figsize=(10, 8))
plt.title(f'k={k}: label 0=red triangle, 1=blue rectangle')

red_triangles = data[labels.ravel() == 0]
plt.scatter(red_triangles[:, 0], red_triangles[:, 1], 200, 'r', '^')    # Label이 0인 point 출력

blue_squares = data[labels.ravel() == 1]
plt.scatter(blue_squares[:, 0], blue_squares[:, 1], 200, 'b', 's')      # Label이 1인 point 출력

for sample in sample_arr:
    plt.scatter(sample[:, 0], sample[:, 1], 200, 'g', 'o')      # 테스트 데이터의 포인트 출력

i = 0
for sample in sample_arr:
    tmp_dist = dist_arr[i]

    dist_max = np.sqrt(tmp_dist[0, k - 1])
    sample = np.int32(sample)

    color = ()
    if ret_arr[i] == 0.0:
        shp=patches.Circle((sample[0, 0], sample[0, 1]), radius=dist_max, color='r', fill=True)
    else:
        shp=patches.Circle((sample[0, 0], sample[0, 1]), radius=dist_max, color='b', fill=True)
    plt.text(sample[0, 0], sample[0, 1], s=str(i), fontsize=20)

    i += 1
    plt.gca().add_patch(shp)

plt.axis('scaled')
plt.grid("on")
plt.show()
