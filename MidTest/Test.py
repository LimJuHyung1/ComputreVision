
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

N = 16
T = 4
k = 3

data = np.random.randint(0, 100, (N, 2)).astype(np.float32)
labels = np.random.randint(0, 2, (N, 1)).astype(np.float32)

sample_arr = []
ret_arr = []
results_arr = []
neighbours_arr =[]
dist_arr = []

for _ in range(T):
    sample = np.random.randint(0, 100, (1, 2)).astype(np.float32)
    sample_arr.append(sample)

    knn = cv2.ml.KNearest_create()
    knn.train(data, cv2.ml.ROW_SAMPLE, labels)
    ret, result, neighbors, dist = knn.findNearest(sample, k)

    ret_arr.append(ret)
    results_arr.append(result)
    neighbours_arr.append(neighbors)
    dist_arr.append(dist)


fig = plt.figure(figsize=(10, 8))
plt.text(x=0, y=110,s=f"{k}-NN:num of green test data points={T}\n"
                      f"Circle color means classfied group\n"
                      f"2019305061 Lim Ju Hyung", fontsize=14, fontweight='bold')

red_triangles = data[labels.ravel() == 0]
plt.scatter(red_triangles[:, 0], red_triangles[:, 1], 200, 'r', '^')

blue_squares = data[labels.ravel() == 1]
plt.scatter(blue_squares[:, 0], blue_squares[:, 1], 200, 'b', 's')

for sample in sample_arr:
    plt.scatter(sample[:, 0], sample[:, 1], 200, 'g', 'o')

i = 0
for sample in sample_arr:
    tmp_dist = dist_arr[i]

    dist_max = np.sqrt(tmp_dist[0, k - 1])
    sample = np.int32(sample)

    color = ()
    if ret_arr[i] == 0.0:
        shp=patches.Circle((sample[0, 0], sample[0, 1]), radius=dist_max, color='r', fill=False)
    else:
        shp=patches.Circle((sample[0, 0], sample[0, 1]), radius=dist_max, color='b', fill=False)
    plt.text(sample[0, 0], sample[0, 1], s=str(i), fontsize=20)
    i += 1
    plt.axis('scaled')
    plt.gca().add_patch(shp)
    plt.grid("on")

plt.show()
