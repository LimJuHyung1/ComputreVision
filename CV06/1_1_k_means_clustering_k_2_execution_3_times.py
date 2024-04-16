"""
K-means clustering algorithm applied to three different 'clusters' of points (k=2)
개요
    x축, y축으로 구성된 2차 평면에 3단계의 랜덤한 분포를 갖는 50x3개의 점을
    K=2로 클러스터링하는 사례를 보인다.
    실습할 때는 실습데이터 1, 2중에 하나를 선택해서 사용한다.
        data set 1: 2개 영역으로 겹치지 않게 분포
        data set 2: 3개 영역으로 중첩하여 분포

함수 설명: 데이터의 배열 중에서 K개의 중심에 가장 가까운 데이터들을 레이블링하여 반환한다.
    cv.kmeans(	data, K, bestLabels, criteria, attempts, flags[, centers]	)
    -> 출력 값: retval, bestLabels, centers
    bestLabels은 None으로 입력을 해주어야 한다. - C++ 함수에서 입출력 겸용 변수임.
    centers=None으로 정의되 있기 때문에 생략해도 된다.

입력 파라미터
    data: 학습 데이터 행렬. numpy.ndarray. shape=(N, d), dtype=numpy.float32.
    K: 군집 개수
    bestLabels: 각 샘플의 군집 번호 행렬. numpy.ndarray. shape=(N, 1), dtype=np.int32.
    criteria: 종료 기준. (type, maxCount, epsilon) 튜플.
    attempts: 다른 초기 레이블을 이용해 반복 실행할 횟수.
    flags: 초기 중앙 설정 방법.
        cv2.KMEANS_RANDOM_CENTERS, cv2.KMEANS_PP_CENTERS, cv2.KMEANS_USE_INITIAL_LABELS 중 하나.

비고
    criteria에 있는 maxCount는 한 번의 시도(초기지점이 정해진 상황)에서 수행할 반복 횟수를 제어하고,
    - center point를 바꿔가며 데이터 포인트를 클러스터에 할당한다. 이 회수는 maxCount로 제어.
    attempts는 여러 번의 시도(초기지점을 바꾸면서 수행하는)를 수행하는 전체적인 시도 횟수를 제어한다.


반환값
centers: 군집 중심을 나타내는 행렬. np.ndarray. shape=(N, d), dtype=np.float32.
retval: Compactness measure
bestLabels: 각 샘플의 군집 번호 행렬. numpy.ndarray. shape=(N, 1), dtype=np.int32.
centers: 군집 중심을 나타내는 행렬. np.ndarray. shape=(N, d), dtype=np.float32.

질문
    cv.kmeans() 함수의 입력 파라미터와 반환값을 기술하시오.
    attempts와 maxCount의 용도가 무엇인지, 이 값이 성능이 clustering 성능에 미치는 영향에 대해서 설명하시오.

미션
    1. 본 프로그램의 2~4화면을 출력과 관련 루틴을 for loop를 사용하여 간략화시키시오.
    --> 문제 발견: cv.kmeans() 함수가 한 프로그램에 2번 호출되면 자동 최적화가 이루어지는 것으로 보인다.
    이를 해결하기 위해 다른 랜덤 데이터를 먼저 클러스터링하고, 다른 시켜보면 좀 개선되는 것으로 보이나, 별 의미를 찾을 수가 없다.
    --> 더 이상의 추적은 포기하는 것이 현명하다고 판단된다. 따라서 아래 2, 3번 미션도 시간투여 대비 얻는 지식에 별로 없어 보인다.
    2. 본 시뮬레이션 프로그램을 임의의 K에 대해서도 수행가능하도록 변경(개선)하시오.
    3. 주요 파라미터에 대한 변화에 따른 영향을 관찰할 수 있도록 해당 파라미터 변경에 다른 성능을 그래프로 표현하는 프로그램을 작성하시오.
      - 성능은 cv.kmeans() 함수의 반환 값 중에서 ret(compactness)를 활용하시오.
      - 다른 상황은 모두 고정한 채 한가지의 파라미터만 바꾸면서 시뮬레이션한다.
      - 이러한 작업을 10여차례 반복한 후 평균 성능을 비교한다.
"""

# Import required packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Create data (three different 'clusters' of points (it should be of np.float32 data type):

# Create data (three different 'clusters' of points (it should be of np.float32 data type):

# data set 1: 2개 영역으로 멀리 떨어져 분포
data1 = np.float32(np.vstack(
    (np.random.randint(0, 40, (50, 2)),
    np.random.randint(80, 120, (50, 2)))))

# data set 2: 2개 영역으로 근접하게 분포
data2 = np.float32(np.vstack(
    (np.random.randint(0, 40, (50, 2)),
    np.random.randint(50, 90, (50, 2)))))

# data set 3: 2개 영역으로 맞닿아 분포
data3 = np.float32(np.vstack(
    (np.random.randint(0, 40, (50, 2)),
    np.random.randint(40, 80, (50, 2)))))

# data set 4: 3개 영역으로 중첩없이 분포
data4 = np.float32(np.vstack(
    (np.random.randint(0, 40, (50, 2)),
     np.random.randint(40, 80, (50, 2)),
     np.random.randint(80, 120, (50, 2)))))

# data set 5: 3개 영역으로 중첩하여 분포
data5 = np.float32(np.vstack(
    (np.random.randint(0, 40, (50, 2)),
     np.random.randint(30, 70, (50, 2)),
     np.random.randint(60, 100, (50, 2)))))

data6 = np.float32(np.vstack(
    (np.random.randint(0, 500, (500, 2)),
    np.random.randint(600, 1200, (500, 2)))))

data = data6.copy()     # 입력은 data set 6이다.

print(f'1. data: {type(data)}, shape={data.shape}, len={len(data)}')   # data: <class 'numpy.ndarray'>, shape=(150, 2), len=150


K = 2       # 클러스터의 수를 정의한다.


# criteria: 종료 기준. (type, maxCount, epsilon) 튜플.
# Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
# 클러스터링 과정은 반복적으로 평균을 업데이트하고 데이터 포인트를 클러스터에 할당한다.
# 아래 2가지는 알고리즘의 수렴을 판단하는 기준으로 사용된다. 2가지 모두를 섞어(OR) 쓸 수있고, 1가지만 쓸 수도 있다.
# maxCount: 이러한 반복의 최대 횟수를 제한하는 파라미터이다.
#   maxCount 회수 만큼 평균을 업데이트하고 데이터 포인트를 클러스터에 할당하는 일을 반복한다.
# epsilon: 클러스터 중심의 이동량이 이 값보다 작아지면 알고리즘이 종료된다.

num_maxCount = 1; eps = 0.0005; num_attempts = 10     # num_attempts = 10 -> initial point 지정을 10번 하겠다
criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, num_maxCount, eps)        # 둘중 의 하나 조건 만족하면 중지

num_maxCount3 = 2; eps3 = 0.0005; num_attempts3 = 10
criteria3 = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, num_maxCount3, eps3)

num_maxCount4 = 10; eps4 = 0.0005; num_attempts4 = 10
criteria4 = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, num_maxCount4, eps4)


# 1번 화면 ---------------------------------------------------------------------------------------
# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 6))
plt.suptitle(f"K-means clustering algorithm: K={K}", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')
# fig.patch.set_alpha(0.2)

# Plot the 'original' data:
ax = plt.subplot(2, 2, 1)
plt.grid('on')
# scatter() 함수는 (x, y) 좌표를 사용한다.
# x 좌표=data[:, 0], y좌표=data[:, 1]
plt.scatter(data[:, 0], data[:, 1], c='c')
plt.title("data")


# 2번 화면 ---------------------------------------------------------------------------------------

# Apply k-means algorithm
# cv.kmeans() 함수
#   https://docs.opencv.org/4.5.4/d5/d38/group__core__cluster.html#ga9a34dc06c6ec9460e90860f15bcd2f88
#  retval, bestLabels, centers = cv.kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
#
#   K, which is the number of clusters to split the set by
#   attempts = 10, which specifies the number of times the algorithm is executed
#       using different initial labellings
#       (the algorithm returns the labels that yield the best compactness):
#       이 만큼의 시도를 해본 후 가장 적은 compactness를 내는 클러스터링 결과를 반환한다.
#   flags: 클러스터의 초기 선정을 어떻게 할 것인가를 결정
#       cv2.KMEANS_RANDOM_CENTERS selects random initial centers in each attempt.
#       cv2.KMEANS_PP_CENTERS: kmeans++ center initialization by Arthu 알고리즘 사용.
#       cv2.KMEANS_USE_INITIAL_LABELS: 프로그래머가 초기 위치를 지정. 이후 랜덤.
#       cv2.KMEANS_*_CENTERS: 선정 알고리즘을 프로그래머가 지정
# ret: compactness=sum of all (sample - center)**2
# bestLabels: 해당 샘플이 어느 군집에 속한다고 판단하는 것이 좋은지에 대한 레이블링.
#   해당 샘플과 똑 같은 길이의 라벨 자료는 (샘플의 수 x 1)의 차원을 갖는다.
#   라벨 자료는 샘플과 같은 순서로 배열되어 있다.
#   라벨은 K=2이면 라벨은 0, 1의 값을 갖는다.
#   라벨은 K=3이면 라벨은 0, 1, 2의 값을 갖는다.
#   K가 커진다고 해서 라벨의 차원이 달라질 것은 없다. 라벨데이터는 int32형으로 관리된다.
# center: K개의 클러스터의 중심점. K개 만큼 필요한데, 각 중심점이 (x, y) 좌표를 가지므로 (k, 2)의 차원을 갖는다.
#

ret, label, center = cv2.kmeans(data, K, None, criteria,
                                num_attempts, cv2.KMEANS_RANDOM_CENTERS)

print(f'2. 2번 화면 ret, compactness={ret:#.2f}')                                           # ret: 65391.29
#print(f'ret: {ret:#.2f}, normalized compactness={ret/(50*3 * 2):#.2f}')
# 2. ret, compactness=68776.15

print(f'3. label: type={type(label)}, shape={label.shape}, dtype={label.dtype}')
# 3. label: type=<class 'numpy.ndarray'>, shape=(150, 1), dtype=int32

print(f'4. center: type={type(center)}, shape={center.shape}')
# 4. center: type=<class 'numpy.ndarray'>, shape=(2, 2)

print('5. center:\n', center)
# 5. center:
#  [[65.639534 71.523254]
#  [21.390625 25.171875]]

# Now separate the data using label output (stores the cluster indices for every sample)
# Therefore, we split the data to different clusters depending on their labels:
# ravel() 함수를 이용해서 데이터 샘플을 1차원 한줄로 나열한다. 총 150개이므로 shape=(150,)
print(f'6. label.ravel(): shape={label.ravel().shape}')
# 6. label.ravel(): shape=(150,)

# K=2이므로 2개의 그룹 A, B로 레이블링을 한다. 0 혹은 1로 lavel이 부여된다.
A = data[label.ravel() == 0]    # data[150,2] 중에서 첫번째 원소에 대해 label 값이 0인 원소만 A에 배정
B = data[label.ravel() == 1]

# 각 그룹의 shape를 살펴본다.
# 그중, row는 합해서 샘플의 총 갯수가 되어야 한다.
# 랜덤하게 샘플데이터가 자리잡기 때문에 실험을 할 때마다 그룹의 길이가 달라질 것이다.
print(f'7. A: shape={A.shape}')    # A: shape=(78, 2)
print(f'8. B: shape={B.shape}')    # B: shape=(72, 2)

# Plot the 'clustered' data and the centroids
ax = plt.subplot(2, 2, 2)

# 그룹 A, B의 해당 좌표에 blue, green으로 점을 찍는다.
plt.scatter(A[:, 0], A[:, 1], c='b')
plt.scatter(B[:, 0], B[:, 1], c='g')

# 각 그룹의 중심점에 magenta 색상으로 squre, +를 크기 100으로 마킹한다.
plt.scatter(center[:, 0], center[:, 1], s=200, c='m', marker='+', linewidths=2)
#plt.title("clustered data and centroids (K = 2)")
plt.title(f"maxCount={num_maxCount}, epsilon={eps}, attempts={num_attempts}")


# 3번 화면 ---------------------------------------------------------------------------------------
ret, label, center = cv2.kmeans(data, K, None, criteria3,
                                num_attempts3, cv2.KMEANS_RANDOM_CENTERS)

A = data[label.ravel() == 0]
B = data[label.ravel() == 1]

ax = plt.subplot(2, 2, 3)

# 그룹 A, B의 해당 좌표에 blue, green으로 점을 찍는다.
plt.scatter(A[:, 0], A[:, 1], c='b')
plt.scatter(B[:, 0], B[:, 1], c='g')

# 각 그룹의 중심점에 magenta 색상으로 squre, +를 크기 100으로 마킹한다.
plt.scatter(center[:, 0], center[:, 1], s=200, c='m', marker='+', linewidths=2)
plt.title(f"maxCount={num_maxCount3}, epsilon={eps3}, attempts={num_attempts3}")
print(f'3번 화면. ret, compactness={ret:#.2f}')                                           # ret: 65391.29


# 4번 화면 ---------------------------------------------------------------------------------------
ret, label, center = cv2.kmeans(data, K, None, criteria4,
                                num_attempts4, cv2.KMEANS_RANDOM_CENTERS)


A = data[label.ravel() == 0]
B = data[label.ravel() == 1]

ax = plt.subplot(2, 2, 4)

# 그룹 A, B의 해당 좌표에 blue, green으로 점을 찍는다.
plt.scatter(A[:, 0], A[:, 1], c='b')
plt.scatter(B[:, 0], B[:, 1], c='g')

# 각 그룹의 중심점에 magenta 색상으로 squre, +를 크기 100으로 마킹한다.
plt.scatter(center[:, 0], center[:, 1], s=200, c='m', marker='+', linewidths=2)
plt.title(f"maxCount={num_maxCount4}, epsilon={eps4}, attempts={num_attempts4}")
print(f'4번 화면. ret, compactness={ret:#.2f}')                                           # ret: 65391.29


# Show the Figure:
plt.show()
