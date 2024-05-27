# 03조 (임주형,이세비,최하은)

"""
8.3 - 다른 예제(ex. 2_1_knn_handwritten_digits_recognition_varying_k.py)에서 사용했던
get_accuracy() 함수가 여기서(1_1_checking_knn_model_accuracy.py)는
계속 50%대의 정확도로 고정되어 출력된다.
어떤 오류가 숨겨져 있는지 원인을 파악하여 해결해 보자.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

print('\nStep 1: Making random training data & labels..')
L = 400
data = np.random.randint(0, 100, (L, 2)).astype(np.float32)
print('data for training:', type(data), data.shape)

labels = np.random.randint(0, 2, (L, 1)).astype(np.float32)
print('Labels for training:', type(labels), labels.shape)

print('\nStep 2: Creating & training a knn model..')
knn = cv2.ml.KNearest_create()

import time

s_time = time.time()
knn.train(data, cv2.ml.ROW_SAMPLE, labels)
print(f'training time={time.time() - s_time:#.2f}')

print('\nStep 3: Checking the knn model according to k selection..')
from collections import Counter

K = [1]


def print_ret_values(ret_val, rslt_v='no', nfg_v='no', dst_v='no'):
    retval, results, neighbours, dist = ret_val
    print('results:', type(results), results.shape)
    if rslt_v != 'no': print(results)
    print('neighbours:', type(neighbours), neighbours.shape)
    if nfg_v != 'no': print(neighbours)
    print('dist:', type(dist), dist.shape)
    if dst_v != 'no': print(dist)


def get_accuracy(predictions, labels):
    """
    정확도를 계산하는 함수
    :param predictions: 예측된 레이블
    :param labels: 실제 레이블
    :return: 정확도 (퍼센트)
    """
    # 결과의 shape를 출력
    print(f'results의 shape:             {predictions.shape} ')
    print(f'np.squeeze(results)의 shape: {np.squeeze(predictions).shape} ')
    print(f'labels의 shape:              {labels.shape} ')

    # 예측된 레이블과 실제 레이블이 같은지 비교하여 정확도를 계산
    accuracy = (np.squeeze(predictions) == labels).mean()

    # 정확도를 퍼센트로 반환
    return accuracy * 100


for k in K:
    print(f'\ntest=train: k={k}, num of test data={len(data)} -------')
    s_time = time.time()
    ret_val = knn.findNearest(data, k)
    e_time = time.time()
    print(f'testing time: whole={e_time - s_time:#.2f}, unit={(e_time - s_time) / len(data):#.2f}')

    print_ret_values(ret_val)

    ret, results, neighbours, dist = ret_val

    cmp = labels == results

    cmp_f = cmp.flatten()

    dict = Counter(cmp_f)

    print(f'test=train: L={L}, k={k}: Accuracy={dict[True] * 100 / len(cmp):#.2f}%')

    acc = get_accuracy(results, labels)
    print(f"k={k}: Accuracy2={acc:#6.2f}")
exit(0)
