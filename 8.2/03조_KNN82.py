# 03조 (임주형,이세비,최하은)

"""
8.2 - knn은 학습한 데이터는 모두 인식하는가?
=> 학습데이터의 수와 data의 랜덤값을 적게 설정하여 출력하여 data 값에 따라서 labels와 neighbours의 값을 비교
=> digit.png에서도 label를 출력하여 랜덤데이터와 문자데이터에 따라 학습한데이터가 어떻게 다른지 확인
"""

import cv2
import numpy as np


# 1. 학습데이터(랜덤데이터)로 정확도 출력

L = 10
data = np.random.randint(0, 2, (L, 2)).astype(np.float32)
labels = np.random.randint(0, 2, (L, 1)).astype(np.float32)

knn = cv2.ml.KNearest_create()

import time
s_time = time.time()
knn.train(data, cv2.ml.ROW_SAMPLE, labels)

from collections import Counter
K = [1]
def print_ret_values(ret_val, rslt_v='no', nfg_v='no', dst_v='no'):
    retval, results, neighbours, dist = ret_val
    print('data:',type(data),'\n',data)
    print('results:', type(results), results.shape)
    if rslt_v != 'no': print(results)
    print('neighbours:', type(neighbours), neighbours.shape)
    if nfg_v != 'no': print(neighbours)
    print('dist:', type(dist), dist.shape)
    if dst_v != 'no': print(dist)

def get_accuracy(predictions, labels):
    accuracy = (np.squeeze(predictions) == labels).mean()

for k in K:
    print(f'\ntest=train: k={k}, num of test data={len(data)} -------')
    s_time = time.time()
    ret_val = knn.findNearest(data, k)
    e_time = time.time()
    print(f'testing time: whole={e_time - s_time:#.2f}, unit={(e_time - s_time)/len(data):#.2f}')
    print_ret_values(ret_val, rslt_v='yes', nfg_v='yes', dst_v='yes')
    print('labels:',type(labels),'\n',labels)
    ret, results, neighbours, dist = ret_val

    cmp = labels == results
    cmp_f = cmp.flatten()
    dict = Counter(cmp_f)
    print(f'test=train: L={L}, k={k}: Accuracy={dict[True] * 100 / len(cmp):#.2f}%')


# 2. 학습데이터(문자데이터인 digit.png)로 정확도 출력

SIZE_IMAGE_ROW = 20
SIZE_IMAGE_COL = 20
NUMBER_CLASSES = 10
def load_digits_and_labels(big_image):
    digits_img = cv2.imread(big_image, 0)
    number_cols = digits_img.shape[1] / SIZE_IMAGE_COL
    number_rows = digits_img.shape[0] / SIZE_IMAGE_ROW
    rows = np.vsplit(digits_img, number_rows)

    digits = []
    for row in rows:
        row_cells = np.hsplit(row, number_cols)
        for digit in row_cells:
            digits.append(digit)
    digits = np.array(digits)

    labels = np.repeat(np.arange(NUMBER_CLASSES), len(digits) / NUMBER_CLASSES)
    return digits, labels
def get_accuracy(predictions, labels):
    accuracy = (np.squeeze(predictions) == labels).mean()
    return accuracy * 100

digits, labels = load_digits_and_labels('../data/digits.png')
for i in digits[0]:
    print(i)

shuffle = np.random.permutation(len(digits))
digits, labels = digits[shuffle], labels[shuffle]

raw_descriptors = []
for img in digits:
    raw_descriptors.append(np.float32(img.flatten()))

raw_descriptors = np.array(raw_descriptors)
partition = int(0.5 * len(raw_descriptors))

raw_descriptors_train, raw_descriptors_test = np.split(raw_descriptors, [partition])
labels_train, labels_test = np.split(labels, [partition])

knn = cv2.ml.KNearest_create()
knn.train(raw_descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)

ret, result, neighbours, dist = knn.findNearest(raw_descriptors_test, 1)
acc = get_accuracy(result, labels_test)     # test 데이터를 분류했을 때의 정확도를 확인
print(f"k={1}: Accuracy for test data: {acc:#6.2f}")

ret, result, neighbours, dist = knn.findNearest(raw_descriptors_train, 1)
acc = get_accuracy(result, labels_train)  # 학습데이터를 분류했을 때의 정확도를 확인
print(f"k={1}: Accuracy for train data: {acc:#6.2f}")