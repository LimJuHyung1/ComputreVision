# 03조 (임주형,이세비,최하은)

"""
8.4 - knn 모델은 학습한 데이터를 모두 저장하는가?
=> knn 모델을 xml로 저장하여 파일을 저장하고 읽어들여 파일의 크기를 출력했습니다.
=> 학습전과 학습후 이후에 sys.getsizeof()를 이용하여 모델 안에 학습데이터가 포함되는지 출력했습니다.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# L = 400
L = 40000       # 작은 수를 사용하면 프린트하기 수월하다.
data = np.random.randint(0, 100, (L, 2)).astype(np.float32)
labels = np.random.randint(0, 2, (L, 1)).astype(np.float32)
print('data for training:', type(data), data.shape)
print('Labels for training:', type(labels), labels.shape)

knn = cv2.ml.KNearest_create()

import time

s_time = time.time()
knn.train(data, cv2.ml.ROW_SAMPLE, labels)

import os
import sys
from os.path import getsize
import os

#파일 저장
file_path = 'knn_data.xml'
knn.save(file_path)
f_size = getsize("knn_data.xml")

print(f'knn file size before training={f_size}')

print(f'model size before training= {sys.getsizeof(knn)}')
file_size = os.path.getsize(file_path)
print(f"Model file size: {file_size} bytes")

# 파일 존재 여부 확인
if os.path.exists(file_path):
    # 파일 읽기 권한 확인
    if os.access(file_path, os.R_OK):
        print("File exists and is readable")
        knn_loaded = cv2.ml.KNearest_create()
        knn_loaded.load(file_path)
    else:
        print("File is not readable")
else:
    print("File does not exist")

# 전 후의 크기는 변함이 보이지 않는다

# 학습 데이터의 양 출력 (원본 데이터 크기를 기반으로)
print("Number of training samples:", len(data))

print(f'model size after training= {sys.getsizeof(knn_loaded)}')
file_size = os.path.getsize(file_path)
print(f"Model file size: {file_size} bytes")
