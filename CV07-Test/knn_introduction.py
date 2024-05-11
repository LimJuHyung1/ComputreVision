import cv2

# 훈련 데이터 및 레이블 생성
raw_descriptors_train = [[1, 2], [2, 3], [3, 4], [4, 5]]  # 각 훈련 데이터의 특징
labels_train = [0, 0, 1, 1]  # 각 훈련 데이터의 클래스 레이블

# k-NN 분류기 생성
knn = cv2.ml.KNearest_create()

# 훈련 데이터로 분류기 훈련
knn.train(raw_descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)

# 예측을 위한 테스트 데이터 생성
test_data = [[1.5, 2.5], [4.5, 5.5]]  # 테스트 데이터의 특징

# 테스트 데이터에 대한 예측 수행
_, results, _, _ = knn.findNearest(test_data, k=1)

# 결과 출력
print("Predictions:", results.flatten())
