"""

개요
    '../data/digits.png' 파일에서 5000개의 손글씨(20x20) 숫자 0~9를 읽어들여
    이것의 반은 KNN 학습에 사용하고, 나머지는 분류 정확성을 판단하는 데 사용한다.

주요 사항: KNN 학습모델을 만들고, 분류작업을 하기 위해서는 5단계부터 사용되는 다음 3가지의 함수 활용이 필요하다.
        1. cv2.ml.KNearest_create()
        2. knn.train(학습 데이터, cv2.ml.ROW_SAMPLE, 학습 데이터 정답)
        3. knn.findNearest(테스트 데이터, k)
        knn = cv2.ml.KNearest_create()
        knn.train(학습데이터, cv2.ml.ROW_SAMPLE, 학습_데이터_라벨)
        knn.findNearest(테스트_데이터, k)

처리 절차
    1. 영상 파일을 읽어들이고 라벨링을 행한다.
        digits.png에는 20x20으로 이루어진 손글씨 폰트가 100x50(가로x세로)개 나열되어 있다. 레이블(0~9)은 프로그램으로 자체 생성한다.
        영상파일을 가로로 자른 후 다시 세로로 잘라 (5000, 20, 20)의 어레이 변수에 저장한다.

    2~4. 학습에 적합한 데이터로 정렬한다.
        2. 학습을 위해 영상 데이터를 랜덤 변수를 이용하여 뒤섞는다.
        3. 파일로 존재하였던 영상데이터를 학습 데이터의 개수(5000개) 길이의 리스트로 재편성한다.
        4. png 파일에서 제공한 당초의 학습용 데이터를 knn 모델링하기 위한 학습 데이터와 테스트 데이터로 반씩 나눈다.
    5. knn 모델을 생성하고, 학습데이터로 학습시킨다. OpenCV에서는 두 함수가 필요하다. - create(), train()
    6. 나머지 반의 데이터로 테스트를 행하고 분류 성능을 분석한다.
        사용되는 함수: 결과=findNearest(테스트_데이터, k)
        k변화에 따른 정확도 변화를 관찰한다. 학습된데이터를 테스팅하니 K=1이면 100% 정확도를 보였다.

"""

# Import required packages:
import cv2
import numpy as np

# Constants:
ROW_SIZE = 20     # 글자 하나의 세로 해상도
COL_SIZE = 20     # 글자 하나의 가로 해상도
NUMBER_CLASSES = 10     # 영상 파일 안에 있는 글자의 클래스 10종(0~9)


def load_digits_and_labels(big_image):
    """Returns all the digits from the 'big' image and creates the corresponding labels for each image"""
    digits_img = cv2.imread(big_image, 0)

    number_cols = digits_img.shape[1] / COL_SIZE    # 세로 - 100글자
    number_rows = digits_img.shape[0] / ROW_SIZE    # 가로 - 50글자

    # (3) vsplit(어레이, indices_or_sections) 함수는 전체 영상을 세로로 잘라 number_rows(50)줄로 나눈 영상을 리스트 구조체로 반환한다.
    rows = np.vsplit(digits_img, number_rows)   # 영상을 가로(50개)로 나눈다.
    # rows[0].shape = (20, 2000)

    # 여기에 한 글자씩 담음
    digits = []

    # hsplit() 함수를 이용해 각 row(가로 축)에 대해 20 픽셀 단위로 잘라 list 구조체로 만든다. => 20 x20 화소 추출
    for row in rows:
        row_cells = np.hsplit(row, number_cols)
        for digit in row_cells:
            digits.append(digit)

    digits = np.array(digits)   # shape = (5000, 20, 20)


    # Create the labels for each image:
    # >>> np.repeat(3, 4) => array([3, 3, 3, 3])
    # np.arange(NUMBER_CLASSES) => [0 1 2 3 4 5 6 7 8 9]

    # [0~9]를 0을 500번, 1을 500번, ..., 9를 500번, 총 5,000개의 레이블을 생성한다.
    # labels = np.repeat([0 1 2 ... 9], 500) -> 각 클래스당 500번해서 리스트로 생성
    labels = np.repeat(np.arange(NUMBER_CLASSES), len(digits) / NUMBER_CLASSES) # (5000,)개의 영상과 레이블을 반환한다.

    return digits, labels


def get_accuracy(predictions, labels):
    """Returns the accuracy based on the coincidences between predictions and labels"""

    accuracy = (np.squeeze(predictions) == labels).mean()

    # 본 정확도는 test 데이터에 대해서만 환산됨에 유의...
    accuracy = (np.squeeze(predictions) == labels).mean()   # 'mean()' returns the average of the array elements.
    return accuracy * 100   # labels가 10개라서 맞힌 갯수를 평균(mean)을 구해서 100을 곱하면 정확도가 나온다.


# =======================================================================================================
# 프로그램의 시작
# =======================================================================================================

print('\n@@1. trainin data(20x20 image) and labels(0~9) will be produced; ')
digits, labels = load_digits_and_labels('digits.png')

# Randomly permute the sequence: 0~4999의 수를 랜덤하게 재배열한다.
#rand = np.random.RandomState(1234)      # Constructs a random number generator:
#shuffle = rand.permutation(len(digits))        # 1) 컨스트럭터를 이용한 셔플링
shuffle = np.random.permutation(len(digits))    # 2) 함수를 이용한 셔플링

# 데이터 셋을 섞는다
shuffle = np.random.permutation(len(digits))
print('1) type(shuffle)=', type(shuffle), 'shuffle.shape=', shuffle.shape)
# 0~4999의 뒤섞인 수로 영상과 레이블의 순서를 pair 상태로 순서를 뒤 섞는다.
digits, labels = digits[shuffle], labels[shuffle]


# ------------------------------------------------------------------------------------------------------
# 3. 3차원 학습용 영상에서 영상 부분을 직렬화시켜 이를 feature vector로 사용한 raw_descriptors를 만든다.
# digit.shape=(5000, 20, 20) -> raw_descriptors.shape(5000, 400)
#
# Compute the descriptors for all the images.
# In this case, the raw pixels are the feature descriptors
# ------------------------------------------------------------------------------------------------------
print('\n@@3. Arrange the training data as sequence list for knn-modeling\n'
      'Elements of the list are flattened images.')
# raw_descriptors: knn 모델링을 위한 학습 데이터(영상).
# 20x20의 영상을 1차원으로 나열한 영상 데이터가 5000개 있는 리스트 자료이다.
# 이것으로 ndarray로 변환해서 raw_descriptors.shape= (5000, 400)가 되게 할 예정이다.

# 1) 일단 영상을 직렬화(400,)시킨 리스트 자료(5000개의 원소)를 만든다.
raw_descriptors = []
for img in digits:      # digits.shape= (5000, 20, 20). 5000번 loop를 수행.
    raw_descriptors.append(np.float32(img.flatten()))  # 5000개의 1차원 데이터를 리스트에 추가

# raw_descriptors의 길이는 5000, 각 원소는 400 개의 요소를 가진 1차원 배열
print('1) raw_descriptors: type=', type(raw_descriptors), '| length=', len(raw_descriptors))    # 5000
print('2) raw_descriptors[0].shape=', raw_descriptors[0].shape)     # 400

# (5000, 400) 모양의 2차원 ndarray가 됨
raw_descriptors = np.array(raw_descriptors)     # 위의 스퀴즈 함수를 써도 되지만 이해가기 쉬운 방식으로 리스트 자료를 ndarray 타입으로 만들었다.
print('3) raw_descriptors: type=', type(raw_descriptors), '| shape=', raw_descriptors.shape)    # (5000, 400)

partition = int(0.5 * len(raw_descriptors))     # 중간 부분을 지정하는 index 번호
raw_descriptors_train, raw_descriptors_test = np.split(raw_descriptors, [partition])
labels_train, labels_test = np.split(labels, [partition])

print('1) labels_train: type=', type(labels_train), '| shape=', labels_train.shape)     # (2500, )
print('2) labels_test: type=', type(labels_test), '| shape=', labels_test.shape)        # (2500, )


print('\n@@5. Training KNN model - raw pixels as features')
knn = cv2.ml.KNearest_create()
knn.train(raw_descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)   # row로 구성된 학습데이터와 레이블을 제공한다.


k = 3
print(f'\n@@6. Test the model with k={k}.')

# ret - 성공할 경우 0 반환
# result - 테스트 데이터 포인트에 대한 이웃의 인덱스를 나타냄
# neighbours - 이웃과의 거리를 나타냄
# dist - 이웃과의 거리의 제곱을 나타냄

ret, result, neighbours, dist = knn.findNearest(raw_descriptors_test, k)
print('1) type(ret)=', type(ret), 'ret=', ret)     # 0이 아니면 됨.

print('2) result.shape=', result.shape, '| result[0, 0]=', result[0, 0])
# (2500, 1) | result[0, 0]= [1.] 0번째 결과는 문자 1이라고 말한다. ret와 같음.

print('3) neighbours.shape=', neighbours.shape,'| neighbours[0]=', neighbours[0])
# (2500, k) | neighbours[0]= [1. 4. 1.] k=3일 경우 사례: 주변의 k=3개가 2개는 1, 1개는 4라고 말한다.

print('4) dist.shape=', dist.shape, '| dist[0]=', dist[0])
# (2500, 3) | dist[0]= [ 88687. 116511. 144120.] 주변 k=3개와 test 점과의 개별 거리.

# 예측한(테스트)한 결과와 실제 값(레이블)을 일부분만 출력해 보도록 하자.
pr_start = 0; pr_end = 20
print(f"5) Sampling test_result vs labels_test: {pr_start} to {pr_end}")
print(f"   result: ", end='')
for i in range(pr_start, pr_end):
    print(f"{int(result[i, 0])}", end=' ')
print(f"\n    label: ", end='')
for i in range(pr_start, pr_end):
    print(f"{int(labels_test[i])}", end=' ')

acc = get_accuracy(result, labels_test)
print(f"\n6) Accuracy(k={k}) for test data: {acc:#6.2f}")

print(f'\n@@7. Accuracy for test and train data with varying k')

# k값을 바꾸어 가면 전체 정확도를 확인해 본다.
# 1) 학습된 데이터와 다른 test 데이터를 적용해서 KNN으로 가장 가까운 라벨을 추론했을 경우의 정확도
for k in [1, 3, 5]:    # 동률을 이루지 않도록 k는 홀수로 선정한다.
    ret, result, neighbours, dist = knn.findNearest(raw_descriptors_test, k)
    # Compute the accuracy:
    acc = get_accuracy(result, labels_test)     # test 데이터를 분류했을 때의 정확도를 확인
    print(f"k={k}: Accuracy for test data: {acc:#6.2f}")


print()
# 2) 학습된 데이터와 같은 학습 데이터를 적용해서 KNN으로 가장 가까운 라벨을 추론했을 경우의 정확도
for k in [1, 3, 5]:    # 동률을 이루지 않도록 k는 홀수로 선정한다.
    ret, result, neighbours, dist = knn.findNearest(raw_descriptors_train, k)
    # Compute the accuracy:
    acc = get_accuracy(result, labels_train)  # 학습데이터를 분류했을 때의 정확도를 확인
    print(f"k={k}: Accuracy for train data: {acc:#6.2f}")

