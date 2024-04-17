"""
Handwritten digits recognition using KNN and raw pixels as features and varying both k and the number of
training/testing images with pre-processing of the images

주의:
    이 프로그램의 수행을 위해서는 현재 폴더 위의 data 폴더에 다음 파일이 준비되어 있어야 한다.
        '../data/digits.png'



"""

# Import required packages:
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Constants:
SIZE_IMAGE = 20
NUMBER_CLASSES = 10


def load_digits_and_labels(big_image):
    """ Returns all the digits from the 'big' image and creates the corresponding labels for each image"""

    # Load the 'big' image containing all the digits:
    digits_img = cv2.imread(big_image, 0)

    # Get all the digit images from the 'big' image:
    number_rows = digits_img.shape[1] / SIZE_IMAGE
    rows = np.vsplit(digits_img, digits_img.shape[0] / SIZE_IMAGE)

    digits = []
    for row in rows:
        row_cells = np.hsplit(row, number_rows)
        for digit in row_cells:
            digits.append(digit)
    digits = np.array(digits)

    # Create the labels for each image:
    labels = np.repeat(np.arange(NUMBER_CLASSES), len(digits) / NUMBER_CLASSES)
    return digits, labels

"""Pre-processing of the images"""
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SIZE_IMAGE * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SIZE_IMAGE, SIZE_IMAGE),
                         flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def get_accuracy(predictions, labels):
    """Returns the accuracy based on the coincidences between predictions and labels"""

    accuracy = (np.squeeze(predictions) == labels).mean()
    return accuracy * 100


def raw_pixels(img):
    """Return raw pixels as feature from the image"""

    return img.flatten()



# =================================================================
# 시작
if __name__ == '__main__':
    # Load all the digits and the corresponding labels:
    digits, labels = load_digits_and_labels('digits.png')

    # 1) 데이터 뒤섞기: 영상 데이터와 레이블 뒤섞기 ---------------------------------------
    # Constructs a random number generator:
    rand = np.random.RandomState(1234)
    # Randomly permute the sequence:
    shuffle = rand.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]

    # 2) 영상 데이터의 skew를 바로 잡아서 1차원 배열로 만들기 ------------------------------
    # Compute the descriptors for all the images.
    # In this case, the raw pixels are the feature descriptors
    raw_descriptors = []
    for img in digits:
        raw_descriptors.append(np.float32(raw_pixels(deskew(img))))
    raw_descriptors = np.squeeze(raw_descriptors)

    # 3) 데이터를 training용과 testing으로 나눌 준비를 하고, knn 객체를 생성한다.  ----------------------

    # 나누는 지점의 단계를 정한다. 10% ~ 90%
    split_values = np.arange(0.1, 1, 0.1)

    # Create a dictionary to store the accuracy when testing:
    # KNN의 분류 데이터를 담을 수 있는 사전형 자료를 미리 지정한다. value는 list 구조체이다.
    results = defaultdict(list)

    # Create KNN:
    knn = cv2.ml.KNearest_create()

    # 4) k가 변할 때마다 학습과 테스트의 비중을 바꾸어 가며 학습/테스팅을 수행하면서 정확도를 산출하여 사전형 자료(results)에 담는다.
    for split_value in split_values:        # split_values=[0.1, ..., 0.9]

        # Split the data into training and testing:
        partition = int(split_value * len(raw_descriptors))     # [0.1, ..., 0.9] * 5,000
        raw_descriptors_train, raw_descriptors_test = np.split(raw_descriptors, [partition])
        labels_train, labels_test = np.split(labels, [partition])

        # Train KNN model
        #print('Training KNN model - raw pixels as features')
        print(f'\n\nTraining KNN(skew correction) model using {partition} training data, which is {split_value*100:#.2f}% ...')
        knn.train(raw_descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)

        print(f'{len(labels_test)}: k:Acc => ', end='')
        # k를 증가시켜가면서 분류작업을 시행하고, 그때의 정확도를 기록한다.:
        for k in np.arange(1, 10):
            ret, result, neighbours, dist = knn.findNearest(raw_descriptors_test, k)
            acc = get_accuracy(result, labels_test)
            #print(" {}".format("%.2f" % acc))
            print(f"{k}:{acc:#.1f}", end=' ')
            results[int(split_value * 100)].append(acc)

    # Show all results using matplotlib capabilities:
    # Create the dimensions of the figure and set title:
    fig = plt.figure(figsize=(12, 5))
    plt.suptitle("k-NN handwritten digits recognition", fontsize=14, fontweight='bold')
    #fig.patch.set_facecolor('silver')

    ax = plt.subplot(1, 1, 1)
    ax.set_xlim(0, 10)
    dim = np.arange(1, 10)

    for key in results:
        ax.plot(dim, results[key], linestyle='--', marker='o', label=str(key) + "%")

    plt.legend(loc='upper left', title="% training")
    plt.title('Accuracy of the k-NN model varying both k and the percentage of images to train/test with pre-processing')
    plt.xlabel("number of k")
    plt.ylabel("accuracy")
    plt.show()
