"""
Handwritten digits recognition using KNN and HoG features and varying both k and the number of
training/testing images with pre-processing of the images


참고: HOG descriptor 링크
        Feature Engineering for Images: A Valuable Introduction to the HOG Feature Descriptor
        https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/

성능:
    50%를 학습에 사용했을 때 정확도 97.8%(k=3)
    90%를 학습에 사용했을 때 정확도 98.6%(k=2)

실험 결과: 학습에 사용한 데이터의 비율, K값의 선정에 따른..

        Training KNN(skew correction & Hog) model using 500 training data, which is 10.00% ...
        4500: k:Acc => 1:95.9 2:95.6 3:96.0 4:96.1 5:96.0 6:95.8 7:95.8 8:95.7 9:95.8

        Training KNN(skew correction & Hog) model using 1000 training data, which is 20.00% ...
        4000: k:Acc => 1:96.7 2:96.7 3:97.1 4:97.4 5:97.0 6:97.1 7:96.9 8:96.9 9:96.8

        Training KNN(skew correction & Hog) model using 1500 training data, which is 30.00% ...
        3500: k:Acc => 1:97.0 2:96.9 3:97.2 4:97.4 5:97.3 6:97.4 7:97.3 8:97.3 9:97.1

        Training KNN(skew correction & Hog) model using 2000 training data, which is 40.00% ...
        3000: k:Acc => 1:97.4 2:97.0 3:97.5 4:97.6 5:97.7 6:97.8 7:97.8 8:97.6 9:97.5

        Training KNN(skew correction & Hog) model using 2500 training data, which is 50.00% ...
        2500: k:Acc => 1:97.5 2:97.3 3:97.8 4:97.8 5:97.7 6:97.8 7:97.9 8:97.7 9:97.6

        Training KNN(skew correction & Hog) model using 3000 training data, which is 60.00% ...
        2000: k:Acc => 1:97.7 2:97.5 3:97.8 4:97.8 5:97.8 6:97.7 7:98.0 8:97.9 9:97.8

        Training KNN(skew correction & Hog) model using 3500 training data, which is 70.00% ...
        1500: k:Acc => 1:97.6 2:97.5 3:97.9 4:97.8 5:97.9 6:97.8 7:98.0 8:97.9 9:98.0

        Training KNN(skew correction & Hog) model using 4000 training data, which is 80.00% ...
        1000: k:Acc => 1:97.3 2:97.6 3:97.5 4:97.4 5:97.4 6:97.5 7:97.5 8:97.5 9:97.7

        Training KNN(skew correction & Hog) model using 4500 training data, which is 90.00% ...
        500: k:Acc => 1:97.6 2:98.6 3:98.0 4:98.2 5:97.6 6:98.0 7:97.6 8:97.8 9:98.0

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


def deskew(img):
    """Pre-processing of the images"""

    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SIZE_IMAGE * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SIZE_IMAGE, SIZE_IMAGE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def get_accuracy(predictions, labels):
    """Returns the accuracy based on the coincidences between predictions and labels"""

    accuracy = (np.squeeze(predictions) == labels).mean()
    return accuracy * 100


def get_hog():
    """ Get hog descriptor """

    # cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
    # L2HysThreshold, gammaCorrection, nlevels, signedGradient)
    hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE),   # winSize,
                            (8, 8),     # blockSize,
                            (4, 4),     # blockStride,
                            (8, 8),     # cellSize,
                            9,          # nbins,
                            1,          # derivAperture,
                            -1,         # winSigma,
                            0,          # histogramNormType,
                            0.2,        # L2HysThreshold,
                            1,          # gammaCorrection,
                            64,         # nlevels,
                            True)       # signedGradient
    print("hog descriptor size: '{}'".format(hog.getDescriptorSize()))
    return hog


def raw_pixels(img):
    """Return raw pixels as feature from the image"""

    return img.flatten()



# =================================================================
# 시작

# Load all the digits and the corresponding labels:
digits, labels = load_digits_and_labels('digits.png')

# Shuffle data
# Constructs a random number generator:
rand = np.random.RandomState(1234)
# Randomly permute the sequence:
shuffle = rand.permutation(len(digits))
digits, labels = digits[shuffle], labels[shuffle]

# HoG feature descriptor:
hog = get_hog()

# Compute the descriptors for all the images.
# In this case, the HoG descriptor is calculated
hog_descriptors = []
for img in digits:
    hog_ret = hog.compute(deskew(img))
    #print(f"type(hog_ret)={type(hog_ret)}, hog_ret.shape={hog_ret.shape}"); exit()
    hog_descriptors.append(hog_ret)
hog_descriptors = np.squeeze(hog_descriptors)

# Split data into training/testing:
split_values = np.arange(0.1, 1, 0.1)

# Create a dictionary to store the accuracy when testing:
results = defaultdict(list)

# Create KNN:
knn = cv2.ml.KNearest_create()


for split_value in split_values:
    # Split the data into training and testing:
    partition = int(split_value * len(hog_descriptors))
    hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [partition])
    labels_train, labels_test = np.split(labels, [partition])

    # Train KNN model
    #print('Training KNN model - HOG features')
    print(f'\n\nTraining KNN(skew correction & Hog) model using {partition} training data'
          f', which is {split_value*100:#.2f}% ...')
    knn.train(hog_descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)

    print(f'{len(labels_test)}: k:Acc => ', end='')
    # Store the accuracy when testing:
    for k in np.arange(1, 10):
        ret, result, neighbours, dist = knn.findNearest(hog_descriptors_test, k)
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
plt.title('Accuracy of the k-NN model varying both k and the percentage of images to train/test with pre-processing '
          'and HoG features')
plt.xlabel("number of k")
plt.ylabel("accuracy")
plt.show()
