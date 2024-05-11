"""
svm01_introduction.py를 불필요한 함수 정의를 자제하고, 나름 단순하게 만들어 본 프로그램
5개의 2차원, 2종 데이터로 SVM을 학습시켜 보고, 그 모델의 분류 능력을 검증해 본다.
- 위 데이터 5개로 학습시키고,
    학습한 것은 분류가 잘되는지,
    학습하지 않은 데이터에 대해서는 어떻게 분류해 낼 것인지를 테스트 결과를 색상(노란색, 남색)으로 표현한다.
Simple introduction to Support Vector Machine (SVM) technique with OpenCV

OpenCV Python Tutorial: Support Vector Machines (SVM)
    https://docs.opencv.org/4.x/d3/d02/tutorial_py_svm_index.html


유사한 프로그램이 OpenCV에 있음,
    https://docs.opencv.org/4.5.5/d1/d73/tutorial_introduction_to_svm.html

"""

# Import required packages:
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_svm_response(model, image):
    """Show the prediction for every pixel of the image, the training data and the support vectors"""

    # 레이블 1에 대해서는 BG(cyan), 레이블 -1에 대해서는 GR(yellow) 색상으로 표현한다.
    colors = {1: (255, 255, 0), -1: (0, 255, 255)}      # BGR 평면으로 가정한다.

    # Show the prediction for every pixel of the image:
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            sample = np.matrix([[j, i]], dtype=np.float32)
            response = svm_predict(model, sample)

            image[i, j] = colors[response.item(0)]

    # Show the training data:
    # Show samples with class 1:
    cv2.circle(image, (500, 10), 10, (255, 0, 0), -1)
    cv2.circle(image, (550, 100), 10, (255, 0, 0), -1)
    # Show samples with class -1:
    cv2.circle(image, (300, 10), 10, (0, 255, 0), -1)
    cv2.circle(image, (500, 300), 10, (0, 255, 0), -1)
    cv2.circle(image, (10, 600), 10, (0, 255, 0), -1)

    # Show the support vectors:
    support_vectors = model.getUncompressedSupportVectors()
    for i in range(support_vectors.shape[0]):
        print(support_vectors[i, 0])
        cv2.circle(image, (int(support_vectors[i, 0]), int(support_vectors[i, 1])), 15, (0, 0, 255), 6)

def show_img_with_matplotlib(color_img, title, pos):
    """ Shows an image using matplotlib capabilities """

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_svm_response2(model, image):
    """Show the prediction for every pixel of the image, the training data and the support vectors"""

    # 사전형 자료.
    print("\n6) 레이블 1 평면은 BG(cyan), 레이블 -1 평면은 GR(yellow) 색상으로 표현한다.")
    colors = {1: (255, 255, 0), -1: (0, 255, 255)}      # BGR 평면. cyan:레이블 1, yellow:레이블 -1

    # Show the prediction for every pixel of the image:

    # cv.ml_StatModel.predict(samples[, results[, flags]]	) -> retval, results
    #   samples: The input samples, floating-point matrix
    #   results: The optional output matrix of results.
    #   flags: The optional flags, model-dependent.
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            sample = np.array([[j, i]], dtype=np.float32)   # (x, y) 좌표 정보를 ndarray로 전달한다.
            #response = svm_model.predict(sample)[1].ravel()    # retval, results 중에서 첫 번째.
            _, response = svm_model.predict(sample)   # response: (1, 1) ndarray
            #print(1, type(response), response.shape)    # 1 <class 'numpy.ndarray'> (1, 1)
            response = response.ravel()     # response: (1,) ndarray
            #print(2, type(response), response.shape)    # 2 <class 'numpy.ndarray'> (1,)

            # ndarray.item(*args)
            #   Copy an element of an array to a standard Python scalar and return it.
            #   int_type: this argument is interpreted as a flat index into the array, specifying which element to copy and return
            it = response.item(0)       # 반환 값을 scalar로 만들기 위해 numpy.item() 함수를 사용하였다.
            #print(3, type(it), it)      # 3 <class 'float'> -1.0
            image[i, j] = colors[it]    # 사전형 자료 colors를 key=it로 액세스하여 컬러값을 인출함.
            #exit()

    # Show the training data:
    print("class 1 학습 데이터는 blue 색상으로 표현된다.")
    cv2.circle(image, (500, 10), 10, (255, 0, 0), -1)
    cv2.circle(image, (550, 100), 10, (255, 0, 0), -1)

    # Show samples with class -1: green
    print("class -1 학습 데이터는 green 색상으로 표현된다.")
    cv2.circle(image, (300, 10), 10, (0, 255, 0), -1)
    cv2.circle(image, (500, 300), 10, (0, 255, 0), -1)
    cv2.circle(image, (10, 600), 10, (0, 255, 0), -1)

    # Show the support vectors: 빨간색 원으로 표시한다.
    print("학습테이터 중 support vector들은 빨간색 원으로 마킹한다.")
    # getUncompressedSupportVectors()
    #   Retrieves all the uncompressed support vectors of a linear SVM.
    #
    # The method returns all the uncompressed support vectors of a linear SVM
    # that the compressed support vector, used for prediction, was derived from.
    # They are returned in a floating-point matrix,
    # where the support vectors are stored as matrix rows.
    support_vectors = model.getUncompressedSupportVectors()
    print("support vectors in (x, y) form:")
    for i in range(support_vectors.shape[0]):
        center = (int(support_vectors[i, 0]), int(support_vectors[i, 1]))
        center_text = (int(support_vectors[i, 0]), int(support_vectors[i, 1])+50)

        print(f"{i}: {center}")
        cv2.circle(image, center, 15, (0, 0, 255), 6)
        cv2.putText(image, f"{i}", center_text, cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.2, color=(0, 0, 255), thickness=3)




# 메인 루틴 시작  ----------------------------------------------------------------------------------------
if __name__ == '__main__':
    # 1) Set up training data: ----------------------------

    # 2차 평면의 특정 위치를 가진 5개의 학습 데이터를 정의한다. 학습데이터는 samples로 칭하기로 한다.
    samples = np.matrix([[500, 10], [550, 100], [300, 10], [500, 300], [10, 600]], dtype=np.float32)

    # 5개의 학습용 데이터(인스턴스)가 1, -1 로 이름붙은 2종의 레이블을 갖는다. --> responses와 같은 것이다.
    labels = np.array([1, 1, -1, -1, -1])

    print(f"1) Training data set: num={len(samples)}, samples.shape={samples.shape},"
          f"\nlabels는 responses와 같은 의미: labels.shape={labels.shape}, labels={labels}")

    # 데이터(좌표)와 레이블, 컬러를 출력한다
    for i in range(len(labels)):
        color = lambda x: "blue" if x == 1 else "green"
        print(samples[i], labels[i], color(labels[i]))
    # [[500.  10.]] 1 blue
    # [[550. 100.]] 1 blue
    # [[300.  10.]] -1 green
    # [[500. 300.]] -1 green
    # [[ 10. 600.]] -1 green


    # 2) Initialize the SVM model: -----------------------------
    # Creates empty model and assigns main parameters
    svm_model = cv2.ml.SVM_create()
    C_ = 12.5
    gamma_ = 0.50625
    print(f"2) SVM model created: C={C_}, gamma={gamma_}")

    # SVM kernel type
    svm_model.setKernel(cv2.ml.SVM_LINEAR)
    # svm_model.setKernel(cv2.ml.SVM_RBF)        # 오류 발생. 'NoneType' object has no attribute 'shape'
    # svm_model.setKernel(cv2.ml.SVM_POLY)        # 오류 발생. 모두 위의 오류 발생
    # svm_model.setKernel(cv2.ml.SVM_SIGMOID)        # 되는 것을 발견 못함

    # C_SVC:
    # C-Support Vector Classification. n-class classification (n ≥ 2),
    # allows imperfect separation of classes with penalty multiplier C for outliers.
    # 외곽 인스탄스에 대해 C 변수의 곱에 해당하는 패널티를 가지고 불안전한 분리를 허락한다.
    svm_model.setType(cv2.ml.SVM_C_SVC)

    svm_model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))


    # 3) Train the SVM: ---------------------------------
    # 다음 1), 2) 중의 하나를 이용해 학습된 모델을 확보한다.

    #"""
    # 방법 1) train 메서드를 사용해서 학습 데이터와 레이블을 이용해서 학습시킨다.
    #svm_train(svm_model, samples, labels)         # 함수 기반의 학습
    svm_model.train(samples, cv2.ml.ROW_SAMPLE, labels)
    # 학습한 모델을 다음 save 함수로 저장할 수도 있다.
    svm_model.save('my_svm.xml')
    #"""

    """
    # 방법 2)는 기 학습해서 저장한 모델을 불러와 활용하는 방법을 보인다.
    # 기 학습한 모델을 로드해서 사용할 수도 있다.
    svm_model = cv2.ml.SVM_create()
    svm_model = svm_model.load('my_svm.xml')
    """

    # 4) 일단 학습한 데이터에 대한 predict 결과를 보기로 하자. -학습한 데이터를 어떻게 분류하는지 보자.
    packed_return_values = svm_model.predict(samples)   # samples 자리에는 원래 test_data를 입력해야 한다.
    print(f"4.1) packed return values of predict(): type(packed_return_values)={type(packed_return_values)}")   # <class 'tuple'>
    ret, results = packed_return_values
    print(f"4.2) unpacked return values of predict(): ret={ret}, results.shape={results.shape}")    # ret=0.0, results.shape=(5, 1)
    print(f"results.ravel()={results.ravel()}")     # results.ravel()=[ 1.  1. -1. -1. -1.]
    print(f"results.flatten()={results.flatten()}") # results.flatten()=[ 1.  1. -1. -1. -1.]
    # results가 labels와 일치하는 것을 보니 학습이 잘 되었다.


    # 5) 이제 분류 결과 그림으로 살펴 보자 - 도시할 2차 평면(영상)을 정의한다.
    # 이곳에 학습 데이터를 분리하는 hyper plane(2차 평면상의 인스턴스에 대해서는 1차원 직선)을 도시한다.
    # Create the canvas (black image with three channels)
    # This image will be used to show the prediction for every pixel:
    img_output = np.zeros((640, 640, 3), dtype="uint8")


    # 6) Show the SVM response: --------------------
    # 모델의 학습 결과(support vectors, separated planes)를 제시하는 평면에 도시하여 반환한다.
    # support vectors에 의해 2개의 큰 평면(cyan, yellow)로 나뉜 모습을 볼 수 있다.
    # 이 동작을 아래 함수에 구현하였다.
    show_svm_response2(svm_model, img_output)   # img_output는 입력 영상이면서 함수의 반환값(영상)이기도 하다.

    # 7) Create the dimensions of the figure and set title: --------
    fig = plt.figure(figsize=(8, 6))
    plt.suptitle("SVM Training: class 1=blue, class -1=green \n-> support vectors=red", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')
    fig.patch.set_alpha(0.2)

    # 8) Plot the images: -------------------------
    show_img_with_matplotlib(img_output, "SVM Testing: class 1=cyan, class -1=yellow", 1)

    # Show the Figure:
    plt.show()
