import cv2
import numpy as np
from matplotlib import pyplot as plt

# 이미지 로드
image = cv2.imread("Images/basketball.png")

def show_img_with_matplotlib(color_img, title, pos):
    """Matplotlib을 사용하여 이미지를 표시합니다."""
    # BGR 형식을 RGB 형식으로 변환
    img_RGB = color_img[:, :, ::-1]
    # 서브플롯 생성
    ax = plt.subplot(1, 1, pos)
    # 이미지 표시
    plt.imshow(img_RGB)
    # 제목 설정
    plt.title(title)
    # 축 숨기기
    plt.axis('off')

# 클래스 이름을 로드
print("1. Load the names of the classes:")
rows = open('../CV12/1_1_opencv_object_classification_alexnet/synset_words.txt').read().strip().split('\n')
classes = [r[r.find(' ') + 1:].split(',')[0] for r in rows]

# Caffe 모델 로드
print("2. Load the serialized caffe model from disk:")
net = cv2.dnn.readNetFromCaffe("../CV12/1_1_opencv_object_classification_alexnet/bvlc_alexnet.prototxt",
                               "../CV12/1_1_opencv_object_classification_alexnet/bvlc_alexnet.caffemodel")

# 블롭 생성
print("3. Create the blob with a size of (227,227), mean subtraction values (104, 117, 123)")
blob = cv2.dnn.blobFromImage(image, 1, (227, 227), (104, 117, 123))
print(f"blob.shape={blob.shape}-> (입력 영상의 수, 채널의 수, 높이, 넓이)")

# 입력 블롭을 네트워크에 설정하고 추론 수행
print("4. Feed the input blob to the network, perform inference and get the output:")
net.setInput(blob)
preds = net.forward()
print(f"preds.shape={preds.shape}-> (입력 영상의 수, 클래스의 수)")

# 추론 시간 출력
print("5. Get inference time:")
t, _ = net.getPerfProfile()
print(f'Inference time: {(t * 1000.0 / cv2.getTickFrequency()):.2f} ms')

# 확률이 높은 상위 10개 클래스의 인덱스 가져오기
print("6. Get the 10 indexes with the highest probability (in descending order):")
indexes = np.argsort(preds[0])[::-1][:10]
print(f"type(indexes)={type(indexes)}, len(indexes)={len(indexes)}\nindexes={indexes}")

# 이미지에 최고 예측 클래스와 확률을 표시
text = f"label: {classes[indexes[0]]}\nprobability: {preds[0][indexes[0]] * 100:.2f}%"
print(f"top prediction=>" + text)

y0, dy = 30, 30
for i, line in enumerate(text.split('\n')):
    y = y0 + i * dy
    cv2.putText(image, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

# 상위 10개 클래스와 확률 출력
print("출력노드(preds[0])의 값이 큰 순서대로 10개를 추려 출력한다.")
for (index, idx) in enumerate(indexes):
    print(f"{index}) label: {classes[idx]}, probability: {preds[0][idx]:.10f}")

# Figure 설정 및 이미지 표시
fig = plt.figure(figsize=(10, 6))
plt.suptitle("Image classification with OpenCV using AlexNet", fontsize=14, fontweight='bold')
show_img_with_matplotlib(image, "Pre-trained Caffe models using ImageNet(1,000 classes image DB for classification)", 1)
plt.show()
