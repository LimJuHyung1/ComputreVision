import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    """Matplotlib을 사용하여 이미지를 표시합니다"""
    # BGR 형식을 RGB 형식으로 변환
    img_RGB = color_img[:, :, ::-1]

    # 서브플롯 생성
    ax = plt.subplot(1, 4, pos)
    # 이미지 표시
    plt.imshow(img_RGB)
    # 제목 설정
    plt.title(title)
    # 축 숨기기
    plt.axis('off')

# 사전 훈련된 모델 로드
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# 이미지 로드 및 리스트에 저장
image = cv2.imread("Images/The_Quintessential_Quintuplets.jpg")
image2 = cv2.imread("Images/The_Quintessential_Quintuplets2.jpg")
images = [image.copy(), image2.copy()]

# cv2.dnn.blobFromImages() 함수 호출
blob_images = cv2.dnn.blobFromImages(images, 1.0, (300, 300), [104., 117., 123.], False, False)

# 블롭을 입력으로 설정하고 탐지 결과 얻기
net.setInput(blob_images)
detections = net.forward()

'''
detections[0, 0, i, 0]: 이미지 ID
detections[0, 0, i, 1]: 클래스를 나타내는 값 (얼굴 탐지의 경우, 일반적으로 1)
detections[0, 0, i, 2]: 신뢰도 (confidence)
detections[0, 0, i, 3:7]: 경계 상자의 좌표 (x1, y1, x2, y2)
'''

# 모든 탐지를 반복
for i in range(0, detections.shape[2]):
    # 각 탐지가 속한 이미지 ID 가져오기
    img_id = int(detections[0, 0, i, 0])    # i는 객체의 인덱스
    # 이 예측의 신뢰도 가져오기
    confidence = detections[0, 0, i, 2]

    # 약한 예측 필터링
    if confidence > 0.25:
        # 현재 이미지의 크기 가져오기
        (h, w) = images[img_id].shape[:2]

        # 탐지의 (x,y) 좌표 가져오기
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # 경계 상자와 신뢰도 표시
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(images[img_id], (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(images[img_id], text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Figure의 크기 설정 및 제목 지정
fig = plt.figure(figsize=(14, 8))
# fig = plt.figure()
plt.suptitle("OpenCV DNN face detector when feeding several images", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# 입력 이미지 및 탐지 결과 출력
show_img_with_matplotlib(image, "input img 1", 1)
show_img_with_matplotlib(image2, "input img 2", 3)
show_img_with_matplotlib(images[0], "output img 1", 2)
show_img_with_matplotlib(images[1], "output img 2", 4)

# Figure 표시
plt.show()
