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

def get_cropped_imgs(imgs):
    """이미지를 잘라서 반환합니다"""
    imgs_cropped = []  # 자른 이미지를 저장할 리스트

    for img in imgs:
        # 이미지 복사본 생성
        img_copy = img.copy()

        # 결과 이미지의 크기 계산
        size = min(img_copy.shape[1], img_copy.shape[0])

        # 좌표 계산
        x1 = int(0.5 * (img_copy.shape[1] - size))
        y1 = int(0.5 * (img_copy.shape[0] - size))

        # 이미지를 자르고 리스트에 추가
        imgs_cropped.append(img_copy[y1:(y1 + size), x1:(x1 + size)])

    return imgs_cropped

# 사전 훈련된 모델 로드
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# 이미지 로드 및 리스트에 저장
image = cv2.imread("Images/The_Quintessential_Quintuplets.jpg")
image2 = cv2.imread("Images/The_Quintessential_Quintuplets2.jpg")
images = [image, image2]

# 자른 원본 이미지 가져오기
images_cropped = get_cropped_imgs(images)

# cv2.dnn.blobFromImages() 함수 호출
blob_cropped = cv2.dnn.blobFromImages(images, 1.0, (300, 300), [104., 117., 123.], False, True)

# 블롭을 입력으로 설정하고 탐지 결과 얻기
net.setInput(blob_cropped)
detections = net.forward()

# 모든 탐지를 반복
for i in range(0, detections.shape[2]):
    # 각 탐지가 속한 이미지 ID 가져오기
    img_id = int(detections[0, 0, i, 0])
    # 이 예측의 신뢰도 가져오기
    confidence = detections[0, 0, i, 2]

    # 약한 예측 필터링
    if confidence > 0.25:
        # 현재 이미지의 크기 가져오기
        (h, w) = images_cropped[img_id].shape[:2]

        # 탐지의 (x,y) 좌표 가져오기
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # 경계 상자와 신뢰도 표시
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10

        # 자른 이미지에 결과 표시
        cv2.rectangle(images_cropped[img_id], (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(images_cropped[img_id], text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Figure의 크기 설정 및 제목 지정
fig = plt.figure(figsize=(14, 8))
plt.suptitle("OpenCV DNN face detector when feeding several images and cropping", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# 입력 이미지 및 탐지 결과 출력
show_img_with_matplotlib(image, "input img 1", 1)
show_img_with_matplotlib(image2, "input img 2", 3)
show_img_with_matplotlib(images_cropped[0], "output cropped img 1", 2)
show_img_with_matplotlib(images_cropped[1], "output cropped img 2", 4)

# Figure 표시
plt.show()
