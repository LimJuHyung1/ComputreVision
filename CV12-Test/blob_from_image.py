import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Matplotlib을 사용하여 이미지를 표시하는 함수"""
    img_RGB = color_img[:, :, ::-1]  # BGR을 RGB로 변환
    ax = plt.subplot(1, 4, pos)  # 1행 4열의 서브플롯 설정
    plt.imshow(img_RGB)  # 이미지를 플롯에 표시
    plt.title(title)  # 제목 설정
    plt.axis('off')  # 축을 표시하지 않음


def get_image_from_blob(blob_img, scalefactor, dim, mean, swap_rb, mean_added):
    """블롭에서 이미지를 추출하는 함수"""
    images_from_blob = cv2.dnn.imagesFromBlob(blob_img)  # 블롭에서 이미지 추출
    image_from_blob = np.reshape(images_from_blob[0], dim) / scalefactor  # 이미지를 원래 크기로 변환
    image_from_blob_mean = np.uint8(image_from_blob)  # 이미지의 화소값을 8비트로 변환
    image_from_blob = image_from_blob_mean + np.uint8(mean)  # 평균값을 더함

    if mean_added is True:
        if swap_rb:
            image_from_blob = image_from_blob[:, :, ::-1]
        return image_from_blob
    else:
        if swap_rb:
            image_from_blob_mean = image_from_blob_mean[:, :, ::-1]
        return image_from_blob_mean


# Load image:
image = cv2.imread("Images/lum.png")

# Call cv2.dnn.blobFromImage():
blob_image = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104., 117., 123.], False, False)

# 블롭 이미지의 형태는 (1, 3, 300, 300)일 것
print(blob_image.shape)  # (1, 3, 300, 300) -> (N, Channel, Height, Width)

# 블롭에서 다른 이미지 추출
img_from_blob = get_image_from_blob(blob_image, 1.0, (300, 300, 3), [104., 117., 123.], False, True)
img_from_blob_swap = get_image_from_blob(blob_image, 1.0, (300, 300, 3), [104., 117., 123.], True, True)
img_from_blob_mean = get_image_from_blob(blob_image, 1.0, (300, 300, 3), [104., 117., 123.], False, False)
img_from_blob_mean_swap = get_image_from_blob(blob_image, 1.0, (300, 300, 3), [104., 117., 123.], True, False)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(16, 4))
plt.suptitle("cv2.dnn.blobFromImage() visualization", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Show the created images:
show_img_with_matplotlib(img_from_blob, "img from blob " + str(img_from_blob.shape), 1)
show_img_with_matplotlib(img_from_blob_swap, "img from blob swap " + str(img_from_blob.shape), 2)
show_img_with_matplotlib(img_from_blob_mean, "img from blob mean " + str(img_from_blob.shape), 3)
show_img_with_matplotlib(img_from_blob_mean_swap, "img from blob mean swap " + str(img_from_blob.shape), 4)

# Show the Figure:
plt.show()

