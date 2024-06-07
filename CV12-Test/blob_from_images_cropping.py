import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_cropped_img(img):
    """이미지의 중앙 부분을 자르고 반환합니다."""

    # 이미지의 복사본을 만듭니다.
    img_copy = img.copy()

    # 결과 이미지의 크기를 계산합니다.
    size = min(img_copy.shape[1], img_copy.shape[0])

    # x1과 y1 좌표를 계산합니다.
    x1 = int(0.5 * (img_copy.shape[1] - size))
    y1 = int(0.5 * (img_copy.shape[0] - size))

    # 이미지를 자르고 반환합니다.
    return img_copy[y1:(y1 + size), x1:(x1 + size)]


def show_img_with_matplotlib(color_img, title, pos):
    """Matplotlib을 사용하여 이미지를 표시합니다."""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def get_images_from_blob(blob_imgs, scalefactor, dim, mean, swap_rb, mean_added):
    """블롭에서 이미지를 추출합니다."""

    images_from_blob = cv2.dnn.imagesFromBlob(blob_imgs)
    imgs = []

    for image_blob in images_from_blob:
        # 블롭 이미지를 원래 크기로 변환하고 스케일링합니다.
        image_from_blob = np.reshape(image_blob, dim) / scalefactor
        image_from_blob_mean = np.uint8(image_from_blob)
        image_from_blob = image_from_blob_mean + np.uint8(mean)

        # 평균값이 추가되었는지 여부에 따라 처리합니다.
        if mean_added is True:
            if swap_rb:
                # BGR에서 RGB로 색상 순서를 변경합니다.
                image_from_blob = image_from_blob[:, :, ::-1]
            imgs.append(image_from_blob)
        else:
            if swap_rb:
                # BGR에서 RGB로 색상 순서를 변경합니다.
                image_from_blob_mean = image_from_blob_mean[:, :, ::-1]
            imgs.append(image_from_blob_mean)

    return imgs


# Load images and get the list of images:
image = cv2.imread("Images/benten.png")
image2 = cv2.imread("Images/oyuki.png")
images = [image, image2]

# 잘라내기 동작을 확인하기 위해 blobFromImage()와 blobFromImages() 함수가 수행하는
# 잘라내기 공식을 하나의 입력 이미지에 적용합니다.
cropped_img = get_cropped_img(image)
# cv2.imwrite("cropped_img.jpg", cropped_img)

# Call cv2.dnn.blobFromImages():
blob_images = cv2.dnn.blobFromImages(images, 1.0, (300, 300), [104., 117., 123.], False, False)
blob_blob_images_cropped = cv2.dnn.blobFromImages(images, 1.0, (300, 300), [104., 117., 123.], False, True)

# Get different images from the blob:
imgs_from_blob = get_images_from_blob(blob_images, 1.0, (300, 300, 3), [104., 117., 123.], False, True)
imgs_from_blob_cropped = get_images_from_blob(blob_blob_images_cropped, 1.0, (300, 300, 3), [104., 117., 123.], False,
                                              True)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 8))
plt.suptitle("cv2.dnn.blobFromImages() visualization with cropping", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Show the input images
show_img_with_matplotlib(imgs_from_blob[0], "img 1 from blob " + str(imgs_from_blob[0].shape), 1)
show_img_with_matplotlib(imgs_from_blob[1], "img 2 from blob " + str(imgs_from_blob[1].shape), 2)
show_img_with_matplotlib(imgs_from_blob_cropped[0], "img 1 from blob cropped " + str(imgs_from_blob[1].shape), 3)
show_img_with_matplotlib(imgs_from_blob_cropped[1], "img 2 from blob cropped " + str(imgs_from_blob[1].shape), 4)

# Show the Figure:
plt.show()
