import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    """Matplotlib을 사용하여 이미지를 표시합니다."""

    img_RGB = color_img[:, :, ::-1]
    # img_RGB = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    ax = plt.subplot(2, 4, pos)
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
image = cv2.imread("Images/sakura.png")
image2 = cv2.imread("Images/shinobu.png")
images = [image, image2]

# Call cv2.dnn.blobFromImages():
blob_images = cv2.dnn.blobFromImages(images, 1.0, (300, 300), [30., 40., 60.], False, False)
print(blob_images.shape)

# Get different images from the blob:
imgs_from_blob = get_images_from_blob(blob_images, 1.0, (300, 300, 3), [30., 40., 60.], False, True)
imgs_from_blob_swap = get_images_from_blob(blob_images, 1.0, (300, 300, 3), [30., 40., 60.], True, True)
imgs_from_blob_mean = get_images_from_blob(blob_images, 1.0, (300, 300, 3), [30., 40., 60.], False, False)
imgs_from_blob_mean_swap = get_images_from_blob(blob_images, 1.0, (300, 300, 3), [30., 40., 60.], True, False)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(16, 8))
plt.suptitle("cv2.dnn.blobFromImages() visualization", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Show the input images
show_img_with_matplotlib(imgs_from_blob[0], "img from blob " + str(imgs_from_blob[0].shape), 1)
show_img_with_matplotlib(imgs_from_blob_swap[0], "img from blob swap " + str(imgs_from_blob_swap[0].shape), 2)
show_img_with_matplotlib(imgs_from_blob_mean[0], "img from blob mean " + str(imgs_from_blob_mean[0].shape), 3)
show_img_with_matplotlib(imgs_from_blob_mean_swap[0],
                         "img from blob mean swap " + str(imgs_from_blob_mean_swap[0].shape), 4)
show_img_with_matplotlib(imgs_from_blob[1], "img from blob " + str(imgs_from_blob[1].shape), 5)
show_img_with_matplotlib(imgs_from_blob_swap[1], "img from blob swap " + str(imgs_from_blob_swap[1].shape), 6)
show_img_with_matplotlib(imgs_from_blob_mean[1], "img from blob mean " + str(imgs_from_blob_mean[1].shape), 7)
show_img_with_matplotlib(imgs_from_blob_mean_swap[1],
                         "img from blob mean swap " + str(imgs_from_blob_mean_swap[1].shape), 8)

# Show the Figure:
plt.show()
