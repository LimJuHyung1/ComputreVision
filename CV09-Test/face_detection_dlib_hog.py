import cv2
import dlib

from matplotlib import pyplot as plt

path = 'Images/'
file = "mushoku_tensei2.png"


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_detection(image, faces):
    """Draws a rectangle over each detected face"""
    for face in faces:
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 5)
    return image


# Load image and convert to grayscale:
fullName = path + file
img = cv2.imread(fullName)
print('original img.shape=', img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Load frontal face detector from dlib:
detector = dlib.get_frontal_face_detector()
# detector은 callable object이다. 이 오브젝트는 함수처럼 파라미터를 지정하여 호출할 수 있다.
print('detector callable?: ', callable(detector))   # True


# Detect faces:
import time
s_time = time.time()
up_scale1 = 1    # 입력 영상 크기 그대로 사용하여 검출

rects_1 = detector(gray, up_scale1)
e1_time = time.time() - s_time

up_scale2 = 2    # 입력 영상의 2배 크기로 확장하여 검출. 작은 사진일 경우 필요할 수도..

rects_2 = detector(gray, up_scale2)
e2_time = time.time() - s_time
#print(f"execution time0 = {e1_time-s_time:.4f}, execution time1 = {e2_time-e1_time:.4f}")

# Draw face detections:
img_faces_1 = show_detection(img.copy(), rects_1)
img_faces_2 = show_detection(img.copy(), rects_2)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 6))
plt.suptitle(f"dlib HoG face detector: file={file}, shape={img.shape}", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(img_faces_1, f"detector(gray, {up_scale1}): num={len(rects_1)}, time={e1_time:#.4f}", 1)
show_img_with_matplotlib(img_faces_2, f"detector(gray, {up_scale2}): num={len(rects_2)}, time={e2_time:#.4f}", 2)

# Show the Figure:
plt.show()
