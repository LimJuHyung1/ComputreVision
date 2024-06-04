import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

def show_detection(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return image

# 애니메이션 얼굴 인식을 위한 특화된 캐스케이드 파일 경로
anime_cascade_path = 'cascades/lbpcascades/lbpcascade_animeface.xml'

# 애니메이션 얼굴 인식을 위한 캐스케이드 클래스 로드
anime_face_cascade = cv2.CascadeClassifier(anime_cascade_path)

# 이미지 로드 및 그레이스케일 변환
path = 'Images/'
file = 'fff.png'
img = cv2.imread(path + file)
if img is None:
    raise FileNotFoundError(f"Could not read image: {path + file}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

scaleFactorList = [1.1, 1.2, 1.3, 1.5]
minNeighborsList = [1, 1, 1,  3]
minSizeList = [(8, 8), (8, 8), (8, 8), (8, 8)]
# minSizeList = [(4, 4), (8, 8), (16, 16), (24, 24)]
# minSizeList = [(8, 8), (16, 16), (20, 20), (24, 24)]
faces_anime = []
img_faces_anime = []

for i in range(4):
    faces_anime.append(anime_face_cascade.detectMultiScale(gray,
                                                         scaleFactor=scaleFactorList[i],
                                                         minNeighbors=minNeighborsList[i],
                                                         minSize=minSizeList[i]))

    # 얼굴 검출 결과 시각화
    img_faces_anime.append(show_detection(img.copy(), faces_anime[i]))

# 결과 출력
fig = plt.figure(figsize=(16, 8))
plt.suptitle("Face detection using anime face cascade classifier", fontsize=14, fontweight='bold')

for j in range(4):
    show_img_with_matplotlib(img_faces_anime[j],
                             "Anime Face Detection: " + str(len(faces_anime[j]))
                             + "\nScaleFactor: " + str(scaleFactorList[j])
                             + "\nminNeighbors: " + str(minNeighborsList[j])
                             + "\nminSize: " + str(minSizeList[j]),
                             j + 1)

plt.show()
