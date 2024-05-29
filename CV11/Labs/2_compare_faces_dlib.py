"""
This script makes used of dlib library to calculate the 128D descriptor to be used for face recognition
and compare the faces using some distance metrics
    http://dlib.net/python/index.html

개요
    얼굴에서 추출한 128개의 벡터로 구성된 dlib face descriptor를 이용하여
    4명의 인물과 1명의 신원 불명의 사람 얼굴 사진으로 미지 인물이 4인 중 누구에 가장 가까운지 맞추는 사례 프로그램

    5인의 얼굴 디스크립터를 구하여 미지 인물의 것이 나머지 4인의 face descriptor의 누구 것과 가장
    가까운지를 norm(유클리디언 거리)으로 판단한다.
    이 프로그램은 dlib library를 사용한다. 내부 처리과정에서 dlib hog로 얼굴을 검출한다.

비고
    이 프로그램은 1회성의 프로그램이라 범용으로 사용하기 위해서는 대대적인 수정이 필요한 것을 감안하여 분석하기 바랍니다.
    -> 정리하여 ver.2를 만들었으나 자율적 학습을 위해 소스는 공개하지 않는다.

"""

# Import required packages:
import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt

# path for dlib model
model_path = '../data/dlib_face_recog/'
def show_img_with_matplotlib(color_img, title, row, col, pos):
    """Shows an image using matplotlib capabilities"""
    #img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(row, col, pos)
    plt.imshow(color_img)
    plt.title(title)
    plt.axis('off')



# landmark 검출기(shape predictor-검출기)를 다운로드하고 그 객체(callable object)를 생성한다.
pose_predictor_5_point = dlib.shape_predictor(model_path + "shape_predictor_5_face_landmarks.dat")
# callable object는 함수로 사용할 수 있다.
# 이 검출기는 랜드마크 검출 예제(2_@landmarks_detection_dlib_from_videofile.py)에 쓰던 것과 다른 것이다.
# 모델 파일도 같은 위치에 저장되어 있다.
# model down load: https://github.com/davisking/dlib-models/blob/master/shape_predictor_5_face_landmarks.dat.bz2


# face enconder를 다운로드하고, 그 객체(callable object)를 생성한다.
face_encoder = dlib.face_recognition_model_v1(model_path + "dlib_face_recognition_resnet_model_v1.dat")
# callable object는 함수로 사용할 수 있다.
# model down load: https://github.com/davisking/dlib-models/blob/master/dlib_face_recognition_resnet_model_v1.dat.bz2

# 얼굴 검출기 객체(callable object)를 생성한다.
detector = dlib.get_frontal_face_detector()     # dlib hog face detector을 사용하였음.
# callable object는 함수로 사용할 수 있다.


def compare_faces(encodings, encoding_to_check):
    # Returns the distances when comparing a list of face encodings against a candidate to check
    # 입력:
    #   encodings: 누군지 알고 있는 얼굴에 대해 dlib 함수로 적용하여 추출한 128차원의 face descriptor들의 list 자료
    #   encoding_to_check: 비교하고자 하는 신원 미상의 dlib face descriptor
    # 반환값:
    #   encodings list에 있는 여러 개의 encoding과 1개의 encoding_to_check의 인코딩 값을
    #   각 차원별로 뺀 유클리디언 거리를 리스트로 반환한다.
    #   작을 수록 encoding_to_check의 얼굴과 가깝다.
    return list(np.linalg.norm(encodings - encoding_to_check, axis=1))
    # linalg.norm 링크: 두 디스크립터간의 차이에 대한 norm을 계산한다.
    # If axis is an integer, it specifies the axis of x along which to compute the vector norms
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

def compare_faces_ordered(encodings, face_names, encoding_to_check):
    # 위와 같은 함수인데 반환할 때 norm의 값을 크기 순으로 소팅(작은 값부터..)하여 반환한다.
    # distances: 매칭값 순으로 작은 값부터 나열하여 반환한다. ...  작은 값이 가장 가까운 얼굴이다.
    # face_names: 매칭값에 따라 face_names 순서도 바꾸어 반환한다.
    distances = list(np.linalg.norm(encodings - encoding_to_check, axis=1))
    return zip(*sorted(zip(distances, face_names)))


def face_encodings(face_image, number_of_times_to_upsample=1, num_jitters=1):
    # Returns the 128D dlib face descriptor for each face in the image
    gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    face_locations = detector(gray, number_of_times_to_upsample) # 그레이 영상으로 얼굴을 검출한다.

    # 주어진 얼굴의 위치로 5 지점의 landmark를 검출한다.
    raw_landmarks = [pose_predictor_5_point(face_image, face_location) for face_location in face_locations]

    # Calculate the face encoding for every detected face using the detected landmarks for each one:
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for
            raw_landmark_set in raw_landmarks]

# 1) 실험 데이터 선택: 다음 4세트 중의 하나만 주석문을 해제하시오. 파일 확장자는 jpg를 가정한다.
# 선택된 사람이 위 4인 중에서 누구와 가장 가까운 것인가를 128차원 descriptor 정보로 결정한다.
data_path = "face_files/"    # 영상 파일의 위치
set0 = ["jared_1.jpg", "jared_2.jpg", "jared_3.jpg", "obama.jpg", "jared_4.jpg"]
set1 = ["jared_1.jpg", "jared_2.jpg", "jared_3.jpg", "obama.jpg", "obama2.jpg"]
set2 = ["obama2.jpg", "obama3.jpg", "obama4.jpg", "obama5.jpg", "obama6.jpg"]
set3 = ["obama2.jpg", "obama3.jpg", "obama5.jpg", "obama6.jpg", "obama4.jpg"]
data_set_list = [set0, set1, set2, set3]
set_num = 0
names = data_set_list[set_num]
print(f"사용된 영상 실험 세트={set_num}, 미지의 인물 파일명={names[4]}")

# 2) 미리 누구인지 아는 사람들의 사진을 순서대로 읽어들인다.
# 맨 마지막 사진은 test 얼굴 영상이다.
# 이것과 나머지 사진들의 fcae descreptor 비교를 행한다.
known_image_1 = cv2.imread(data_path + names[0])
known_image_2 = cv2.imread(data_path + names[1])
known_image_3 = cv2.imread(data_path + names[2])
known_image_4 = cv2.imread(data_path + names[3])

# 3) 모르는 인물의 사진을 읽어들인다.
unknown_image = cv2.imread(data_path + names[4])

# 4) from BGR(OpenCV format) to RGB(dlib format):
known_image_1 = known_image_1[:, :, ::-1]
known_image_2 = known_image_2[:, :, ::-1]
known_image_3 = known_image_3[:, :, ::-1]
known_image_4 = known_image_4[:, :, ::-1]
unknown_image = unknown_image[:, :, ::-1]

# face_encodings은 검출된 인물들의 descriptor 정보를 각 사람마다 ndarray 데이터로 만들어
# 리스트 자료형으로 반환한다.
# 여러 사람이 있는 사진일 경우를 대비해 그 중 0번째를 반환 받는다.
# 미지의 인물 사진도 0번째 인물 사진에 대해 비교할 것이다.

# 5) Create the encodings for both known images & unknown image:
known_image_1_encoding = face_encodings(known_image_1)[0]
known_image_2_encoding = face_encodings(known_image_2)[0]
known_image_3_encoding = face_encodings(known_image_3)[0]
known_image_4_encoding = face_encodings(known_image_4)[0]
known_encodings = [known_image_1_encoding, known_image_2_encoding,
                   known_image_3_encoding, known_image_4_encoding]
print(f"5) type(known_encodings)={type(known_encodings)}, len(known_encodings)={len(known_encodings)}")
unknown_encoding = face_encodings(unknown_image)[0]         # 여러 사람일 때는 index 번호를 바꿀 수 있다.
print(f"type(unknown_encoding)={type(unknown_encoding)}, unknown_encoding.shape={unknown_encoding.shape}")

all_encodings = known_encodings + [unknown_encoding]    # 모르는 인물의 인코딩까지 포함한 5인의 인코딩 생성
print(f"type(all_encodings)={type(all_encodings)}, len(all_encodings)={len(all_encodings)}")


# 6) Compare faces: 5개의 얼굴 디스크립터와 맨 마지막 디스크립터와의 유클리디언 거리를 계산한다.
# compare_faces() 함수는 unknown_encoding을 known_encodings list에 있는
# 여러개의 encoding에 비교한 유클리디언 거리(각 차원별 오차를 제곱한 것을 더해서 root를 취함)를 리스트로 반환한다.
# 유클리디언 거리가 0.6이하면 동일 인물로 본다.
# all_encodings[-1]은 미지의 얼굴의 인코딩(unknown_encoding)이다.
computed_distances = compare_faces(all_encodings, all_encodings[-1])

# 7) Print obtained results: 이름순으로 나열...
#print(names)
print("\n7) computed_distances = compare_faces(all_encodings, unknown_encoding)")
print(f'name              =', end=' ')
for n in names:        # 맨 마지막은 미지의 인물이 맨 나중에 나옴.
    print(f"{n[0:-4].rjust(12)}", end=' ')      # 파일이름에서 확장자는 제거.
print()

print(f'matching distance =', end=' ')
for v in computed_distances:
    print(f"{v:#12.7f}", end=' ')         # 파일이름에서 확장자는 제거.
print()

# 8) Print obtained results: 매칭값 순으로 나열하여 반환한다. ... 작은 값부터
# 매칭값에 따라 names 순서도 바꾸어 반환한다.
computed_distances_ordered, ordered_names = compare_faces_ordered(known_encodings, names, unknown_encoding)
print("\n8) computed_distances_ordered, ordered_names"
      "\n = compare_faces_ordered(known_encodings, names, unknown_encoding)")

#print(ordered_names)
print(f'name              =', end=' ')
for n in ordered_names:
    print(f"{n[0:-4].rjust(12)}", end=' ')
    #
print()

#print("computed_distances_ordered = ", computed_distances_ordered)
print(f'ordered distance  =', end=' ')
for v in computed_distances_ordered:
    print(f"{v:#12.7f}", end=' ')
print(f"\nThe unknown person is identified as '{ordered_names[0]}'.")


# Plot the images:
fig = plt.figure(figsize=(10, 7))
plt.suptitle(f"face recognition using dlib face detector & descriptor\n"
             f"test data set number={0}: unknown face file={names[4]}", fontsize=12, fontweight='bold')
#fig.patch.set_facecolor('silver')

all_images = [known_image_1, known_image_2, known_image_3, known_image_4]

for i in range(4):
    show_img_with_matplotlib(all_images[i], f"{names[i]}: {computed_distances[i]:.7f}", 2, 4, i+1)


show_img_with_matplotlib(unknown_image, f"{names[4]}: {computed_distances[i+1]:.7f}", 2, 3, 5)      # 미지의 인물.

# Show the Figure:
plt.show()
