import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt

# dlib 모델 경로
model_path = 'dlib_face_recog/'

def show_img_with_matplotlib(color_img, title, row, col, pos):
    """matplotlib을 사용하여 이미지를 보여줌"""
    ax = plt.subplot(row, col, pos)
    plt.imshow(color_img)
    plt.title(title)
    plt.axis('off')

# 랜드마크 검출기 초기화
pose_predictor_5_point = dlib.shape_predictor(model_path + "shape_predictor_5_face_landmarks.dat")

# 얼굴 인코더 초기화
face_encoder = dlib.face_recognition_model_v1(model_path + "dlib_face_recognition_resnet_model_v1.dat")

# 얼굴 검출기 초기화
detector = dlib.get_frontal_face_detector()

def compare_faces(encodings, encoding_to_check):
    """
    알려진 얼굴 인코딩과 비교하려는 얼굴 인코딩 간의 유클리드 거리를 계산하여 리스트로 반환
    """
    return list(np.linalg.norm(encodings - encoding_to_check, axis=1))

def compare_faces_ordered(encodings, face_names, encoding_to_check):
    """
    얼굴 인코딩을 비교하고 거리와 얼굴 이름을 거리 순으로 정렬하여 반환
    """
    distances = list(np.linalg.norm(encodings - encoding_to_check, axis=1))
    return zip(*sorted(zip(distances, face_names)))

def face_encodings(face_image, number_of_times_to_upsample=1, num_jitters=1):
    """
    이미지에 있는 각 얼굴에 대해 128D dlib 얼굴 디스크립터를 반환
    """
    try:
        # 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        print("이미지를 그레이스케일로 변환했습니다.")

        # 그레이스케일 이미지에서 얼굴 검출
        face_locations = detector(gray, number_of_times_to_upsample)
        print(f"이미지에서 {len(face_locations)}개의 얼굴을 검출했습니다.")

        # 검출된 얼굴 위치에서 랜드마크 검출
        raw_landmarks = [pose_predictor_5_point(face_image, face_location) for face_location in face_locations]
        print("각 얼굴에 대한 랜드마크를 검출했습니다.")

        # 각 얼굴에 대해 128차원 얼굴 디스크립터를 계산하여 리스트로 반환
        face_descriptors = [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]
        print("각 얼굴에 대한 얼굴 디스크립터를 계산했습니다.")

        return face_descriptors
    except Exception as e:
        print(f"얼굴 인코딩 중 오류 발생: {e}")
        return []

# 실험 데이터 세트 선택
data_path = "Images/"
set0 = ["shinobu.jpg", "nezuko1.png", "nezuko2.jpg", "nezuko3.png", "nezuko4.png"]
set1 = []
set2 = []
set3 = []
data_set_list = [set0, set1, set2, set3]
set_num = 0
names = data_set_list[set_num]
print(f"사용된 실험 세트={set_num}, 미지의 인물 파일명={names[4]}")

# 알려진 이미지 읽기
known_image_1 = cv2.imread(data_path + names[0])
known_image_2 = cv2.imread(data_path + names[1])
known_image_3 = cv2.imread(data_path + names[2])
known_image_4 = cv2.imread(data_path + names[3])

# 미지의 인물 이미지 읽기
unknown_image = cv2.imread(data_path + names[4])

# BGR (OpenCV 포맷)에서 RGB (dlib 포맷)으로 변환
known_image_1 = known_image_1[:, :, ::-1]
known_image_2 = known_image_2[:, :, ::-1]
known_image_3 = known_image_3[:, :, ::-1]
known_image_4 = known_image_4[:, :, ::-1]
unknown_image = unknown_image[:, :, ::-1]

# 알려진 이미지와 미지의 이미지 인코딩 생성
known_image_1_encoding = face_encodings(known_image_1)[0]
known_image_2_encoding = face_encodings(known_image_2)[0]
known_image_3_encoding = face_encodings(known_image_3)[0]
known_image_4_encoding = face_encodings(known_image_4)[0]
known_encodings = [known_image_1_encoding, known_image_2_encoding, known_image_3_encoding, known_image_4_encoding]
print(f"알려진 인코딩 생성됨, 타입={type(known_encodings)}, 길이={len(known_encodings)}")
unknown_encoding = face_encodings(unknown_image)[0]
print(f"미지의 인코딩 생성됨, 타입={type(unknown_encoding)}, 모양={unknown_encoding.shape}")

all_encodings = known_encodings + [unknown_encoding]
print(f"모든 인코딩 생성됨, 타입={type(all_encodings)}, 길이={len(all_encodings)}")

# 얼굴 비교: 알려진 인코딩과 미지의 인코딩 간의 거리 계산
computed_distances = compare_faces(all_encodings, all_encodings[-1])

# 결과 출력
print("\n알려진 얼굴과 미지의 얼굴 간의 거리 계산 결과")
print(f'name              =', end=' ')
for n in names:
    print(f"{n[0:-4].rjust(12)}", end=' ')
print()
print(f'matching distance =', end=' ')
for v in computed_distances:
    print(f"{v:#12.7f}", end=' ')
print()

# 거리 순으로 얼굴 비교
computed_distances_ordered, ordered_names = compare_faces_ordered(known_encodings, names, unknown_encoding)
print("\n거리 순으로 정렬된 거리와 이름")
print(f'name              =', end=' ')
for n in ordered_names:
    print(f"{n[0:-4].rjust(12)}", end=' ')
print()
print(f'ordered distance  =', end=' ')
for v in computed_distances_ordered:
    print(f"{v:#12.7f}", end=' ')
print(f"\n미지의 인물은 '{ordered_names[0]}'로 확인되었습니다.")

# 이미지 플롯
fig = plt.figure(figsize=(10, 7))
plt.suptitle(f"Face recognition using dlib face detector & descriptor\nTest data set number={0}: "
             f"unknown face file={names[4]}", fontsize=12, fontweight='bold')

all_images = [known_image_1, known_image_2, known_image_3, known_image_4]

for i in range(4):
    show_img_with_matplotlib(all_images[i], f"{names[i]}:\n{computed_distances[i]:.7f}", 2, 4, i+1)

show_img_with_matplotlib(unknown_image, f"{names[4]}:\n{computed_distances[i+1]:.7f}", 2, 3, 5)

# 플롯 표시
plt.show()
