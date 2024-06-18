import cv2
import dlib
import numpy as np

# 얼굴 검출기와 랜드마크 검출기 초기화
detector = dlib.get_frontal_face_detector()
pose_predictor_5_point = dlib.shape_predictor("dlib_face_recog/shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recog/dlib_face_recognition_resnet_model_v1.dat")

def face_encodings(face_image, number_of_times_to_upsample=1, num_jitters=1):
    # Returns the 128D dlib face descriptor for each face in the image

    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)

    # 그레이스케일 이미지에서 얼굴을 검출
    face_locations = detector(gray, number_of_times_to_upsample)

    # 검출된 얼굴 위치에서 5개의 랜드마크를 검출
    raw_landmarks = [pose_predictor_5_point(face_image, face_location) for face_location in face_locations]

    # 각 얼굴에 대해 128차원 얼굴 디스크립터를 계산하여 리스트로 반환
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

def process_images(image_files):
    all_encodings = []
    for file in image_files:
        image = cv2.imread(file)
        encodings = face_encodings(image)
        if len(encodings) > 0:
            all_encodings.append(encodings)
            print(f"Processed {file}: Found {len(encodings)} face(s).")
        else:
            print(f"Processed {file}: No faces found.")
    return all_encodings

# 이미지 파일 목록
image_files = ["Images/shinobu.jpg", "Images/nezuko2.jpg", "Images/nezuko4.png"]  # 이미지 파일 경로들
set0 = ["shinobu.jpg", "nezuko1.png", "nezuko2.jpg", "nezuko3.png", "nezuko4.jpg"]
# 이미지 처리 및 디스크립터 계산
all_encodings = process_images(image_files)

# 결과 확인
for i, encodings in enumerate(all_encodings):
    print(f"Image {i+1} has {len(encodings)} face encodings.")
