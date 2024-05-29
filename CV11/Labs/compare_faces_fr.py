"""
This script makes used of face_recognition package to calculate the 128D descriptor to be used for face recognition
and compare the faces using some distance metrics

face_recognition 모듈은 내부적으로는 dlib를 사용한다.
dlib 얼굴인식기는 모델 파일을 따로 제공해야 하는 불편함이 있는 반면에
face_recognition은 다음의 장점이 있다.
    - dlib face 5 landmark 모델이 내장되어 있다.
    - dlib 얼굴 검출기도 내장되어 있다.
    - 이런 디테일 과정이 모두 감추어져 있다.
그러나 대신 아래의 단점을 각오해야 한다.
    - 얼굴 인식할 때 True 혹은 false로만 말해주어 다른 성능 개선을 도모할 여지가 없다.
    - 더 옵션 기능이 있을 수 있겠지만, 확인해 보지 않았음.


"""

# Import required packages:
import face_recognition
path = "face_files/"    # 영상 파일의 위치

# Load known images (remember that these images are loaded in RGB order):
known_image_1 = face_recognition.load_image_file(path + "jared_1.jpg")
known_image_2 = face_recognition.load_image_file(path + "jared_2.jpg")
known_image_3 = face_recognition.load_image_file(path + "jared_3.jpg")
known_image_4 = face_recognition.load_image_file(path + "obama.jpg")

# Crate names for each loaded image:
names = ["jared_1.jpg", "jared_2.jpg", "jared_3.jpg", "obama.jpg"]

# Load unknown image (this image is going to be compared against all the previous loaded images):
unknown_image = face_recognition.load_image_file(path + "jared_4.jpg")

# Calculate the encodings for every of the images:
known_image_1_encoding = face_recognition.face_encodings(known_image_1)[0]
known_image_2_encoding = face_recognition.face_encodings(known_image_2)[0]
known_image_3_encoding = face_recognition.face_encodings(known_image_3)[0]
known_image_4_encoding = face_recognition.face_encodings(known_image_4)[0]
known_encodings = [known_image_1_encoding, known_image_2_encoding,
                   known_image_3_encoding, known_image_4_encoding]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare the faces:
results = face_recognition.compare_faces(known_encodings, unknown_encoding)

# Print the results:
print(results)
