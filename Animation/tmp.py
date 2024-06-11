import dlib
import cv2
import os


def test_face_detection_with_dlib(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)

    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Face Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_face_detection_with_dlib_video(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = dlib.get_frontal_face_detector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Face Detection Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 예시로 저장된 이미지의 경로
# test_face_detection_with_dlib("frozen_characters/tmp/Elsa/frame_0_face_0.jpg")

# 예시로 엘사의 비디오에 대해 face detection 수행
test_face_detection_with_dlib_video("[FHD] 6. Frozen(겨울왕국) - For The First Time In Forever (Reprise) (영어+한글자막).mp4]")