from Modules import cv2
from Modules import face_cascade as m_face_cascade


# 얼굴 검출기 목록
cascade_files = {
    "Default": "cascades/haarcascades/haarcascade_frontalface_default.xml",
    "Animeface": 'cascades/lbpcascades/lbpcascade_animeface.xml'
}


# 캐스케이더를 변경
def change_cascade(cascade):
    global face_cascade, selected_cascade
    selected_cascade = cascade
    face_cascade = cv2.CascadeClassifier(cascade_files[cascade])