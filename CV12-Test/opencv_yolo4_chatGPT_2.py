import cv2
import numpy as np

# 이미지 로드
image = cv2.imread("Images/bocchi.jpg")
height, width, channels = image.shape

# YOLO4 모델 로드
net = cv2.dnn.readNet("../CV12/1_2_opencv_object_detection_yolo4/yolov4.weights",
                      "../CV12/1_2_opencv_object_detection_yolo4/yolov4.cfg")

# 클래스 이름 로드
with open("../CV12/1_2_opencv_object_detection_yolo4/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 네트워크의 출력 레이어 이름 가져오기
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

# 객체 탐지를 위한 블롭 생성
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# 추론 수행
import time
s_time = time.time()
outs = net.forward(output_layers)
e_time = time.time() - s_time

# 탐지된 객체 정보 처리
class_ids = []
confidences = []
boxes = []
confidence_threshold = 0.5

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > confidence_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-max suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

# 결과 이미지에 박스와 레이블 표시
num_obj = 0
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label2 = f"{num_obj}:" + label + f"({confidence:.2f})"
        cv2.putText(image, label2, (x + 5, y + 20), font, 1.25, (0, 0, 0), 6)
        cv2.putText(image, label2, (x + 5, y + 20), font, 1.25, (255, 255, 255), 2)
        num_obj += 1

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
