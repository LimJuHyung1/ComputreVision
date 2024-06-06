## face detection using xml
+ ##### cv2.CascadeClassifier(anime_cascade_path) - 캐스케이드 파일을 통한 애니메이션 얼굴 인식
+ ##### anime_face_cascade.detectMultiScale() - 이미지 또는 비디오에서 객체를 검출
![face detection using xml](./Images/face_detection_xml_mushoku_tensei.PNG)
- - -
## face detection using SSD
+ ##### cv.dnn.readNetFromCaffe(prototxt_path, caffe_model_path) - caffe 모델 구조와 prototxt 파일을 읽어 딥러닝 신경망을 생성
+ ##### confidence_threshold 수치 이상의 신뢰도를 가진 객체를 검출 시 화면에 표시
![face detection using SSD](./Images/face_detection_using_SSD.gif)
