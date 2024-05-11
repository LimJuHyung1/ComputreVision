## k-means color quantization example
+ ##### cv2.kmeans(데이터, 군집 수, None, 알고리즘 수렴 기준, 중심 포인트 선택 횟수, 초기 중심 포인트 탐색 방법)
+ ##### data = np.float32(image).reshape((-1, 3)) -> data.shape() => (가로 X 세로, 3채널)
+ ##### criteria = (몰라, 알고리즘이 반복하는 횟수, 수렴 기준)
![K-means color quantization mikoto](./Images/K-means_clustering_mikoto.PNG)
#### PSNR이 30 이상이라면 원본과 거의 비슷함
- - -
