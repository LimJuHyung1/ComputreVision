## contourArea example
+ ##### cv2.contourArea(contour) 함수로 contours의 각 contour 영역의 넓이를 구함.
+ ##### cv2.putText() 함수를 통해 각 contour 그룹에 해당하는 contourArea를 출력함.
![Origin lovelive](./Images/origin_love_live.PNG)
![Gray lovelive](./Images/gray_love_live.PNG)
![ContourArea lovelive](./Images/contourArea_love_live.PNG)
- - -
## contours sort example
+ ##### contoursArea()의 결과를 정렬시켜 크기 순서를 화면에 출력
+ ##### 숫자가 클수록 면적이 넓다
![Contours Sort lovelive](./Images/contour_sort_love_live.PNG)
- - -
## arcLength example
+ ##### cv2.arcLength(contour, closed=True) 함수를 통해 윤곽선의 길이를 구함.
+ ##### cv2.drawContours(contour_image, [contour], ...) 함수를 통해 컨투어 별로 색상을 다르게 생성
![Origin saenai](./Images/origin_saenai.PNG)
![Gray saenai](./Images/gray_saenai.PNG)
![ArcLength saenai](./Images/arc_length_saenai.PNG)
- - -
