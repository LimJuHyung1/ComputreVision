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
## Affine example
+ ##### cv2.getAffineTransform(pts_src, pts_dst) 함수를 통해 affine 행렬을 반환 받음.
+ ##### cv2.invertAffineTransform(matrix), cv2.warpAffine(..., flags=cv2.WARP_INVERSE_MAP) 등의 함수로 변형된 영상을 복원함
![Affine nanami](./Images/Affine_nanami.PNG)
- - -
## Perspective example
+ ##### cv2.getPerspectiveTransform(src, dst) 함수를 통해 perspective 행렬을 반환 받음.
+ ##### cv2.getPerspectiveTransform(dst, src) 함수로 변형된 영상을 복원함
![Perspective misaki](./Images/Perspective_misaki.PNG)
- - -


