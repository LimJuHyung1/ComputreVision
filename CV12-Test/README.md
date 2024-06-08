# 0_1 blob from images
> ## blob from image
> > + ##### cv2.dnn.blobFromImage() 함수를 이용하여 이미지를 블롭으로 변환 - 딥러닝 모델의 입력으로 사용됨
> > + ##### swap - BGR -> RGB 로 변경을 의미(딥러닝에서 RGB로 사용)
> > ![blob from image lum](./Images/blobFromImage_lum.PNG)
- - -
> ## blob from images
> > + ##### cv2.dnn.blobFromImages() 에 이미지 리스트를 입력
> > ![blob from images sakura and shinobu](./Images/blobFromImages_sakura_and_shinobu.PNG)
- - -
> ## blob from images cropping
> > + ##### 이미지에서 가로, 세로 중 작은 축을 기준으로 cropping
> > + ##### blob 에서 이미지로 변경되는 경우의 결과물은 crop 되지 않고 이미지가 전반적으로 가운데로 모임
> > + ##### crop 된 이미지는 위와 같은 가운데로 모이는 현상 없이 잘려진 이미지가 출력되는 것을 볼 수 있음
> > ![blob from images cropping benten and oyuki](./Images/blobFromImages_cropping_benten_and_oyuki.PNG)
