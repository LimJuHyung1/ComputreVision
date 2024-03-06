import cv2 as cv

# 1. 현재의 위치에서 상위 폴더의 data 폴더에는 이번 시간 다룰 샘플 영상과 비디오 동영상을 저장하였다.
route = "C:/CV/01/CV00/data"
a = cv.imread("../data/monarch.bmp")
b = cv.imread("../data/smooth.jpg")
# print(f"type(a)={type(a)}, a.shape={a.shape}, a.dtype={a.dtype}")
# print('type(a)={}, a.shape={}, a.dtype={}'.format(type(a), a.shape, a.dtype))
cv.imshow(route + '00', a)  # 영상 보이기. 창의 이름(test2)에 어레이 영상(a)를 출력한다.
cv.imshow(route + '01', b)  # 영상 보이기. 창의 이름(test2)에 어레이 영상(a)를 출력한다.

cv.waitKey()  # 5000[ms]=5초 기다리기. 중간에 키입력 들어오면 빠져나간다.

c = cv.imread("../data/capture.png")
cv.imshow(route + '02', c)
cv.waitKey()
