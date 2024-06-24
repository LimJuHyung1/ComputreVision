from Modules import cv2, np
from Modules import cap, show_snapshot

# Geotrans 함수들
def translation():
    ret, frame = cap.read()
    if ret:
        rows, cols, _ = frame.shape
        M = np.float32([[1, 0, 50], [0, 1, 50]])                # 포인트
        translated = cv2.warpAffine(frame, M, (cols, rows))     # 포인트
        show_snapshot(translated)


def rotation():
    ret, frame = cap.read()
    if ret:
        rows, cols, _ = frame.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)    # 포인트
        rotated = cv2.warpAffine(frame, M, (cols, rows))            # 포인트
        show_snapshot(rotated)


def affine_transformation():
    ret, frame = cap.read()
    if ret:
        rows, cols, _ = frame.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])         # 포인트
        pts2 = np.float32([[10, 100], [230, 80], [100, 250]])       #
        M = cv2.getAffineTransform(pts1, pts2)                      #
        transformed = cv2.warpAffine(frame, M, (cols, rows))        # 포인트
        show_snapshot(transformed)


def perspective_transformation():
    ret, frame = cap.read()
    if ret:
        rows, cols, _ = frame.shape
        pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])     # 포인트
        pts2 = np.float32([[0, 0], [250, 0], [0, 300], [250, 300]])         #
        M = cv2.getPerspectiveTransform(pts1, pts2)                         #
        transformed = cv2.warpPerspective(frame, M, (cols, rows))           # 포인트
        show_snapshot(transformed)



