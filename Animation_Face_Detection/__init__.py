from Modules import Menu
from Modules import root, take_snapshot
import Video_Control as V
import Threshold as T
import Select_Cascade as S
import Geotrans as G
import Contour as C
import K_means as K

# 메뉴 생성
menu = Menu(root)
root.config(menu=menu)

file_menu = Menu(menu)
menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Exit", command=root.quit)

# 시작 및 중지 버튼 추가
video_control_menu = Menu(menu)
menu.add_cascade(label="Video Control", menu=video_control_menu)
video_control_menu.add_command(label="Start", command=V.start_video)
video_control_menu.add_command(label="Stop", command=V.stop_video)
video_control_menu.add_command(label="Visualization Rect", command=V.turn_on_detection_rect)
video_control_menu.add_command(label="Unvisualization Rect", command=V.turn_off_detection_rect)
video_control_menu.add_command(label="Visualization Confidence", command=V.turn_on_confidence)
video_control_menu.add_command(label="Unvisualization Confidence", command=V.turn_off_confidence)

# 얼굴 검출기 선택 메뉴 추가
cascade_menu = Menu(menu)
menu.add_cascade(label="Select Cascade", menu=cascade_menu)

# 스냅샷 메뉴 추가
snapshot_menu = Menu(menu)
menu.add_cascade(label="Threshold", menu=snapshot_menu)
snapshot_menu.add_command(label="Take Snapshot", command=take_snapshot)
snapshot_menu.add_command(label="cv2.threshold", command=lambda: T.threshold_image("cv2.threshold"))
snapshot_menu.add_command(label="cv2.adaptiveThreshold", command=lambda: T.threshold_image("cv2.adaptiveThreshold"))
snapshot_menu.add_command(label="threshold_otsu", command=lambda: T.threshold_image("threshold_otsu"))

for cascade in S.cascade_files:
    cascade_menu.add_command(label=cascade, command=lambda c=cascade: S.change_cascade(c))

# Geotrans 메뉴 추가
geotrans_menu = Menu(menu)
menu.add_cascade(label="Geotrans", menu=geotrans_menu)
geotrans_menu.add_command(label="Translation", command=G.translation)
geotrans_menu.add_command(label="Rotation", command=G.rotation)
geotrans_menu.add_command(label="Affine Transformation", command=G.affine_transformation)
geotrans_menu.add_command(label="Perspective Transformation", command=G.perspective_transformation)

# Contour 메뉴 추가
contour_menu = Menu(menu)
menu.add_cascade(label="Contour", menu=contour_menu)
contour_menu.add_command(label="Draw Contours", command=C.draw_contours)
contour_menu.add_command(label="Contours Centroid", command=C.moments)

# K-Means 메뉴 추가
kmeans_menu = Menu(menu)
menu.add_cascade(label="K-Means", menu=kmeans_menu)
# K값에 따른 K-Means Color Quantization 버튼 추가
k_values = [2, 4, 8, 16, 32]
for k in k_values:
    kmeans_menu.add_command(label=f"Apply K-Means (k={k})", command=K.create_kmeans_button(k))

# 초기화 함수 호출
V.show_frame()
root.mainloop()
