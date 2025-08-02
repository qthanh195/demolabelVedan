from ai_hander import model_obb_label, model_segment_label
import cv2
from api_handler import ApiHandler
import numpy as np

ai_handler = ApiHandler()

ai_handler.api_open_camera()
image = ai_handler.get_image()
ai_handler.api_close_camera()
# In id và class_name của model

if image is None:
    print("Không lấy được ảnh từ camera!")

# Đảm bảo ảnh là BGR
if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

image_copy = image.copy()
results = model_segment_label(image, conf=0.5, retina_masks =True)
for result in results:
    masks = result.masks
    for i, poly in enumerate(masks.xy):
        poly = poly.astype(np.int32)

        # Tính bounding box nghiêng nhỏ nhất
        rect = cv2.minAreaRect(poly)
        box = cv2.boxPoints(rect)  # 4 điểm
        box = np.int32(box)

        # Tính kích thước của box (width, height)
        width = int(rect[1][0])
        height = int(rect[1][1])

        if width == 0 or height == 0:
            continue  # bỏ qua box không hợp lệ

        # Xác định thứ tự 4 điểm box: [top-left, top-right, bottom-right, bottom-left]
        box_ordered = np.array(cv2.convexHull(box), dtype="float32")

        # Chuẩn hóa lại thứ tự nếu cần (có thể dùng sort contour nếu kết quả bị xoay sai)
        # Hoặc dùng trực tiếp box như sau:
        dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

        # Perspective Transform để cắt theo hình chữ nhật
        M = cv2.getPerspectiveTransform(box.astype("float32"), dst_pts)
        warped = cv2.warpPerspective(image_copy, M, (width, height))

        # Lưu hoặc hiển thị ảnh đã cắt
        cv2.imwrite(f"crop_{i}.png", warped)
        # Vẽ box
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
        cv2.polylines(image, [poly], isClosed=True, color=(0,255,0), thickness=2)
    result.show() 
    
# import supervision as sv
# detections = sv.Detections.from_ultralytics(results[0])
# print("detections:", detections)

# oriented_box_annotator = sv.OrientedBoxAnnotator()
# annotated_frame = oriented_box_annotator.annotate(
#     scene=image,
#     detections=detections
# )

# sv.plot_image(image=annotated_frame)
