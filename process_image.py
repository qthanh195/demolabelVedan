import cv2
import numpy as np
from rapidfuzz import process, fuzz
from PIL import Image, ImageDraw, ImageFont

class ProcessImage:
    def __init__(self):
        pass

    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Tính toán ma trận xoay
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Lấy kích thước mới sau khi xoay
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Cập nhật ma trận xoay để dịch ảnh đúng vào trung tâm
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]

        # Xoay ảnh với kích thước mới
        rotated = cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def transform_point(self, p, image, angle):
        # Chuyển đổi tọa độ điểm p từ tọa độ gốc sang tọa độ mới
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        # Tính toán ma trận xoay
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        new_x = rotation_matrix[0, 0] * p[0] + rotation_matrix[0, 1] * p[1] + rotation_matrix[0, 2]
        new_y = rotation_matrix[1, 0] * p[0] + rotation_matrix[1, 1] * p[1] + rotation_matrix[1, 2]
        new_point = (int(new_x), int(new_y))
        return new_point

    def crop_rotated_contour(self, image, contour):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="float32")

        # Sắp xếp box theo thứ tự: top-left, top-right, bottom-right, bottom-left
        width = int(rect[1][0])
        height = int(rect[1][1])
        print("width:", width)
        print("height:", height)
        if height < width:
            # width, height = height, width

            # Xoay box theo chiều kim đồng hồ 90 độ
            box = np.roll(box, -1, axis=0)
        if width < height:
            width, height = height, width

        if width == 0 or height == 0:
            return None  # tránh lỗi chia 0
        
        # Nếu height > width thì hoán đổi và xoay box cho đúng hướng
        
        
        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        # Tính ma trận transform và áp dụng
        M = cv2.getPerspectiveTransform(box, dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))

        return warped

    def get_best_match(self, text, valid_list, score_cutoff=70):
        match = process.extractOne(text, valid_list, scorer=fuzz.ratio, score_cutoff=score_cutoff)
        return match[0] if match else None

    def draw_text_with_pillow(self, image, text, position, font_path="simsun.ttc", font_size=20, color=(0, 255, 0)):
        """
        Vẽ văn bản Unicode lên ảnh bằng Pillow.
        
        Args:
            image: Ảnh OpenCV (numpy.ndarray).
            text: Văn bản Unicode cần vẽ.
            position: Tọa độ (x, y) để vẽ văn bản.
            font_path: Đường dẫn đến file font (ví dụ: simsun.ttc cho tiếng Trung).
            font_size: Kích thước font.
            color: Màu văn bản (BGR).
        
        Returns:
            Ảnh OpenCV với văn bản đã vẽ.
        """
        # Chuyển đổi ảnh OpenCV sang định dạng Pillow
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        # Tải font
        font = ImageFont.truetype(font_path, font_size)

        # Vẽ văn bản
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))  # Đảo thứ tự màu từ BGR sang RGB

        # Chuyển đổi ảnh Pillow trở lại OpenCV
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        return image

    def crop_rotated_contour_Dung(self, image, contour):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="float32")

        # Sắp xếp box theo thứ tự: top-left, top-right, bottom-right, bottom-left
        width = int(rect[1][0])
        height = int(rect[1][1])
        print("width:", width)
        print("height:", height)
        if height > width:
            # width, height = height, width

            # Xoay box theo chiều kim đồng hồ 90 độ
            box = np.roll(box, -1, axis=0)
        if width > height:
            width, height = height, width

        if width == 0 or height == 0:
            return None  # tránh lỗi chia 0
        
        # Nếu height > width thì hoán đổi và xoay box cho đúng hướng
        
        
        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        # Tính ma trận transform và áp dụng
        M = cv2.getPerspectiveTransform(box, dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))

        return warped