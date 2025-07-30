import cv2
import numpy as np
from rapidfuzz import process, fuzz
from PIL import Image, ImageDraw, ImageFont
import os
from skimage.metrics import structural_similarity as ssim
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import imagehash

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
    
    def compare_image(self, image, name_image_in_folder, image_folder_path = "data_list"):
        """
        So sánh hai ảnh và trả về độ tương đồng.
        
        Args:
            image: Ảnh cần so sánh (numpy.ndarray).
            name_image_in_folder: Tên ảnh trong thư mục để so sánh.
            image_folder_path: Đường dẫn đến thư mục chứa ảnh để so sánh.
        
        Returns:
            Độ tương đồng giữa hai ảnh (float).
        """
        
        name_image_in_folder = name_image_in_folder + ".jpg"
        # Đọc ảnh từ thư mục
        image_in_folder = cv2.imread(os.path.join(image_folder_path, name_image_in_folder))
        if image_in_folder is None:
            return 0.0

        max_score = 0.0
        angles = [0, 90, 180, 270]

        for angle in angles:
            # Xoay ảnh trong thư mục
            if angle == 0:
                rotated = image_in_folder
            else:
                rotated = self.rotate_image(image_in_folder, angle)

            # Resize về cùng kích thước với ảnh đầu vào
            if rotated.shape != image.shape:
                rotated = cv2.resize(rotated, (image.shape[1], image.shape[0]))

            # Chuyển về float32 nếu cần
            # So sánh 2 ảnh
            embed1 = self._get_embedding(image)
            embed2 = self._get_embedding(rotated)

            score = cosine_similarity([embed1], [embed2])[0][0]
            print(f"Angle {angle} score {score:.4f}")

            if score > max_score:
                max_score = score

        return max_score

    def _get_embedding(self, image):
        # convert image to rgb
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Dùng model ResNet50 pretrained (bỏ phần fully-connected)
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # bỏ phần cuối
        model.eval()

        # Chuyển ảnh về chuẩn đầu vào
        transform = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)  # batch size 1
        with torch.no_grad():
            embedding = model(input_tensor).squeeze().numpy()  # 2048 chiều
        return embedding
    
    def compare_image_hash(self, image, name_image_in_folder, image_folder_path = "data_list"):
        """
        So sánh hai ảnh bằng cách sử dụng hash và trả về độ tương đồng.
        
        Args:
            image: Ảnh cần so sánh (numpy.ndarray).
            name_image_in_folder: Tên ảnh trong thư mục để so sánh.
            image_folder_path: Đường dẫn đến thư mục chứa ảnh để so sánh.
        
        Returns:
            Độ tương đồng giữa hai ảnh (float).
        """
        
        name_image_in_folder = name_image_in_folder + ".jpg"
        # Đọc ảnh từ thư mục
        image_in_folder = cv2.imread(os.path.join(image_folder_path, name_image_in_folder))
        if image_in_folder is None:
            return 0.0

        # Chuyển đổi ảnh sang định dạng PIL
        pil_image = Image.fromarray(image)
        pil_image_in_folder = Image.fromarray(image_in_folder)

        # Tính hash cho cả hai ảnh
        hash1 = imagehash.phash(pil_image)
        hash2 = imagehash.phash(pil_image_in_folder)

        # Tính độ tương đồng dựa trên khoảng cách Hamming 
        score = (hash1 - hash2) / len(hash1.hash.flatten())
        return score