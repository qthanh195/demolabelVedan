import os
import cv2
import json
import numpy as np

def save_image_sizes_to_json(input_folder, output_json="image_sizes.json"):
    image_info = {}
    border_size = 5
    

    # duyệt qua tất cả file trong thư mục
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)

        # chỉ xử lý file ảnh (jpg, png, jpeg…)
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            img = cv2.imread(filepath)
            if img is not None:
                expanded_image_sample = cv2.copyMakeBorder(img,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # pixel đen
    )
                
                # Tìm contours Áp dụng threshold
                gray_img = cv2.cvtColor(expanded_image_sample, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=2)
                cv2.imwrite("threshold.jpg", thresh)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filtered_contours = [c for c in contours if cv2.contourArea(c) < ((img.shape[0]+10) * (img.shape[1]+10))]
                if filtered_contours:
                    rect = cv2.minAreaRect(max(filtered_contours, key=cv2.contourArea))
                    
                    h, w = rect[1][0], rect[1][1]
                    image_info[filename] = {"width": w, "height": h}

    # lưu kết quả ra file json
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(image_info, f, ensure_ascii=False, indent=4)

    print(f"✅ Đã lưu thông tin kích thước ảnh vào: {output_json}")

# --- Ví dụ sử dụng ---
save_image_sizes_to_json("data\SampleData", "data\label_sizes.json")
