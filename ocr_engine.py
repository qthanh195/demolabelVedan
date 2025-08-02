# from rapidfuzz import process, fuzz
import pytesseract
# from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from ai_hander import AiHander, model_logo_unu, model_logo_recycling, model_logo_halal, model_detect_logo_tdc, model_detect_khoiluong_tdc
from process_image import ProcessImage

valid_tdc = ["でん粉「TW-100」", 
             "食品用タピオカでん粉「BK-V」", 
             "食品用タピオカでん粉「BK-V3」",
             "イモのちから",
             "食品用タピオカでん粉「ES-5」",
             "食品用タピオカでん粉「SK-08」", 
             "食品用タピオカでん粉「タピオカV3」",
             "食品用タピオカでん粉「タピオカV」", 
             "食品用タピオカでん粉「FM-5」", 
             "食品用タピオカでん粉「RT-90」",
             "食品用タピオカでん粉「タピオカV2」", 
             "食品用タピオカでん粉「BK-V7」",]

valid_tdc_kg = ["20kg", "25kg", "12.5kg", "18kg", "12kg",]

valid_label_recyling = ["TINH BỘT ACETAT (TINH BỘT BIẾN TÍNH BS (A)) Dùng cho thực phẩm",
                        "DISTARCH PHOSPHAT ACETYLAT (TINH BỘT BIẾN TÍNH CB) Dùng cho thực phẩm", 
                        "TINH BỘT NATRI OCTENYL SUCCINAT (TINH BỘT BIẾN TÍNH KS) Dùng cho thực phẩm", 
                        "TINH BỘT OXY HÓA (TINH BỘT BIẾN TÍNH ET) Dùng cho thực phẩm", 
                        "DISTARCH ADIPAT ACETYLAT (TINH BỘT BIẾN TÍNH CBA) Dùng cho thực phẩm", 
                        "DISTARCH PHOSPHAT (TINH BỘT BIẾN TÍNH CT) Dùng cho thực phẩm",
                        "ACETYLATED OXYDIZED STARCH (TINH BỘT BIẾN TÍNH EB) Dùng cho thực phẩm",
                        "CATIONIC STARCH (TINH BỘT BIẾN TÍNH JT) Cấp Công Nghiệp",
                        "STARCH ACETATE ESTE HÓA VỚI VINYL ACETATE (TINH BỘT BIẾN TÍNH BS) Dùng cho thực phẩm"]

valid_label_halal = ["AL-69 (E1412) (FOOD GRADE)",
                     "AL-43F (E1450) (FOOD GRADE)", 
                     "AL-58 (E1422) (FOOD GRADE)", 
                     "AL-94 (FOOD GRADE)",]

valid_label_unu = ["514", "510", "TS01V"]

class OCR_Engine(ProcessImage, AiHander):
    def __init__(self):
        super().__init__()
        
    def classifi_tdc_with_ocr(self, image): 
        # image = cv2.imread(image)
        new_img = None
        w, h = image.shape[1], image.shape[0]
        # tạo mask với kích thước lớn hơn ảnh gốc 10
        mask = np.zeros((h+10, w+10, 3), dtype=np.uint8)
        
        #ghép ảnh vào giữa mask
        mask[5:h+5, 5:w+5] = image
        
        results = model_detect_logo_tdc.predict(source=image)
        if results is None or len(results) == 0 or results[0].boxes is None:
            return False, None
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                if center_x < w // 2 and center_y < h // 2:
                    print("Nhan tren trai")
                    new_img = mask
                elif center_x > w // 2 and center_y < h // 2:
                    print("Nhan tren phai")
                    new_img = self.rotate_image(mask, 90)
                    center_x, center_y = self.transform_point((center_x, center_y), image, 90)
                elif center_x < w // 2 and center_y > h // 2:
                    print("Nhan duoi trai")
                    new_img = self.rotate_image(mask, -90)
                    center_x, center_y = self.transform_point((center_x, center_y), image, -90)
                elif center_x > w // 2 and center_y > h // 2:
                    print("Nhan duoi phai")
                    new_img = self.rotate_image(mask, 180)
                    center_x, center_y = self.transform_point((center_x, center_y), image, 180)

                # Chuyển đổi ảnh sang grayscale
                gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
                # Áp dụng threshold
                _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # Tìm contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Lọc contours theo diện tích nhỏ hơn max_area
                max_area = (w+10) * (h+10)
                filtered_contours = [contour for contour in contours if cv2.contourArea(contour) < max_area]

                if filtered_contours:
                    # Tìm contour lớn nhất trong danh sách đã lọc
                    contour = max(filtered_contours, key=cv2.contourArea)
                else:
                    print("Không có contour nào thỏa mãn điều kiện.")
                    contour = None
                    
                if contour is not None:
                    image_crop = self.crop_rotated_contour(new_img, contour)
                    results = model_detect_khoiluong_tdc.predict(source=image_crop)
                    if results is None or len(results) == 0 or results[0].boxes is None:
                        return False, None
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            image_crop1 = image_crop[y1-4:y2+4, x1:x2]
                            image_crop2 = image_crop[y1-98:y2-80, x1-5:x2+400]

                            text1 = pytesseract.image_to_string(image_crop1, config=r'--oem 3 --psm 8 -l eng')
                            text2 = pytesseract.image_to_string(image_crop2, config=r'--oem 3 --psm 7 -l jpn')

                            idx_text1 = self.get_best_match(''.join(text1.split()).strip().lower(), valid_tdc_kg)
                            idx_text2 = self.get_best_match(''.join(text2.split()).strip().upper(), valid_tdc)

                            if text1 is not None:
                                image_crop = self.draw_text_with_pillow(image_crop, text1, (x1, y1-20), font_path="simsun.ttc", font_size=20, color=(0, 255, 0))
                            if text2 is not None:
                                image_crop = self.draw_text_with_pillow(image_crop, text2, (x1, y1-110), font_path="simsun.ttc", font_size=20, color=(0, 255, 0))
                            
                            print("text1:", text1)
                            print("text2:", text2)
                            
                            match (idx_text1, idx_text2):
                                case (0, 0):  # "20kg", "でん粉「TW-100」"
                                    return "Label30", image_crop
                                case (3, 1):  # "18kg", "食品用タピオカでん粉「BK-V」"
                                    return "Label49", image_crop
                                case (0, 1):  # "20kg", "食品用タピオカでん粉「BK-V」"
                                    return "Label50", image_crop
                                case (1, 2):  # "25kg", "食品用タピオカでん粉「BK-V3」"
                                    return "Label51", image_crop
                                case (1, 3):  # "25kg", "イモのちから"
                                    return "Label52", image_crop
                                case (0, 4):  # "20kg", "食品用タピオカでん粉「ES-5」"
                                    return "Label53", image_crop
                                case (1, 5):  # "25kg", "食品用タピオカでん粉「SK-08」"
                                    return "Label54", image_crop
                                case (1, 6):  # "25kg", "食品用タピオカでん粉「タピオカV3」"
                                    return "Label55", image_crop
                                case (2, 7):  # "12.5kg", "食品用タピオカでん粉「タピオカV」"
                                    return "Label56", image_crop
                                case (0, 8):  # "20kg", "食品用タピオカでん粉「FM-5」"
                                    return "Label57", image_crop
                                case (0, 7):  # "20kg", "食品用タピocaでん粉「タピオカV」"
                                    return "Label58", image_crop
                                case (0, 9):  # "20kg", "食品用タピオカでん粉「RT-90」"
                                    return "Label59", image_crop
                                case (1, 7):  # "25kg", "食品用タピオカでん粉「タピオカV」"
                                    return "Label60", image_crop
                                case (1, 1):  # "25kg", "食品用タピオカでん粉「BK-V」"
                                    return "Label61", image_crop
                                case (1, 10): # "25kg", "食品用タピオカでん粉「タピオカV2」"
                                    return "Label62", image_crop
                                case (1, 11): # "25kg", "食品用タピオカでん粉「BK-V7」"
                                    return "Label63", image_crop
                                case _:
                                    return "", image_crop
        return "", new_img

    def classify_label_logo_recycling(self, image):
        """"
            1. Phát hien logo recycling
            2. Xoay ảnh thẳng đứng
            3. Cắt đúng vùng ảnh
            4. lấy vùng đọc chữ
        """
        # Đọc ảnh
        new_img = None
        # image = cv2.imread(image_path)
        w, h = image.shape[1], image.shape[0]
        # Dự đoán ảnh
        results = model_logo_recycling.predict(source=image)
        if results is None or len(results) == 0 or results[0].boxes is None:
            return False, None
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                if center_x < w // 2 and center_y < h // 2:
                    print("Nhan tren trai")
                    new_img = self.rotate_image(image, -90)
                elif center_x > w // 2 and center_y < h // 2:
                    print("Nhan tren phai")
                    new_img = image
                elif center_x < w // 2 and center_y > h // 2:
                    print("Nhan duoi trai")
                    new_img = self.rotate_image(image, 180)
                elif center_x > w // 2 and center_y > h // 2:
                    print("Nhan duoi phai")
                    new_img = self.rotate_image(image, 90)
                    
                w, h = new_img.shape[1], new_img.shape[0]
                # tạo mask với kích thước lớn hơn ảnh gốc 10
                mask = np.zeros((h+10, w+10, 3), dtype=np.uint8)
                #ghép ảnh vào giữa mask
                mask[5:h+5, 5:w+5] = new_img
                center_x, center_y = center_x + 5, center_y + 5
                
                # Chuyển đổi ảnh sang grayscale
                gray_img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                # Áp dụng thresholdx
                _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # Tìm contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Lọc contours theo diện tích nhỏ hơn max_area
                max_area = (w+10) * (h+10)
                filtered_contours = [contour for contour in contours if cv2.contourArea(contour) < max_area]

                if filtered_contours:
                    # Tìm contour lớn nhất trong danh sách đã lọc
                    contour = max(filtered_contours, key=cv2.contourArea)
                else:
                    print("Không có contour nào thỏa mãn điều kiện.")
                    contour = None
                    
                if contour is not None:
                    image_crop = self.crop_rotated_contour_Dung(mask, contour)
                    results = model_logo_recycling.predict(source=image_crop)
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            zone_text = image_crop[y1+120:y1+305, x1-780:x1+80]
                            text = pytesseract.image_to_string(zone_text, config=r'--oem 3 --psm 6 -l vie')
                            text = text.replace("\n", "")
                            # text = text.replace(" ", "")
                            idx_text = self.get_best_match(text, valid_label_recyling)
                            # idx_text = valid_label_recyling.index(text) if text in valid_label_recyling else None
                            print("text:", text)
                            match idx_text:
                                case 3:
                                    return "Label64", image_crop
                                case 7:
                                    return "Label65", image_crop
                                case 5:
                                    return "Label66", image_crop
                                case 4:
                                    return "Label67", image_crop
                                case 1:
                                    return "Label68", image_crop
                                case 6:
                                    return "Label69", image_crop
                                case 8:
                                    return "Label70", image_crop
                                case 2:
                                    return "Label71", image_crop
                                case 0:
                                    return "Label72", image_crop

        return "", new_img
    
    def classify_label_logo_halal(self, image):
        """"
            1. Phát hien logo halal
            2. Xoay ảnh thẳng đứng
            3. Cắt đúng vùng ảnh
            4. lấy vùng đọc chữ
        """
        # Đọc ảnh
        new_img = None
        text = ""
        w, h = image.shape[1], image.shape[0]
        # Dự đoán ảnh
        results = model_logo_halal.predict(source=image)
        if results is None or len(results) == 0 or results[0].boxes is None:
            return False, None

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                if center_x < w // 2 and center_y < h // 2:
                    print("Nhan tren trai")
                    new_img = self.rotate_image(image, -90)
                    
                elif center_x > w // 2 and center_y < h // 2:
                        print("Nhan tren phai")
                        new_img = image
                elif center_x < w // 2 and center_y > h // 2:
                    print("Nhan duoi trai")
                    new_img = self.rotate_image(image, 180)

                elif center_x > w // 2 and center_y > h // 2:
                    print("Nhan duoi phai")
                    new_img = self.rotate_image(image, 90)
                    
                w, h = new_img.shape[1], new_img.shape[0]
                # tạo mask với kích thước lớn hơn ảnh gốc 10
                mask = np.zeros((h+10, w+10, 3), dtype=np.uint8)
                #ghép ảnh vào giữa mask
                mask[5:h+5, 5:w+5] = new_img
                # Chuyển đổi ảnh sang grayscale
                gray_img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                # Áp dụng thresholdx
                _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # Tìm contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Lọc contours theo diện tích nhỏ hơn max_area
                max_area = (w+10) * (h+10)
                filtered_contours = [contour for contour in contours if cv2.contourArea(contour) < max_area]
                
                if filtered_contours:
                    # Tìm contour lớn nhất trong danh sách đã lọc
                    contour = max(filtered_contours, key=cv2.contourArea)
                else:
                    print("Không có contour nào thỏa mãn điều kiện.")
                    contour = None

                if contour is not None:
                    image_crop = self.crop_rotated_contour(mask, contour)
                    zone_text = image_crop[230:330, 70:650]
                    text = pytesseract.image_to_string(zone_text, config=r'--oem 3 --psm 7 -l eng')
                    idx_text = self.get_best_match(text, valid_label_halal)
                    image_crop = self.draw_text_with_pillow(image_crop, text, (120, 230), font_size=28, color=(0, 255, 0))
                    
                    match idx_text:
                        case 0:  # AL-69 (E1412) (FOOD GRADE)
                            return "Label73", image_crop
                        case 1:  # AL-43F (E1450) (FOOD GRADE)
                            return "Label74", image_crop
                        case 2:  # AL-58 (E1422) (FOOD GRADE)
                            return "Label75", image_crop
                        case 3:  # AL-94 (FOOD GRADE)
                            return "Label45", image_crop
                        case _:
                            return "", image_crop

        return "", new_img
    
    def classify_label_logo_unu(self, image):
        """"
            1. Phát hien logo unu
            2. Xoay ảnh thẳng đứng
            3. Cắt đúng vùng ảnh
            4. lấy vùng đọc chữ
        """
        # Đọc ảnh
        new_img = None
        text = ""
        w, h = image.shape[1], image.shape[0]
        # Dự đoán ảnh
        results = model_logo_unu.predict(source=image)
        if results is None or len(results) == 0 or results[0].boxes is None:
            return "", image

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                if center_x < w // 2 and center_y < h // 2:
                    print("Nhan tren trai")
                    new_img = image
                elif center_x > w // 2 and center_y < h // 2:
                    print("Nhan tren phai")
                    new_img = self.rotate_image(image, 90)
                elif center_x < w // 2 and center_y > h // 2:
                    print("Nhan duoi trai")
                    new_img = self.rotate_image(image, -90)

                elif center_x > w // 2 and center_y > h // 2:
                    print("Nhan duoi phai")
                    new_img = self.rotate_image(image, 180)
                    
                w, h = new_img.shape[1], new_img.shape[0]
                # tạo mask với kích thước lớn hơn ảnh gốc 10
                mask = np.zeros((h+10, w+10, 3), dtype=np.uint8)
                #ghép ảnh vào giữa mask
                mask[5:h+5, 5:w+5] = new_img
                # Chuyển đổi ảnh sang grayscale
                gray_img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                # Áp dụng thresholdx
                _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # Tìm contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Lọc contours theo diện tích nhỏ hơn max_area
                max_area = (w+10) * (h+10)
                filtered_contours = [contour for contour in contours if cv2.contourArea(contour) < max_area]
                if filtered_contours:
                    # Tìm contour lớn nhất trong danh sách đã lọc
                    contour = max(filtered_contours, key=cv2.contourArea)
                else:
                    print("Không có contour nào thỏa mãn điều kiện.")
                    contour = None

                if contour is not None:
                    image_crop = self.crop_rotated_contour(mask, contour)
                    # zone_text = image_crop[160:350, 180:1300]
                    zone_text = image_crop[160:350, 800:1300]
                    text = pytesseract.image_to_string(zone_text, config=r'--oem 3 --psm 7 -l eng')
                    idx_text = self.get_best_match(text, valid_label_unu)
                    text = f"サナス{text}"
                    image_crop = self.draw_text_with_pillow(image_crop, text, (300, 130),font_size=50, color=(0, 255, 0))
                    
                    print("text:", text)
                    
                    match idx_text:
                        case 0:  # "サナス514"
                            return "Label34", image_crop
                        case 1:  # "サナス510"
                            return "Label76", image_crop
                        case 2:  # "サナスTS01V"
                            return "Label77", image_crop
                        case _:
                            return "", image_crop

        return "", new_img