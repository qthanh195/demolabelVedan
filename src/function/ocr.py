import pytesseract
import cv2
import numpy as np
from src.function.yolo import AiHander
from PIL import Image
from rapidfuzz import process, fuzz
from PIL import Image, ImageDraw, ImageFont

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

valid_label_unu = ["サナス514", "サナス510", "サナスTS01V"]

valid_label_36 = [
"""
IMPORTER:
VETIA INTERNATIONAL
10 RUE DES HALLES
75001 PARIS - FRANCE]
""",
"""
MODIFIED STARCH CBA-8866
NET WEIGHT: 25KGS
GROSS WEOHT: 25.35KGS
""",]

valid_label_37 = [
"""
QUICKFLOW Modified Starch,
FS-85 Modified Starch,
"""
]


class OCR_Engine(AiHander):
    def __init__(self):
        super().__init__()
        
    def classifi_tdc_with_ocr(self, image, height, width): 

        confidence_ocr = 0
        text_returned =""
        weight = ""
        results = self.model_detect_khoiluong_tdc.predict(source=image)
        if results is None or len(results) == 0 or results[0].boxes is None:
            return False, None
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                # Chuyển sang ảnh xám
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()
                # Tiền xử lý theo đoạn code của bạn
                gray = cv2.medianBlur(gray, 3)
                image_crop1 = gray[y1-4:y2+4, x1:x2]
                image_crop2 = gray[y1-int((98)*height/2748):y2-int((80)*height/2748), x1-int((5)*width/3840):x2+int((400)*width/3840)]
                # image_crop1 = cv2.threshold(image_crop1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                # image_crop2 = cv2.threshold(image_crop2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                cv2.imwrite("image_crop1.jpg", image_crop1)
                cv2.imwrite("image_crop2.jpg", image_crop2)
                # cv2.rectangle(image, (x1-int((5)*(image.shape[1]/1081)), y1-int((98)*(image.shape[0]/750))), (x2+int((400)*(image.shape[1]/1081)),y2-int((80)*(image.shape[0]/750))), (0,255,0), thickness= 3)
                # cv2.imwrite("image_rec.jpg", image)
                # image_crop1_upsize = cv2.resize(image_crop1, (0,0), fx= 4, fy = 4, interpolation=cv2.INTER_LANCZOS4)
                # image_crop1_upsize = cv2.threshold(image_crop1_upsize, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                # cv2.imwrite("image_crop1_upsize.jpg", image_crop1_upsize)
                
                text1 = pytesseract.image_to_string(image_crop1, config=r'--oem 3 --psm 8 -l eng')
                text2 = pytesseract.image_to_string(image_crop2, config=r'--oem 3 --psm 7 -l jpn')
                print
                idx_text1, confidence_text1 = self.get_best_match(''.join(text1.split()).strip().lower(), valid_tdc_kg)
                idx_text2, confidence_text2 = self.get_best_match(''.join(text2.split()).strip().upper(), valid_tdc)

                print("text1:", text1)
                print("text2:", text2)
                weight = text1 if text1 else ""
                print("weight ", weight)
                text_returned = f"{text2}"
                if confidence_text1 is None:
                    confidence_text1 = 0.00
                if confidence_text2 is None:
                    confidence_text2 = 0.00
                confidence_ocr = (confidence_text1 + confidence_text2) / 2
                
                match (idx_text1, idx_text2):
                    case (0, 0):  # "20kg", "でん粉「TW-100」"
                        return "Label-30",  confidence_ocr, text_returned, weight
                    case (3, 1):  # "18kg", "食品用タピオカでん粉「BK-V」"
                        return "Label-49",  confidence_ocr, text_returned, weight
                    case (0, 1):  # "20kg", "食品用タピオカでん粉「BK-V」"
                        return "Label-50", confidence_ocr, text_returned, weight
                    case (1, 2):  # "25kg", "食品用タピオカでん粉「BK-V3」"
                        return "Label-51", confidence_ocr, text_returned, weight
                    case (1, 3):  # "25kg", "イモのちから"
                        return "Label-52", confidence_ocr, text_returned, weight
                    case (0, 4):  # "20kg", "食品用タピオカでん粉「ES-5」"
                        return "Label-53", confidence_ocr, text_returned, weight
                    case (1, 5):  # "25kg", "食品用タピオカでん粉「SK-08」"
                        return "Label-54", confidence_ocr, text_returned, weight
                    case (1, 6):  # "25kg", "食品用タピオカでん粉「タピオカV3」"
                        return "Label-55", confidence_ocr, text_returned, weight
                    case (2, 7):  # "12.5kg", "食品用タピオカでん粉「タピオカV」"
                        return "Label-56", confidence_ocr, text_returned, weight
                    case (0, 8):  # "20kg", "食品用タピオカでん粉「FM-5」"
                        return "Label-57", confidence_ocr, text_returned, weight
                    case (0, 7):  # "20kg", "食品用タピocaでん粉「タピオカV」"
                        return "Label-58", confidence_ocr, text_returned, weight
                    case (0, 9):  # "20kg", "食品用タピオカでん粉「RT-90」"
                        return "Label-59", confidence_ocr, text_returned, weight
                    case (1, 7):  # "25kg", "食品用タピオカでん粉「タピオカV」"
                        return "Label-60", confidence_ocr, text_returned, weight
                    case (1, 1):  # "25kg", "食品用タピオカでん粉「BK-V」"
                        return "Label-61", confidence_ocr, text_returned, weight
                    case (1, 10): # "25kg", "食品用タピオカでん粉「タピオカV2」"
                        return "Label-62", confidence_ocr, text_returned, weight
                    case (1, 11): # "25kg", "食品用タピオカでん粉「BK-V7」"
                        return "Label-63", confidence_ocr, text_returned, weight
                    case _:
                        return "Group-04 (TDC)", confidence_ocr, text_returned, weight
        return "Group-04 (TDC)", confidence_ocr, text_returned, weight

    def classify_label_logo_recycling(self, image, height, width):
        confidence_ocr = 0
        weight = ""
        text = ""
        results = self.model_logo_recycling.predict(source=image)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, _, _ = map(int, box.xyxy[0])
                zone_text = image[y1+int(115*height/2748):y1+int(305*height/2748), x1-int(780*width/3840):x1+int(80*width/3840)]
                # zone_text = image[194:379, int(image.shape[1]/2-430):int(image.shape[1]/2+430)]
                cv2.imwrite("zone_text.jpg", zone_text)
                
                text = pytesseract.image_to_string(zone_text, config=r'--oem 3 --psm 6 -l vie')
                text = text.replace("\n", "")
                
                idx_text, confidence_ocr = self.get_best_match(text, valid_label_recyling)
                print("text:", text)
                match idx_text:
                    case 3:
                        return "Label-64", confidence_ocr, text, weight
                    case 7:
                        return "Label-65", confidence_ocr, text, weight
                    case 5:
                        return "Label-66", confidence_ocr, text, weight
                    case 4:
                        return "Label-67", confidence_ocr, text, weight
                    case 1:
                        return "Label-68", confidence_ocr, text, weight
                    case 6:
                        return "Label-69", confidence_ocr, text, weight
                    case 8:
                        return "Label-70", confidence_ocr, text, weight
                    case 2:
                        return "Label-71", confidence_ocr, text, weight
                    case 0:
                        return "Label-72", confidence_ocr, text, weight
                    case _:
                        return "Group-01", confidence_ocr, "", weight
        return "Group-01", confidence_ocr, "", weight

    def classify_label_logo_halal(self, image, height, width):
        weight = ""
        confidence_ocr = 0

        # zone_text = image[235:330, 60:650]
        zone_text = image[int(235*height/2748):int(330*height/2748), int(60*width/3840):int(650*width/3840)]
        cv2.imwrite("zone_text.jpg", zone_text)
        
        text = pytesseract.image_to_string(zone_text, config=r'--oem 3 --psm 7 -l eng')
        idx_text, confidence_ocr = self.get_best_match(text, valid_label_halal)
        
        match idx_text:
            case 0:  # AL-69 (E1412) (FOOD GRADE)
                return "Label-73", confidence_ocr, text, weight
            case 1:  # AL-43F (E1450) (FOOD GRADE)
                return "Label-74", confidence_ocr, text, weight
            case 2:  # AL-58 (E1422) (FOOD GRADE)
                return "Label-75", confidence_ocr, text, weight
            case 3:  # AL-94 (FOOD GRADE)
                return "Label-45", confidence_ocr, text, weight
            case _:
                return "Group-02", confidence_ocr, "", weight

    def classify_label_logo_unu(self, image, height, width):

        confidence_ocr = 0
        weight = ""
        # zone_text = image[160:350, 800:1300]
        zone_text = image[int(160*height/2748):int(350*height/2748), int(800*width/3840):int(1300*width/3840)]
        cv2.imwrite("zone_text.jpg", zone_text)
        text = pytesseract.image_to_string(zone_text, config=r'--oem 3 --psm 7 -l eng')
        text = f"サナス{text}"
        idx_text, confidence_ocr = self.get_best_match(text, valid_label_unu)
        
        print("text:", text)
        
        match idx_text:
            case 0:  # "サナス514"
                return "Label-34", confidence_ocr, text, weight
            case 1:  # "サナス510"
                return "Label-76", confidence_ocr, text, weight
            case 2:  # "サナスTS01V"
                return "Label-77", confidence_ocr, text, weight
            case _:
                return "Group-03", confidence_ocr, "", weight
    
    def classify_label_36(self, image, height, width):
        image_crop = image[0:int(260*height/2748), 0:int(image.shape[1]/2)]  
        text = pytesseract.image_to_string(image_crop, config=r'--oem 3 --psm 3 -l eng')
        idx_text, _ = self.get_best_match(text, valid_label_36)
        print("text:", text)
        if idx_text == 0:
            return "Label-78"
        else:
            return "Label-36"
        
    def get_best_match(self, text, valid_list, score_cutoff=50):
        match = process.extractOne(text, valid_list, scorer=fuzz.ratio, score_cutoff=score_cutoff)
        if match:
            # match[2]: index trong valid_list, match[1]: score (độ tương đồng)
            print("OCR Confidence: ", match[1]/100)
            return match[2], match[1]/100
        else:
            return None, None

# import pytesseract 
# import cv2  
# import numpy as np
# from src.function.yolo import AiHander
# from PIL import Image
# from rapidfuzz import process, fuzz
# from PIL import Image, ImageDraw, ImageFont

# valid_tdc = ["でん粉「TW-100」", 
#              "食品用タピオカでん粉「BK-V」", 
#              "食品用タピオカでん粉「BK-V3」",
#              "イモのちから",
#              "食品用タピオカでん粉「ES-5」",
#              "食品用タピオカでん粉「SK-08」", 
#              "食品用タピオカでん粉「タピオカV3」",
#              "食品用タピオカでん粉「タピオカV」", 
#              "食品用タピオカでん粉「FM-5」", 
#              "食品用タピオカでん粉「RT-90」",
#              "食品用タピオカでん粉「タピオカV2」", 
#              "食品用タピオカでん粉「BK-V7」",]

# valid_tdc_kg = ["20kg", "25kg", "12.5kg", "18kg", "12kg",]

# valid_label_recyling = ["TINH BỘT ACETAT (TINH BỘT BIẾN TÍNH BS (A)) Dùng cho thực phẩm",
#                         "DISTARCH PHOSPHAT ACETYLAT (TINH BỘT BIẾN TÍNH CB) Dùng cho thực phẩm", 
#                         "TINH BỘT NATRI OCTENYL SUCCINAT (TINH BỘT BIẾN TÍNH KS) Dùng cho thực phẩm", 
#                         "TINH BỘT OXY HÓA (TINH BỘT BIẾN TÍNH ET) Dùng cho thực phẩm", 
#                         "DISTARCH ADIPAT ACETYLAT (TINH BỘT BIẾN TÍNH CBA) Dùng cho thực phẩm", 
#                         "DISTARCH PHOSPHAT (TINH BỘT BIẾN TÍNH CT) Dùng cho thực phẩm",
#                         "ACETYLATED OXYDIZED STARCH (TINH BỘT BIẾN TÍNH EB) Dùng cho thực phẩm",
#                         "CATIONIC STARCH (TINH BỘT BIẾN TÍNH JT) Cấp Công Nghiệp",
#                         "STARCH ACETATE ESTE HÓA VỚI VINYL ACETATE (TINH BỘT BIẾN TÍNH BS) Dùng cho thực phẩm"]

# valid_label_halal = ["AL-69 (E1412) (FOOD GRADE)",
#                      "AL-43F (E1450) (FOOD GRADE)", 
#                      "AL-58 (E1422) (FOOD GRADE)", 
#                      "AL-94 (FOOD GRADE)",]

# valid_label_unu = ["サナス514", "サナス510", "サナスTS01V"]

# valid_label_36 = [
# """
# IMPORTER:
# VETIA INTERNATIONAL
# 10 RUE DES HALLES
# 75001 PARIS - FRANCE]
# """,
# """
# MODIFIED STARCH CBA-8866
# NET WEIGHT: 25KGS
# GROSS WEOHT: 25.35KGS
# """,]

# valid_label_37 = [
# """
# QUICKFLOW Modified Starch,
# FS-85 Modified Starch,
# """
# ]


# class OCR_Engine(AiHander):
#     def __init__(self):
#         super().__init__()
        
#     def classifi_tdc_with_ocr(self, image): 
#         # image = cv2.imread(image)
#         new_img = None
#         confidence_ocr = 0
#         w, h = image.shape[1], image.shape[0]
#         # tạo mask với kích thước lớn hơn ảnh gốc 10
#         mask = np.zeros((h+10, w+10, 3), dtype=np.uint8)
        
#         #ghép ảnh vào giữa mask
#         mask[5:h+5, 5:w+5] = image
        
#         results = self.model_detect_logo_tdc.predict(source=image)
#         if results is None or len(results) == 0 or results[0].boxes is None:
#             return False, None
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 if center_x < w // 2 and center_y < h // 2:
#                     print("Nhan tren trai")
#                     new_img = mask
#                 elif center_x > w // 2 and center_y < h // 2:
#                     print("Nhan tren phai")
#                     new_img = self.rotate_image(mask, 90)
#                     center_x, center_y = self.transform_point((center_x, center_y), image, 90)
#                 elif center_x < w // 2 and center_y > h // 2:
#                     print("Nhan duoi trai")
#                     new_img = self.rotate_image(mask, -90)
#                     center_x, center_y = self.transform_point((center_x, center_y), image, -90)
#                 elif center_x > w // 2 and center_y > h // 2:
#                     print("Nhan duoi phai")
#                     new_img = self.rotate_image(mask, 180)
#                     center_x, center_y = self.transform_point((center_x, center_y), image, 180)

#                 # Chuyển đổi ảnh sang grayscale
#                 gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
#                 # Áp dụng threshold
#                 _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                 # Tìm contours
#                 contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 # Lọc contours theo diện tích nhỏ hơn max_area
#                 max_area = (w+10) * (h+10)
#                 filtered_contours = [contour for contour in contours if cv2.contourArea(contour) < max_area]

#                 if filtered_contours:
#                     # Tìm contour lớn nhất trong danh sách đã lọc
#                     contour = max(filtered_contours, key=cv2.contourArea)
#                 else:
#                     print("Không có contour nào thỏa mãn điều kiện.")
#                     contour = None
                    
#                 if contour is not None:
#                     image_crop = self.crop_rotated_contour(new_img, contour)
#                     results = self.model_detect_khoiluong_tdc.predict(source=image_crop)
#                     if results is None or len(results) == 0 or results[0].boxes is None:
#                         return False, None
#                     for result in results:
#                         boxes = result.boxes
#                         for box in boxes:
#                             x1, y1, x2, y2 = map(int, box.xyxy[0])

#                             image_crop1 = image_crop[y1-4:y2+4, x1:x2]
#                             image_crop2 = image_crop[y1-98:y2-80, x1-5:x2+400]

#                             text1 = pytesseract.image_to_string(image_crop1, config=r'--oem 3 --psm 8 -l eng')
#                             text2 = pytesseract.image_to_string(image_crop2, config=r'--oem 3 --psm 7 -l jpn')

#                             idx_text1, confidence_text1 = self.get_best_match(''.join(text1.split()).strip().lower(), valid_tdc_kg)
#                             idx_text2, confidence_text2 = self.get_best_match(''.join(text2.split()).strip().upper(), valid_tdc)

#                             # if text1 is not None:
#                             #     image_crop = self.draw_text_with_pillow(image_crop, text1, (x1, y1-20), font_path="simsun.ttc", font_size=20, color=(0, 255, 0))
#                             # if text2 is not None:
#                             #     image_crop = self.draw_text_with_pillow(image_crop, text2, (x1, y1-110), font_path="simsun.ttc", font_size=20, color=(0, 255, 0))
                            
#                             print("text1:", text1)
#                             print("text2:", text2)
#                             weight = text1 if text1 else ""
#                             text_returned = f"{text2}"
#                             if confidence_text1 is None:
#                                 confidence_text1 = 0.00
#                             if confidence_text2 is None:
#                                 confidence_text2 = 0.00
#                             confidence_ocr = (confidence_text1 + confidence_text2) / 2
                            
#                             match (idx_text1, idx_text2):
#                                 case (0, 0):  # "20kg", "でん粉「TW-100」"
#                                     return "Label-30", image_crop, confidence_ocr, text_returned, weight
#                                 case (3, 1):  # "18kg", "食品用タピオカでん粉「BK-V」"
#                                     return "Label-49", image_crop, confidence_ocr, text_returned, weight
#                                 case (0, 1):  # "20kg", "食品用タピオカでん粉「BK-V」"
#                                     return "Label-50", image_crop, confidence_ocr, text_returned, weight
#                                 case (1, 2):  # "25kg", "食品用タピオカでん粉「BK-V3」"
#                                     return "Label-51", image_crop, confidence_ocr, text_returned, weight
#                                 case (1, 3):  # "25kg", "イモのちから"
#                                     return "Label-52", image_crop, confidence_ocr, text_returned, weight
#                                 case (0, 4):  # "20kg", "食品用タピオカでん粉「ES-5」"
#                                     return "Label-53", image_crop, confidence_ocr, text_returned, weight
#                                 case (1, 5):  # "25kg", "食品用タピオカでん粉「SK-08」"
#                                     return "Label-54", image_crop, confidence_ocr, text_returned, weight
#                                 case (1, 6):  # "25kg", "食品用タピオカでん粉「タピオカV3」"
#                                     return "Label-55", image_crop, confidence_ocr, text_returned, weight
#                                 case (2, 7):  # "12.5kg", "食品用タピオカでん粉「タピオカV」"
#                                     return "Label-56", image_crop, confidence_ocr, text_returned, weight
#                                 case (0, 8):  # "20kg", "食品用タピオカでん粉「FM-5」"
#                                     return "Label-57", image_crop, confidence_ocr, text_returned, weight
#                                 case (0, 7):  # "20kg", "食品用タピocaでん粉「タピオカV」"
#                                     return "Label-58", image_crop, confidence_ocr, text_returned, weight
#                                 case (0, 9):  # "20kg", "食品用タピオカでん粉「RT-90」"
#                                     return "Label-59", image_crop, confidence_ocr, text_returned, weight
#                                 case (1, 7):  # "25kg", "食品用タピオカでん粉「タピオカV」"
#                                     return "Label-60", image_crop, confidence_ocr, text_returned, weight
#                                 case (1, 1):  # "25kg", "食品用タピオカでん粉「BK-V」"
#                                     return "Label-61", image_crop, confidence_ocr, text_returned, weight
#                                 case (1, 10): # "25kg", "食品用タピオカでん粉「タピオカV2」"
#                                     return "Label-62", image_crop, confidence_ocr, text_returned, weight
#                                 case (1, 11): # "25kg", "食品用タピオカでん粉「BK-V7」"
#                                     return "Label-63", image_crop, confidence_ocr, text_returned, weight
#                                 case _:
#                                     return "Group-04 (TDC)", image_crop, confidence_ocr, "", ""
#         return "Group-04 (TDC)", new_img, confidence_ocr, "", ""

#     def classify_label_logo_recycling(self, image):
#         """"
#             1. Phát hien logo recycling
#             2. Xoay ảnh thẳng đứng
#             3. Cắt đúng vùng ảnh
#             4. lấy vùng đọc chữ
#         """
#         confidence_ocr = 0
#         weight = ""
#         # Đọc ảnh
#         new_img = None
#         # image = cv2.imread(image_path)
#         w, h = image.shape[1], image.shape[0]
#         # Dự đoán ảnh
#         results = self.model_logo_recycling.predict(source=image)
#         if results is None or len(results) == 0 or results[0].boxes is None:
#             return False, None
        
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
                
#                 if center_x < w // 2 and center_y < h // 2:
#                     print("Nhan tren trai")
#                     new_img = self.rotate_image(image, -90)
#                 elif center_x > w // 2 and center_y < h // 2:
#                     print("Nhan tren phai")
#                     new_img = image
#                 elif center_x < w // 2 and center_y > h // 2:
#                     print("Nhan duoi trai")
#                     new_img = self.rotate_image(image, 180)
#                 elif center_x > w // 2 and center_y > h // 2:
#                     print("Nhan duoi phai")
#                     new_img = self.rotate_image(image, 90)
                    
#                 w, h = new_img.shape[1], new_img.shape[0]
#                 # tạo mask với kích thước lớn hơn ảnh gốc 10
#                 mask = np.zeros((h+10, w+10, 3), dtype=np.uint8)
#                 #ghép ảnh vào giữa mask
#                 mask[5:h+5, 5:w+5] = new_img
#                 center_x, center_y = center_x + 5, center_y + 5
                
#                 # Chuyển đổi ảnh sang grayscale
#                 gray_img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#                 # Áp dụng thresholdx
#                 _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                 # Tìm contours
#                 contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 # Lọc contours theo diện tích nhỏ hơn max_area
#                 max_area = (w+10) * (h+10)
#                 filtered_contours = [contour for contour in contours if cv2.contourArea(contour) < max_area]

#                 if filtered_contours:
#                     # Tìm contour lớn nhất trong danh sách đã lọc
#                     contour = max(filtered_contours, key=cv2.contourArea)
#                 else:
#                     print("Không có contour nào thỏa mãn điều kiện.")
#                     contour = None
                    
#                 if contour is not None:
#                     image_crop = self.crop_rotated_contour_Dung(mask, contour)
#                     results = self.model_logo_recycling.predict(source=image_crop)
#                     for result in results:
#                         boxes = result.boxes
#                         for box in boxes:
#                             x1, y1, x2, y2 = map(int, box.xyxy[0])
#                             zone_text = image_crop[y1+120:y1+305, x1-780:x1+80]
#                             text = pytesseract.image_to_string(zone_text, config=r'--oem 3 --psm 6 -l vie')
#                             text = text.replace("\n", "")
#                             # text = text.replace(" ", "")
#                             idx_text, confidence_ocr = self.get_best_match(text, valid_label_recyling)
#                             # idx_text = valid_label_recyling.index(text) if text in valid_label_recyling else None
#                             print("text:", text)
#                             match idx_text:
#                                 case 3:
#                                     return "Label-64", image_crop, confidence_ocr, text, weight
#                                 case 7:
#                                     return "Label-65", image_crop, confidence_ocr, text, weight
#                                 case 5:
#                                     return "Label-66", image_crop, confidence_ocr, text, weight
#                                 case 4:
#                                     return "Label-67", image_crop, confidence_ocr, text, weight
#                                 case 1:
#                                     return "Label-68", image_crop, confidence_ocr, text, weight
#                                 case 6:
#                                     return "Label-69", image_crop, confidence_ocr, text, weight
#                                 case 8:
#                                     return "Label-70", image_crop, confidence_ocr, text, weight
#                                 case 2:
#                                     return "Label-71", image_crop, confidence_ocr, text, weight
#                                 case 0:
#                                     return "Label-72", image_crop, confidence_ocr, text, weight

#         return "Group-01", new_img, confidence_ocr, "", weight

#     def classify_label_logo_halal(self, image):
#         """"
#             1. Phát hien logo halal
#             2. Xoay ảnh thẳng đứng
#             3. Cắt đúng vùng ảnh
#             4. lấy vùng đọc chữ
#         """
#         weight = ""
#         confidence_ocr = 0
#         # Đọc ảnh
#         new_img = None
#         text = ""
#         w, h = image.shape[1], image.shape[0]
#         # Dự đoán ảnh
#         results = self.model_logo_halal.predict(source=image)
#         if results is None or len(results) == 0 or results[0].boxes is None:
#             return False, None

#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
                
#                 if center_x < w // 2 and center_y < h // 2:
#                     print("Nhan tren trai")
#                     new_img = self.rotate_image(image, -90)
                    
#                 elif center_x > w // 2 and center_y < h // 2:
#                         print("Nhan tren phai")
#                         new_img = image
#                 elif center_x < w // 2 and center_y > h // 2:
#                     print("Nhan duoi trai")
#                     new_img = self.rotate_image(image, 180)

#                 elif center_x > w // 2 and center_y > h // 2:
#                     print("Nhan duoi phai")
#                     new_img = self.rotate_image(image, 90)
                    
#                 w, h = new_img.shape[1], new_img.shape[0]
#                 # tạo mask với kích thước lớn hơn ảnh gốc 10
#                 mask = np.zeros((h+10, w+10, 3), dtype=np.uint8)
#                 #ghép ảnh vào giữa mask
#                 mask[5:h+5, 5:w+5] = new_img
#                 # Chuyển đổi ảnh sang grayscale
#                 gray_img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#                 # Áp dụng thresholdx
#                 _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                 # Tìm contours
#                 contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 # Lọc contours theo diện tích nhỏ hơn max_area
#                 max_area = (w+10) * (h+10)
#                 filtered_contours = [contour for contour in contours if cv2.contourArea(contour) < max_area]
                
#                 if filtered_contours:
#                     # Tìm contour lớn nhất trong danh sách đã lọc
#                     contour = max(filtered_contours, key=cv2.contourArea)
#                 else:
#                     print("Không có contour nào thỏa mãn điều kiện.")
#                     contour = None

#                 if contour is not None:
#                     image_crop = self.crop_rotated_contour(mask, contour)
#                     zone_text = image_crop[230:330, 70:650]
#                     cv2.imwrite("zone_text.jpg", zone_text)
#                     text = pytesseract.image_to_string(zone_text, config=r'--oem 3 --psm 7 -l eng')
#                     idx_text, confidence_ocr = self.get_best_match(text, valid_label_halal)
#                     # image_crop = self.draw_text_with_pillow(image_crop, text, (120, 230), font_size=28, color=(0, 255, 0))
                    
#                     match idx_text:
#                         case 0:  # AL-69 (E1412) (FOOD GRADE)
#                             return "Label-73", image_crop, confidence_ocr, text, weight
#                         case 1:  # AL-43F (E1450) (FOOD GRADE)
#                             return "Label-74", image_crop, confidence_ocr, text, weight
#                         case 2:  # AL-58 (E1422) (FOOD GRADE)
#                             return "Label-75", image_crop, confidence_ocr, text, weight
#                         case 3:  # AL-94 (FOOD GRADE)
#                             return "Label-45", image_crop, confidence_ocr, text, weight
#                         case _:
#                             return "Group-02", image_crop, confidence_ocr, "", weight

#         return "Group-02", new_img, confidence_ocr, "", weight

#     def classify_label_logo_unu(self, image):
#         """"
#             1. Phát hien logo unu
#             2. Xoay ảnh thẳng đứng
#             3. Cắt đúng vùng ảnh
#             4. lấy vùng đọc chữ
#         """
#         confidence_ocr = 0
#         weight = ""
#         # Đọc ảnh
#         new_img = None
#         text = ""
#         w, h = image.shape[1], image.shape[0]
#         # Dự đoán ảnh
#         results = self.model_logo_unu.predict(source=image)
#         if results is None or len(results) == 0 or results[0].boxes is None:
#             return "", image

#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
                
#                 if center_x < w // 2 and center_y < h // 2:
#                     print("Nhan tren trai")
#                     new_img = image
#                 elif center_x > w // 2 and center_y < h // 2:
#                     print("Nhan tren phai")
#                     new_img = self.rotate_image(image, 90)
#                 elif center_x < w // 2 and center_y > h // 2:
#                     print("Nhan duoi trai")
#                     new_img = self.rotate_image(image, -90)

#                 elif center_x > w // 2 and center_y > h // 2:
#                     print("Nhan duoi phai")
#                     new_img = self.rotate_image(image, 180)
                    
#                 w, h = new_img.shape[1], new_img.shape[0]
#                 # tạo mask với kích thước lớn hơn ảnh gốc 10
#                 mask = np.zeros((h+10, w+10, 3), dtype=np.uint8)
#                 #ghép ảnh vào giữa mask
#                 mask[5:h+5, 5:w+5] = new_img
#                 # Chuyển đổi ảnh sang grayscale
#                 gray_img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#                 # Áp dụng thresholdx
#                 _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                 # Tìm contours
#                 contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 # Lọc contours theo diện tích nhỏ hơn max_area
#                 max_area = (w+10) * (h+10)
#                 filtered_contours = [contour for contour in contours if cv2.contourArea(contour) < max_area]
#                 if filtered_contours:
#                     # Tìm contour lớn nhất trong danh sách đã lọc
#                     contour = max(filtered_contours, key=cv2.contourArea)
#                 else:
#                     print("Không có contour nào thỏa mãn điều kiện.")
#                     contour = None

#                 if contour is not None:
#                     image_crop = self.crop_rotated_contour(mask, contour)
#                     # zone_text = image_crop[160:350, 180:1300]
#                     zone_text = image_crop[160:350, 800:1300]
#                     text = pytesseract.image_to_string(zone_text, config=r'--oem 3 --psm 7 -l eng')
#                     text = f"サナス{text}"
#                     idx_text, confidence_ocr = self.get_best_match(text, valid_label_unu)
#                     # image_crop = self.draw_text_with_pillow(image_crop, text, (300, 130),font_size=50, color=(0, 255, 0))
                    
#                     print("text:", text)
                    
#                     match idx_text:
#                         case 0:  # "サナス514"
#                             return "Label-34", image_crop, confidence_ocr, text, weight
#                         case 1:  # "サナス510"
#                             return "Label-76", image_crop, confidence_ocr, text, weight
#                         case 2:  # "サナスTS01V"
#                             return "Label-77", image_crop, confidence_ocr, text, weight
#                         case _:
#                             return "Group-03", image_crop, confidence_ocr, "", weight

#         return "Group-03", new_img, confidence_ocr, "", weight
    
#     def classify_label_36(self, image):
#         image_crop = image[0:260, 0:int(image.shape[1]/2)]  
#         text = pytesseract.image_to_string(image_crop, config=r'--oem 3 --psm 3 -l eng')
#         idx_text, _ = self.get_best_match(text, valid_label_36)
#         print("text:", text)
#         if idx_text == 0:
#             return "Label-78"
#         else:
#             return "Label-36"
        
#     def classify_label_37(self, image):
#         # image_crop = image[0:190, :]  
#         # cv2.imwrite("image_crop.png", image_crop)
#         text = pytesseract.image_to_string(image, config=r'--oem 3 --psm 3 -l eng')
#         # idx_text, _ = self.get_best_match(text, valid_label_36)
#         print("text:", text)
#         # if idx_text == 0:
#         #     return "Label-78"
#         # else:
#             # return "Label-36"
            
#     def read_ocr_label(self, image):
#         import time
#         start_time = time.time()
#         text = pytesseract.image_to_string(image, config=r'--oem 3 --psm 3 -l eng+jpn+vie')
#         print(f"Thời gian đọc toàn bộ nhãn: {time.time()-start_time} giây.")
#         print("text:", text)
#         cv2.imwrite("label_image.jpg",image)

#     def rotate_image(self, image, angle):
#         (h, w) = image.shape[:2]
#         center = (w // 2, h // 2)

#         # Tính toán ma trận xoay
#         matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

#         # Lấy kích thước mới sau khi xoay
#         cos = np.abs(matrix[0, 0])
#         sin = np.abs(matrix[0, 1])

#         new_w = int((h * sin) + (w * cos))
#         new_h = int((h * cos) + (w * sin))

#         # Cập nhật ma trận xoay để dịch ảnh đúng vào trung tâm
#         matrix[0, 2] += (new_w / 2) - center[0]
#         matrix[1, 2] += (new_h / 2) - center[1]

#         # Xoay ảnh với kích thước mới
#         rotated = cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
#         return rotated

#     def transform_point(self, p, image, angle):
#         # Chuyển đổi tọa độ điểm p từ tọa độ gốc sang tọa độ mới
#         (h, w) = image.shape[:2]
#         center = (w // 2, h // 2)
#         # Tính toán ma trận xoay
#         rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#         new_x = rotation_matrix[0, 0] * p[0] + rotation_matrix[0, 1] * p[1] + rotation_matrix[0, 2]
#         new_y = rotation_matrix[1, 0] * p[0] + rotation_matrix[1, 1] * p[1] + rotation_matrix[1, 2]
#         new_point = (int(new_x), int(new_y))
#         return new_point

#     def crop_rotated_contour(self, image, contour):
#         rect = cv2.minAreaRect(contour)
#         box = cv2.boxPoints(rect)
#         box = np.array(box, dtype="float32")

#         # Sắp xếp box theo thứ tự: top-left, top-right, bottom-right, bottom-left
#         width = int(rect[1][0])
#         height = int(rect[1][1])
#         print("width:", width)
#         print("height:", height)
#         if height < width:
#             # width, height = height, width

#             # Xoay box theo chiều kim đồng hồ 90 độ
#             box = np.roll(box, -1, axis=0)
#         if width < height:
#             width, height = height, width

#         if width == 0 or height == 0:
#             return None  # tránh lỗi chia 0
        
#         # Nếu height > width thì hoán đổi và xoay box cho đúng hướng
        
        
#         dst_pts = np.array([
#             [0, 0],
#             [width - 1, 0],
#             [width - 1, height - 1],
#             [0, height - 1]
#         ], dtype="float32")

#         # Tính ma trận transform và áp dụng
#         M = cv2.getPerspectiveTransform(box, dst_pts)
#         warped = cv2.warpPerspective(image, M, (width, height))

#         return warped

#     def get_best_match(self, text, valid_list, score_cutoff=50):
#         match = process.extractOne(text, valid_list, scorer=fuzz.ratio, score_cutoff=score_cutoff)
#         if match:
#             # match[2]: index trong valid_list, match[1]: score (độ tương đồng)
#             print("OCR Confidence: ", match[1]/100)
#             return match[2], match[1]/100
#         else:
#             return None, None

#     def draw_text_with_pillow(self, image, text, position, font_path="simsun.ttc", font_size=20, color=(0, 255, 0)):
#         """
#         Vẽ văn bản Unicode lên ảnh bằng Pillow.
        
#         Args:
#             image: Ảnh OpenCV (numpy.ndarray).
#             text: Văn bản Unicode cần vẽ.
#             position: Tọa độ (x, y) để vẽ văn bản.
#             font_path: Đường dẫn đến file font (ví dụ: simsun.ttc cho tiếng Trung).
#             font_size: Kích thước font.
#             color: Màu văn bản (BGR).
        
#         Returns:
#             Ảnh OpenCV với văn bản đã vẽ.
#         """
#         # Chuyển đổi ảnh OpenCV sang định dạng Pillow
#         image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         draw = ImageDraw.Draw(image_pil)

#         # Tải font
#         font = ImageFont.truetype(font_path, font_size)

#         # Vẽ văn bản
#         draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))  # Đảo thứ tự màu từ BGR sang RGB

#         # Chuyển đổi ảnh Pillow trở lại OpenCV
#         image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
#         return image

#     def crop_rotated_contour_Dung(self, image, contour):
#         rect = cv2.minAreaRect(contour)
#         box = cv2.boxPoints(rect)
#         box = np.array(box, dtype="float32")

#         # Sắp xếp box theo thứ tự: top-left, top-right, bottom-right, bottom-left
#         width = int(rect[1][0])
#         height = int(rect[1][1])
#         print("width:", width)
#         print("height:", height)
#         if height > width:
#             # width, height = height, width

#             # Xoay box theo chiều kim đồng hồ 90 độ
#             box = np.roll(box, -1, axis=0)
#         if width > height:
#             width, height = height, width

#         if width == 0 or height == 0:
#             return None  # tránh lỗi chia 0
        
#         # Nếu height > width thì hoán đổi và xoay box cho đúng hướng
        
        
#         dst_pts = np.array([
#             [0, 0],
#             [width - 1, 0],
#             [width - 1, height - 1],
#             [0, height - 1]
#         ], dtype="float32")

#         # Tính ma trận transform và áp dụng
#         M = cv2.getPerspectiveTransform(box, dst_pts)
#         warped = cv2.warpPerspective(image, M, (width, height))

#         return warped