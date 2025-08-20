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
        
    def classifi_tdc_with_ocr(self, image): 
        confidence_ocr = 0
        image_crop1 = image[251:290, 327:425]
        image_crop2 = image[163:202, 327:803]
        # cv2.imwrite("image_crop1.jpg", image_crop1)
        # cv2.imwrite("image_crop2.jpg", image_crop2)

        text1 = pytesseract.image_to_string(image_crop1, config=r'--oem 3 --psm 8 -l eng')
        text2 = pytesseract.image_to_string(image_crop2, config=r'--oem 3 --psm 7 -l jpn')

        idx_text1, confidence_text1 = self.get_best_match(''.join(text1.split()).strip().lower(), valid_tdc_kg)
        idx_text2, confidence_text2 = self.get_best_match(''.join(text2.split()).strip().upper(), valid_tdc)

        print("text1:", text1)
        print("text2:", text2)
        weight = text1 if text1 else ""
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
                return "Group-04 (TDC)", confidence_ocr, "", ""

    def classify_label_logo_recycling(self, image):
        confidence_ocr = 0
        weight = ""
        zone_text = image[194:379, int(image.shape[1]/2-430):int(image.shape[1]/2+430)]
        # cv2.imwrite("zone_text.jpg", zone_text)
        
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

    def classify_label_logo_halal(self, image):
        weight = ""
        confidence_ocr = 0
        
        zone_text = image[230:330, 70:650]
        
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

    def classify_label_logo_unu(self, image):

        confidence_ocr = 0
        weight = ""
        zone_text = image[160:350, 800:1300]
        # cv2.imwrite("zone_text.jpg", zone_text)
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
    
    def classify_label_36(self, image):
        image_crop = image[0:260, 0:int(image.shape[1]/2)]  
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
