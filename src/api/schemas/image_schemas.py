from pydantic import BaseModel

class ImageCaptureRequest(BaseModel):
    name_a: str
    name_b: str
    name_c: str
    name_d: str
    name_e: str
    name_f: str
    thresh_a_object: float
    thresh_a_group: float
    thresh_a_ocr: float
    thresh_b_object: float
    thresh_b_group: float
    thresh_b_ocr: float
    thresh_c_object: float
    thresh_c_group: float
    thresh_c_ocr: float
    thresh_d_object: float
    thresh_d_group: float
    thresh_d_ocr: float
    thresh_e_object: float
    thresh_e_group: float
    thresh_e_ocr: float
    thresh_f_object: float
    thresh_f_group: float
    thresh_f_ocr: float
    thresh_area: float
