from pydantic import BaseModel

class ImageCaptureRequest(BaseModel):
    name_a: str
    name_b: str
    name_c: str
    name_d: str
    name_e: str
    name_f: str
    thresh_a: float
    thresh_b: float
    thresh_c: float
    thresh_d: float
    thresh_e: float
    thresh_f: float