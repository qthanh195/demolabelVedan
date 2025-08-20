from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from src.api.schemas.image_schemas_copy import ImageCaptureRequest
# from src.handler.api_handler import ApiHandler
import uvicorn

import random
import cv2
import base64

# api_handel = ApiHandler()
app = FastAPI()

# THÊM middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả domain
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả phương thức (GET, POST, PUT, DELETE,...)
    allow_headers=["*"],  # Cho phép tất cả headers
)

@app.post("/image_capture")
def capture_image(req: ImageCaptureRequest):
    print("aaaaaaaaa")
    pallet_infos = [
            (req.name_a, req.thresh_a_object, req.thresh_a_group, req.thresh_a_ocr),
            (req.name_b, req.thresh_b_object, req.thresh_b_group, req.thresh_b_ocr),
            (req.name_c, req.thresh_c_object, req.thresh_c_group, req.thresh_c_ocr),
            (req.name_d, req.thresh_d_object, req.thresh_d_group, req.thresh_d_ocr),
            (req.name_e, req.thresh_e_object, req.thresh_e_group, req.thresh_e_ocr),
            (req.name_f, req.thresh_f_object, req.thresh_f_group, req.thresh_f_ocr),
            (req.thresh_area)
        ]
    print(pallet_infos)
    def image_to_base64(image_np: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image_np)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    # results = api_handel.analyze_image(pallet_infos)
    return JSONResponse(content={
        "label_detected": f"Label-{random.randint(1, 78)}",
        "pallet_detect": f"A",
        "confidence_detect": f"{random.random():.2f}",
        "confidence_classify": f"{random.random():.2f}",
        "confidence_ocr": f"{random.random():.2f}",
        "origin_image": image_to_base64(cv2.imread("E:/2. GE/22. Vedan Vision Ocr\Image0505\image30\img_20250506_165238.png")),
        "cropped_image": image_to_base64(cv2.imread("data\Label-62.jpg")),
        "text": "Ahihiiiiiiiiiiiiiii",
        "weight": "20kg",
        "percent_area": "98%",
    })


    # return JSONResponse(content={
    #     "label_detected": (results["label_detected"] if results["label_detected"] else "None"),
    #     "pallet_detect": (results["pallet_detected"] if results["pallet_detected"] else "F"),
    #     # "pallet_detect": "C",
    #     "confidence_detect": f"{(results['confidence_detect'] or 0):.2f}",
    #     "confidence_classify": f"{(results['confidence_classify'] or 0):.2f}",
    #     "confidence_ocr": f"{(results['confidence_ocr'] or 0):.2f}",
    #     "confidence": f"{(results['confidence'] or 0):.2f}",
    #     "origin_image": (
    #         api_handel.image_to_base64(results["origin_image"])
    #         if results["origin_image"] is not None and isinstance(results["origin_image"], np.ndarray) and results["origin_image"].size > 0 else ""
    #     ),
    #     "cropped_image": (
    #         api_handel.image_to_base64(results["label_image"])
    #         if results["label_image"] is not None and isinstance(results["label_image"], np.ndarray) and results["label_image"].size > 0 else ""
    #     ),
    #     "text": (results["text"] if results["text"] else ""),
    #     "weight": (results["weight"] if results["weight"] else ""),
    # })

# @app.on_event("startup")
# def startup_event():
#     print("Khởi động server và mở camera...")
#     api_handel.api_open_camera()

# @app.on_event("shutdown")
# def shutdown_event():
#     print("Đang tắt server và đóng camera...")
#     api_handel.close_camera()

def run_server():
    uvicorn.run(
        "src.api.api_copy:app",          # <tên_file>:<tên_app>
        host="0.0.0.0",      # hoặc "127.0.0.1" nếu chỉ chạy trên máy local
        port=8005,
        reload=True
    )