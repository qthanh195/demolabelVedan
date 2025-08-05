from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from schemas.image_schemas import ImageCaptureRequest
from api_handler import ApiHandler
# from log import logging

api_handel = ApiHandler()
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
    pallet_infos = [
            (req.name_a, req.thresh_a),
            (req.name_b, req.thresh_b),
            (req.name_c, req.thresh_c),
            (req.name_d, req.thresh_d),
            (req.name_e, req.thresh_e),
            (req.name_f, req.thresh_f),
        ]
    results = api_handel.analyze_image(pallet_infos)

    print("label_detected: ", results["label_detected"])
    print("pallet_detect: ", results["pallet_detected"])
    print("confidence_detect: ", f"{(results['confidence_detect'] or 0):.2f}")
    print("confidence_classify: ", f"{(results['confidence_classify'] or 0):.2f}")
    print("confidence_ocr: ", f"{(results['confidence_ocr'] or 0):.2f}")
    print("confidence: ", f"{(results['confidence'] or 0):.2f}")

    return JSONResponse(content={
        "label_detected": (results["label_detected"] if results["label_detected"] else "None"),
        "pallet_detect": (results["pallet_detected"] if results["pallet_detected"] else "F"),
        # "pallet_detect": "C",
        "confidence_detect": f"{(results['confidence_detect'] or 0):.2f}",
        "confidence_classify": f"{(results['confidence_classify'] or 0):.2f}",
        "confidence_ocr": f"{(results['confidence_ocr'] or 0):.2f}",
        "confidence": f"{(results['confidence'] or 0):.2f}",
        "origin_image": (
            api_handel._image_to_base64(results["origin_image"])
            if results["origin_image"] is not None and isinstance(results["origin_image"], np.ndarray) and results["origin_image"].size > 0 else ""
        ),
        "cropped_image": (
            api_handel._image_to_base64(results["label_image"])
            if results["label_image"] is not None and isinstance(results["label_image"], np.ndarray) and results["label_image"].size > 0 else ""
        )
    })

@app.on_event("startup")
def startup_event():
    print("Khởi động server và mở camera...")
    api_handel.api_open_camera()

@app.on_event("shutdown")
def shutdown_event():
    print("Đang tắt server và đóng camera...")
    api_handel.close_camera()