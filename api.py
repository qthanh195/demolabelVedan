from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from schemas.image_schemas import ImageCaptureRequest
from api_handler import ApiHandler

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
    label_detected, pallet_detect, confidence, origin_image, cropped_label = api_handel.analyze_image(
        req.name_a, req.name_b, req.name_c, 
        req.name_d, req.name_e, req.name_f, 
        req.thresh_a, req.thresh_b, req.thresh_c, 
        req.thresh_d, req.thresh_e, req.thresh_f
    )
    print("label_detected: ", label_detected)
    print("pallet_detect: ", pallet_detect)
    print("confidence: ", f"{confidence:.2f}")
    return JSONResponse(content={
        "label_detected": label_detected,
        "pallet_detect": pallet_detect,
        "confidence": f"{confidence:.2f}",
        "origin_image": origin_image,
        "cropped_image": cropped_label
    })

@app.on_event("startup")
def startup_event():
    print("Khởi động server và mở camera...")
    api_handel.api_open_camera()

@app.on_event("shutdown")
def shutdown_event():
    print("Đang tắt server và đóng camera...")
    api_handel.close_camera()