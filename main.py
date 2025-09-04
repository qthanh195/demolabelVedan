# from src.api.api import run_server

# if __name__ == "__main__":
#     run_server()
import time
import cv2
import numpy as np  
from src.function.full_ocr import LabelOCRPipeline
import cv2

pipeline = LabelOCRPipeline(tesseract_lang='jpn',
                            max_workers=4,
                            enable_logo_detection=True)

def ocr_whole_label(label_image):
    return pipeline.run_full_ocr(label_image,
                                 max_chunk_height=360,
                                 super_chunk_target=3,
                                 parallel_min_chunks=6,
                                 fast_mode=True)

if __name__ == "__main__":
    start_time = time.time()
    img = cv2.imread("data/SampleData/Label-70.jpg")
    r = ocr_whole_label(img)
    print(r['full_text'])
    print(f"Processing time: {time.time() - start_time} seconds")