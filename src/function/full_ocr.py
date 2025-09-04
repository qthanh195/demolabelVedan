# ...existing code...

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import cv2
import numpy as np
import pytesseract
import os
import time

TESS_CONFIG_BASE = '--oem 1 -l eng+vie+jpn -c load_system_dawg=0 -c load_freq_dawg=0'

# ===== REUSE POOL TO AVOID RESPAWN COST =====
# Tạo 1 pool dùng chung (tối đa 1 lần). Điều kiện __name__ để tránh issues khi import.
GLOBAL_OCR_POOL = ProcessPoolExecutor(max_workers=os.cpu_count()//2 or 2) if __name__ != '__main__' else None

class LabelOCRPipeline:
    """
    Pipeline OCR toàn nhãn:
      - detect logo (optional) -> mask
      - detect text boxes
      - group -> lines -> chunks
      - parallel OCR (multiprocessing)
    """
    def __init__(self,
                 tesseract_lang='vie+eng',
                 max_workers=os.cpu_count() // 2 or 2,
                 enable_logo_detection=True):
        self.lang = tesseract_lang
        self.max_workers = max_workers
        self.enable_logo_detection = enable_logo_detection

    # ============== STEP 1: LOGO ==============
    def detect_logos(self, image: np.ndarray):
        """
        Trả về list[(x1,y1,x2,y2)] logo. Placeholder: bạn gắn model YOLO logo tại đây.
        """
        # TODO: tích hợp self.model_logo.predict(...)
        return []  # hiện chưa có → bỏ qua

    def mask_logos(self, image: np.ndarray, logo_boxes, mode='fill'):
        """
        Che vùng logo để tránh OCR đọc nhầm.
        mode='fill' tô màu nền xám trung bình; mode='white' = trắng.
        """
        if not logo_boxes:
            return image
        out = image.copy()
        if len(out.shape) == 2:
            h_img, w_img = out.shape
        else:
            h_img, w_img = out.shape[:2]
        for (x1, y1, x2, y2) in logo_boxes:
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(w_img, x2); y2 = min(h_img, y2)
            if mode == 'fill':
                patch = out[y1:y2, x1:x2]
                val = int(np.median(patch)) if patch.size else 200
                out[y1:y2, x1:x2] = val
            else:
                out[y1:y2, x1:x2] = 255
        return out

    # ============== STEP 2: TEXT BOXES ==============
    def detect_text_boxes_fallback(self, gray: np.ndarray):
        """
        Fallback morphology + contour. Trả về list boxes (x1,y1,x2,y2).
        """
        bin_img = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # Co & dãn để nối ký tự cùng dòng
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        dil = cv2.dilate(bin_img, kernel, iterations=1)
        contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h_img, w_img = gray.shape
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h < 8 or w < 15:
                continue
            if h > 0.4 * h_img:  # bỏ vùng quá lớn bất thường
                continue
            boxes.append((x, y, x + w, y + h))
        return boxes

    def detect_text_boxes(self, image: np.ndarray):
        """
        Nếu có model text detector (CRAFT/EAST) thì dùng, không thì fallback.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # TODO: tích hợp model CRAFT/EAST ở đây (trả về boxes)
        boxes = self.detect_text_boxes_fallback(gray)
        # Sắp xếp từ trên xuống
        boxes.sort(key=lambda b: (b[1], b[0]))
        return boxes

    # ============== STEP 3: GROUP LINES / CHUNKS ==============
    def group_boxes_to_lines(self, boxes, y_thresh=10):
        lines = []
        for b in boxes:
            x1, y1, x2, y2 = b
            cy = (y1 + y2) // 2
            placed = False
            for line in lines:
                # line structure: {'cy':..., 'boxes':[...] }
                if abs(cy - line['cy']) <= y_thresh:
                    line['boxes'].append(b)
                    # update center
                    ys = [ (bb[1]+bb[3])//2 for bb in line['boxes']]
                    line['cy'] = int(sum(ys)/len(ys))
                    placed = True
                    break
            if not placed:
                lines.append({'cy': cy, 'boxes': [b]})
        # sort boxes inside line
        for ln in lines:
            ln['boxes'].sort(key=lambda bb: bb[0])
        # final lines sorted
        lines.sort(key=lambda d: d['cy'])
        return lines

    def build_chunks_from_lines(self, lines, max_chunk_height=450, pad=4, image_shape=None):
        """
        Gom các line (list dict {'cy':..., 'boxes':[...]}) thành các chunk dọc.
        Tự động nới max_chunk_height nếu quá nhiều line.
        """
        if not lines:
            return []
        if len(lines) > 25:
            max_chunk_height = int(max_chunk_height * 1.4)

        H = image_shape[0] if image_shape is not None else 10_000
        chunks = []
        cur_lines = []
        cur_top = cur_bot = None

        for ln in lines:
            ys = [bb[1] for bb in ln['boxes']] + [bb[3] for bb in ln['boxes']]
            top, bot = min(ys), max(ys)
            if cur_top is None:
                cur_top, cur_bot = top, bot
                cur_lines.append(ln)
                continue

            proposed_top = min(cur_top, top)
            proposed_bot = max(cur_bot, bot)
            if (proposed_bot - proposed_top) <= max_chunk_height:
                cur_top, cur_bot = proposed_top, proposed_bot
                cur_lines.append(ln)
            else:
                chunks.append((cur_top, cur_bot, cur_lines))
                cur_lines = [ln]
                cur_top, cur_bot = top, bot

        if cur_lines:
            chunks.append((cur_top, cur_bot, cur_lines))

        final = []
        for idx, (t, b, ls) in enumerate(chunks):
            t_pad = max(0, t - pad)
            b_pad = min(H, b + pad)
            final.append({
                'index': idx,
                'y1': t_pad,
                'y2': b_pad,
                'lines': ls
            })
        return final

    def merge_chunks(self, chunks, target_count=3):
        """
        Gộp các chunk liền kề để giảm số lần gọi OCR.
        Giữ thứ tự theo trục Y. target_count là số chunk mong muốn (xấp xỉ).
        """
        if len(chunks) <= target_count:
            return chunks
        merged = []
        cur = None
        for ck in chunks:
            if cur is None:
                cur = ck.copy()
                continue
            # nếu còn nhiều chunk hơn target thì tiếp tục gộp
            if len(merged) < target_count - 1:
                cur['y1'] = min(cur['y1'], ck['y1'])
                cur['y2'] = max(cur['y2'], ck['y2'])
            else:
                merged.append(cur)
                cur = ck.copy()
        if cur:
            merged.append(cur)
        # Reindex
        for i, m in enumerate(merged):
            m['index'] = i
        return merged
    # ============== STEP 4: OCR ONE CHUNK ==============
    @staticmethod
    def _ocr_single_chunk(args):
        """
        Hàm static để dùng trong multiprocessing.
        args: (chunk_dict, image, lang)
        """
        chunk, image, lang = args
        y1, y2 = chunk['y1'], chunk['y2']
        crop = image[y1:y2, :]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape)==3 else crop
        # Một bước threshold duy nhất
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        h = y2 - y1
        if h < 70: psm = 7
        elif h < 160: psm = 6
        else: psm = 4
        config = f'{TESS_CONFIG_BASE} --psm {psm} -c preserve_interword_spaces=1'
        try:
            txt = pytesseract.image_to_string(thr, config=config).strip()
        except Exception:
            txt = ""
        return {'chunk_index': chunk['index'], 'y1': y1, 'y2': y2, 'text': txt}

    # ============== STEP 5: FULL PIPE ==============
    def run_full_ocr(self, label_image: np.ndarray,
                     max_chunk_height=450,
                     remove_logo=True,
                     super_chunk_target=3,
                     parallel_min_chunks=6,
                     fast_mode=False):
        """
        fast_mode=True: bỏ chunking, OCR toàn ảnh (nhanh nhất).
        """
        orig = label_image.copy()
        # Fast mode
        if fast_mode:
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY) if len(orig.shape)==3 else orig
            thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            txt = pytesseract.image_to_string(thr, config=f'{TESS_CONFIG_BASE} --psm 4').strip()
            return {'full_text': txt, 'chunks': [{'index':0,'y1':0,'y2':orig.shape[0],'text':txt}],
                    'logo_boxes': [], 'mode': 'fast'}

        logo_boxes = self.detect_logos(orig) if self.enable_logo_detection else []
        processed = self.mask_logos(orig, logo_boxes) if (remove_logo and logo_boxes) else orig

        boxes = self.detect_text_boxes(processed)
        if not boxes:
            full_text = pytesseract.image_to_string(
                processed, config=f'{TESS_CONFIG_BASE} --psm 6'
            ).strip()
            return {'full_text': full_text,'chunks': [],'logo_boxes': logo_boxes,'mode': 'fallback'}

        lines = self.group_boxes_to_lines(boxes)
        chunks = self.build_chunks_from_lines(lines,
                                              max_chunk_height=max_chunk_height,
                                              image_shape=processed.shape)
        chunks = self.merge_chunks(chunks, target_count=super_chunk_target)

        # QUYẾT ĐỊNH SONG SONG HAY TUẦN TỰ
        use_parallel = len(chunks) >= parallel_min_chunks

        tasks = [(ck, processed, self.lang) for ck in chunks]

        t0 = time.time()
        if use_parallel and GLOBAL_OCR_POOL:
            futures = [GLOBAL_OCR_POOL.submit(self._ocr_single_chunk, t) for t in tasks]
            results = [f.result() for f in futures]
        else:
            # Tuần tự (ít chunk → nhanh hơn vì không spawn + scheduling)
            results = [self._ocr_single_chunk(t) for t in tasks]

        results.sort(key=lambda r: r['y1'])
        full_text = "\n".join([r['text'] for r in results if r['text']])
        elapsed = time.time() - t0
        print(f"OCR time ({'parallel' if use_parallel else 'sequential'}): {elapsed:.3f}s | chunks={len(chunks)}")

        return {
            'full_text': full_text,
            'chunks': results,
            'logo_boxes': logo_boxes,
            'mode': 'chunk'
        }
        
# import time
# import cv2
# import numpy as np  
# from src.function.full_ocr import LabelOCRPipeline
# import cv2

# pipeline = LabelOCRPipeline(tesseract_lang='jpn',
#                             max_workers=4,
#                             enable_logo_detection=True)

# def ocr_whole_label(label_image):
#     return pipeline.run_full_ocr(label_image,
#                                  max_chunk_height=360,
#                                  super_chunk_target=3,
#                                  parallel_min_chunks=6,
#                                  fast_mode=True)

# if __name__ == "__main__":
#     start_time = time.time()
#     img = cv2.imread("data/SampleData/Label-70.jpg")
#     r = ocr_whole_label(img)
#     print(r['full_text'])
#     print(f"Processing time: {time.time() - start_time} seconds")