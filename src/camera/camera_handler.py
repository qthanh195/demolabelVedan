from pypylon import pylon
import numpy as np
import cv2
import time
from collections import deque
import threading

  
class BaslerCamera():
    def __init__(self):
        super().__init__()
        self.camera = None
        self.is_open = False
        
    def get_image(self):
        return self.single_shot()

    def setup_camera(self):
        """Thiết lập cấu hình camera."""
        
        self.camera.PixelFormat.Value = "Mono8" # Đặt định dạng pixel thành Mono8
        self.camera.ExposureAuto.Value = "Off" ## Đặt chế độ tự động điều chỉnh độ sáng thành Once
        self.camera.BalanceWhiteAuto.Value = "Off" # Đặt chế độ tự động điều chỉnh màu trắng thành Once
        # self.print_camera_settings()
        
    def open_camera(self):
        """Mở camera và tải user set nếu có."""
        try:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
            self.is_open = True
            print("Camera đã mở.")
            # self.setup_camera()
        except Exception as e:
            print(f"Lỗi khi mở camera: {e}")

    def close_camera(self):
        """Đóng camera Basler an toàn."""
        if self.is_open:
            # Dừng grabbing nếu đang chạy
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            self.camera.Close()
            self.is_open = False
            print("Camera đã đóng.")
        # Đóng mọi cửa sổ hiển thị OpenCV nếu có
        cv2.destroyAllWindows()

    def single_shot(self):
        """Chụp một hình ảnh."""
        image = None
        if not self.is_open:
            print("Camera chưa được mở.")
            return image
        try:
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            grab_result = self.camera.RetrieveResult(20000, pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():
                # Chuyển đổi hình ảnh sang định dạng OpenCV
                image = grab_result.Array
                grab_result.Release()
                self.camera.StopGrabbing()
                return image
            else:
                print("Lỗi khi chụp hình.")
        except Exception as e:
            print(f"Lỗi khi chụp hình: {e}")

    def get_camera_info(self):
        """Lấy thông tin camera."""
        if not self.is_open:
            print("Camera chưa được mở.")
            return None

        try:
            exposure_time = self.camera.ExposureTime.Value if self.camera.ExposureTime.IsReadable else None
            gain = self.camera.Gain.Value if self.camera.Gain.IsReadable else None
            frame_rate = self.camera.AcquisitionFrameRate.Value if self.camera.AcquisitionFrameRate.IsReadable else None
            return exposure_time, gain, frame_rate
        except Exception as e:
            print(f"Lỗi khi lấy thông tin camera: {e}")
            return None
            
    def print_camera_settings(self):
        """In ra các thông số của camera."""
        try:
            print(f"Pixel Format: {self.camera.PixelFormat.Value}")
            print(f"Exposure Time: {self.camera.ExposureTime.Value}")
            print(f"Balance White Auto: {self.camera.BalanceWhiteAuto.Value}")
            print(f"Gain: {self.camera.Gain.Value}")
            print(f"Gamma: {self.camera.Gamma.Value}")
            print(f"Frame Rate: {self.camera.AcquisitionFrameRate.Value}")
        except Exception as e:
            print(f"Lỗi khi in thông số camera: {e}")
          
    def start_continuous_grabbing(self):
        """Bắt đầu chế độ chụp liên tục."""
        if not self.is_open:
            print("Camera chưa được mở.")
            return

        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        while True:
            try:
                grab_result = self.camera.RetrieveResult(20000, pylon.TimeoutHandling_ThrowException)

                if grab_result is not None and grab_result.GrabSucceeded():
                    # Chuyển đổi hình ảnh sang định dạng OpenCV
                    image = grab_result.Array
                    image = self._draw_calib(image)
                    img_resize = cv2.resize(image, (1920, image.shape[0] * 1920 // image.shape[1]))
                    cv2.imshow("cam", img_resize)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        grab_result.Release()
                        break
                else:
                    print("Lỗi khi chụp hình hoặc không có dữ liệu.")
                grab_result.Release()
            except pylon.TimeoutException as e:
                print(f"Lỗi timeout khi lấy khung hình: {e}")
                break
            except Exception as e:
                print(f"Lỗi khi lấy khung hình: {e}")
                break

        self.camera.StopGrabbing()
        cv2.destroyAllWindows()
    
    def stop_continuous_grabbing(self):
        """Dừng chế độ chụp liên tục."""
        if self.camera is not None and self.camera.IsGrabbing():
            self.camera.StopGrabbing()
            print("Đã dừng chế độ chụp liên tục.")

    def _draw_calib(self, image):
        image_cop = image.copy()
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        gray = cv2.medianBlur(image, 3)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [c for c in contours if cv2.contourArea(c) < ((gray.shape[0]) * (gray.shape[1]))]
        if filtered_contours:
            rect = cv2.minAreaRect(max(filtered_contours, key=cv2.contourArea))
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.putText(image_cop, f"Width: {rect[1][1]:.2f}", (200,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), thickness = 4,)
            cv2.putText(image_cop, f"Height: {rect[1][0]:.2f}", (200,400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), thickness = 4)
            cv2.drawContours(image_cop,[box],0,(0,0,255),2)
        return image_cop
            
class CameraWebcam:
    def __init__(self, cam_index=0):
        super().__init__()
        self.cam_index = cam_index
        self.camera = None
        self.is_open = False

        # ===== Added for anti-stale-frame =====
        self._capture_thread = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_ts = 0.0
        self._warmup_frames = 8  # số frame bỏ sau khi mở

    def open_camera(self):
        if self.camera is None:
            print("Opening camera...")
            self.camera = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
            # Thiết lập cơ bản
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            # Có thể thử (không phải backend nào cũng hỗ trợ):
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.camera.isOpened():
                raise RuntimeError(f"Không thể mở camera index {self.cam_index}")

            # Flush vài frame “cũ” ngay sau open
            for _ in range(self._warmup_frames):
                self.camera.read()

            self.is_open = True
            print("Camera opened.")

            # Khởi động thread đọc liên tục
            self._start_capture_thread()

    def _start_capture_thread(self):
        if self._capture_thread and self._capture_thread.is_alive():
            return
        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def _capture_loop(self):
        """
        Luồng đọc liên tục giữ frame mới nhất (drop frame cũ).
        """
        while not self._stop_event.is_set():
            if not self.is_open or self.camera is None:
                time.sleep(0.01)
                continue
            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.005)
                continue
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_ts = time.time()
            # Giảm tải CPU: ngủ rất ngắn; điều chỉnh theo FPS mong muốn
            time.sleep(0.001)

    def get_image(self):
        """
        Lấy frame mới nhất (copy) – tránh stale frame.
        """
        if not self.is_open:
            print("Camera chưa được mở.")
            return None
        # Đợi nếu chưa có frame (ví dụ ngay sau khi mở)
        start_wait = time.time()
        while self._latest_frame is None and time.time() - start_wait < 1.0:
            time.sleep(0.01)
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def capture_image(self):
        """
        API tương thích cũ – dùng get_image().
        """
        return self.get_image()

    def capture_single_fresh(self, flush_reads=3):
        """
        Nếu KHÔNG muốn dùng thread: flush vài frame rồi lấy một.
        (Giữ làm lựa chọn khác)
        """
        if not self.is_open or self.camera is None:
            raise RuntimeError("Camera chưa mở")
        for _ in range(flush_reads):
            self.camera.grab()
        ret, frame = self.camera.read()
        return frame if ret else None

    def close_camera(self):
        # Dừng thread trước
        self._stop_event.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        if self.is_open and self.camera is not None:
            self.camera.release()
            self.is_open = False
            print("Camera released.")
        cv2.destroyAllWindows()

    def start_continuous_grabbing(self):
        """
        Hiển thị liên tục frame mới nhất (không lag).
        """
        if not self.is_open:
            raise RuntimeError("Camera chưa mở")
        while True:
            frame = self.get_image()
            if frame is None:
                continue
            image = self._draw_calib(frame)
            img_resize = cv2.resize(image, (1920, image.shape[0] * 1920 // image.shape[1]))
            cv2.imshow("Camera", img_resize)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def _draw_calib(self, image):
        image_cop = image.copy()
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = cv2.medianBlur(gray, 3)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [c for c in contours if cv2.contourArea(c) < ((gray.shape[0]) * (gray.shape[1]))]
        if filtered_contours:
            rect = cv2.minAreaRect(max(filtered_contours, key=cv2.contourArea))
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.putText(image_cop, f"Width: {rect[1][1]:.2f}", (200,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), thickness = 4,)
            cv2.putText(image_cop, f"Height: {rect[1][0]:.2f}", (200,400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), thickness = 4)
            cv2.drawContours(image_cop,[box],0,(0,0,255),2)
        return image_cop