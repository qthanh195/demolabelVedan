from pypylon import pylon
import numpy as np
import cv2
  
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
    def __init__(self, cam_index=2):
        super().__init__()
        self.cam_index = cam_index
        self.camera = None
        self.is_open = False

    def open_camera(self):
        if self.camera is None:
            print("Opening camera...")
            self.camera = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.is_open = True

            if not self.camera.isOpened():
                raise RuntimeError(f"Không thể mở camera index {self.cam_index}")
            print("Camera opened.")
            
    def get_image(self):
        return self.capture_image()

    def capture_image(self):
        if self.camera is None:
            raise RuntimeError("Camera chưa mở, gọi open_camera() trước.")
        image = None
        if not self.is_open:
            print("Camera chưa được mở.")
            return image

        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Không đọc được frame từ camera.")
        # name_image = f"captured_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        # cv2.imwrite(f"{save_folder}/{name_image}", frame)
        return frame

    def close_camera(self):
        if self.is_open: 
            self.camera.release()
            self.is_open = False
            print("Camera released.")
            
    def start_continuous_grabbing(self):
        if self.camera is None:
            raise RuntimeError("Camera chưa mở, gọi open_camera() trước.")
        image = None
        if not self.is_open:
            print("Camera chưa được mở.")
            return image
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Không đọc được frame từ camera.")
                break
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