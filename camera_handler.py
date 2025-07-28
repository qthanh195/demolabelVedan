from pypylon import pylon
import numpy as np
        
class BaslerCamera():
    def __init__(self):
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
        """Đóng camera Basler."""
        if self.is_open:
            # if self.is_continuous:
            #     self.stop_continuous_grabbing()
            self.camera.Close()
            self.is_open = False
            print("Camera đã đóng.")

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
            