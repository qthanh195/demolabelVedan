import cv2
import numpy as np
import glob



def calibrate_camera(image_folder, chessboard_size=(9,6), square_size=1.0):
    """
    Calibrate a 2D camera (Basler, webcam, etc.) using chessboard images.
    
    :param image_folder: Thư mục chứa ảnh bàn cờ (vd: "./images/*.jpg")
    :param chessboard_size: Số góc trong (inner corners) (columns, rows) -> (9,6) nghĩa là 9 cột, 6 hàng
    :param square_size: Kích thước thật mỗi ô vuông (mm/cm). Nếu chỉ undistort thì để =1
    :return: camera_matrix, dist_coeffs
    """

    # termination criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Chuẩn bị object points, ví dụ (0,0,0), (1,0,0), ..., (8,5,0)
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # nhân kích thước thật

    objpoints = []  # 3D points ngoài đời thực
    imgpoints = []  # 2D points trong ảnh

    images = glob.glob(image_folder)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Tìm góc bàn cờ
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)

            # Tinh chỉnh góc chính xác hơn
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Vẽ kết quả
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imwrite('Chessboard.jpg', img)
            # cv2.waitKey(200)

    # cv2.destroyAllWindows()

    # Calibration
    if len(objpoints) == 0:
        raise ValueError("Không tìm thấy bàn cờ nào trong ảnh, kiểm tra lại dữ liệu!")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("Kết quả calibration:")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)

    return camera_matrix, dist_coeffs
