import cv2
import numpy as np
import os

def run_calibration(cam_idx, chessboard_size=(9, 14), square_size=0.025, save_path=None):
    """
    Captures 20 successful chessboard detections from camera `cam_idx`,
    performs camera calibration, and returns (K, dist, rvec, tvec).
    Optionally saves params to `save_path` (.npz).
    """
    # Prepare object points based on real chessboard dimensions
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {cam_idx}")
        return None

    count = 0
    print(f"[*] Starting calibration for camera {cam_idx}. Press ESC to abort.")
    while count < 20:
        ret, frame = cap.read()
        if not ret:
            print(f"WARNING: Frame grab failed for cam{cam_idx}")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if found:
            # Refine corner locations
            term = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
            objpoints.append(objp)
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(frame, chessboard_size, corners2, found)
            count += 1
            print(f"Captured {count}/20 chessboard for cam{cam_idx}")

        cv2.imshow(f"Calibrating Camera {cam_idx}", frame)
        if cv2.waitKey(500) & 0xFF == 27:
            print("Calibration aborted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 5:
        print("ERROR: Not enough valid detections for reliable calibration.")
        return None

    # Calibrate
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    if not ret:
        print("ERROR: Calibration computation failed.")
        return None

    # Use first view's extrinsics as reference
    rvec, tvec = rvecs[0], tvecs[0]

    # Save if requested
    if save_path:
        np.savez(save_path, K=K, dist=dist, rvec=rvec, tvec=tvec)

    return K, dist, rvec, tvec


if __name__ == "__main__":
    # Directory for saving params
    base = os.path.dirname(__file__)
    param_dir = os.path.join(base, "params")
    os.makedirs(param_dir, exist_ok=True)

    for idx in range(4):
        out_file = os.path.join(param_dir, f"cam{idx}.npz")
        result = run_calibration(idx, save_path=out_file)
        if result is None:
            print(f"Camera {idx} calibration failed.")
        else:
            print(f"Saved calibration to {out_file}")
