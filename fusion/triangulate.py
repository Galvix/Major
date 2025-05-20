import cv2
import numpy as np
import pymvg.camera_model as _cm
from pymvg.camera_model import CameraModel
from pymvg.multi_camera_system import MultiCameraSystem

# Monkey-patch CameraModel.undistort to allow copies (avoids NumPy 2.0 copy=False issue)
_orig_undistort = _cm.CameraModel.undistort

def _patched_undistort(self, nparr):
    # Ensure input is an ndarray, allowing copies
    return _orig_undistort(self, np.asarray(nparr))

_cm.CameraModel.undistort = _patched_undistort

def load_cameras(param_files, image_size):
    """
    Load camera parameters from .npz files and return a MultiCameraSystem.
    param_files: list of file paths to .npz files containing K, dist, rvec, tvec
    image_size: tuple (width, height)
    """
    cams = []
    for idx, p in enumerate(param_files):
        data = np.load(p)
        K = data['K']
        dist_coeffs = data['dist'].flatten()[:5]
        rvec = data['rvec']
        tvec = data['tvec']

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        # Build 3×4 projection matrix
        M = K @ np.hstack((R, tvec))

        # Instantiate camera with distortion coefficients
        cam = CameraModel.load_camera_from_M(
            M,
            width=image_size[0],
            height=image_size[1],
            name=f"cam{idx}",
            distortion_coefficients=dist_coeffs
        )
        cams.append(cam)

    # Pass list of CameraModel instances directly
    return MultiCameraSystem(cams)


def triangulate(camsys, detections_per_cam):
    """
    Perform multi-view triangulation of detected 2D centers.

    camsys:     MultiCameraSystem instance
    detections_per_cam: list of lists of 2D centers per camera
        e.g. [[(u1,v1),(u2,v2),...], [(u1',v1'),...], ...]

    Returns:
        np.ndarray of shape (N, 3) containing 3D points.
    """
    pts3d = []
    # No valid detections
    if not detections_per_cam or not detections_per_cam[0]:
        return np.zeros((0, 3))

    num_objs = len(detections_per_cam[0])
    for i in range(num_objs):
        obs = []
        for cam_idx, centers in enumerate(detections_per_cam):
            if len(centers) > i:
                cam_name = f"cam{cam_idx}"
                cam = camsys._cameras.get(cam_name)
                if cam:
                    u, v = centers[i]
                    obs.append((cam_name, [float(u), float(v)]))
        # Triangulate when seen by ≥2 cameras
        if len(obs) >= 2:
            X = camsys.find3d(obs)
            pts3d.append(X)

    return np.array(pts3d)
