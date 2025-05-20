import cv2
import glob
import numpy as np
from fusion.triangulate import load_cameras, triangulate
from detect.yolo_detect import detect_objects

# Configuration
IMAGE_SIZE = (640, 480)
PARAM_GLOB = "calib/params/cam*.npz"
CAM_IDS = [0, 1, 2, 3]  # List your actual camera indices


def main():
    # 1) Initialize calibrated multi-camera system
    camsys = load_cameras(glob.glob(PARAM_GLOB), image_size=IMAGE_SIZE)

    # 2) Open all camera streams using default backend
    caps = []
    for cam_id in CAM_IDS:
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera {cam_id}")
        else:
            # Set uniform resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])
        caps.append(cap)

    try:
        while True:
            frames = []
            detections_per_cam = []

            # 3) Capture & detect for each camera
            for ci, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"Warning: Camera {CAM_IDS[ci]} frame grab failed; using blank image")
                    frame = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
                    dets = []
                else:
                    # Run YOLO inference
                    dets = detect_objects(frame)

                # Draw bounding boxes and labels
                for d in dets:
                    x1, y1, x2, y2 = map(int, d['bbox'])
                    label = f"{d['label']}:{d['conf']:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Collect frames and detection centers
                frames.append(frame)
                centers = [d['centre'] for d in dets]
                detections_per_cam.append(centers)

            # 4) (Optional) Triangulate 3D points
            # pts3d = triangulate(camsys, detections_per_cam)
            # print("3D points:", pts3d)

            # 5) Create 2×2 mosaic of frames
            if len(frames) >= 4:
                top = cv2.hconcat(frames[0:2])
                bottom = cv2.hconcat(frames[2:4])
                mosaic = cv2.vconcat([top, bottom])
            else:
                mosaic = cv2.hconcat(frames)

            cv2.imshow("Multi-View YOLO Detections", mosaic)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
