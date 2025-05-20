import cv2
import glob
import numpy as np
from fusion.triangulate import load_cameras, triangulate
from detect.yolo_detect import detect_objects

# Configuration
IMAGE_SIZE = (320, 240)
PARAM_GLOB = "calib/params/cam*.npz"
NUM_CAMERAS = 4


def main():
    # 1) Initialize calibrated multi-camera system
    camsys = load_cameras(glob.glob(PARAM_GLOB), image_size=IMAGE_SIZE)

    # 2) Open all camera streams
    caps = [cv2.VideoCapture(i) for i in range(NUM_CAMERAS)]
    for idx, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera {idx}")

    try:
        while True:
            frames = []
            detections_per_cam = []

            # 3) Capture frames & run YOLO detection
            for ci, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Use blank frame if grab fails
                    frame = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
                    dets = []
                else:
                    frame = cv2.resize(frame, IMAGE_SIZE)
                    dets = detect_objects(frame)

                frames.append(frame)
                detections_per_cam.append(dets)

            # 4) Extract 2D centers and triangulate
            centers_list = [[d['centre'] for d in dets] for dets in detections_per_cam]
            pts3d = triangulate(camsys, centers_list)
            # (Optional: Send pts3d to 3D viewer here)

            # 5) Draw bounding boxes with labels
            vis_frames = []
            for frame, dets in zip(frames, detections_per_cam):
                for d in dets:
                    x1, y1, x2, y2 = map(int, d['bbox'])
                    label = f"{d['label']}:{d['conf']:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                vis_frames.append(frame)

            # 6) Create 2x2 mosaic
            top = cv2.hconcat(vis_frames[:2]) if len(vis_frames) >= 2 else vis_frames[0]
            bottom = cv2.hconcat(vis_frames[2:4]) if len(vis_frames) >= 4 else None
            if bottom is not None:
                mosaic = cv2.vconcat([top, bottom])
            else:
                mosaic = top

            cv2.imshow("Multi-View YOLO Detections", mosaic)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
