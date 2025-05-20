# detect/yolo_detect.py
from ultralytics import YOLO
import cv2

# 1) Load your YOLO model (change path to your custom weights if any)
model = YOLO("yolov5su.pt")  # or "yolov8n.pt", or "./runs/train/exp/weights/best.pt"

def detect_objects(frame):
    """
    Runs YOLO inference on a single frame.
    Returns a list of dicts: [{'centre':(u,v), 'bbox':[x1,y1,x2,y2], 'label':str, 'conf':float}, ...].
    """
    results = model(frame)[0]             # only first batch
    detections = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = box.tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        detections.append({
            "centre": (cx, cy),
            "bbox":   [x1, y1, x2, y2],
            "label":  model.names[int(cls)],
            "conf":   float(conf)
        })
    return detections

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # test single camera
    while True:
        ret, frame = cap.read()
        if not ret: break
        dets = detect_objects(frame)
        for d in dets:
            x1,y1,x2,y2 = map(int, d["bbox"])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, f"{d['label']}:{d['conf']:.2f}",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow("YOLO Demo", frame)
        if cv2.waitKey(1)==27: break
    cap.release()
    cv2.destroyAllWindows()
