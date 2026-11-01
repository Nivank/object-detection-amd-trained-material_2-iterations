import cv2
import json
import time
import paho.mqtt.client as mqtt
from ultralytics import YOLO
import numpy as np

# -----------------------
# MQTT Setup
# -----------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "webcam/detections"

client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# -----------------------
# YOLO Model
# -----------------------
model = YOLO("yolov8n.pt")  # CPU-friendly

# -----------------------
# Webcam Setup
# -----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# JSON file
json_file_path = "detections.json"

# Previous frame object centers for movement tracking
prev_centers = []

# Simple nearest-neighbor movement tracking
def get_direction(prev, curr, threshold=5):
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    if abs(dx) < threshold and abs(dy) < threshold:
        return "stationary"
    dir_x = ""
    dir_y = ""
    if dy < -threshold:
        dir_y = "up"
    elif dy > threshold:
        dir_y = "down"
    if dx > threshold:
        dir_x = "right"
    elif dx < -threshold:
        dir_x = "left"
    if dir_x and dir_y:
        return f"{dir_y}-{dir_x}"
    return dir_x or dir_y

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False)
    detected_objects = []
    current_centers = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            current_centers.append((cx, cy))

            # Determine movement direction using nearest previous center
            if prev_centers:
                distances = [np.linalg.norm(np.array((cx, cy)) - np.array(pc)) for pc in prev_centers]
                nearest_idx = int(np.argmin(distances))
                direction = get_direction(prev_centers[nearest_idx], (cx, cy))
            else:
                direction = "unknown"

            detected_objects.append({
                "object": model.names[int(cls_id)],
                "confidence": float(score),
                "center": [cx, cy],
                "direction": direction
            })

            label = f"{model.names[int(cls_id)]}:{score:.2f} {direction}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    prev_centers = current_centers.copy()

    # Prepare JSON message
    if detected_objects:
        message = {
            "timestamp": time.time(),
            "detections": detected_objects
        }

        client.publish(MQTT_TOPIC, json.dumps(message))
        with open(json_file_path, "a") as f:
            f.write(json.dumps(message) + "\n")

    cv2.imshow("YOLO Detection with Movement", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Detections saved in {json_file_path}")
