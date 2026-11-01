# 🛸 Drone vs Bird Detection & Tracking System

An end-to-end, real-time object detection, movement tracking, and communication pipeline designed to classify and monitor aerial objects (Drones, Birds, Airplanes, and Helicopters) using **YOLOv8** and **MQTT**.

---

## 📌 Project Overview
This project unifies disjointed classification and detection datasets, trains a custom YOLOv8 model optimized for small-object detection, and provides a real-time tracking application that stream coordinates and motion vectors over MQTT.

```
                                          +-------------------------+
                                          |   real-time webcam/vid  |
                                          +------------+------------+
                                                       |
                                                       v
+------------------+     +----------------+     +------+------+     +-----------------------+
|  Drone Dataset   | --> |  Combined      | --> |   YOLOv8    | --> | MQTT Broker (QoS 1)   |
|  (YOLO format)   |     |  Dataset       |     |  Inference  |     | topic: webcam/detect  |
+------------------+     +-------+--------+     +------+------+     +-----------+-----------+
                                 ^                     |                        |
+------------------+             |                     v                        v
|  Bird Dataset    | ------------+              +------+------+     +-----------+-----------+
|  (Auto-labeled)  |                            | Local JSON  |     | Reordered receiver.py |
+------------------+                            | Logging     |     | (Heap-based buffer)   |
                                                +-------------+     +-----------------------+
```

---

## 🛠️ System Architecture & Codebase

The repository is structured as follows:

| Script | Purpose | Key Details |
| :--- | :--- | :--- |
| **[`prepare_data.py`](file:///d:/Web%20Dev/Yolo%20Project/prepare_data.py)** | Preprocessing & Unification | • Integrates `drone-dataset` (AirPlane, Drone, Helicopter) and `birds-dataset` (pure classification image files).<br>• Runs pretrained `yolov8n.pt` over raw bird images, extracts bird bounding boxes (COCO class 14), maps them to Class 3 (Bird), and builds a unified 80/10/10 split under `dataset_combined`. |
| **[`train.py`](file:///d:/Web%20Dev/Yolo%20Project/train.py)** | Model Training | • Sets training resolution to `imgsz=1024` for small drones.<br>• Runs early-stopping with patience=20.<br>• Saves checkpoints to `drone_bird_detection_experiment` and exports to ONNX format. |
| **[`detection.py`](file:///d:/Web%20Dev/Yolo%20Project/detection.py)** | Real-time Tracking & Publisher | • Performs frame-by-frame inference on webcam.<br>• Tracks object centroids and estimates direction (e.g. `up-left`, `down`, `stationary`).<br>• Publishes structured JSON payloads to MQTT topic `webcam/detections` and logs events in `detections.json`. |
| **[`recevier.py`](file:///d:/Web%20Dev/Yolo%20Project/recevier.py)** | Message Consumer | • Subscribes to MQTT topic `webcam/detection`.<br>• Implements a min-heap reorder buffer to correct network jitters and out-of-order sequence arrivals.<br>• Detects missing sequences/packet drops and prints smoothed movement logs. |
| **[`video.py`](file:///d:/Web%20Dev/Yolo%20Project/video.py)** | Video Demo | • Demonstrates model inference on sample video streams (`WhatsApp Video 2025-10-12 at 16.59.45.mp4`). |
| **[`demo.py`](file:///d:/Web%20Dev/Yolo%20Project/demo.py)** | Basic Webcam Demo | • A lightweight test script to quickly verify webcam functionality. |

---

## 📊 Training Experiments & Results

The custom model was trained on the combined dataset using the configuration below:

- **Image Size (`imgsz`)**: `1024` (Optimized for micro-UAV and bird detections at distances)
- **Batch Size**: Auto-adaptive (`batch=-1`)
- **Patience**: `20` epochs (Early stopping)

All output weights, performance curves, and visual verification frames are saved under **[`drone_bird_detection_experiment`](file:///d:/Web%20Dev/Yolo%20Project/drone_bird_detection_experiment)**:

- **`weights/best.pt`**: PyTorch weights for deployment.
- **`weights/best.onnx`**: Exported ONNX format for web/embedded runtime optimizations.
- **`confusion_matrix.png`**: Multi-class confusion matrix.
- **`BoxPR_curve.png` & `BoxF1_curve.png`**: Precision-Recall curves indicating the class thresholds.
- **`val_batch0_pred.jpg`**: Sample ground-truth vs. prediction overlay comparisons.

---

## 📡 MQTT Message Format
Real-time tracking notifications are published with the following schema:

```json
{
  "timestamp": 1760123456.789,
  "detections": [
    {
      "object": "Drone",
      "confidence": 0.92,
      "center": [640, 360],
      "direction": "up-right"
    },
    {
      "object": "Bird",
      "confidence": 0.81,
      "center": [200, 150],
      "direction": "left"
    }
  ]
}
```

---

## 🚀 How to Run the Project

### 1. Installation
Ensure Python 3.9+ is installed, then install required packages:
```bash
pip install ultralytics opencv-python paho-mqtt tqdm numpy
```

### 2. Preprocess & Merge Datasets
To prepare the unified YOLO dataset from the raw images:
```bash
python prepare_data.py
```
This merges the datasets, performs auto-labeling on the bird classification pictures, and writes out `dataset_combined/dataset.yaml`.

### 3. Model Training
To start training the custom YOLOv8 model:
```bash
python train.py
```

### 4. Real-time Object Detection, Tracking & Communication
Make sure your local MQTT broker (like Mosquitto) is running on port 1883.
1. Run the consumer to print out stabilized, re-ordered detection streams:
   ```bash
   python recevier.py --topic webcam/detections
   ```
2. Start the tracking and publishing pipeline:
   ```bash
   python detection.py
   ```

### 5. Running Video Inference
To test the detector on the pre-loaded sample video:
```bash
python video.py
```

---

## 🏆 Model Performance Summary
Based on early validation runs:
* **Classes Trained**: `['AirPlane', 'Drone', 'Helicopter', 'Bird']`
* **mAP@0.5**: ~`0.781`
* **Precision**: ~`0.721`
* **Recall**: ~`0.745`
