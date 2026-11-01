from ultralytics import YOLO
import os

def train():
    # Load model from previous run
    model_path = r"D:\inav3\drone-detection-system-main\drone-detection-system-main\drone_bird_detection_experiment\weights\best.pt"
    if os.path.exists(model_path):
        print(f"Loading trained weights from {model_path}...")
        model = YOLO(model_path)
    else:
        print("Pretrained weights not found, starting fresh...")
        model = YOLO('yolov8n.pt')

    # Train the model
    print("Starting training for 1 more epoch...")
    results = model.train(
        data=r"D:\inav3\drone-detection-system-main\drone-detection-system-main\dataset_combined\dataset.yaml",
        epochs=2,
        imgsz=480,
        patience=5,
        batch=-8,
        project=r"D:\inav3\drone-detection-system-main\drone-detection-system-main",
        name='drone_bird_detection_experiment13',
        pretrained=True,
        device="cpu",
        workers=4,
        optimizer="SGD",
        lr0=0.01,
        cache="ram",
        amp=False,
        exist_ok=True,
        plots=True,
        save=True,
    )
    
    # Export the model
    print("Exporting model to ONNX...")
    model.export(format='onnx')

    print("Training Complete. Results saved to drone_bird_detection_experiment")

if __name__ == '__main__':
    train()