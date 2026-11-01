import cv2
from ultralytics import YOLO

def main():
    # Load the YOLO model
    model = YOLO("yolov8n.pt") # You can choose other models like yolov8s.pt, yolov8m.pt, etc.

    # Initialize webcam capture
    cap = cv2.VideoCapture(0) # 0 for default webcam, change if you have multiple cameras
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set webcam resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Perform object detection
        results = model(frame, stream=True)  # stream=True for generator output

        # Process results and draw bounding boxes
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Get confidence
                confidence = float(box.conf[0])
                # Get class
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Webcam Object Detection", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()