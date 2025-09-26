import cv2
import numpy as np
import time
import os

# File paths (adjusted for repository structure)
weights_path = "yolov4/yolov4.weights"
cfg_path = "data/yolov4.cfg"
names_path = "data/coco.names"

# Check if files exist
for file_path in [weights_path, cfg_path, names_path]:
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        exit()

# Load YOLOv4 model
try:
    net = cv2.dnn.readNet(weights_path, cfg_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use DNN_TARGET_CUDA for GPU
    print("YOLOv4 model loaded successfully!")
except cv2.error as e:
    print(f"Error loading model: {e}")
    exit()

# Get output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
try:
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(classes)} classes")
except Exception as e:
    print(f"Error loading class names: {e}")
    exit()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_skip = 2  # Process every 2nd frame for speed
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        cv2.imshow("YOLOv4 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # YOLOv4 uses 416x416
    net.setInput(blob)

    start_time = time.time()
    outs = net.forward(output_layers)
    end_time = time.time()

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Higher threshold for YOLOv4 due to better accuracy
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    print(f"Detections found: {len(boxes)}")
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            object_name = classes[class_ids[i]]
            accuracy = confidences[i] * 100
            label = f"{object_name}: {accuracy:.1f}%"
            x, y = max(0, x), max(0, y)
            w, h = min(w, width - x), min(h, height - y)
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 0.6, color, 2)
            print(f"Detected: {label} at ({x}, {y}, {w}, {h})")
    else:
        print("No objects detected after NMS.")

    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.imshow("YOLOv4 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()