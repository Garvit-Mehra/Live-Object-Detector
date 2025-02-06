import cv2
import numpy as np
from config import CONFIG_PATH, WEIGHTS_PATH, CLASSES_PATH, CONFIDENCE_THRESHOLD, NMS_THRESHOLD
from utils import generate_color, get_text_color  # Import the updated generate_color function

# Load YOLO model
with open(CLASSES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)


def detect_objects(frame):
    """
    Detects objects in a frame using the YOLO model and draws bounding boxes.
    """
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            class_name = classes[class_ids[i]]
            label = f"{class_name}"

            # Use the updated generate_color function
            color = generate_color(class_ids[i])  # Get the color for this class ID

            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)

            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_width, label_height = label_size

            cv2.rectangle(frame, (x, y - label_height - 10), (x + label_width, y), color, -1)
            text_color = get_text_color(color)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    return frame
