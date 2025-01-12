import cv2
import numpy as np
import os

# Load YOLO model
config_path = "model_data/yolov3.cfg"  # Path to your .cfg file
weights_path = "model_data/yolov3.weights"  # Path to your .weights file
classes_path = "model_data/coco.names"  # Path to your .names file

# Load class names
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO network
net = cv2.dnn.readNet(weights_path, config_path)

# Set backend (use GPU if available)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)


# Function to perform detection
def detect_objects(frame):
    height, width, _ = frame.shape

    # Preprocess frame
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    # Process detections
    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Bounding box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to reduce overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            # Increased font size and positioning
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


# Ask user whether to use webcam or upload an image
choice = input("Choose '1' to run detection on webcam or '2' to upload an image: ")

if choice == '1':
    # Open webcam feed
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        result_frame = detect_objects(frame)

        # Show the frame
        cv2.imshow("YOLO Detection", result_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

elif choice == '2':
    # Get the image file path
    image_path = input("Enter the image file path: ")
    expanded_path = os.path.expanduser(image_path)

    image = cv2.imread(expanded_path)

    if image is None:
        print("Error: Image not found.")
    else:
        # Perform detection
        result_image = detect_objects(image)

        # Resize the result to a larger size (e.g., 2x larger)
        result_image_resized = cv2.resize(result_image, (result_image.shape[1] * 5, result_image.shape[0] * 5))

        # Save the result
        cv2.imwrite("result.jpg", result_image_resized)
        print("Detection completed. The result has been saved as result.jpg.")

else:
    print("Invalid choice. Please select '1' or '2'.")
