import cv2
import numpy as np
import os

# Load YOLO model
config_path = "model_data/yolov3.cfg"
weights_path = "model_data/yolov3.weights"
classes_path = "model_data/coco.names"

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


# Ask user whether to upload a video
choice = input("Choose '1' to run detection on webcam, '2' to upload an image, or '3' to upload a video: ")

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

elif choice == '3':
    # Get the video file path
    video_path = input("Enter the video file path: ")
    expanded_path = os.path.expanduser(video_path)

    # Open video file
    cap = cv2.VideoCapture(expanded_path)

    if not cap.isOpened():
        print("Error: Video not found or unable to open.")
    else:
        # Get video information (width, height, frames per second)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object
        out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection
            result_frame = detect_objects(frame)

            # Write the frame to the output video
            out.write(result_frame)

            # Display the frame
            cv2.imshow("YOLO Detection", result_frame)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release everything when done
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Detection completed. The result has been saved as output_video.mp4.")

else:
    print("Invalid choice. Please select '1', '2', or '3'.")
