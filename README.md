# YOLO Object Detection with Video/Camera Input

This project uses the YOLO (You Only Look Once) object detection model to detect objects in real-time from a webcam, image, or video file. The object detection is done using OpenCV and YOLOv3, and it displays the bounding boxes with class labels, each with a unique color.

## Features:
- Detects objects in images, videos, or webcam feed.
- Uses YOLOv3 for real-time object detection.
- Bounding boxes are drawn around detected objects with unique colors for each class.
- Each class label is displayed with a background color (confidence value hidden).
- Supports both image and video input.

## Requirements:
- Python 3.x
- OpenCV
- Numpy

You can install the required dependencies using `pip`:

```pip install opencv-python numpy```

You also need the following files for YOLOv3:
	• yolov3.cfg (YOLOv3 configuration file)
	• yolov3.weights (YOLOv3 pre-trained weights)
	• coco.names (Text file containing class names)
Make sure to download the necessary files from the model_data folder.

## Setup:
1. Download the YOLOv3 model configuration, weights, and class names:
	- YOLOv3 CFG
	- YOLOv3 WEIGHTS
	- COCO.NAMES
2. Place these files in the model_data/ directory in the same folder as this script.
3. Make sure to have the correct file paths to the model files in the script.
 
## How to Use:
1. Run Detection on Webcam Feed:
When you run the script, you’ll be prompted to choose an option. To run the detection on your webcam, select option 1.
python detect.py
 	- Press q to stop the webcam feed.
2. Run Detection on an Image:
Select option 2 when prompted. Enter the file path of the image when asked. The result will be saved as result.jpg.
python detect.py
   	- Enter the path to the image file when prompted.
	- The result will be saved as result.jpg in the working directory.
3. Run Detection on a Video:
Select option 3 when prompted. Enter the file path of the video when asked. The result will be saved as output_video.mp4.
python detect.py
	- Enter the path to the video file when prompted.
	- The result will be saved as output_video.mp4 in the working directory.
4. Input and Output:
	- Input: You can either choose to use your webcam, upload an image, or upload a video.
	- Output: The detected objects will be marked with bounding boxes and labels. The processed video or image will be saved with bounding boxes and labels applied.
  
## How it Works:
- The script uses YOLOv3, which is a deep learning-based object detection algorithm, to detect objects in the input image or video.
- The script applies the following steps:
	1. Load the YOLOv3 model and its configuration files.
	2. Preprocess the input data (image/video frame).
	3. Perform forward pass through the YOLOv3 model to get detections.
	4. Post-process the detections, including non-maxima suppression to remove overlapping boxes.
	5. Draw bounding boxes around detected objects with unique colors for each class.
	6. Display the processed frame/image or save the result to an output file.
 
## Color Coding:
- Each detected class (e.g., “person”, “car”, “dog”) will have a unique color assigned to its bounding box and label. The color is deterministic for each class, meaning the same class will always have the same color.
Label Background:
- The class label will appear inside a background rectangle, making it easier to read.
  
## Troubleshooting:
- If the video or image file is not found, make sure the file path is correct.
- Ensure you have the necessary YOLO model files (yolov3.cfg, yolov3.weights, coco.names) in the correct location.

