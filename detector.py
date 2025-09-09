import os
import cv2
from detection import YoloV7Detector

detector = YoloV7Detector(weights_path="YOLOv7 w6.pt", device="cpu", classes=None)

choice = input(
    "Choose '1' to run detection on webcam, '2' to upload an image, or '3' to upload a video: "
)

if choice == "1":
    detector.run_live(cam_index=0, view_img=True, save_txt=False)

elif choice == "2":
    path = input("Enter the path to your image: ").strip()
    expath = os.path.expanduser(path)

    image = cv2.imread(expath)

    if image is None:
        print("Error: Image is not found")
    else:
        detector.run_image(expath)
        print("Image Processed")

elif choice == "3":
    path = input("Enter the path to your video: ").strip()
    expath = os.path.expanduser(path)

    image = cv2.imread(expath)

    if image is None:
        print("Error: Image is not found")
    else:
        detector.run_video(expath)
        print("Video Processed")

else:
    print("Invalid choice. Please select '1', '2', or '3'.")
