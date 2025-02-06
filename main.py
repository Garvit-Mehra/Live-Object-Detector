import cv2
import os
from detection import detect_objects


def main():
    choice = input("Choose '1' for webcam, '2' for image, or '3' for video: ")

    if choice == '1':
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result_frame = detect_objects(frame)
            cv2.imshow("YOLO Detection", result_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif choice == '2':
        image_path = input("Enter image file path: ")
        expanded_path = os.path.expanduser(image_path)
        image = cv2.imread(expanded_path)

        if image is None:
            print("Error: Image not found.")
        else:
            result_image = detect_objects(image)
            cv2.imwrite("result.jpg", result_image)
            print("Detection completed. Result saved as result.jpg.")

    elif choice == '3':
        video_path = input("Enter video file path: ")
        expanded_path = os.path.expanduser(video_path)
        cap = cv2.VideoCapture(expanded_path)

        if not cap.isOpened():
            print("Error: Video not found.")
        else:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result_frame = detect_objects(frame)
                out.write(result_frame)
                cv2.imshow("YOLO Detection", result_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print("Detection completed. Result saved as output_video.mp4.")
    else:
        print("Invalid choice. Please select '1', '2', or '3'.")


if __name__ == "__main__":
    main()
