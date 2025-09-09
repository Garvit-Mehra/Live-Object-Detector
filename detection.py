import os
import sys
import time
import logging
from typing import List, Optional, Set, Union
from pathlib import Path

import cv2
import torch
import numpy as np

class YoloV7Detector:
    def __init__(
        self,
        weights_path: str = "/Users/garvitmehra/PycharmProjects/pythonProject/Object_detector/YOLOv7 w6.pt",
        device: str = "cpu",  # 'cuda', 'mps', 'cpu'
        img_size: int = 1280,  # Default for w6 weights
        conf_thres: float = 0.1,  # Lowered for better detection
        iou_thres: float = 0.3,  # Lowered for better detection
        classes: Optional[Union[List[int], List[str]]] = None,  # No default filtering
        project: str = "runs/detect",
        name: str = "exp",
        exist_ok: bool = False,
    ) -> None:
        # Configuration
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.debug = os.getenv("YOLO_DEBUG", "").strip().lower() in ("1", "true", "yes")

        # Logger setup
        log_level = os.getenv("YOLO_LOG_LEVEL", "DEBUG" if self.debug else "INFO").upper()
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov7_detection.log")
        self.logger = logging.getLogger("yolov7_detector")
        if not self.logger.handlers:
            self.logger.setLevel(getattr(logging, log_level, logging.INFO))
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            try:
                fh = logging.FileHandler(log_file)
                fh.setFormatter(fmt)
                self.logger.addHandler(fh)
            except Exception:
                self.logger.warning("Failed to set up file logging")
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self.logger.addHandler(sh)

        # Setup YOLOv7 runtime path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yolov7_dir = os.path.join(script_dir, "yolov7_runtime")
        if yolov7_dir not in sys.path:
            sys.path.append(yolov7_dir)

        # Imports
        try:
            from models.experimental import attempt_load
            from utils.datasets import letterbox
            from utils.general import check_img_size, non_max_suppression, scale_coords
            from utils.torch_utils import select_device
        except Exception as e:
            self.logger.error(f"Failed to import YOLOv7 modules from {yolov7_dir}: {e}")
            raise RuntimeError(f"Failed to import YOLOv7 modules: {e}")

        self.letterbox = letterbox
        self.nms = non_max_suppression
        self.scale_coords = scale_coords
        self.select_device = select_device

        # Device selection
        self.device = select_device(device)
        self.half = self.device.type == "cuda" and torch.cuda.is_available()
        self.logger.info(f"Using device: {self.device}, half={self.half}")

        # Weights
        self.weights = weights_path if os.path.exists(weights_path) else os.path.join(script_dir, weights_path)
        if not os.path.exists(self.weights):
            self.logger.error(f"Weights not found at {self.weights}")
            raise FileNotFoundError(f"YOLOv7 w6.pt not found at {self.weights}")

        # Load model
        self.logger.info(f"Loading model from {self.weights}")
        t0 = time.monotonic()
        try:
            # Try loading without weights_only for compatibility
            self.model = attempt_load(self.weights, map_location=self.device)
            self.names = self.model.module.names if hasattr(self.model, "module") else self.model.names
        except Exception as e:
            self.logger.warning(f"attempt_load failed: {e}, trying manual rebuild")
            try:
                from models.yolo import Model
                yaml_path = os.path.join(yolov7_dir, "cfg", "training", "yolov7-w6.yaml")
                if not os.path.exists(yaml_path):
                    self.logger.error(f"YOLOv7-W6 YAML not found at {yaml_path}. Please download it from https://github.com/WongKinYiu/yolov7")
                    raise FileNotFoundError(f"YOLOv7-W6 YAML not found at {yaml_path}")
                ckpt = torch.load(self.weights, map_location=self.device)
                self.model = Model(yaml_path).to(self.device)
                model_state = ckpt.get("ema", ckpt.get("model", ckpt))
                if hasattr(model_state, "state_dict"):
                    model_state = model_state.state_dict()
                self.model.load_state_dict(model_state, strict=False)  # Allow partial loading
                self.names = ckpt.get("names", [])
                if not self.names:
                    self.logger.warning("No class names in checkpoint, using default COCO names")
                    self.names = [f"class{i}" for i in range(80)]  # Fallback for COCO
            except Exception as e2:
                self.logger.error(f"Model rebuild failed: {e2}")
                raise RuntimeError(f"Failed to load model: {e2}")
        self.logger.info(f"Model load time: {(time.monotonic() - t0) * 1000:.1f} ms")

        if self.half:
            self.model.half()

        # Model properties
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)
        self.logger.info(f"Stride={self.stride}, img_size={self.img_size}, classes={len(self.names)}")

        # Random colors for visualization
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Class filtering
        self.allowed_indices: Set[int] = set()
        if classes:
            name_to_idx = {str(n).lower(): i for i, n in enumerate(self.names)}
            for c in classes:
                if isinstance(c, int):
                    self.allowed_indices.add(c)
                else:
                    c_str = str(c).strip().lower()
                    if c_str.isdigit():
                        self.allowed_indices.add(int(c_str))
                    elif c_str in name_to_idx:
                        self.allowed_indices.add(name_to_idx[c_str])
            self.logger.info(f"Filtering classes: {sorted(self.allowed_indices)} ({[self.names[i] for i in sorted(self.allowed_indices) if i < len(self.names)]})")
        else:
            self.logger.info("No class filtering applied")

        # Directories
        self.save_dir = Path(self._increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving results to {self.save_dir}")

        # Warmup
        if self.device.type != "cpu":
            t0 = time.monotonic()
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))
            self.logger.info(f"Warmup done in {(time.monotonic() - t0) * 1000:.1f} ms")

    def _increment_path(self, path: Path, exist_ok: bool = False) -> Path:
        path = Path(path)
        if path.exists() and not exist_ok:
            for i in range(100):
                new_path = Path(f"{path}{i if i else ''}")
                if not new_path.exists():
                    return new_path
        return path

    def detect_frame(self, frame: np.ndarray, save_txt: bool = False, txt_path: Optional[str] = None) -> np.ndarray:
        if frame is None or frame.size == 0:
            self.logger.error("Invalid input frame")
            return frame
        im0 = frame.copy()
        self.logger.debug(f"Input frame shape: {im0.shape}")

        # Preprocess
        img, _, _ = self.letterbox(im0, new_shape=self.img_size, stride=self.stride)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img_t = torch.from_numpy(img).to(self.device)
        img_t = img_t.half() if self.half else img_t.float()
        img_t /= 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)
        self.logger.debug(f"Input tensor: {img_t.shape}, {img_t.dtype}")

        # Inference
        t0 = time.monotonic()
        with torch.no_grad():
            pred = self.model(img_t, augment=False)[0]
        infer_time = (time.monotonic() - t0) * 1000
        self.logger.debug(f"Inference: {infer_time:.1f} ms")

        # NMS
        t1 = time.monotonic()
        pred = self.nms(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres, classes=list(self.allowed_indices) if self.allowed_indices else None)
        nms_time = (time.monotonic() - t1) * 1000
        self.logger.debug(f"NMS: {nms_time:.1f} ms")

        # Process detections
        total_boxes = 0
        detection_str = ""
        if len(pred) and pred[0] is not None and len(pred[0]):
            pred[0][:, :4] = self.scale_coords(img_t.shape[2:], pred[0][:, :4], im0.shape).round()
            for c in pred[0][:, -1].unique():
                n = (pred[0][:, -1] == c).sum()
                detection_str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
            for *xyxy, conf, cls in pred[0]:
                cls_idx = int(cls)
                class_name = self.names[cls_idx] if cls_idx < len(self.names) else f"class{cls_idx}"
                conf_score = float(conf)
                self.logger.debug(f"Detection: {class_name} (idx={cls_idx}), conf={conf_score:.2f}, box={xyxy}")

                # Draw box
                x1, y1, x2, y2 = map(int, xyxy)
                if x2 <= x1 or y2 <= y1:
                    continue
                label = f"{class_name} {conf_score:.2f}"
                color = self.colors[cls_idx]
                cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(im0, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                cv2.putText(im0, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                total_boxes += 1

                # Save txt
                if save_txt and txt_path:
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # whwh
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls_idx, *xywh, conf_score)
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

        self.logger.info(f"Detections: {detection_str or 'None'}")
        if total_boxes == 0:
            self.logger.info("No detections on this frame")
        else:
            cv2.imwrite(f"{self.save_dir}/debug_frame_{int(time.time())}.jpg", im0)
        return im0

    def run_live(self, cam_index: int = 0, view_img: bool = True, save_txt: bool = False) -> None:
        self.logger.info(f"Starting live detection on camera {cam_index}")
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            self.logger.error("Cannot open camera")
            raise RuntimeError("Cannot open camera")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame_count = 0
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.warning("Failed to read frame")
                break
            frame_count += 1
            txt_path = str(self.save_dir / 'labels' / f'frame_{frame_count}') if save_txt else None
            if save_txt:
                (self.save_dir / 'labels').mkdir(exist_ok=True)
            result = self.detect_frame(frame, save_txt, txt_path)
            if view_img:
                display_frame = cv2.resize(result, (960, 540))
                cv2.imshow("YOLOv7 Detection (press 'q' to quit)", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
        self.logger.info(f"Processed {frame_count} frames in {(time.time() - t0):.1f}s")

    def run_image(self, image_path: str, save_img: bool = True, save_txt: bool = False) -> str:
        save_path = str(self.save_dir / Path(image_path).name)
        txt_path = str(self.save_dir / 'labels' / Path(image_path).stem) if save_txt else None
        self.logger.info(f"Detecting in image: {image_path} -> {save_path}")
        img = cv2.imread(os.path.expanduser(image_path))
        if img is None:
            self.logger.error(f"Cannot load image: {image_path}")
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        if save_txt:
            (self.save_dir / 'labels').mkdir(exist_ok=True)
        result = self.detect_frame(img, save_txt, txt_path)
        if save_img:
            cv2.imwrite(save_path, result)
            self.logger.info(f"Saved to {save_path}")
        return save_path

    def run_video(self, video_path: str, save_video: bool = True, view_img: bool = True, save_txt: bool = False) -> str:
        save_path = str(self.save_dir / (Path(video_path).stem + '.mp4'))
        self.logger.info(f"Detecting in video: {video_path} -> {save_path}")
        cap = cv2.VideoCapture(os.path.expanduser(video_path))
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)) if save_video else None
        frame_count = 0
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            txt_path = str(self.save_dir / 'labels' / f'frame_{frame_count}') if save_txt else None
            if save_txt:
                (self.save_dir / 'labels').mkdir(exist_ok=True)
            result = self.detect_frame(frame, save_txt, txt_path)
            if save_video:
                writer.write(result)
            if view_img:
                cv2.imshow("YOLOv7 Detection", cv2.resize(result, (960, 540)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        if save_video:
            writer.release()
        cv2.destroyAllWindows()
        self.logger.info(f"Processed {frame_count} frames in {(time.time() - t0):.1f}s")
        return save_path

def xyxy2xywh(xyxy):
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    x1, y1, x2, y2 = xyxy
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]

if __name__ == "__main__":
    detector = YoloV7Detector(
        weights_path="/Users/garvitmehra/PycharmProjects/pythonProject/Object_detector/YOLOv7 w6.pt",
        device="cpu",
        img_size=1280,
        conf_thres=0.1,
        iou_thres=0.3,
        classes=None,  # Detect all COCO classes
    )
    detector.run_live(cam_index=0, view_img=True, save_txt=False)
    # detector.run_image("test.jpg", save_img=True, save_txt=False)
    # detector.run_video("test.mp4", save_video=True, view_img=True, save_txt=False)
