"""
Bolt AI Helmet Detection Prototype - Core Detector Module
Vishwakarma Awards Engineering Project

This module encapsulates model loading, inference, and classification
logic for motorcycle rider helmet compliance.

Status categories:
- HELMET DETECTED
- NO HELMET DETECTED
- UNCERTAIN / RIDER NOT VISIBLE

Important safety & compliance note:
A rider is NEVER categorized as unhelmeted solely because no helmet
is detected, or due to poor lighting, distance, or camera failure.
If the rider's head cannot be assessed with high confidence, the system
reports UNCERTAIN / RIDER NOT VISIBLE.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

# Status Constants
STATUS_HELMET_DETECTED = "HELMET DETECTED"
STATUS_NO_HELMET = "NO HELMET DETECTED"
STATUS_UNCERTAIN = "UNCERTAIN / RIDER NOT VISIBLE"

# Model Class Names for iam-tsr/yolov8n-helmet-detection
CLASS_WITH_HELMET = "With Helmet"
CLASS_WITHOUT_HELMET = "Without Helmet"


class HelmetDetector:
    """
    YOLOv8-based helmet detector and compliance evaluator.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.50,
        device: str = "cpu"
    ) -> None:
        """
        Initialize the detector.

        Args:
            model_path: Absolute or relative path to the trained YOLO model weights (.pt).
            conf_threshold: Minimum confidence score to accept a detection (0.0 to 1.0).
            device: Compute device ('cpu', 'cuda', etc.).
        """
        self.conf_threshold = float(conf_threshold)
        self.device = device
        self.model = None
        self.model_path = model_path
        self.classes: Dict[int, str] = {}

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """
        Load YOLO weights from the specified path.

        Raises:
            FileNotFoundError: If the model weights file does not exist.
            RuntimeError: If model loading fails.
        """
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model weights file not found at: {model_path}. "
                "Ensure the model is downloaded to the project models/ directory."
            )

        try:
            # Deferred import so non-model test scenarios do not fail if dependencies differ
            from ultralytics import YOLO
            self.model = YOLO(str(path))
            self.model_path = str(path)
            self.classes = getattr(self.model, "names", {})
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from '{model_path}': {e}") from e

    def evaluate_detections(
        self,
        detections: List[Dict[str, Any]]
    ) -> str:
        """
        Determine overall compliance status from a list of detections.

        Rules:
        1. If no detections meet confidence threshold -> UNCERTAIN / RIDER NOT VISIBLE.
        2. If 'With Helmet' is detected with confidence >= threshold and no conflicting
           high-confidence 'Without Helmet' is present on the same rider -> HELMET DETECTED.
        3. If 'Without Helmet' is detected with confidence >= threshold and no 'With Helmet'
           is present -> NO HELMET DETECTED.
        4. If both or conflicting detections exist without clear resolution -> UNCERTAIN / RIDER NOT VISIBLE.
        """
        valid_detections = [
            d for d in detections if d.get("confidence", 0.0) >= self.conf_threshold
        ]

        if not valid_detections:
            return STATUS_UNCERTAIN

        has_helmet = any(
            d.get("class_name", "").strip().lower() == CLASS_WITH_HELMET.lower()
            for d in valid_detections
        )
        has_no_helmet = any(
            d.get("class_name", "").strip().lower() == CLASS_WITHOUT_HELMET.lower()
            for d in valid_detections
        )

        if has_helmet and not has_no_helmet:
            return STATUS_HELMET_DETECTED
        elif has_no_helmet and not has_helmet:
            return STATUS_NO_HELMET
        else:
            # Ambiguous: multiple heads or conflicting bounding boxes
            return STATUS_UNCERTAIN

    def detect(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run helmet detection on a single image frame (BGR format).

        Args:
            frame: OpenCV image array (numpy ndarray, uint8, BGR).
            conf: Optional confidence override for this frame.

        Returns:
            Dictionary containing:
                - status: One of the three status constants.
                - detections: List of detection dicts (box, class_id, class_name, confidence).
                - annotated_frame: Copy of frame with bounding boxes and status banner.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Input frame is invalid or empty.")

        if self.model is None:
            raise RuntimeError("Cannot perform detection: No model is loaded.")

        effective_conf = conf if conf is not None else self.conf_threshold

        # Run YOLO inference
        results = self.model.predict(
            source=frame,
            conf=effective_conf,
            device=self.device,
            verbose=False
        )

        detections: List[Dict[str, Any]] = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    cls_name = self.classes.get(cls_id, f"class_{cls_id}")

                    detections.append({
                        "box": [int(coord) for coord in xyxy],
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(score, 4)
                    })

        status = self.evaluate_detections(detections)
        annotated_frame = self.annotate_frame(frame, detections, status)

        return {
            "status": status,
            "detections": detections,
            "annotated_frame": annotated_frame
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        status: str
    ) -> np.ndarray:
        """
        Draw detection bounding boxes, confidence tags, and status overlay.
        """
        output = frame.copy()
        height, width = output.shape[:2]

        # Draw individual detection boxes
        for det in detections:
            box = det.get("box", [0, 0, 0, 0])
            x1, y1, x2, y2 = box
            cls_name = det.get("class_name", "")
            score = det.get("confidence", 0.0)

            if cls_name.strip().lower() == CLASS_WITH_HELMET.lower():
                color = (0, 200, 0)      # Green for helmet
            elif cls_name.strip().lower() == CLASS_WITHOUT_HELMET.lower():
                color = (0, 0, 230)      # Red for no helmet
            else:
                color = (255, 165, 0)    # Amber/Cyan for unknown

            # Bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Label tag
            label = f"{cls_name} ({score * 100:.1f}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_label = max(y1 - 5, th + 5)
            cv2.rectangle(output, (x1, y_label - th - 4), (x1 + tw + 6, y_label + 2), color, -1)
            cv2.putText(
                output,
                label,
                (x1 + 3, y_label - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # Draw top banner for overall compliance status
        if status == STATUS_HELMET_DETECTED:
            banner_color = (0, 160, 0)      # Dark Green
            text_color = (255, 255, 255)
        elif status == STATUS_NO_HELMET:
            banner_color = (0, 0, 200)      # Strong Red
            text_color = (255, 255, 255)
        else:
            banner_color = (0, 140, 255)    # Orange/Amber
            text_color = (0, 0, 0)

        # Banner background
        banner_h = 42
        cv2.rectangle(output, (0, 0), (width, banner_h), banner_color, -1)
        cv2.line(output, (0, banner_h), (width, banner_h), (255, 255, 255), 1)

        # Banner text
        title = f"BOLT AI HELMET MONITOR: {status}"
        cv2.putText(
            output,
            title,
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
            cv2.LINE_AA
        )

        return output
