"""
Automated Test Suite for Bolt AI Helmet Detection Prototype
Vishwakarma Awards Engineering Project

Covers:
1. Missing and invalid model files
2. Invalid image and video input paths
3. Unavailable camera handling
4. Compliance status classification logic and uncertainty thresholding
5. Real model integrity (SHA-256 checksum and class verification)
6. Synthetic frame inference pipeline
"""

import hashlib
import os
from pathlib import Path
import pytest
import numpy as np
import cv2

from detector import (
    HelmetDetector,
    STATUS_HELMET_DETECTED,
    STATUS_NO_HELMET,
    STATUS_UNCERTAIN,
    CLASS_WITH_HELMET,
    CLASS_WITHOUT_HELMET
)
from video_processor import VideoProcessor

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
EXPECTED_SHA256 = "c8eb324e365cf4faeab491d9cc301535ec745171b55e8b1acadea62be5101a9d"


class TestDetectorInitialization:
    """Tests for model loading and error handling."""

    def test_missing_model_file_raises_error(self):
        """Ensure attempting to load a nonexistent model path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            HelmetDetector(model_path="nonexistent_directory/fake_model.pt")
        assert "Model weights file not found" in str(exc_info.value)

    def test_detector_requires_model_for_inference(self):
        """Ensure calling detect without a loaded model raises RuntimeError."""
        detector = HelmetDetector(model_path=None)
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError) as exc_info:
            detector.detect(blank)
        assert "No model is loaded" in str(exc_info.value)

    def test_invalid_frame_input_raises_value_error(self):
        """Ensure None or empty frame raises ValueError."""
        detector = HelmetDetector(model_path=None)
        with pytest.raises(ValueError):
            detector.detect(None)
        with pytest.raises(ValueError):
            detector.detect(np.array([]))


class TestStatusClassificationLogic:
    """
    Tests for status evaluation and uncertainty handling.
    Crucial: Rider must NOT be labeled as NO HELMET when head is not detected.
    """

    @pytest.fixture
    def detector(self):
        return HelmetDetector(model_path=None, conf_threshold=0.50)

    def test_empty_detections_returns_uncertain(self, detector):
        """Empty detections must yield UNCERTAIN, never NO HELMET."""
        result = detector.evaluate_detections([])
        assert result == STATUS_UNCERTAIN

    def test_low_confidence_detections_return_uncertain(self, detector):
        """Detections below confidence threshold must yield UNCERTAIN."""
        low_conf = [
            {"class_name": CLASS_WITH_HELMET, "confidence": 0.35, "box": [10, 10, 50, 50]},
            {"class_name": CLASS_WITHOUT_HELMET, "confidence": 0.40, "box": [60, 60, 90, 90]}
        ]
        result = detector.evaluate_detections(low_conf)
        assert result == STATUS_UNCERTAIN

    def test_with_helmet_meets_threshold(self, detector):
        """High-confidence With Helmet detection yields HELMET DETECTED."""
        detections = [
            {"class_name": CLASS_WITH_HELMET, "confidence": 0.88, "box": [20, 20, 120, 120]}
        ]
        result = detector.evaluate_detections(detections)
        assert result == STATUS_HELMET_DETECTED

    def test_without_helmet_meets_threshold(self, detector):
        """High-confidence Without Helmet detection yields NO HELMET DETECTED."""
        detections = [
            {"class_name": CLASS_WITHOUT_HELMET, "confidence": 0.82, "box": [20, 20, 120, 120]}
        ]
        result = detector.evaluate_detections(detections)
        assert result == STATUS_NO_HELMET

    def test_conflicting_detections_yield_uncertain(self, detector):
        """If both With Helmet and Without Helmet exist with high confidence, status is UNCERTAIN."""
        conflicting = [
            {"class_name": CLASS_WITH_HELMET, "confidence": 0.75, "box": [10, 10, 80, 80]},
            {"class_name": CLASS_WITHOUT_HELMET, "confidence": 0.78, "box": [10, 10, 80, 80]}
        ]
        result = detector.evaluate_detections(conflicting)
        assert result == STATUS_UNCERTAIN


class TestVideoProcessorHandling:
    """Tests for file and camera input handling."""

    @pytest.fixture
    def processor(self):
        detector = HelmetDetector(model_path=None, conf_threshold=0.50)
        return VideoProcessor(detector)

    def test_missing_image_file_raises_not_found(self, processor):
        """Ensure missing image file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            processor.process_image_file("nonexistent_test_image.jpg", display=False)

    def test_missing_video_file_raises_not_found(self, processor):
        """Ensure missing video file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            processor.process_stream("nonexistent_test_video.mp4", is_camera=False, display=False)

    def test_unavailable_camera_index_handled_gracefully(self, processor):
        """Ensure an unavailable camera index (e.g. 9999) raises descriptive RuntimeError without crash."""
        with pytest.raises(RuntimeError) as exc_info:
            processor.process_stream(source=9999, is_camera=True, display=False)
        assert "Unable to access Camera" in str(exc_info.value)


class TestModelIntegrityAndInference:
    """Tests evaluating the real downloaded model weights."""

    @pytest.mark.skipif(not MODEL_PATH.is_file(), reason="Model file not yet downloaded")
    def test_model_file_sha256_checksum(self):
        """Verify the model binary integrity against the recorded SHA-256 hash."""
        sha256 = hashlib.sha256()
        with open(MODEL_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest()
        assert actual_hash == EXPECTED_SHA256, (
            f"Model checksum mismatch: expected {EXPECTED_SHA256}, got {actual_hash}"
        )

    @pytest.mark.skipif(not MODEL_PATH.is_file(), reason="Model file not yet downloaded")
    def test_model_class_labels(self):
        """Verify that the model classes are exactly With Helmet and Without Helmet."""
        detector = HelmetDetector(model_path=str(MODEL_PATH))
        assert detector.classes.get(0) == CLASS_WITH_HELMET
        assert detector.classes.get(1) == CLASS_WITHOUT_HELMET

    @pytest.mark.skipif(not MODEL_PATH.is_file(), reason="Model file not yet downloaded")
    def test_synthetic_frame_inference_end_to_end(self):
        """
        Run inference on a synthetic blank test frame.
        Since no rider is present, status must be UNCERTAIN / RIDER NOT VISIBLE.
        """
        detector = HelmetDetector(model_path=str(MODEL_PATH), conf_threshold=0.50)
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = detector.detect(blank_frame)
        assert result["status"] == STATUS_UNCERTAIN
        assert isinstance(result["detections"], list)
        assert result["annotated_frame"].shape == (480, 640, 3)
