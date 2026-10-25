# Bolt AI Helmet Detection Prototype
**Vishwakarma Awards Engineering Submission**

## 1. Project Purpose
The Bolt AI Helmet Detection Prototype is an experimental computer-vision system designed to monitor motorcycle rider safety compliance. The objective is to analyze visual input (static images, prerecorded videos, or video streams) and assess whether a rider is wearing a protective helmet.

This project is developed as an initial engineering prototype for a student award submission. It separates camera input acquisition from computer-vision inference, preparing the foundation for future integration with a camera mounted on a physical motorcycle prototype.

---

## 2. System Architecture & File Structure

```
Helmet-detection/
├── .venv/                         # Isolated Python virtual environment (local)
├── models/
│   └── best.pt                    # Pretrained YOLOv8n weights (~6.2 MB)
├── sample_data/
│   ├── README.md                  # Sourcing and validation guide
│   ├── evaluation_manifest.json   # Ground-truth test manifest
│   ├── test_synthetic.jpg         # Synthetic calibration frame
│   ├── helmet1.jpg                # Sample evaluation image (Helmeted)
│   ├── No helmet1.jpg             # Sample evaluation image (Unhelmeted)
│   └── Empty road.jpg             # Sample evaluation image (Empty road)
├── tests/
│   ├── __init__.py
│   └── test_detector.py           # Automated pytest suite (14 test cases)
├── detector.py                    # HelmetDetector class & compliance logic
├── video_processor.py             # Stream & file processor for images, video, webcam
├── main.py                        # CLI & interactive demonstration interface
├── evaluate.py                    # Ground-truth evaluation script
├── requirements.txt               # Project dependencies
├── .gitignore                     # Version control exclusion rules
└── README.md                      # Project documentation
```

### Module Descriptions
- [`detector.py`](file:///D:/Vishwakarma%20awards/Helmet%20detection/detector.py): Implements `HelmetDetector`, encapsulating YOLO inference, head bounding-box processing, confidence thresholding, and compliance status classification.
- [`video_processor.py`](file:///D:/Vishwakarma%20awards/Helmet%20detection/video_processor.py): Implements `VideoProcessor`, handling OpenCV image decoding, video playback with FPS calculation, and live camera capture with clean stop mechanisms (`q` / `ESC`).
- [`main.py`](file:///D:/Vishwakarma%20awards/Helmet%20detection/main.py): Entry point providing both command-line arguments (`--image`, `--video`, `--camera`, `--conf`, `--no-display`) and an interactive demonstration menu.
- [`evaluate.py`](file:///D:/Vishwakarma%20awards/Helmet%20detection/evaluate.py): Evaluates the detector against labeled ground-truth images, computes exact match metrics, and outputs visual overlays.
- [`tests/test_detector.py`](file:///D:/Vishwakarma%20awards/Helmet%20detection/tests/test_detector.py): Automated test suite verifying file validation, camera failure handling, uncertainty thresholding, and model integrity.

---

## 3. Pretrained Model Details

- **Model Identifier**: `iam-tsr/yolov8n-helmet-detection`
- **Repository Source**: [Hugging Face Hub](https://huggingface.co/iam-tsr/yolov8n-helmet-detection)
- **Architecture**: Ultralytics YOLOv8n (Nano)
- **File Size**: ~6.2 MB (`best.pt`)
- **License**: MIT License
- **SHA-256 Checksum**: `c8eb324e365cf4faeab491d9cc301535ec745171b55e8b1acadea62be5101a9d`
- **Verified Classes**:
  - `0`: `With Helmet` (head wearing a helmet)
  - `1`: `Without Helmet` (exposed head without a helmet)

### Detection Mechanism & Uncertainty Handling
Unlike general-purpose models (such as COCO models) that only detect `person` and `motorcycle` without knowing whether a helmet is worn, this model specifically detects the rider's head region and classifies it into `With Helmet` or `Without Helmet`.

The detector implements three clear status categories:
1. **`HELMET DETECTED`**: A head with `With Helmet` is detected with confidence $\ge$ the threshold (default: 0.50).
2. **`NO HELMET DETECTED`**: A head with `Without Helmet` is detected with confidence $\ge$ the threshold, with no helmet detected.
3. **`UNCERTAIN / RIDER NOT VISIBLE`**: Reported when no rider/head is detected, when confidence is below the threshold, or when detections are conflicting.

> [!IMPORTANT]
> **Safety Principle**: A rider is **never** accused of a violation simply because the head is obscured, lighting is poor, or no helmet was detected. If the head cannot be assessed reliably, the system reports `UNCERTAIN / RIDER NOT VISIBLE`.

---

## 4. Installation & Setup

These instructions allow an evaluator or developer to run the project on any Windows computer:

### Step 1: Open Terminal in Project Directory
Clone the repository or navigate to your local project directory:
```powershell
git clone https://github.com/Hippo-delta/Helmet-detection.git
cd Helmet-detection
```

### Step 2: Create and Activate Python Virtual Environment
Python 3.10+ (tested on Python 3.13) is recommended:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating).*

### Step 3: Install Dependencies
Install all required packages into the isolated virtual environment:
```powershell
pip install -r requirements.txt
```

---

## 5. Usage & Input Modes

### A. Interactive Menu
Run without arguments to access the interactive demonstration menu:
```powershell
python main.py
```

### B. Image Mode
Analyze a static photograph:
```powershell
python main.py --image "sample_data\test_synthetic.jpg"
```
Or headless (without opening GUI windows):
```powershell
python main.py --image "sample_data\test_synthetic.jpg" --no-display
```

### C. Video Mode
Analyze a prerecorded video file (.mp4, .avi):
```powershell
python main.py --video "path\to\your_video.mp4"
```
Press `q` or `ESC` in the playback window at any time to cleanly stop playback.

### D. Live Camera Mode
Attempt connection to a connected webcam or USB camera:
```powershell
python main.py --camera 0
```
If no camera is physically connected, the system reports an informative notice without crashing.

### E. Ground-Truth Validation Workflow
Evaluate the detector on labeled images across all 4 operational categories:
```powershell
python evaluate.py
```
This runs the detector against [`sample_data/evaluation_manifest.json`](file:///D:/Vishwakarma%20awards/Helmet%20detection/sample_data/evaluation_manifest.json), logs an evaluation matrix, and saves annotated visual overlays to `sample_data/evaluation_results/`.

---

## 6. Automated Test Results

The test suite was executed inside the virtual environment using `pytest`:

```
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-9.1.1, pluggy-1.7.0
rootdir: D:\Vishwakarma awards\Helmet detection

tests/test_detector.py::TestDetectorInitialization::test_missing_model_file_raises_error PASSED
tests/test_detector.py::TestDetectorInitialization::test_detector_requires_model_for_inference PASSED
tests/test_detector.py::TestDetectorInitialization::test_invalid_frame_input_raises_value_error PASSED
tests/test_detector.py::TestStatusClassificationLogic::test_empty_detections_returns_uncertain PASSED
tests/test_detector.py::TestStatusClassificationLogic::test_low_confidence_detections_return_uncertain PASSED
tests/test_detector.py::TestStatusClassificationLogic::test_with_helmet_meets_threshold PASSED
tests/test_detector.py::TestStatusClassificationLogic::test_without_helmet_meets_threshold PASSED
tests/test_detector.py::TestStatusClassificationLogic::test_conflicting_detections_yield_uncertain PASSED
tests/test_detector.py::TestVideoProcessorHandling::test_missing_image_file_raises_not_found PASSED
tests/test_detector.py::TestVideoProcessorHandling::test_missing_video_file_raises_not_found PASSED
tests/test_detector.py::TestVideoProcessorHandling::test_unavailable_camera_index_handled_gracefully PASSED
tests/test_detector.py::TestModelIntegrityAndInference::test_model_file_sha256_checksum PASSED
tests/test_detector.py::TestModelIntegrityAndInference::test_model_class_labels PASSED
tests/test_detector.py::TestModelIntegrityAndInference::test_synthetic_frame_inference_end_to_end PASSED

============================= 14 passed in 21.63s =============================
```

> [!NOTE]
> These automated tests verify software correctness, input validation, status classification rules, and binary integrity. They **do not constitute real-world accuracy validation** on real road traffic datasets.

---

## 7. Known Limitations & Distinctions

1. **Hardware Absence**: The prototype was verified in software. Since physical camera hardware was unavailable during testing, live-camera functionality has been tested for graceful fallback, but not live motorcycle stream performance.
2. **Model Evaluation Scope**: The model has not been evaluated on benchmark motorcycle datasets (e.g. night riding, heavy rain, adverse angles). Real-world accuracy claims cannot be made without rigorous field evaluation.
3. **Occlusion & Passenger Ambiguity**: If multiple people appear on a motorcycle or in the background, a single-head classifier can produce ambiguous results. Future work will require rider-seat association.

---

## 8. Safety & Future Hardware Integration

- **Safety Boundary**: This software is strictly an experimental prototype for academic evaluation. It must never be connected directly to vehicle ignition, throttle cutoff, or safety-critical mechanisms.
- **Future Integration Plan**:
  1. Mount an IP67-rated wide-angle camera to the motorcycle instrument cluster.
  2. Implement an onboard edge processor (e.g., Raspberry Pi 5 or NVIDIA Jetson Orin Nano).
  3. Integrate temporal filtering (requiring consistent detection over multiple consecutive frames before changing status).
  4. Provide an advisory-only dashboard indicator for rider compliance.
