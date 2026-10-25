# Sample Data & Image Validation Guide
**Bolt AI Helmet Detection Prototype**

This directory contains test media, validation manifests, and evaluation results.

---

## 1. Ground Truth Categories Required for Validation
To evaluate the detector rigorously across realistic operating conditions, collect test photos corresponding to each of the following four test categories:

| Category | Description | Ground Truth Status | Expected Behavior |
| :--- | :--- | :--- | :--- |
| **1. Helmeted Rider** | Clear photo of motorcycle rider wearing a certified helmet (full-face, open-face, or modular). | `HELMET DETECTED` | Bounding box on head labeled `With Helmet`, status banner green. |
| **2. Unhelmeted Rider** | Clear photo of rider with exposed head (no helmet). | `NO HELMET DETECTED` | Bounding box on head labeled `Without Helmet`, status banner red. |
| **3. Obscured / Distant Rider** | Rider head obscured by windshield/shadows, extreme blur, or too far away to assess. | `UNCERTAIN / RIDER NOT VISIBLE` | High uncertainty; must **never** falsely report a violation. |
| **4. Empty Scene / No Rider** | Street, parked bike without rider, or empty scene. | `UNCERTAIN / RIDER NOT VISIBLE` | Zero detections; status remains neutral/uncertain. |

---

## 2. How to Add Your Test Images

1. Place your test images directly into `sample_data/` (e.g. `rider_helmet.jpg`, `rider_no_helmet.jpg`, `rider_distant.jpg`, `empty_road.jpg`).
2. Add corresponding entries to [`sample_data/evaluation_manifest.json`](file:///D:/Vishwakarma%20awards/Helmet%20detection/sample_data/evaluation_manifest.json):

```json
[
  {
    "filename": "sample_data/rider_helmet.jpg",
    "ground_truth": "HELMET DETECTED",
    "category": "Rider wearing helmet",
    "is_synthetic": false,
    "description": "Side-angle photo of rider with full-face helmet in daylight."
  },
  {
    "filename": "sample_data/rider_no_helmet.jpg",
    "ground_truth": "NO HELMET DETECTED",
    "category": "Rider without helmet",
    "is_synthetic": false,
    "description": "Frontal photo of unhelmeted rider."
  },
  {
    "filename": "sample_data/rider_distant.jpg",
    "ground_truth": "UNCERTAIN / RIDER NOT VISIBLE",
    "category": "Distant / obscured rider",
    "is_synthetic": false,
    "description": "Motorcycle rider 30 meters away with low facial/head detail."
  },
  {
    "filename": "sample_data/empty_road.jpg",
    "ground_truth": "UNCERTAIN / RIDER NOT VISIBLE",
    "category": "No rider present",
    "is_synthetic": false,
    "description": "Road scene with no motorcycles or riders."
  }
]
```

---

## 3. Running the Validation Workflow

From the project root:
```powershell
.\.venv\Scripts\python.exe evaluate.py
```

This will:
- Run the detector on each image in `evaluation_manifest.json`.
- Compare predictions against the ground truth labels.
- Print a detailed evaluation matrix to the console.
- Save annotated images and a JSON evaluation report to `sample_data/evaluation_results/`.

---

## 4. Current Directory Contents
- `README.md` — This guide.
- `evaluation_manifest.json` — Ground truth mapping file.
- `test_synthetic.jpg` — Synthetic calibration frame for testing empty-scene uncertainty.
- `evaluation_results/` — Auto-generated directory containing visual detection overlays and reports.
