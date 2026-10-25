"""
Bolt AI Helmet Detection Prototype - Real-Image Evaluation Workflow
Vishwakarma Awards Engineering Project

This script evaluates the trained detector on user-provided or licensed images
with known ground truth labels across the four core categories:
1. Rider wearing a helmet (Ground Truth: HELMET DETECTED)
2. Rider not wearing a helmet (Ground Truth: NO HELMET DETECTED)
3. Rider/head partially obscured or distant (Ground Truth: UNCERTAIN / RIDER NOT VISIBLE)
4. No rider present (Ground Truth: UNCERTAIN / RIDER NOT VISIBLE)

Rules:
- Never treats absent detections as proof of non-compliance.
- Explicitly flags whether each evaluated image is synthetic or real.
- Computes exact match metrics without fabricating results.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import cv2
import numpy as np

from detector import (
    HelmetDetector,
    STATUS_HELMET_DETECTED,
    STATUS_NO_HELMET,
    STATUS_UNCERTAIN
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_MANIFEST = PROJECT_ROOT / "sample_data" / "evaluation_manifest.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "sample_data" / "evaluation_results"


def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load ground-truth evaluation manifest JSON."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Evaluation manifest must contain a JSON list of image entries.")

    return data


def run_evaluation(
    detector: HelmetDetector,
    entries: List[Dict[str, Any]],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run detector on manifest entries and compare against ground truth.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    total_images = len(entries)
    synthetic_count = 0
    real_count = 0
    matches = 0
    mismatches = 0

    print("\n" + "=" * 90)
    print("                BOLT AI HELMET DETECTION - EVALUATION MATRIX")
    print("=" * 90)
    header = (
        f"{'Image':<26} | {'Type':<9} | {'Ground Truth':<18} | "
        f"{'Predicted Status':<18} | {'Conf':<6} | {'Match'}"
    )
    print(header)
    print("-" * 90)

    for item in entries:
        rel_path = item.get("filename", "")
        img_path = PROJECT_ROOT / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        ground_truth = item.get("ground_truth", "").strip()
        category = item.get("category", "Unspecified")
        is_synthetic = bool(item.get("is_synthetic", False))

        if is_synthetic:
            synthetic_count += 1
        else:
            real_count += 1

        if not img_path.is_file():
            print(f"{img_path.name:<26} | {'MISSING FILE':<9} | {ground_truth:<18} | {'N/A':<18} | {'N/A':<6} | ERROR")
            results.append({
                "filename": str(img_path.name),
                "error": "File not found",
                "ground_truth": ground_truth,
                "matched": False
            })
            mismatches += 1
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"{img_path.name:<26} | {'CORRUPT':<9} | {ground_truth:<18} | {'N/A':<18} | {'N/A':<6} | ERROR")
            results.append({
                "filename": str(img_path.name),
                "error": "Could not decode image",
                "ground_truth": ground_truth,
                "matched": False
            })
            mismatches += 1
            continue

        det_result = detector.detect(frame)
        predicted_status = det_result["status"]
        detections = det_result.get("detections", [])

        # Get highest confidence detection for summary display
        top_conf_str = "N/A"
        top_class_str = "None"
        if detections:
            best_det = max(detections, key=lambda d: d.get("confidence", 0.0))
            top_conf_str = f"{best_det.get('confidence', 0.0) * 100:.1f}%"
            top_class_str = best_det.get("class_name", "Unknown")

        is_match = (predicted_status == ground_truth)
        if is_match:
            matches += 1
            match_str = "MATCH (PASS)"
        else:
            mismatches += 1
            match_str = "MISMATCH (FAIL)"

        type_str = "Synthetic" if is_synthetic else "Real"
        print(
            f"{img_path.name[:25]:<26} | {type_str:<9} | {ground_truth[:18]:<18} | "
            f"{predicted_status[:18]:<18} | {top_conf_str:<6} | {match_str}"
        )

        # Save annotated image for manual visual inspection
        out_img_path = output_dir / f"eval_{img_path.name}"
        cv2.imwrite(str(out_img_path), det_result["annotated_frame"])

        results.append({
            "filename": str(img_path.name),
            "filepath": str(img_path),
            "is_synthetic": is_synthetic,
            "category": category,
            "ground_truth": ground_truth,
            "predicted_status": predicted_status,
            "top_detected_class": top_class_str,
            "top_confidence": top_conf_str,
            "detections_count": len(detections),
            "detections": detections,
            "matched": is_match,
            "annotated_output": str(out_img_path)
        })

    print("-" * 90)

    accuracy_rate = (matches / total_images * 100.0) if total_images > 0 else 0.0

    print("\n--- Evaluation Summary ---")
    print(f"Total Images Evaluated: {total_images}")
    print(f"  - Real Images:        {real_count}")
    print(f"  - Synthetic Images:   {synthetic_count}")
    print(f"Ground Truth Matches:   {matches}/{total_images} ({accuracy_rate:.1f}%)")
    print(f"Mismatches / Errors:    {mismatches}/{total_images}")
    print(f"Annotated outputs saved to: {output_dir}")

    if real_count == 0:
        print("\n[IMPORTANT VALIDATION NOTE]")
        print("Only synthetic calibration media was evaluated.")
        print("No real-world images were present in this run.")
        print("Real-world accuracy, precision, and recall CANNOT be claimed without real test images.")

    summary = {
        "total_images": total_images,
        "real_count": real_count,
        "synthetic_count": synthetic_count,
        "matches": matches,
        "mismatches": mismatches,
        "accuracy_rate_percent": round(accuracy_rate, 2),
        "results": results
    }

    # Save summary report JSON
    report_json_path = output_dir / "evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Machine-readable report written to: {report_json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Bolt AI Helmet Detection Prototype - Ground Truth Image Evaluator"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST),
        help=f"Path to JSON manifest of images and ground-truth labels (default: {DEFAULT_MANIFEST})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help=f"Path to trained YOLO model weights (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Confidence threshold (default: 0.50)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help=f"Directory to save annotated evaluation results (default: {DEFAULT_OUT_DIR})"
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)

    print("=" * 68)
    print("      BOLT AI HELMET DETECTION - IMAGE VALIDATION WORKFLOW")
    print("      Vishwakarma Awards Engineering Submission")
    print("=" * 68)
    print(f"Loading Model:     {model_path}")
    print(f"Manifest File:     {manifest_path}")
    print(f"Output Directory:  {output_dir}")
    print(f"Confidence Level:  {args.conf:.2f}")

    try:
        detector = HelmetDetector(model_path=str(model_path), conf_threshold=args.conf)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        sys.exit(1)

    try:
        entries = load_manifest(manifest_path)
    except Exception as e:
        print(f"Error loading manifest: {e}")
        sys.exit(1)

    run_evaluation(detector, entries, output_dir)


if __name__ == "__main__":
    main()
