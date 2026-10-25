"""
Bolt AI Helmet Detection Prototype - Main Entry Point
Vishwakarma Awards Engineering Project

This application demonstrates real-time and static AI helmet detection
for motorcycle rider safety assessment.
"""

import argparse
import os
import sys
from pathlib import Path

from detector import (
    HelmetDetector,
    STATUS_HELMET_DETECTED,
    STATUS_NO_HELMET,
    STATUS_UNCERTAIN
)
from video_processor import VideoProcessor

# Default model weights path inside project
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pt"


def print_banner() -> None:
    print("=" * 68)
    print("      BOLT AI HELMET DETECTION PROTOTYPE")
    print("      Vishwakarma Awards Engineering Submission")
    print("=" * 68)


def run_image_mode(processor: VideoProcessor, image_path: str, display: bool = True) -> None:
    """Run detection on a static image."""
    print(f"\n[Image Mode] Loading: {image_path}")
    try:
        results = processor.process_image_file(image_path, display=display)
        print("\n--- Detection Results ---")
        print(f"Overall Status: {results['status']}")
        detections = results.get("detections", [])
        print(f"Detections found: {len(detections)}")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. Class: {det['class_name']} | Confidence: {det['confidence'] * 100:.1f}% | Box: {det['box']}")
        if not detections:
            print("  (No head/helmet detected with sufficient confidence)")
    except Exception as e:
        print(f"Error processing image: {e}")


def run_video_mode(processor: VideoProcessor, video_path: str, display: bool = True) -> None:
    """Run detection on a prerecorded video file."""
    print(f"\n[Video Mode] Processing video: {video_path}")
    print("Press 'q' or 'ESC' in the playback window to stop at any time.")
    try:
        summary = processor.process_stream(
            source=video_path,
            is_camera=False,
            display=display
        )
        print("\n--- Video Processing Summary ---")
        print(f"Total frames processed: {summary['total_frames_processed']}")
        print(f"Elapsed time: {summary['elapsed_seconds']}s")
        print(f"Average FPS: {summary['average_fps']}")
        print("Status distribution:")
        for st, count in summary.get("status_distribution", {}).items():
            print(f"  - {st}: {count} frames")
    except Exception as e:
        print(f"Error processing video: {e}")


def run_camera_mode(processor: VideoProcessor, camera_index: int = 0, display: bool = True) -> None:
    """Run detection on a live webcam/USB camera."""
    print(f"\n[Camera Mode] Attempting connection to camera device [Index {camera_index}]...")
    print("Note: If no physical camera is connected to the prototype, this will report an error gracefully.")
    print("Press 'q' or 'ESC' in the live window to stop.")
    try:
        summary = processor.process_stream(
            source=camera_index,
            is_camera=True,
            display=display
        )
        print("\n--- Camera Stream Summary ---")
        print(f"Total frames captured: {summary['total_frames_processed']}")
        print(f"Average FPS: {summary['average_fps']}")
    except RuntimeError as e:
        print(f"\n[Camera Status Notice] {e}")
        print("Live camera testing cannot run without physical hardware connected.")
        print("You can test using Mode 1 (Images) or Mode 2 (Prerecorded Video) instead.")
    except Exception as e:
        print(f"Unexpected camera error: {e}")


def interactive_menu(processor: VideoProcessor) -> None:
    """Provide a user-friendly interactive terminal menu."""
    while True:
        print_banner()
        print("Select an operation mode:")
        print("  1. Image Mode       - Analyze a static picture (.jpg, .png)")
        print("  2. Video Mode       - Analyze a prerecorded video (.mp4, .avi)")
        print("  3. Live Camera Mode - Connect to webcam / mounted USB camera")
        print("  4. Exit")
        print("-" * 68)

        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            img_path = input("Enter path to image file: ").strip().strip('"').strip("'")
            if img_path:
                run_image_mode(processor, img_path)
            else:
                print("No path provided.")
        elif choice == "2":
            vid_path = input("Enter path to video file: ").strip().strip('"').strip("'")
            if vid_path:
                run_video_mode(processor, vid_path)
            else:
                print("No path provided.")
        elif choice == "3":
            idx_input = input("Enter camera device index (default 0): ").strip()
            cam_idx = int(idx_input) if idx_input.isdigit() else 0
            run_camera_mode(processor, camera_index=cam_idx)
        elif choice == "4":
            print("Exiting Bolt AI Helmet Detection Prototype. Goodbye!")
            break
        else:
            print("Invalid selection. Please choose 1, 2, 3, or 4.")

        input("\nPress Enter to return to the main menu...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bolt AI Helmet Detection Prototype (Vishwakarma Awards)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path to trained YOLO model weights (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Confidence threshold for detections (0.0 to 1.0, default: 0.50)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to an image to evaluate"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to a video file to evaluate"
    )
    parser.add_argument(
        "--camera",
        type=int,
        help="Camera device index for live capture (e.g. 0)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening GUI windows (headless execution)"
    )

    args = parser.parse_args()

    print_banner()
    print(f"Loading model weights: {args.model}")
    print(f"Confidence threshold: {args.conf:.2f}")

    # Initialize detector
    try:
        detector = HelmetDetector(model_path=args.model, conf_threshold=args.conf)
        print("Model loaded successfully.")
        print(f"Supported classes: {detector.classes}")
    except Exception as e:
        print(f"\n[Warning] Could not initialize model: {e}")
        print("Ensure models/best.pt exists or specify a valid model path with --model.")
        sys.exit(1)

    processor = VideoProcessor(detector)
    display = not args.no_display

    if args.image:
        run_image_mode(processor, args.image, display=display)
    elif args.video:
        run_video_mode(processor, args.video, display=display)
    elif args.camera is not None:
        run_camera_mode(processor, camera_index=args.camera, display=display)
    else:
        interactive_menu(processor)


if __name__ == "__main__":
    main()
