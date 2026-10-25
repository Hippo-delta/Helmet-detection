"""
Bolt AI Helmet Detection Prototype - Video & Stream Processing
Vishwakarma Awards Engineering Project

This module manages image loading, video playback, and live webcam capture.
It isolates input acquisition from detection logic so that future physical
camera integration on the motorcycle prototype can be connected cleanly.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Union, Any, Callable
import cv2
import numpy as np

from detector import HelmetDetector, STATUS_UNCERTAIN


class VideoProcessor:
    """
    Manages image files, prerecorded video files, and camera streams for helmet detection.
    """

    def __init__(self, detector: HelmetDetector) -> None:
        """
        Initialize the processor with an active HelmetDetector instance.
        """
        self.detector = detector

    def process_image_file(
        self,
        image_path: str,
        display: bool = True,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a static image from disk, run helmet detection, and optionally display/save.

        Args:
            image_path: Path to the image file (.jpg, .png, etc.).
            display: If True, opens an OpenCV display window.
            output_path: Optional path to save the annotated output image.

        Returns:
            Dictionary with detection results and processing metadata.
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Image file not found: {image_path}. Please check the path."
            )

        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(
                f"Could not decode image at '{image_path}'. Ensure it is a valid image format."
            )

        # Run detection
        results = self.detector.detect(frame)

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), results["annotated_frame"])
            results["saved_to"] = str(out_p)

        if display:
            win_title = f"Bolt AI Helmet Detection - {results['status']}"
            cv2.imshow(win_title, results["annotated_frame"])
            print("Press any key in the image window to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return results

    def process_stream(
        self,
        source: Union[str, int],
        is_camera: bool = False,
        display: bool = True,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        frame_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Process a video file or live camera stream frame-by-frame.

        Args:
            source: Video file path (str) or camera index (int, e.g. 0).
            is_camera: True if connecting to live webcam/USB camera.
            display: True to render frames in a GUI window.
            output_path: Optional path to write an MP4/AVI recording.
            max_frames: Optional frame limit (useful for headless testing).
            frame_callback: Optional callback receiving result dict per frame.

        Returns:
            Summary dictionary of the stream session.
        """
        source_desc = f"Camera [Index {source}]" if is_camera else f"Video ['{source}']"

        # Validate video file existence
        if not is_camera:
            video_path = Path(str(source))
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"Video file not found: {source}. Please verify the file path."
                )

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            err_msg = (
                f"Unable to access {source_desc}. "
                "Ensure the camera is plugged in and not in use by another application."
                if is_camera else
                f"Unable to open video stream from {source_desc}. Video codec may be unsupported."
            )
            raise RuntimeError(err_msg)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path and width > 0 and height > 0:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))

        frame_count = 0
        status_history: Dict[str, int] = {}
        start_time = time.time()
        window_name = f"Bolt AI Helmet Monitor - {source_desc} (Press 'q' to stop)"

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # End of video or stream disconnected
                    break

                frame_count += 1
                result = self.detector.detect(frame)
                st = result["status"]
                status_history[st] = status_history.get(st, 0) + 1

                annotated = result["annotated_frame"]

                # Add FPS and frame counter overlay
                elapsed = max(time.time() - start_time, 0.001)
                curr_fps = frame_count / elapsed
                cv2.putText(
                    annotated,
                    f"FPS: {curr_fps:.1f} | Frame: {frame_count}",
                    (width - 240, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                if writer is not None:
                    writer.write(annotated)

                if frame_callback is not None:
                    frame_callback(result)

                if display:
                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    # 'q' or ESC (27) to stop playback cleanly
                    if key == ord('q') or key == 27:
                        print("\nPlayback stopped by user.")
                        break

                if max_frames is not None and frame_count >= max_frames:
                    break

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        total_elapsed = time.time() - start_time

        return {
            "source": source_desc,
            "total_frames_processed": frame_count,
            "elapsed_seconds": round(total_elapsed, 2),
            "average_fps": round(frame_count / max(total_elapsed, 0.001), 1),
            "status_distribution": status_history
        }
