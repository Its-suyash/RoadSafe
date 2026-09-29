"""
Inference Pipeline
==================
End-to-end road damage detection, classification, and severity estimation.
Processes single images, directories, video files, or live webcam streams.
Outputs annotated images/video + JSON results.

Usage:
    python inference.py --image train/images/India_000017.jpg
    python inference.py --image-dir test_images/ --output results/
    python inference.py --video dashcam_footage.mp4 --output results/
    python inference.py --webcam --output results/
    python inference.py --webcam 1 --device cpu
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from severity_estimator import SeverityEstimator


# Class color palette (BGR for OpenCV rendering)
CLASS_COLORS_BGR = {
    "Pothole":           (0, 100, 255),     # Orange-red
    "LongitudinalCrack": (255, 150, 0),     # Blue-cyan
    "AlligatorCrack":    (0, 200, 200),     # Yellow-green
    "TransverseCrack":   (200, 0, 200),     # Magenta
}


def detect_road_damage(
    image_path: str,
    model: YOLO,
    severity_estimator: SeverityEstimator,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: str = "",
) -> dict:
    """
    Run full detection + severity pipeline on a single image file.

    Returns:
        dict with keys:
            - image_path: str
            - image_size: [width, height]
            - detections: list of detection dicts
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    result = detect_frame(
        frame=img,
        model=model,
        severity_estimator=severity_estimator,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )
    result["image_path"] = str(image_path)
    return result


def detect_frame(
    frame: np.ndarray,
    model: YOLO,
    severity_estimator: SeverityEstimator,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: str = "",
) -> dict:
    """
    Run detection + severity pipeline on a single frame (numpy array).

    Returns:
        dict with keys:
            - image_size: [width, height]
            - num_detections: int
            - detections: list of detection dicts
    """
    img_h, img_w = frame.shape[:2]

    # Run YOLO detection (GPU or CPU)
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device if device else None,
        verbose=False,
    )

    detections = []
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        for i in range(len(boxes)):
            # Extract detection info
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(float)
            confidence = float(boxes.conf[i].cpu().numpy())
            class_idx = int(boxes.cls[i].cpu().numpy())
            class_name = result.names[class_idx]

            # Estimate severity
            severity = severity_estimator.estimate(
                bbox=(x1, y1, x2, y2),
                class_name=class_name,
                image_width=img_w,
                image_height=img_h,
            )

            detections.append({
                "class": class_name,
                "class_idx": class_idx,
                "confidence": round(confidence, 4),
                "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
                "severity": severity,
            })

    return {
        "image_size": [img_w, img_h],
        "num_detections": len(detections),
        "detections": detections,
    }


def draw_detections(image: np.ndarray, result: dict) -> np.ndarray:
    """Draw bounding boxes, class labels, and severity on an image."""
    img = image.copy()

    for det in result["detections"]:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        class_name = det["class"]
        confidence = det["confidence"]
        severity = det["severity"]

        # Get colors
        class_color = CLASS_COLORS_BGR.get(class_name, (200, 200, 200))
        severity_color = SeverityEstimator.SEVERITY_COLORS_BGR.get(severity, (255, 255, 255))

        # Draw bounding box (class color)
        thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), class_color, thickness)

        # Draw severity indicator bar on top of bbox
        bar_height = 4
        cv2.rectangle(img, (x1, y1 - bar_height), (x2, y1), severity_color, -1)

        # Label text
        label = f"{class_name} {confidence:.2f} [{severity}]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thickness = 1

        # Text background
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_y = y1 - bar_height - 4
        cv2.rectangle(
            img,
            (x1, text_y - th - 4),
            (x1 + tw + 4, text_y + 2),
            class_color,
            -1,
        )
        cv2.putText(
            img, label,
            (x1 + 2, text_y - 2),
            font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA,
        )

    return img


def process_video(
    source,
    model: YOLO,
    severity_estimator: SeverityEstimator,
    output_dir: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: str = "",
    show_preview: bool = True,
):
    """
    Process a video file or webcam stream with real-time detection.

    Args:
        source: Path to a video file (str) or webcam index (int).
        model: Loaded YOLO model.
        severity_estimator: SeverityEstimator instance.
        output_dir: Directory to save output video and JSON log.
        conf_threshold: Confidence threshold for detections.
        iou_threshold: IoU threshold for NMS.
        device: Device to run inference on.
        show_preview: Whether to display a live OpenCV preview window.
    """
    is_webcam = isinstance(source, int)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0

    source_name = "webcam" if is_webcam else Path(str(source)).stem
    print(f"\n  Source: {source}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Input FPS: {fps:.1f}")
    if total_frames > 0:
        duration_s = total_frames / fps
        print(f"  Total frames: {total_frames} ({duration_s:.1f}s)")

    # Output video writer
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"det_{source_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_count = 0
    total_detections = 0
    class_counts = Counter()
    severity_counts = Counter()
    frame_times = []

    print(f"\n  Processing... (press 'q' in the preview window to stop)\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_start = time.perf_counter()

            # Run detection on this frame
            result = detect_frame(
                frame=frame,
                model=model,
                severity_estimator=severity_estimator,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                device=device,
            )

            # Draw detections on frame
            annotated = draw_detections(frame, result)

            # Calculate real-time FPS
            frame_time = time.perf_counter() - frame_start
            frame_times.append(frame_time)
            current_fps = 1.0 / frame_time if frame_time > 0 else 0

            # ── HUD Overlay ──
            # FPS and detection count
            hud_line1 = f"FPS: {current_fps:.1f} | Detections: {result['num_detections']}"
            cv2.putText(annotated, hud_line1, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Progress bar for video files
            if not is_webcam and total_frames > 0:
                progress_text = f"Frame {frame_count}/{total_frames}"
                cv2.putText(annotated, progress_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            # Severity summary strip at bottom
            if result["detections"]:
                sev_summary = " | ".join(
                    f"{d['class'][:8]}: {d['severity']}" for d in result["detections"][:4]
                )
                cv2.putText(annotated, sev_summary, (10, height - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Write output frame
            writer.write(annotated)

            # Show live preview (non-blocking)
            if show_preview:
                try:
                    cv2.imshow("RoadSafe - Live Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("  [*] Stopped by user.")
                        break
                except cv2.error:
                    pass

            # Accumulate stats
            total_detections += result["num_detections"]
            for d in result["detections"]:
                class_counts[d["class"]] += 1
                severity_counts[d["severity"]] += 1

            # Progress log every 100 frames
            if frame_count % 100 == 0:
                avg_fps = 1.0 / np.mean(frame_times[-100:])
                print(f"    [{frame_count:>5d}{'/' + str(total_frames) if total_frames else ''}] "
                      f"Avg FPS: {avg_fps:.1f} | "
                      f"Detections this frame: {result['num_detections']}")

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    # ── Summary ──
    avg_fps = 1.0 / np.mean(frame_times) if frame_times else 0
    print(f"\n{'=' * 60}")
    print(f"  Video Processing Complete")
    print(f"{'=' * 60}")
    print(f"  Frames processed : {frame_count}")
    print(f"  Average FPS      : {avg_fps:.1f}")
    print(f"  Total detections : {total_detections}")
    if class_counts:
        print(f"  By class         : {dict(class_counts)}")
    if severity_counts:
        print(f"  By severity      : {dict(severity_counts)}")
    print(f"  Output video     : {out_path}")

    # Save JSON audit log
    log = {
        "source": str(source),
        "frames_processed": frame_count,
        "average_fps": round(avg_fps, 1),
        "total_detections": total_detections,
        "class_counts": dict(class_counts),
        "severity_counts": dict(severity_counts),
        "output_video": str(out_path),
    }
    log_path = output_dir / f"video_log_{source_name}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Audit log        : {log_path}")


DEFAULT_MODEL = "weights/best.pt" if Path("weights/best.pt").exists() else "runs/detect/road_damage/weights/best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Road Damage Detection Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image")
    group.add_argument("--image-dir", type=str, help="Path to a directory of images")
    group.add_argument("--video", type=str, help="Path to a video file (mp4, avi, mov, etc.)")
    group.add_argument("--webcam", type=int, nargs="?", const=0, default=None,
                        help="Run live detection on webcam (default index: 0)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Path to trained model weights (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for annotated images/video and JSON")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--device", type=str, default="",
                        help="Device to run on: '' for auto, 'cpu' for CPU, '0' for GPU (default: auto)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save annotated images")
    parser.add_argument("--no-show", action="store_true",
                        help="Disable live preview window (faster processing or headless mode)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Road Damage Detection -- Inference")
    print("=" * 60)

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Train first with train.py."
        )
    model = YOLO(str(model_path))
    severity_estimator = SeverityEstimator()

    print(f"  Model: {model_path}")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  IoU threshold: {args.iou}")

    # ── Video / Webcam Mode ──────────────────────────────────────
    if args.video is not None or args.webcam is not None:
        source = args.webcam if args.webcam is not None else args.video
        mode = "Webcam" if args.webcam is not None else "Video"
        print(f"  Mode: {mode}")
        process_video(
            source=source,
            model=model,
            severity_estimator=severity_estimator,
            output_dir=Path(args.output),
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            show_preview=not args.no_show,
        )
        return

    # ── Image Mode ───────────────────────────────────────────────
    # Gather image paths
    if args.image:
        image_paths = [Path(args.image)]
    else:
        img_dir = Path(args.image_dir)
        image_paths = sorted(
            list(img_dir.glob("*.jpg")) +
            list(img_dir.glob("*.jpeg")) +
            list(img_dir.glob("*.png"))
        )

    print(f"  Images to process: {len(image_paths)}")
    print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    skipped = 0

    for i, img_path in enumerate(image_paths):
        print(f"  [{i+1}/{len(image_paths)}] {img_path.name}", end="")

        try:
            result = detect_road_damage(
                image_path=img_path,
                model=model,
                severity_estimator=severity_estimator,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                device=args.device,
            )

            print(f"  -> {result['num_detections']} detections", end="")
            if result["detections"]:
                classes = [d["class"] for d in result["detections"]]
                severities = [d["severity"] for d in result["detections"]]
                print(f"  | Classes: {', '.join(set(classes))}", end="")
                print(f"  | Severities: {', '.join(set(severities))}", end="")
            print()

            # Save annotated image
            if not args.no_save:
                img = cv2.imread(str(img_path))
                annotated = draw_detections(img, result)
                out_img_path = output_dir / f"det_{img_path.name}"
                cv2.imwrite(str(out_img_path), annotated)

            all_results.append(result)

        except Exception as e:
            skipped += 1
            print(f"  [ERROR] Skipped -- {e}")
            continue

    # Save JSON results
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[OK] Done! Results saved to '{output_dir}/'")
    print(f"  [*] JSON results: {json_path}")
    if not args.no_save:
        print(f"  [*] Annotated images: {output_dir}/det_*.jpg")
    if skipped > 0:
        print(f"  [!] Skipped {skipped} image(s) due to errors")

    # Print summary
    total_dets = sum(r["num_detections"] for r in all_results)
    class_counts = Counter()
    severity_counts = Counter()
    for r in all_results:
        for d in r["detections"]:
            class_counts[d["class"]] += 1
            severity_counts[d["severity"]] += 1

    print(f"\n[*] Summary:")
    print(f"  Total detections: {total_dets}")
    print(f"  By class: {dict(class_counts)}")
    print(f"  By severity: {dict(severity_counts)}")


if __name__ == "__main__":
    main()
