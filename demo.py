"""
RoadSafe - Interactive Quickstart Demo
======================================
Runs an end-to-end demonstration on a sample road damage image,
performing detection, classification, and severity estimation.

Usage:
    python demo.py
    python demo.py --image path/to/image.jpg
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import argparse
from pathlib import Path
import cv2

from ultralytics import YOLO
from severity_estimator import SeverityEstimator
from inference import detect_road_damage, draw_detections

DEFAULT_MODEL = "weights/best.pt" if Path("weights/best.pt").exists() else "runs/detect/road_damage/weights/best.pt"


def run_demo(image_path: Path, model_path: Path, output_dir: Path):
    print("=" * 65)
    print("  RoadSafe: Road Damage Detection & Severity Assessment Demo")
    print("=" * 65)
    print(f"  Model Weights : {model_path}")
    print(f"  Input Image   : {image_path}")
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  Execution Device: {device_name}")
    print("-" * 65)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at '{model_path}'. Please check weights directory.")
    if not image_path.exists():
        raise FileNotFoundError(f"Sample image not found at '{image_path}'.")

    model = YOLO(str(model_path))
    severity_estimator = SeverityEstimator()

    output_dir.mkdir(parents=True, exist_ok=True)

    result = detect_road_damage(
        image_path=image_path,
        model=model,
        severity_estimator=severity_estimator,
        conf_threshold=0.25,
        iou_threshold=0.45,
        device="" if torch.cuda.is_available() else "cpu",
    )

    num_dets = result["num_detections"]
    print(f"\n[*] Detection Complete: Found {num_dets} defect(s)\n")

    for idx, det in enumerate(result["detections"], 1):
        cls_name = det["class"]
        conf = det["confidence"]
        severity = det["severity"]
        bbox = det["bbox"]
        print(f"  [{idx}] Class: {cls_name:<18} | Confidence: {conf:.2%} | Severity: {severity:<6} | BBox: {bbox}")

    # Draw and save
    img = cv2.imread(str(image_path))
    annotated = draw_detections(img, result)
    out_file = output_dir / f"demo_output_{image_path.name}"
    cv2.imwrite(str(out_file), annotated)

    print("-" * 65)
    print(f"[OK] Annotated output saved to: {out_file}")
    print("=" * 65)


def parse_args():
    parser = argparse.ArgumentParser(description="RoadSafe Quickstart Demo")
    default_img = Path("dataset/test/images/India_009605.jpg")
    if not default_img.exists():
        default_img = Path("sample_images/India_009605.jpg")
    parser.add_argument("--image", type=str, default=str(default_img),
                        help="Path to test road image")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to trained YOLOv8 model")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(Path(args.image), Path(args.model), Path(args.output))
