"""
Evaluation Script
=================
Comprehensive evaluation of the trained road damage detection model.
Computes standard object detection metrics and generates visualizations.

Usage:
    python evaluate.py
    python evaluate.py --model runs/detect/road_damage/weights/best.pt
    python evaluate.py --model best.pt --save-plots
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


DEFAULT_MODEL = "weights/best.pt" if Path("weights/best.pt").exists() else "runs/detect/road_damage/weights/best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Road Damage Detection Model")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Path to trained model (default: {DEFAULT_MODEL})")
    parser.add_argument("--data", type=str, default="dataset.yaml",
                        help="Path to dataset YAML")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for evaluation")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save evaluation plots")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run inference speed benchmark")
    return parser.parse_args()


def run_yolo_validation(model: YOLO, data_yaml: str, split: str, imgsz: int, conf: float):
    """Run YOLO's built-in validation to get mAP metrics."""
    print("\n[*] Running YOLO validation...")
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        conf=conf,
        verbose=True,
        plots=True,
    )
    return results


def benchmark_inference_speed(model: YOLO, dataset_dir: Path, split: str, n_images: int = 50):
    """Benchmark inference speed on a set of images."""
    print(f"\n[*]  Benchmarking inference speed on {n_images} images...")

    img_dir = dataset_dir / split / "images"
    image_paths = sorted(list(img_dir.glob("*.jpg")))[:n_images]

    if not image_paths:
        print("  No images found for benchmarking.")
        return {}

    # Warm up
    for _ in range(3):
        model.predict(str(image_paths[0]), verbose=False)

    # Benchmark
    times = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        start = time.perf_counter()
        model.predict(img, verbose=False)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    stats = {
        "n_images": len(times),
        "mean_ms": float(np.mean(times) * 1000),
        "std_ms": float(np.std(times) * 1000),
        "min_ms": float(np.min(times) * 1000),
        "max_ms": float(np.max(times) * 1000),
        "fps": float(1.0 / np.mean(times)),
    }

    print(f"  Mean inference time: {stats['mean_ms']:.1f} +/- {stats['std_ms']:.1f} ms")
    print(f"  FPS: {stats['fps']:.1f}")
    print(f"  Min/Max: {stats['min_ms']:.1f} / {stats['max_ms']:.1f} ms")

    return stats


def analyze_severity_distribution(model: YOLO, dataset_dir: Path, split: str, conf: float):
    """Analyze severity distribution on the evaluation split."""
    print(f"\n[*] Analyzing severity distribution on {split} set...")

    severity_est = SeverityEstimator()
    img_dir = dataset_dir / split / "images"
    image_paths = sorted(list(img_dir.glob("*.jpg")))

    class_severity = {}  # class_name -> Counter of severities
    total_dets = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        results = model.predict(img, conf=conf, verbose=False)

        if results and len(results) > 0:
            result = results[0]
            for i in range(len(result.boxes)):
                x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(float)
                cls_idx = int(result.boxes.cls[i].cpu().numpy())
                cls_name = result.names[cls_idx]

                severity = severity_est.estimate(
                    bbox=(x1, y1, x2, y2),
                    class_name=cls_name,
                    image_width=img_w,
                    image_height=img_h,
                )

                if cls_name not in class_severity:
                    class_severity[cls_name] = Counter()
                class_severity[cls_name][severity] += 1
                total_dets += 1

    print(f"  Total detections: {total_dets}")
    for cls_name in sorted(class_severity.keys()):
        counts = class_severity[cls_name]
        total = sum(counts.values())
        print(f"  {cls_name}: Low={counts.get('Low', 0)}, "
              f"Medium={counts.get('Medium', 0)}, "
              f"High={counts.get('High', 0)} (total={total})")

    return class_severity


def main():
    args = parse_args()

    print("=" * 60)
    print("Road Damage Detection -- Model Evaluation")
    print("=" * 60)

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Train first with train.py."
        )

    model = YOLO(str(model_path))
    print(f"  Model: {model_path}")

    # Model size
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  Model size: {model_size_mb:.2f} MB")

    # Dataset
    data_yaml = str(Path(args.data).resolve())
    dataset_dir = Path("dataset")
    print(f"  Dataset: {data_yaml}")
    print(f"  Eval split: {args.split}")

    # Count images
    img_dir = dataset_dir / args.split / "images"
    if not img_dir.exists():
        print(f"\n[INFO] Dataset split '{args.split}' not found at '{img_dir}'.")
        print("Full validation requires the RDD2022 dataset (see README.md to download).")
        print("Displaying pre-computed evaluation metrics from 'evaluation_report.json':\n")
        report_file = Path("evaluation_report.json")
        if report_file.exists():
            with open(report_file) as f:
                saved = json.load(f)
            print(f"  mAP@0.5      : {saved.get('mAP50', 0):.4f}")
            print(f"  mAP@0.5:0.95 : {saved.get('mAP50_95', 0):.4f}")
            if "per_class_ap50" in saved:
                print("  Per-Class AP@0.5:")
                for c, ap in saved["per_class_ap50"].items():
                    print(f"    - {c:<18}: {ap:.4f}")
            if "speed" in saved:
                print(f"  Inference Speed: {saved['speed'].get('fps', 0):.1f} FPS ({saved['speed'].get('mean_ms', 0):.1f} ms/frame)")
        return

    n_images = len(list(img_dir.glob("*.jpg")))
    print(f"  Images in {args.split}: {n_images}")

    # ── 1. YOLO Validation (mAP, Precision, Recall) ──
    val_results = run_yolo_validation(
        model, data_yaml, args.split, args.imgsz, args.conf
    )

    # ── 2. Inference Speed Benchmark ──
    if args.benchmark:
        speed_stats = benchmark_inference_speed(
            model, dataset_dir, args.split
        )

    # ── 3. Severity Distribution ──
    severity_dist = analyze_severity_distribution(
        model, dataset_dir, args.split, args.conf
    )

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    # mAP results
    if val_results:
        print(f"\n  [*] Detection Metrics ({args.split} set):")
        print(f"     mAP@0.5:      {val_results.box.map50:.4f}")
        print(f"     mAP@0.5:0.95: {val_results.box.map:.4f}")

        # Per-Class metrics
        class_names = model.names
        class_metrics = {}
        if hasattr(val_results.box, 'ap50') and val_results.box.ap50 is not None:
            print(f"\n  [*] Per-Class AP@0.5:")
            class_indices = getattr(val_results.box, 'ap_class_index', range(len(val_results.box.ap50)))
            for idx, ap in zip(class_indices, val_results.box.ap50):
                name = class_names.get(int(idx), f"class_{idx}")
                class_metrics[name] = round(float(ap), 4)
                print(f"     {name:20s}: {ap:.4f}")

    print(f"\n  [*] Model Info:")
    print(f"     Size: {model_size_mb:.2f} MB")

    if args.benchmark and speed_stats:
        print(f"\n  [*]  Inference Speed:")
        print(f"     {speed_stats['fps']:.1f} FPS ({speed_stats['mean_ms']:.1f} ms/image)")

    # Save evaluation report
    report = {
        "model": str(model_path),
        "model_size_mb": round(model_size_mb, 2),
        "split": args.split,
        "n_images": n_images,
        "conf_threshold": args.conf,
    }

    if val_results:
        report["mAP50"] = float(val_results.box.map50)
        report["mAP50_95"] = float(val_results.box.map)
        report["per_class_ap50"] = class_metrics

    if args.benchmark and speed_stats:
        report["speed"] = speed_stats

    report_path = Path("evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  [*] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
