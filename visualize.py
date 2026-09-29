"""
Visualization Utilities
=======================
Functions for visualizing detections, training results,
and dataset statistics.

Usage:
    python visualize.py --results results/results.json --show-grid 9
    python visualize.py --dataset dataset/ --class-distribution
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from severity_estimator import SeverityEstimator


# Consistent color scheme (RGB normalized for matplotlib)
CLASS_COLORS = {
    "Pothole":           "#FF6400",   # Orange-red
    "LongitudinalCrack": "#0096FF",   # Blue
    "AlligatorCrack":    "#00C8C8",   # Teal
    "TransverseCrack":   "#C800C8",   # Magenta
}

SEVERITY_COLORS = {
    "Low":    "#00C800",   # Green
    "Medium": "#FFA500",   # Orange
    "High":   "#FF0000",   # Red
}


def plot_detection_grid(
    results_json: str,
    image_dir: str = None,
    n_images: int = 9,
    save_path: str = None,
):
    """Plot a grid of detected images with annotations."""
    with open(results_json) as f:
        results = json.load(f)

    # Filter to images with detections
    with_dets = [r for r in results if r["num_detections"] > 0]
    if not with_dets:
        print("No detections found in results.")
        return

    # Select subset
    selected = with_dets[:n_images]
    n = len(selected)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    severity_est = SeverityEstimator()

    for idx, result in enumerate(selected):
        ax = axes[idx]

        # Load annotated image if available, else original
        img_path = result["image_path"]
        is_annotated = False
        if image_dir:
            stem = Path(img_path).stem
            annotated_path = Path(image_dir) / f"det_{stem}.jpg"
            if annotated_path.exists():
                img_path = str(annotated_path)
                is_annotated = True

        img = cv2.imread(str(img_path))
        if img is None:
            ax.text(0.5, 0.5, "Image not found", ha="center", va="center")
            ax.set_title(Path(result["image_path"]).name)
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)

        # Draw detections with matplotlib only if image is not pre-annotated
        if not is_annotated:
            for det in result["detections"]:
                x1, y1, x2, y2 = det["bbox"]
                w, h = x2 - x1, y2 - y1
                color = CLASS_COLORS.get(det["class"], "#FFFFFF")

                rect = patches.Rectangle(
                    (x1, y1), w, h,
                    linewidth=2, edgecolor=color, facecolor="none"
                )
                ax.add_patch(rect)

                label = f"{det['class'][:4]} {det['confidence']:.2f} [{det['severity']}]"
                ax.text(
                    x1, y1 - 5, label,
                    fontsize=7, color="white", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
                )

        title = f"{Path(result['image_path']).name} ({result['num_detections']} det)"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved grid to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_class_distribution(dataset_dir: str, save_path: str = None):
    """Plot class distribution across train/val/test splits."""
    dataset_path = Path(dataset_dir)
    class_names = {0: "Pothole", 1: "LongCrack", 2: "AlligCrack", 3: "TransCrack"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    splits = ["train", "val", "test"]

    for ax, split in zip(axes, splits):
        label_dir = dataset_path / split / "labels"
        if not label_dir.exists():
            ax.set_title(f"{split} -- NOT FOUND")
            continue

        class_counter = Counter()
        for label_file in label_dir.glob("*.txt"):
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_idx = int(parts[0])
                        class_counter[cls_idx] += 1

        indices = sorted(class_counter.keys())
        names = [class_names.get(i, f"cls_{i}") for i in indices]
        counts = [class_counter[i] for i in indices]
        colors = [list(CLASS_COLORS.values())[i] for i in indices]

        bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{split.upper()} ({sum(counts)} objects)", fontsize=12, weight="bold")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=9,
            )

    plt.suptitle("Class Distribution by Split", fontsize=14, weight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved class distribution to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_severity_distribution(results_json: str, save_path: str = None):
    """Plot severity distribution from inference results."""
    with open(results_json) as f:
        results = json.load(f)

    class_severity = {}
    for r in results:
        for d in r["detections"]:
            cls = d["class"]
            sev = d["severity"]
            if cls not in class_severity:
                class_severity[cls] = Counter()
            class_severity[cls][sev] += 1

    if not class_severity:
        print("No detections found.")
        return

    classes = list(class_severity.keys())
    severities = ["Low", "Medium", "High"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.25

    for i, sev in enumerate(severities):
        counts = [class_severity[c].get(sev, 0) for c in classes]
        bars = ax.bar(
            x + i * width, counts, width,
            label=sev, color=SEVERITY_COLORS[sev],
            edgecolor="white", linewidth=0.5,
        )
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(count), ha="center", fontsize=8,
                )

    ax.set_xlabel("Defect Class")
    ax.set_ylabel("Count")
    ax.set_title("Severity Distribution by Class", fontsize=14, weight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=15)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved severity distribution to: {save_path}")
    else:
        plt.show()
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualization Utilities")
    parser.add_argument("--results", type=str, help="Path to results.json")
    parser.add_argument("--image-dir", type=str, help="Directory with annotated images")
    parser.add_argument("--dataset", type=str, help="Path to YOLO dataset directory")
    parser.add_argument("--show-grid", type=int, default=0,
                        help="Show N detection results in a grid")
    parser.add_argument("--class-distribution", action="store_true",
                        help="Plot class distribution")
    parser.add_argument("--severity-distribution", action="store_true",
                        help="Plot severity distribution")
    parser.add_argument("--save-dir", type=str, default="plots",
                        help="Directory to save plots")
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.show_grid and args.results:
        print("[*] Generating detection grid...")
        plot_detection_grid(
            args.results,
            image_dir=args.image_dir,
            n_images=args.show_grid,
            save_path=str(save_dir / "detection_grid.png"),
        )

    if args.class_distribution and args.dataset:
        print("[*] Generating class distribution...")
        plot_class_distribution(
            args.dataset,
            save_path=str(save_dir / "class_distribution.png"),
        )

    if args.severity_distribution and args.results:
        print("[*] Generating severity distribution...")
        plot_severity_distribution(
            args.results,
            save_path=str(save_dir / "severity_distribution.png"),
        )


if __name__ == "__main__":
    main()
