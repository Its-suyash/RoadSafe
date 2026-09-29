"""
Training Script
===============
Fine-tunes YOLOv8-nano on the 4-class road damage dataset.
Supports resuming from checkpoints and configurable hyperparameters.

Usage:
    python train.py                        # Default: 100 epochs
    python train.py --epochs 50 --batch 8  # Custom settings
    python train.py --resume               # Resume last run
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Road Damage Dataset")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Pretrained model to start from (default: yolov8n.pt)")
    parser.add_argument("--data", type=str, default="dataset.yaml",
                        help="Path to dataset YAML config")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (default: 15)")
    parser.add_argument("--device", type=str, default="",
                        help="Device: '' for auto, '0' for GPU 0, 'cpu' for CPU")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--name", type=str, default="road_damage",
                        help="Run name for saving results")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Road Damage Detection -- YOLOv8 Training")
    print("=" * 60)
    print(f"  Model:     {args.model}")
    print(f"  Dataset:   {args.data}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Patience:  {args.patience}")
    print(f"  Device:    {args.device or 'auto'}")
    print()

    # Verify dataset exists
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found at '{data_path}'. "
            "Run data_preparation.py first."
        )

    dataset_dir = Path("dataset")
    for split in ["train", "val"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            raise FileNotFoundError(
                f"Dataset split '{split}' not found at '{img_dir}'. "
                "Run data_preparation.py first."
            )
        n_imgs = len(list(img_dir.glob("*.jpg")))
        n_lbls = len(list(lbl_dir.glob("*.txt")))
        print(f"  {split}: {n_imgs} images, {n_lbls} labels")

    print()

    # Load model
    if args.resume:
        # Resume from last checkpoint
        last_ckpt = Path("runs/detect") / args.name / "weights/last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found at '{last_ckpt}'")
        model = YOLO(str(last_ckpt))
        print(f"[*] Resuming from {last_ckpt}")
    else:
        model = YOLO(args.model)
        print(f"[*] Loaded pretrained model: {args.model}")

    # Train
    print("\n[*] Starting training...\n")
    results = model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device if args.device else None,
        name=args.name,
        exist_ok=True,
        resume=args.resume,

        # Augmentation (aggressive for small dataset with class imbalance)
        hsv_h=0.015,        # HSV-Hue augmentation
        hsv_s=0.7,          # HSV-Saturation augmentation
        hsv_v=0.4,          # HSV-Value augmentation
        degrees=10.0,       # Rotation (+/- degrees)
        translate=0.1,      # Translation (+/- fraction)
        scale=0.5,          # Scale (+/- gain)
        shear=2.0,          # Shear (+/- degrees)
        flipud=0.0,         # Disabled: dashcam footage is never upside-down
        fliplr=0.5,         # Horizontal flip probability
        mosaic=1.0,         # Mosaic augmentation probability
        mixup=0.1,          # Mixup augmentation probability

        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,           # Final LR = lr0 * lrf (cosine schedule)
        weight_decay=0.0005,
        warmup_epochs=3.0,

        # Logging
        verbose=True,
        plots=True,
        save=True,
        save_period=-1,     # Save only best and last
    )

    # Summary
    best_model = Path("runs/detect") / args.name / "weights/best.pt"
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Best model saved at: {best_model}")
    if best_model.exists():
        size_mb = best_model.stat().st_size / (1024 * 1024)
        print(f"  Model size: {size_mb:.1f} MB")
    print(f"  Results saved in: runs/detect/{args.name}/")
    print()

    return results


if __name__ == "__main__":
    main()
