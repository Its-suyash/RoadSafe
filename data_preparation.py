"""
Data Preparation Script
=======================
Converts RDD2022 Pascal VOC XML annotations to YOLO format,
maps class codes to 4 target categories, and creates
train/val/test splits (80/10/10, stratified by dominant class).
"""

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# ── Configuration ──────────────────────────────────────────────
SRC_IMG_DIR = Path("train/images")
SRC_ANN_DIR = Path("train/annotations/xmls")
DST_ROOT = Path("dataset")

# RDD2022 code -> target class index
# 0 = Pothole, 1 = LongitudinalCrack, 2 = AlligatorCrack, 3 = TransverseCrack
CLASS_MAP = {
    "D40": 0,   # Pothole
    "D44": 0,   # Pothole (variant)
    "D43": 0,   # Pothole (variant)
    "D00": 1,   # Longitudinal Crack
    "D01": 1,   # Longitudinal Crack (wheel-mark)
    "D20": 2,   # Alligator Crack
    "D10": 3,   # Transverse Crack
    "D11": 3,   # Transverse Crack (wheel-mark)
}

CLASS_NAMES = {0: "Pothole", 1: "LongitudinalCrack", 2: "AlligatorCrack", 3: "TransverseCrack"}
EXCLUDED_CLASSES = {"D50"}  # 8 samples -- doesn't map to any of the 4 required classes

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
RANDOM_SEED = 42


def parse_voc_xml(xml_path: Path) -> list[dict]:
    """Parse a Pascal VOC XML annotation file and return a list of objects."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text

        if class_name in EXCLUDED_CLASSES:
            continue
        if class_name not in CLASS_MAP:
            print(f"  [WARN] Unknown class '{class_name}' in {xml_path.name}, skipping")
            continue

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Convert to YOLO format: x_center, y_center, width, height (all normalized 0-1)
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        objects.append({
            "class_idx": CLASS_MAP[class_name],
            "class_name": class_name,
            "x_center": x_center,
            "y_center": y_center,
            "width": w,
            "height": h,
        })

    return objects


def write_yolo_label(objects: list[dict], label_path: Path):
    """Write YOLO-format label file."""
    with open(label_path, "w") as f:
        for obj in objects:
            f.write(f"{obj['class_idx']} {obj['x_center']:.6f} {obj['y_center']:.6f} "
                    f"{obj['width']:.6f} {obj['height']:.6f}\n")


def get_dominant_class(objects: list[dict]) -> int:
    """Return the most frequent class index for stratification."""
    if not objects:
        return 0  # fallback
    counter = Counter(obj["class_idx"] for obj in objects)
    return counter.most_common(1)[0][0]


def parse_args():
    parser = argparse.ArgumentParser(description="RDD2022 Dataset Preparation for YOLOv8")
    parser.add_argument("--raw-dir", type=str, default="train",
                        help="Path to unzipped raw RDD2022 folder containing images/ and annotations/")
    parser.add_argument("--output-dir", type=str, default="dataset",
                        help="Destination directory for YOLO format dataset")
    return parser.parse_args()


def main():
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    src_img_dir = raw_dir / "images"
    src_ann_dir = raw_dir / "annotations" / "xmls"
    dst_root = Path(args.output_dir)

    print("=" * 60)
    print("Road Damage Dataset -- Data Preparation")
    print("=" * 60)
    print(f"  Raw Source Dir : {raw_dir}")
    print(f"  Target YOLO Dir: {dst_root}")

    if not src_img_dir.exists() or not src_ann_dir.exists():
        print(f"\n[ERROR] Source directories not found!")
        print(f"  Expected images at:      {src_img_dir}")
        print(f"  Expected annotations at: {src_ann_dir}")
        print("Please download and unzip the RDD2022 India dataset into the specified directory.")
        return

    # ── Step 1: Parse all annotations ──
    print("\n[*] Parsing annotations...")
    all_data = {}  # stem -> list of objects
    class_counter = Counter()

    xml_files = sorted(src_ann_dir.glob("*.xml"))
    for xml_path in xml_files:
        stem = xml_path.stem
        objects = parse_voc_xml(xml_path)
        if objects:  # Only keep images that have at least one valid annotation
            all_data[stem] = objects
            for obj in objects:
                class_counter[obj["class_idx"]] += 1

    print(f"  [OK] Parsed {len(all_data)} images with valid annotations")
    print(f"\n  Class distribution (after mapping):")
    for idx in sorted(class_counter.keys()):
        print(f"    [{idx}] {CLASS_NAMES[idx]}: {class_counter[idx]}")

    # ── Step 2: Stratified split ──
    print(f"\n[*] Splitting dataset (train={SPLIT_RATIOS['train']:.0%}, "
          f"val={SPLIT_RATIOS['val']:.0%}, test={SPLIT_RATIOS['test']:.0%})...")

    stems = sorted(all_data.keys())
    dominant_classes = [get_dominant_class(all_data[s]) for s in stems]

    # First split: train vs (val+test)
    train_stems, temp_stems, train_labels, temp_labels = train_test_split(
        stems, dominant_classes,
        test_size=(SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"]),
        stratify=dominant_classes,
        random_state=RANDOM_SEED,
    )

    # Second split: val vs test (50/50 of the remaining 20%)
    # Use try/except because rare classes (e.g. TransverseCrack with only ~6
    # samples in temp) may have too few members for stratification.
    try:
        val_stems, test_stems = train_test_split(
            temp_stems,
            test_size=0.5,
            stratify=temp_labels,
            random_state=RANDOM_SEED,
        )
    except ValueError:
        print("  (Note: falling back to non-stratified val/test split due to rare classes)")
        val_stems, test_stems = train_test_split(
            temp_stems,
            test_size=0.5,
            random_state=RANDOM_SEED,
        )

    splits = {"train": train_stems, "val": val_stems, "test": test_stems}

    for split_name, split_stems in splits.items():
        split_classes = Counter(get_dominant_class(all_data[s]) for s in split_stems)
        print(f"  {split_name:5s}: {len(split_stems):4d} images | "
              + " | ".join(f"{CLASS_NAMES[k]}={v}" for k, v in sorted(split_classes.items())))

    # ── Step 3: Create directory structure and copy files ──
    print(f"\n[*] Creating YOLO dataset structure at '{dst_root}'...")

    # Clean existing
    if dst_root.exists():
        shutil.rmtree(dst_root)

    for split_name, split_stems in splits.items():
        img_dir = dst_root / split_name / "images"
        lbl_dir = dst_root / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for stem in split_stems:
            # Copy image
            src_img = src_img_dir / f"{stem}.jpg"
            dst_img = img_dir / f"{stem}.jpg"
            shutil.copy2(src_img, dst_img)

            # Write YOLO label
            dst_lbl = lbl_dir / f"{stem}.txt"
            write_yolo_label(all_data[stem], dst_lbl)

    # ── Step 4: Verify ──
    print("\n[*] Verification:")
    for split_name in splits:
        img_count = len(list((dst_root / split_name / "images").glob("*.jpg")))
        lbl_count = len(list((dst_root / split_name / "labels").glob("*.txt")))
        print(f"  {split_name:5s}: {img_count} images, {lbl_count} labels"
              + (" [OK]" if img_count == lbl_count else " [MISMATCH]"))

    print("\n[OK] Data preparation complete!")


if __name__ == "__main__":
    main()
