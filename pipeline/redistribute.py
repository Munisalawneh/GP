"""
Cross-Class Redistribution — copies images to other class folders
based on what objects Roboflow actually detected in them.

Example: if apple_002.png has a bed detected in it, this script copies
apple_002.png → bed_201.png (with its full label) so the bed class
gets extra training data.

Also fixes:
  - Resizes images with wrong dimensions to 1408x768
  - Re-labels images where target class was not detected (no_target_class)
  - Re-labels empty label files
"""

import os
import sys
import shutil
import base64
import requests
from collections import defaultdict

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    CLASSES, IMAGES_DIR, LABELS_DIR, ROBOFLOW_API_KEY,
    ROBOFLOW_MODEL_URL, CLASS_NAME_TO_ID,
)
from generate_images import sanitize_name
from label_images import label_image, save_label

TARGET_SIZE = (1408, 768)


def get_next_index(class_name: str) -> int:
    """Find the next available image index for a class."""
    safe = sanitize_name(class_name)
    img_dir = os.path.join(IMAGES_DIR, safe)
    if not os.path.isdir(img_dir):
        return 1
    existing = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    if not existing:
        return 1
    indices = []
    for f in existing:
        try:
            idx = int(f.replace(safe + "_", "").replace(".png", ""))
            indices.append(idx)
        except ValueError:
            pass
    return max(indices) + 1 if indices else 1


def parse_label_file(label_path: str) -> list[tuple[int, str]]:
    """Parse a YOLO label file. Returns list of (class_id, full_line)."""
    results = []
    if not os.path.exists(label_path):
        return results
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                try:
                    cid = int(parts[0])
                    results.append((cid, line))
                except ValueError:
                    pass
    return results


def resize_if_needed(img_path: str) -> bool:
    """Resize image to TARGET_SIZE if dimensions are wrong. Returns True if resized."""
    img = Image.open(img_path)
    if img.size != TARGET_SIZE:
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        img.save(img_path, optimize=True)
        return True
    return False


def redistribute():
    """Main redistribution logic."""
    print("=" * 70)
    print("  STEP 1: Fix wrong-size images")
    print("=" * 70)

    resized_count = 0
    for cid, cname in CLASSES.items():
        safe = sanitize_name(cname)
        img_dir = os.path.join(IMAGES_DIR, safe)
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            if not fname.endswith(".png"):
                continue
            img_path = os.path.join(img_dir, fname)
            if resize_if_needed(img_path):
                resized_count += 1

    print(f"  Resized {resized_count} images to {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    print(f"\n{'=' * 70}")
    print("  STEP 2: Re-label empty and no-target-class images")
    print("=" * 70)

    relabel_count = 0
    relabel_success = 0
    relabel_still_empty = 0
    relabel_still_no_target = 0

    for cid, cname in CLASSES.items():
        safe = sanitize_name(cname)
        img_dir = os.path.join(IMAGES_DIR, safe)
        lbl_dir = os.path.join(LABELS_DIR, safe)
        if not os.path.isdir(img_dir):
            continue

        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith(".png"):
                continue
            basename = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, basename + ".txt")

            # Check if this image needs re-labeling
            detections = parse_label_file(lbl_path)
            is_empty = len(detections) == 0
            has_target = any(d[0] == cid for d in detections)

            if not is_empty and has_target:
                continue  # This image is fine

            # Re-label via Roboflow
            relabel_count += 1
            try:
                yolo_lines, _ = label_image(img_path)
                if yolo_lines:
                    save_label(yolo_lines, lbl_path)
                    # Check if target class now detected
                    new_detections = parse_label_file(lbl_path)
                    new_has_target = any(d[0] == cid for d in new_detections)
                    if new_has_target:
                        relabel_success += 1
                        print(f"  ✓ {cname}/{fname} — target class now detected")
                    else:
                        relabel_still_no_target += 1
                        print(f"  ~ {cname}/{fname} — re-labeled but still no target class")
                else:
                    relabel_still_empty += 1
                    print(f"  ✗ {cname}/{fname} — still empty after re-label")
            except Exception as e:
                print(f"  ✗ {cname}/{fname} — error: {e}")
                relabel_still_empty += 1

    print(f"\n  Re-label summary: {relabel_count} images processed")
    print(f"    Target class recovered: {relabel_success}")
    print(f"    Still no target class:  {relabel_still_no_target}")
    print(f"    Still empty:            {relabel_still_empty}")

    print(f"\n{'=' * 70}")
    print("  STEP 3: Cross-class redistribution")
    print("=" * 70)

    # Track next available index for each class
    next_idx = {}
    for cid, cname in CLASSES.items():
        next_idx[cid] = get_next_index(cname)

    # Track copies to avoid duplicates (source_path -> set of target_classes)
    copied = defaultdict(set)
    copy_count = defaultdict(int)
    total_copies = 0

    for src_cid, src_cname in CLASSES.items():
        src_safe = sanitize_name(src_cname)
        src_img_dir = os.path.join(IMAGES_DIR, src_safe)
        src_lbl_dir = os.path.join(LABELS_DIR, src_safe)

        if not os.path.isdir(src_img_dir):
            continue

        for fname in sorted(os.listdir(src_img_dir)):
            if not fname.endswith(".png"):
                continue
            basename = os.path.splitext(fname)[0]
            src_img = os.path.join(src_img_dir, fname)
            src_lbl = os.path.join(src_lbl_dir, basename + ".txt")

            if not os.path.exists(src_lbl):
                continue

            detections = parse_label_file(src_lbl)
            if not detections:
                continue

            # Find which OTHER classes (40-59) are detected in this image
            detected_classes = set(d[0] for d in detections)
            other_classes = detected_classes - {src_cid}
            # Only redistribute to classes in our 40-59 range
            other_classes = {c for c in other_classes if c in CLASSES}

            for target_cid in other_classes:
                # Avoid copying the same source image to the same target class twice
                copy_key = (src_img, target_cid)
                if copy_key in copied:
                    continue
                copied[copy_key] = True

                target_cname = CLASSES[target_cid]
                target_safe = sanitize_name(target_cname)
                target_img_dir = os.path.join(IMAGES_DIR, target_safe)
                target_lbl_dir = os.path.join(LABELS_DIR, target_safe)

                os.makedirs(target_img_dir, exist_ok=True)
                os.makedirs(target_lbl_dir, exist_ok=True)

                # Generate target filename
                idx = next_idx[target_cid]
                target_fname = f"{target_safe}_{idx:03d}.png"
                target_lbl_fname = f"{target_safe}_{idx:03d}.txt"
                next_idx[target_cid] = idx + 1

                # Copy image
                shutil.copy2(src_img, os.path.join(target_img_dir, target_fname))
                # Copy full label (all detections — this is correct for YOLO)
                shutil.copy2(src_lbl, os.path.join(target_lbl_dir, target_lbl_fname))

                copy_count[target_cid] += 1
                total_copies += 1

        print(f"  Processed {src_cname} — found cross-class detections")

    print(f"\n  Cross-class redistribution summary:")
    print(f"  {'Class':<15} {'Original':>8} {'Added':>6} {'Total':>6}")
    print(f"  {'-'*37}")
    for cid, cname in CLASSES.items():
        safe = sanitize_name(cname)
        img_dir = os.path.join(IMAGES_DIR, safe)
        total = len([f for f in os.listdir(img_dir) if f.endswith(".png")]) if os.path.isdir(img_dir) else 0
        added = copy_count.get(cid, 0)
        original = total - added
        print(f"  {cname:<15} {original:>8} {added:>6} {total:>6}")

    print(f"\n  Total images copied: {total_copies}")
    print("=" * 70)


if __name__ == "__main__":
    redistribute()
