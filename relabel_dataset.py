"""
Relabel all images in dataset/images/ using the local yolo26x.pt model.

Saves ALL detected bounding boxes (all 80 COCO classes, not just the target
class) to dataset/labels/{class_name}/{image_stem}.txt in YOLO format:
    <class_id> <x_center> <y_center> <width> <height>   (all normalized 0–1)

If an image is unreadable or inference fails for that file, the script logs the
failure and continues so long relabel runs can be resumed safely.
"""

import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
MODEL_PATH  = ROOT_DIR / "yolo26x.pt"
IMAGES_DIR  = ROOT_DIR / "dataset" / "images"
LABELS_DIR  = ROOT_DIR / "dataset" / "labels"
FAILED_LOG  = ROOT_DIR / "dataset" / "relabel_failed_images.txt"
DONE_LOG    = ROOT_DIR / "dataset" / "relabel_completed_images.txt"
CONF        = 0.25          # confidence threshold
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relabel a dataset with yolo26x.pt")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT_DIR / "dataset",
        help="Dataset root containing images/ and labels/ subfolders",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONF,
        help="YOLO confidence threshold",
    )
    return parser.parse_args()


def verify_image(image_path: Path) -> None:
    """Raise if an image file is truncated, corrupt, or otherwise unreadable."""
    with Image.open(image_path) as image:
        image.verify()


def append_failure(failed_log: Path, image_path: Path, reason: str) -> None:
    """Record a skipped image so it can be reviewed later."""
    failed_log.parent.mkdir(parents=True, exist_ok=True)
    with failed_log.open("a", encoding="utf-8") as handle:
        handle.write(f"{image_path}\t{reason}\n")


def load_completed(done_log: Path) -> set[str]:
    """Load previously completed images so interrupted runs can resume."""
    if not done_log.exists():
        return set()

    with done_log.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def append_completed(done_log: Path, image_path: Path) -> None:
    """Record a successfully relabelled image."""
    done_log.parent.mkdir(parents=True, exist_ok=True)
    with done_log.open("a", encoding="utf-8") as handle:
        handle.write(f"{image_path}\n")


def relabel_image(model: YOLO, image_path: Path, label_path: Path, conf: float) -> int:
    """Run inference and write a YOLO label file with ALL detections."""
    verify_image(image_path)
    results = model(str(image_path), conf=conf, verbose=False)
    boxes   = results[0].boxes

    label_path.parent.mkdir(parents=True, exist_ok=True)

    if boxes is None or len(boxes) == 0:
        label_path.write_text("")
        return 0

    # xywhn → normalized cx, cy, w, h  (shape N×4)
    xywhn   = boxes.xywhn.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    lines = [
        f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for cls_id, (cx, cy, w, h) in zip(cls_ids, xywhn)
    ]

    label_path.write_text("\n".join(lines))
    return len(lines)


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    failed_log = dataset_root / "relabel_failed_images.txt"
    done_log = dataset_root / "relabel_completed_images.txt"

    model = YOLO(str(MODEL_PATH))
    print(f"Model loaded  : {MODEL_PATH}")
    print(f"Dataset root  : {dataset_root}")
    print(f"Images root   : {images_dir}")
    print(f"Labels root   : {labels_dir}")
    print(f"Failed log    : {failed_log}")
    print(f"Done log      : {done_log}")
    print(f"Conf threshold: {args.conf}\n")

    # Collect all images, preserving class-subfolder structure
    image_paths = sorted(
        p
        for class_dir in sorted(images_dir.iterdir()) if class_dir.is_dir()
        for p in sorted(class_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS
    )

    if not image_paths:
        print("No images found — check IMAGES_DIR path.")
        return

    completed = load_completed(done_log)
    print(f"Total images to relabel: {len(image_paths)}")
    print(f"Already completed in this relabel run: {len(completed)}\n")

    total_boxes = 0
    skipped_completed = 0
    skipped_failed = 0
    for i, img_path in enumerate(image_paths, 1):
        # Mirror structure:  dataset/labels/{class}/{stem}.txt
        rel        = img_path.relative_to(images_dir)   # e.g. apple/apple_001.jpg
        label_path = labels_dir / rel.parent / (img_path.stem + ".txt")

        img_key = str(rel).replace("\\", "/")
        if img_key in completed:
            skipped_completed += 1
            continue

        try:
            n = relabel_image(model, img_path, label_path, args.conf)
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            skipped_failed += 1
            append_failure(failed_log, img_path, str(exc))
            print(f"WARNING Skipping unreadable image {rel}: {exc}")
            continue
        except Exception as exc:
            skipped_failed += 1
            append_failure(failed_log, img_path, str(exc))
            print(f"WARNING Skipping failed inference for {rel}: {exc}")
            continue

        append_completed(done_log, Path(img_key))
        total_boxes += n

        if i % 100 == 0 or i == len(image_paths):
            print(f"[{i:>5}/{len(image_paths)}] {rel}  →  {n} box(es)")

    print(
        f"\nFinished. {len(image_paths)} images scanned, {total_boxes} total boxes written, "
        f"{skipped_completed} already completed skipped, {skipped_failed} failed images logged."
    )


if __name__ == "__main__":
    main()
