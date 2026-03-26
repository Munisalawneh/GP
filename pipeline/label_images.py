"""
Stage 2 — Auto-Labeling using Roboflow COCO detection model.

Sends generated images to Roboflow, converts predictions to YOLO format
with correct COCO class IDs (40-59), and generates boxed preview images.
"""

import os
import sys
import base64
import requests

from PIL import Image, ImageDraw, ImageFont

from config import (
    ROBOFLOW_API_KEY, ROBOFLOW_MODEL_URL, CLASS_NAME_TO_ID,
    IMAGES_DIR, LABELS_DIR, PREVIEWS_DIR, CLASSES,
)
from generate_images import sanitize_name


# ─── Color palette for bounding box visualization ───────────────────────────
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
]


def label_image(image_path: str) -> tuple[list[str], list[dict]]:
    """
    Send image to Roboflow and return:
      - yolo_lines: list of YOLO-format strings with COCO class IDs
      - predictions: raw prediction dicts for metadata
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    img = Image.open(image_path)
    img_w, img_h = img.size

    response = requests.post(
        ROBOFLOW_MODEL_URL,
        params={"api_key": ROBOFLOW_API_KEY},
        data=image_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )
    response.raise_for_status()

    predictions = response.json().get("predictions", [])

    yolo_lines = []
    for pred in predictions:
        det_class = pred["class"].lower()
        # Map to COCO class ID — skip detections not in our class set
        coco_id = CLASS_NAME_TO_ID.get(det_class)
        if coco_id is None:
            continue

        x_center = pred["x"] / img_w
        y_center = pred["y"] / img_h
        bbox_w = pred["width"] / img_w
        bbox_h = pred["height"] / img_h

        yolo_lines.append(
            f"{coco_id} {x_center:.6f} {y_center:.6f} "
            f"{bbox_w:.6f} {bbox_h:.6f}"
        )

    return yolo_lines, predictions


def save_label(yolo_lines: list[str], label_path: str):
    """Save YOLO label file."""
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))


def draw_boxed_preview(image_path: str, predictions: list[dict],
                       output_path: str):
    """Draw bounding boxes on the image and save a preview."""
    img = Image.open(image_path).copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for i, pred in enumerate(predictions):
        det_class = pred["class"].lower()
        if det_class not in CLASS_NAME_TO_ID:
            continue

        cx, cy = pred["x"], pred["y"]
        bw, bh = pred["width"], pred["height"]
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

        color = PALETTE[i % len(PALETTE)]
        label = f"{pred['class']} {pred['confidence']:.0%}"

        for t in range(3):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)

        bbox = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle(
            [bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1],
            fill=color,
        )
        draw.text((x1, y1 - 20), label, fill=(255, 255, 255), font=font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)


def label_class(class_id: int):
    """Label all images for a single class."""
    class_name = CLASSES[class_id]
    folder_name = sanitize_name(class_name)

    img_dir = os.path.join(IMAGES_DIR, folder_name)
    lbl_dir = os.path.join(LABELS_DIR, folder_name)
    prv_dir = os.path.join(PREVIEWS_DIR, folder_name)

    if not os.path.isdir(img_dir):
        print(f"[SKIP] No images found for {class_name}")
        return []

    results = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(img_dir, fname)
        label_name = os.path.splitext(fname)[0] + ".txt"
        label_path = os.path.join(lbl_dir, label_name)
        preview_path = os.path.join(prv_dir, fname)

        print(f"  [LABEL] {fname}")

        yolo_lines, predictions = label_image(image_path)
        save_label(yolo_lines, label_path)
        draw_boxed_preview(image_path, predictions, preview_path)

        label_str = "; ".join(yolo_lines) if yolo_lines else "(no detections)"
        results.append((image_path, label_path, label_str, len(yolo_lines)))

        print(f"    → {len(yolo_lines)} detections, preview saved")

    return results


def main():
    class_ids = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else list(CLASSES.keys())

    print("=" * 60)
    print("  STAGE 2 — Auto-Labeling with Roboflow COCO Model")
    print("=" * 60)

    for cid in class_ids:
        cname = CLASSES[cid]
        print(f"\n[CLASS] {cname} ({cid})")
        label_class(cid)

    print("\nLabeling complete.")


if __name__ == "__main__":
    main()
