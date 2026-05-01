"""
Roboflow Upload — Upload labeled images to a Roboflow project for manual
annotation adjustments.

Yes, you CAN upload images with their YOLO labels to Roboflow for review
and adjustment. This script handles that workflow:
  1. Creates/uses a project on Roboflow
  2. Uploads images with their corresponding label files (parallel)
  3. You can then refine annotations in Roboflow's web editor
  4. Re-export the corrected dataset when ready
"""

import os
import sys
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import ROBOFLOW_API_KEY, IMAGES_DIR, LABELS_DIR, CLASSES
from generate_images import sanitize_name

WORKERS = 10
_lock = threading.Lock()
_stats = {"ok": 0, "fail": 0, "total": 0}

# Remap COCO class IDs (40-59) → 0-indexed IDs matching the class names list
# Roboflow expects 0-based class IDs that map to a names list
_COCO_TO_IDX = {cid: i for i, cid in enumerate(sorted(CLASSES.keys()))}
CLASS_NAMES = [CLASSES[cid] for cid in sorted(CLASSES.keys())]


def _remap_annotation(annotation_text: str) -> str:
    """Remap COCO class IDs to 0-indexed IDs in YOLO annotation text."""
    lines = []
    for line in annotation_text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            old_id = int(parts[0])
            if old_id in _COCO_TO_IDX:
                parts[0] = str(_COCO_TO_IDX[old_id])
                lines.append(" ".join(parts))
            # skip lines with class IDs outside our 40-59 range
    return "\n".join(lines)


def _upload_one(project_id: str, cname: str, image_path: str, label_path: str):
    """Upload a single image + annotation. Thread-safe."""
    fname = os.path.basename(image_path)
    label_name = os.path.splitext(fname)[0] + ".txt"

    try:
        upload_url = (
            f"https://api.roboflow.com/dataset/{project_id}/upload"
            f"?api_key={ROBOFLOW_API_KEY}"
            f"&name={fname}"
            f"&split=train"
        )
        with open(image_path, "rb") as img_file:
            resp = requests.post(
                upload_url,
                files={"file": (fname, img_file, "image/png")},
                timeout=120,
            )

        if resp.status_code != 200:
            with _lock:
                _stats["fail"] += 1
                print(f"  ✗ {cname}/{fname} — {resp.status_code}")
            return

        image_id = resp.json().get("id")

        # Upload annotation (with remapped class IDs)
        if os.path.exists(label_path):
            with open(label_path, "r") as lf:
                raw_text = lf.read().strip()
            annotation_text = _remap_annotation(raw_text)
            if annotation_text:
                annot_url = (
                    f"https://api.roboflow.com/dataset/{project_id}"
                    f"/annotate/{image_id}"
                    f"?api_key={ROBOFLOW_API_KEY}"
                    f"&name={label_name}"
                )
                annot_resp = requests.post(
                    annot_url,
                    data=annotation_text,
                    headers={"Content-Type": "text/plain"},
                    timeout=60,
                )
                if annot_resp.status_code != 200:
                    with _lock:
                        _stats["fail"] += 1
                        print(f"  ~ {cname}/{fname} — image ok, annotation failed")
                    return

        with _lock:
            _stats["ok"] += 1
            done = _stats["ok"] + _stats["fail"]
            if done % 50 == 0 or done == _stats["total"]:
                print(f"  [{done}/{_stats['total']}] ✓ {_stats['ok']}  ✗ {_stats['fail']}")

    except Exception as e:
        with _lock:
            _stats["fail"] += 1
            print(f"  ✗ {cname}/{fname} — {e}")


def upload_to_roboflow(project_id: str, class_ids: list[int] | None = None):
    """
    Upload images + labels to a Roboflow project (parallel).
    """
    targets = {cid: CLASSES[cid] for cid in (class_ids or CLASSES.keys())}

    # Build task list
    tasks = []
    for cid, cname in targets.items():
        folder = sanitize_name(cname)
        img_dir = os.path.join(IMAGES_DIR, folder)
        lbl_dir = os.path.join(LABELS_DIR, folder)
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            image_path = os.path.join(img_dir, fname)
            label_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
            tasks.append((cname, image_path, label_path))

    _stats["total"] = len(tasks)
    _stats["ok"] = 0
    _stats["fail"] = 0

    print("=" * 60)
    print(f"  Uploading {len(tasks)} images to Roboflow project: {project_id}")
    print(f"  Workers: {WORKERS}")
    print("=" * 60)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [
            pool.submit(_upload_one, project_id, cname, img, lbl)
            for cname, img, lbl in tasks
        ]
        for f in as_completed(futures):
            f.result()  # propagate exceptions

    print("=" * 60)
    print(f"  Done: ✓ {_stats['ok']}  ✗ {_stats['fail']}  total {_stats['total']}")
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_to_roboflow.py <project_id> [class_ids...]")
        print()
        print("Steps:")
        print("  1. Go to https://app.roboflow.com and create a new project")
        print("  2. Copy the project ID from the URL")
        print("  3. Run this script with the project ID")
        print()
        print("Example:")
        print("  python upload_to_roboflow.py syn-coco-40-59")
        print("  python upload_to_roboflow.py syn-coco-40-59 46 47 48")
        sys.exit(1)

    project_id = sys.argv[1]
    class_ids = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else None

    upload_to_roboflow(project_id, class_ids)


if __name__ == "__main__":
    main()
