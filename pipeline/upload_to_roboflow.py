"""
Roboflow Upload — Upload labeled images to a Roboflow project for manual
annotation adjustments.

Yes, you CAN upload images with their YOLO labels to Roboflow for review
and adjustment. This script handles that workflow:
  1. Creates/uses a project on Roboflow
  2. Uploads images with their corresponding label files
  3. You can then refine annotations in Roboflow's web editor
  4. Re-export the corrected dataset when ready
"""

import os
import sys
import requests

from config import ROBOFLOW_API_KEY, IMAGES_DIR, LABELS_DIR, CLASSES
from generate_images import sanitize_name


ROBOFLOW_UPLOAD_URL = "https://api.roboflow.com/dataset/{project_id}/upload"


def upload_to_roboflow(project_id: str, class_ids: list[int] | None = None):
    """
    Upload images + labels to a Roboflow project for annotation adjustment.
    
    Usage:
        1. Create a project on https://app.roboflow.com
        2. Get the project ID from the URL (e.g., "syn-coco-40-59")
        3. Run: python upload_to_roboflow.py <project_id> [class_ids...]
    """
    targets = {cid: CLASSES[cid] for cid in (class_ids or CLASSES.keys())}

    print("=" * 60)
    print(f"  Uploading to Roboflow project: {project_id}")
    print("=" * 60)

    for cid, cname in targets.items():
        folder = sanitize_name(cname)
        img_dir = os.path.join(IMAGES_DIR, folder)
        lbl_dir = os.path.join(LABELS_DIR, folder)

        if not os.path.isdir(img_dir):
            print(f"[SKIP] {cname} — no images")
            continue

        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(img_dir, fname)
            label_name = os.path.splitext(fname)[0] + ".txt"
            label_path = os.path.join(lbl_dir, label_name)

            print(f"  [UPLOAD] {cname}/{fname}")

            # Upload image
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

            if resp.status_code == 200:
                result = resp.json()
                image_id = result.get("id")
                print(f"    ✓ Image uploaded (id={image_id})")

                # Upload annotation if label file exists
                if os.path.exists(label_path):
                    with open(label_path, "r") as lf:
                        annotation_text = lf.read().strip()

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
                            headers={
                                "Content-Type": "text/plain",
                            },
                            timeout=60,
                        )
                        if annot_resp.status_code == 200:
                            print(f"    ✓ Annotation uploaded")
                        else:
                            print(f"    ✗ Annotation failed: "
                                  f"{annot_resp.status_code}")
            else:
                print(f"    ✗ Upload failed: {resp.status_code} {resp.text}")


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
