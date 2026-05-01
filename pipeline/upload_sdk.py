"""Parallel upload of a relabelled dataset to an existing Roboflow project.

Uploads images from a dataset root's images/ directory with YOLO labels from its
labels/ directory, using coco.yaml as the class-name labelmap so Roboflow receives
class names instead of raw numeric IDs.
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import roboflow
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from config import IMAGES_DIR, LABELS_DIR, ROBOFLOW_API_KEY

# ── Roboflow target ─────────────────────────────────────────────────────
WORKSPACE = "omarabualrub"
PROJECT = "syn-coco-qdonc"
BATCH_NAME = "Asem"
TAG_NAMES = ["Asem"]
SPLIT = "train"
WORKERS = 16
NUM_RETRIES = 3

# ── Paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
LABELMAP_PATH = ROOT / "coco.yaml"
FAILED_RELABEL_LOG = ROOT / "dataset" / "relabel_failed_images.txt"
UPLOAD_FAILED_LOG = ROOT / "dataset" / "upload_failed_images.txt"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

_lock = threading.Lock()
_stats = {"ok": 0, "fail": 0, "total": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a YOLO dataset to Roboflow")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "dataset",
        help="Dataset root containing images/ and labels/ subfolders",
    )
    parser.add_argument(
        "--tag",
        default=TAG_NAMES[0],
        help="Roboflow tag and batch name to attach to uploads",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry only items from the previous upload_failed_images.txt log",
    )
    return parser.parse_args()


def create_ascii_labelmap(output_path: Path) -> Path:
    """Write a minimal ASCII-only label map for Roboflow's Windows upload path."""
    payload = yaml.safe_load(LABELMAP_PATH.read_text(encoding="utf-8"))
    names = payload.get("names", {})

    if isinstance(names, list):
        items = list(enumerate(names))
    else:
        items = sorted((int(key), str(value)) for key, value in names.items())

    lines = ["names:"]
    lines.extend(f"  {index}: {name}" for index, name in items)
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return output_path


def load_skip_set(image_root: Path, failed_relabel_log: Path) -> set[str]:
    """Skip images already known to be unreadable during relabeling."""
    if not failed_relabel_log.exists():
        return set()

    skipped = set()
    with failed_relabel_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            path_text = line.strip().split("\t", 1)[0]
            if not path_text:
                continue
            try:
                rel = Path(path_text).resolve().relative_to(image_root.resolve())
            except Exception:
                continue
            skipped.add(rel.as_posix())
    return skipped


def build_tasks(image_root: Path, label_root: Path, failed_relabel_log: Path) -> list[tuple[str, str, str]]:
    """Collect image/label pairs for upload."""
    skipped = load_skip_set(image_root, failed_relabel_log)
    tasks: list[tuple[str, str, str]] = []

    for image_path in sorted(image_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        rel = image_path.relative_to(image_root).as_posix()
        if rel in skipped:
            continue

        label_path = label_root / image_path.relative_to(image_root).with_suffix(".txt")
        if not label_path.exists():
            continue

        tasks.append((rel, str(image_path), str(label_path)))

    return tasks


def build_failed_tasks(image_root: Path, label_root: Path, upload_failed_log: Path) -> list[tuple[str, str, str]]:
    """Rebuild task list from the previous upload failure log."""
    if not upload_failed_log.exists():
        return []

    tasks: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    with upload_failed_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            rel_path = line.strip().split("\t", 1)[0]
            if not rel_path or rel_path in seen:
                continue
            seen.add(rel_path)

            image_path = image_root / Path(rel_path)
            label_path = label_root / Path(rel_path).with_suffix(".txt")
            if image_path.exists():
                tasks.append((rel_path, str(image_path), str(label_path)))

    return tasks


def append_upload_failure(upload_failed_log: Path, rel_path: str, reason: str) -> None:
    upload_failed_log.parent.mkdir(parents=True, exist_ok=True)
    with upload_failed_log.open("a", encoding="utf-8") as handle:
        handle.write(f"{rel_path}\t{reason}\n")


def upload_one(project, rel_path: str, image_path: str, label_path: str, labelmap_path: Path, batch_name: str, tag_names: list[str], upload_failed_log: Path) -> None:
    last_error = None
    annotation_path = label_path if Path(label_path).exists() and Path(label_path).stat().st_size > 0 else None

    for attempt in range(1, NUM_RETRIES + 1):
        try:
            project.upload(
                image_path=image_path,
                annotation_path=annotation_path,
                annotation_labelmap=str(labelmap_path) if annotation_path else None,
                split=SPLIT,
                batch_name=batch_name,
                tag_names=tag_names,
                num_retry_uploads=1,
            )

            with _lock:
                _stats["ok"] += 1
                done = _stats["ok"] + _stats["fail"]
                if done % 50 == 0 or done == _stats["total"]:
                    print(f"[{done}/{_stats['total']}] uploaded={_stats['ok']} failed={_stats['fail']}")
            return
        except Exception as exc:
            last_error = exc
            if attempt < NUM_RETRIES:
                time.sleep(2 * attempt)

    with _lock:
        _stats["fail"] += 1
        append_upload_failure(upload_failed_log, rel_path, str(last_error))
        print(f"WARNING Upload failed for {rel_path}: {last_error}")


def main() -> None:
    if not ROBOFLOW_API_KEY:
        raise RuntimeError("ROBOFLOW_API_KEY is missing in pipeline/.env")

    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    image_root = dataset_root / "images"
    label_root = dataset_root / "labels"
    failed_relabel_log = dataset_root / "relabel_failed_images.txt"
    upload_failed_log = dataset_root / "upload_failed_images.txt"
    roboflow_labelmap_path = dataset_root / "roboflow_labelmap.yaml"
    batch_name = args.tag
    tag_names = [args.tag]
    labelmap_path = create_ascii_labelmap(roboflow_labelmap_path)

    tasks = (
        build_failed_tasks(image_root, label_root, upload_failed_log)
        if args.retry_failed
        else build_tasks(image_root, label_root, failed_relabel_log)
    )
    _stats["total"] = len(tasks)
    _stats["ok"] = 0
    _stats["fail"] = 0

    print(f"Workspace    : {WORKSPACE}")
    print(f"Project      : {WORKSPACE}/{PROJECT}")
    print(f"Dataset root : {dataset_root}")
    print(f"Batch name   : {batch_name}")
    print(f"Tags         : {tag_names}")
    print(f"Label map    : {labelmap_path}")
    print(f"Workers      : {WORKERS}")
    print(f"Retries/item : {NUM_RETRIES}")
    print(f"Retry mode   : {args.retry_failed}")
    print(f"Images queued: {len(tasks)}")

    if not tasks:
        print("No upload tasks found.")
        return

    if upload_failed_log.exists():
        upload_failed_log.unlink()

    rf = roboflow.Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [
            pool.submit(
                upload_one,
                project,
                rel,
                image,
                label,
                labelmap_path,
                batch_name,
                tag_names,
                upload_failed_log,
            )
            for rel, image, label in tasks
        ]
        for future in as_completed(futures):
            future.result()

    print(
        f"Finished upload. total={_stats['total']} uploaded={_stats['ok']} failed={_stats['fail']}"
    )
    if upload_failed_log.exists():
        print(f"See failed uploads in: {upload_failed_log}")


if __name__ == "__main__":
    main()
