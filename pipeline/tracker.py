"""
Excel Tracker — Logs every generated image with its prompt, labels, and metadata.

Columns: class_id, class_name, image_name, prompt, labels, timestamp, phase
"""

import os
from datetime import datetime
from openpyxl import Workbook, load_workbook
from config import TRACKING_FILE, DATASET_DIR

HEADERS = [
    "class_id",
    "class_name",
    "image_name",
    "prompt",
    "labels",
    "num_detections",
    "timestamp",
    "phase",
]


def _ensure_workbook() -> Workbook:
    """Load existing workbook or create a new one with headers."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    if os.path.exists(TRACKING_FILE):
        return load_workbook(TRACKING_FILE)
    wb = Workbook()
    ws = wb.active
    ws.title = "Tracking"
    ws.append(HEADERS)
    # Set column widths for readability
    widths = [10, 16, 30, 80, 50, 14, 22, 8]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[chr(64 + i)].width = w
    wb.save(TRACKING_FILE)
    return wb


def log_entry(class_id: int, class_name: str, image_name: str,
              prompt: str, labels: str, num_detections: int,
              phase: str = "A"):
    """Append a row to the tracking spreadsheet."""
    wb = _ensure_workbook()
    ws = wb.active
    ws.append([
        class_id,
        class_name,
        image_name,
        prompt,
        labels,
        num_detections,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        phase,
    ])
    wb.save(TRACKING_FILE)


if __name__ == "__main__":
    # Quick test
    log_entry(46, "banana", "banana_001.png",
              "a ripe banana on a counter", "46 0.5 0.5 0.3 0.2", 1, "A")
    print(f"Test entry written to {TRACKING_FILE}")
