"""
Pipeline Configuration — Synthetic COCO Dataset (Classes 40-59)
"""

import os
from dotenv import load_dotenv

# ─── Load .env ───────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
PREVIEWS_DIR = os.path.join(DATASET_DIR, "boxed_previews")
TRACKING_FILE = os.path.join(DATASET_DIR, "tracking.xlsx")

# ─── Model Settings ─────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-3-pro-image-preview"
ROBOFLOW_MODEL_URL = "https://detect.roboflow.com/coco/22"

# ─── COCO Classes 40-59 ─────────────────────────────────────────────────────
CLASSES = {
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
}

# Reverse lookup: class name → COCO ID
CLASS_NAME_TO_ID = {name: cid for cid, name in CLASSES.items()}

# ─── Generation Settings ────────────────────────────────────────────────────
IMAGES_PER_CLASS_PHASE_A = 1   # Phase A: verification
IMAGES_PER_CLASS_PHASE_B = 20  # Phase B: full dataset
