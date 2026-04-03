"""
Pipeline Orchestrator — Run the full synthetic data pipeline end-to-end.

Usage:
    python run_pipeline.py A                  # Phase A: 1 image/class, all 20 classes
    python run_pipeline.py A 46 47 48         # Phase A: specific classes only
    python run_pipeline.py B                  # Phase B: 20 images/class
    python run_pipeline.py label              # Label only (skip generation)
    python run_pipeline.py label 46 47        # Label specific classes only
    python run_pipeline.py upload <project>   # Upload to Roboflow project
"""

import sys
import os

# Ensure pipeline/ is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GEMINI_API_KEYS, CLASSES, DATASET_DIR, IMAGES_DIR, VERTEX_PROJECT, VERTEX_LOCATION
from generate_images import GeminiKeyRotator, run_phase_a, run_phase_b, sanitize_name
from label_images import label_class
from tracker import log_entry
from upload_to_roboflow import upload_to_roboflow


def run_generate_and_label(phase: str, class_ids: list[int] | None = None):
    """Run generation + labeling + tracking for a phase."""
    rotator = GeminiKeyRotator(GEMINI_API_KEYS, vertex_project=VERTEX_PROJECT,
                                vertex_location=VERTEX_LOCATION)
    target_ids = class_ids or list(CLASSES.keys())

    # ── Stage 1: Generate ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  STAGE 1 — Image Generation (Phase {phase.upper()})")
    print("=" * 60)

    if phase.upper() == "A":
        gen_results = run_phase_a(rotator, target_ids)
    else:
        gen_results = run_phase_b(rotator, target_ids)

    print(f"\n✓ Generated {len(gen_results)} images")

    # ── Stage 2: Label ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 2 — Auto-Labeling with Roboflow")
    print("=" * 60)

    all_label_results = []
    for cid in target_ids:
        cname = CLASSES[cid]
        print(f"\n[CLASS] {cname} ({cid})")
        label_results = label_class(cid)
        all_label_results.extend(
            [(cid, cname, r) for r in label_results]
        )

    # ── Stage 3: Track in Excel ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 3 — Logging to Excel")
    print("=" * 60)

    # Build a lookup: image_path → prompt from generation results
    prompt_lookup = {r[2]: r[3] for r in gen_results}

    for cid, cname, (img_path, lbl_path, label_str, n_det) in all_label_results:
        img_name = os.path.basename(img_path)
        prompt = prompt_lookup.get(img_path, "(prompt not recorded)")
        log_entry(cid, cname, img_name, prompt, label_str, n_det, phase.upper())
        print(f"  ✓ Logged: {img_name} ({n_det} detections)")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Images generated : {len(gen_results)}")
    print(f"  Images labeled   : {len(all_label_results)}")
    print(f"  Excel tracking   : dataset/tracking.xlsx")
    print(f"  Boxed previews   : dataset/boxed_previews/")
    print("=" * 60)


def run_label_only(class_ids: list[int] | None = None):
    """Label existing images without regenerating."""
    target_ids = class_ids or list(CLASSES.keys())

    print("\n" + "=" * 60)
    print("  LABEL ONLY — Labeling existing images")
    print("=" * 60)

    for cid in target_ids:
        cname = CLASSES[cid]
        print(f"\n[CLASS] {cname} ({cid})")
        results = label_class(cid)
        for img_path, lbl_path, label_str, n_det in results:
            img_name = os.path.basename(img_path)
            log_entry(cid, cname, img_name, "(existing image)", label_str,
                      n_det, "A")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].upper()

    if command in ("A", "B"):
        class_ids = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else None
        run_generate_and_label(command, class_ids)

    elif command == "LABEL":
        class_ids = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else None
        run_label_only(class_ids)

    elif command == "UPLOAD":
        if len(sys.argv) < 3:
            print("Usage: python run_pipeline.py upload <project_id> [class_ids...]")
            sys.exit(1)
        project_id = sys.argv[2]
        class_ids = [int(x) for x in sys.argv[3:]] if len(sys.argv) > 3 else None
        upload_to_roboflow(project_id, class_ids)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
