"""
Stage 1 — Image Generation using Gemini 3 (Nano Banana).

Generates photorealistic images for COCO classes 40-59.
Phase A: 1 image per class for verification.
Phase B: 20 images per class with prompt variations.
"""

import os
import sys
import base64
import time

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY, GEMINI_MODEL, CLASSES,
    IMAGES_DIR, IMAGES_PER_CLASS_PHASE_A, IMAGES_PER_CLASS_PHASE_B,
)
from prompt_engine import get_baseline_prompt, get_variation_prompts


def sanitize_name(class_name: str) -> str:
    """Convert class name to a filesystem-safe folder/file name."""
    return class_name.replace(" ", "_")


def generate_image(client, prompt: str, output_path: str) -> bool:
    """Call Gemini to generate one image. Returns True on success."""
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                data = part.inline_data.data
                image_bytes = (base64.b64decode(data)
                               if isinstance(data, str) else data)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                return True
        print(f"  [WARN] No image returned for: {output_path}")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def run_phase_a(client, class_ids: list[int] | None = None):
    """Generate 1 image per class for visual verification."""
    targets = {cid: CLASSES[cid] for cid in (class_ids or CLASSES.keys())}
    results = []  # (class_id, class_name, image_path, prompt)

    for cid, cname in targets.items():
        folder = os.path.join(IMAGES_DIR, sanitize_name(cname))
        filename = f"{sanitize_name(cname)}_001.png"
        output_path = os.path.join(folder, filename)

        if os.path.exists(output_path):
            print(f"[SKIP] {cname} — already exists")
            prompt = get_baseline_prompt(cname)
            results.append((cid, cname, output_path, prompt))
            continue

        prompt = get_baseline_prompt(cname)
        print(f"[GEN] {cname} ({cid}) → {filename}")

        success = generate_image(client, prompt, output_path)
        if success:
            print(f"  ✓ Saved: {output_path}")
            results.append((cid, cname, output_path, prompt))
        else:
            print(f"  ✗ Failed: {cname}")

        # Rate limiting — be respectful to the API
        time.sleep(2)

    return results


def run_phase_b(client, class_ids: list[int] | None = None):
    """Generate 20 images per class with prompt variations."""
    targets = {cid: CLASSES[cid] for cid in (class_ids or CLASSES.keys())}
    results = []

    for cid, cname in targets.items():
        folder = os.path.join(IMAGES_DIR, sanitize_name(cname))
        prompts = get_variation_prompts(cname, count=IMAGES_PER_CLASS_PHASE_B)

        for i, prompt in enumerate(prompts, 1):
            filename = f"{sanitize_name(cname)}_{i:03d}.png"
            output_path = os.path.join(folder, filename)

            if os.path.exists(output_path):
                print(f"[SKIP] {cname} #{i} — already exists")
                results.append((cid, cname, output_path, prompt))
                continue

            print(f"[GEN] {cname} ({cid}) [{i}/{IMAGES_PER_CLASS_PHASE_B}]")
            success = generate_image(client, prompt, output_path)

            if success:
                print(f"  ✓ Saved: {output_path}")
                results.append((cid, cname, output_path, prompt))
            else:
                print(f"  ✗ Failed: {cname} #{i}")

            time.sleep(2)

    return results


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "A"
    # Optional: pass specific class IDs like "46 47 48"
    class_ids = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else None

    client = genai.Client(api_key=GEMINI_API_KEY)

    if phase.upper() == "A":
        print("=" * 60)
        print("  PHASE A — Generating 1 image per class for verification")
        print("=" * 60)
        results = run_phase_a(client, class_ids)
    elif phase.upper() == "B":
        print("=" * 60)
        print("  PHASE B — Generating 20 images per class with variations")
        print("=" * 60)
        results = run_phase_b(client, class_ids)
    else:
        print(f"Unknown phase: {phase}. Use 'A' or 'B'.")
        sys.exit(1)

    print(f"\nDone. Generated {len(results)} images.")
    return results


if __name__ == "__main__":
    main()
