"""
Dataset Validation Script — Checks images, labels, duplicates, and prompt coverage.

Validates:
  1. Image integrity: every PNG opens and has correct dimensions (1408x768)
  2. Label validity: YOLO format, correct class IDs, normalized coords in [0,1]
  3. Label-image pairing: every image has a label and vice versa
  4. Duplicate detection: perceptual hashing to find near-identical images
  5. Empty/missing labels: labels with no detections
  6. Class ID correctness: labels only contain the expected class ID for that folder
"""

import os
import sys
import hashlib
from collections import defaultdict
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from config import CLASSES, IMAGES_DIR, LABELS_DIR, IMAGES_PER_CLASS_PHASE_B
from generate_images import sanitize_name

TARGET_SIZE = (1408, 768)


def compute_image_hash(path: str) -> str:
    """Compute a perceptual hash by resizing to 16x16 grayscale and hashing."""
    try:
        img = Image.open(path).convert("L").resize((16, 16), Image.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if p > avg else "0" for p in pixels)
        return hashlib.md5(bits.encode()).hexdigest()
    except Exception:
        return ""


def compute_file_hash(path: str) -> str:
    """Exact file hash (MD5)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_label_file(label_path: str, expected_cid: int) -> dict:
    """Validate a single YOLO label file. Returns dict of issues found."""
    issues = []
    detections = 0
    wrong_class = []
    
    with open(label_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    if not lines:
        issues.append("empty_label")
        return {"issues": issues, "detections": 0, "wrong_class": [], "lines": 0}
    
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
            issues.append(f"line {i+1}: expected 5 values, got {len(parts)}")
            continue
        
        try:
            cid = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            issues.append(f"line {i+1}: non-numeric values")
            continue
        
        # Check class ID
        if cid != expected_cid:
            wrong_class.append(cid)
        
        # Check normalized coordinates
        for val, name in [(x_center, "x"), (y_center, "y"), (width, "w"), (height, "h")]:
            if val < 0 or val > 1:
                issues.append(f"line {i+1}: {name}={val:.4f} out of [0,1]")
        
        detections += 1
    
    return {
        "issues": issues,
        "detections": detections,
        "wrong_class": wrong_class,
        "lines": len(lines),
    }


def validate_class(cid: int, class_name: str) -> dict:
    """Full validation for one class."""
    safe_name = sanitize_name(class_name)
    img_folder = os.path.join(IMAGES_DIR, safe_name)
    lbl_folder = os.path.join(LABELS_DIR, safe_name)
    
    result = {
        "class": class_name,
        "cid": cid,
        "total_images": 0,
        "total_labels": 0,
        "missing_labels": [],
        "orphan_labels": [],
        "bad_images": [],
        "wrong_size": [],
        "empty_labels": [],
        "label_errors": [],
        "wrong_class_ids": [],
        "exact_duplicates": [],
        "perceptual_duplicates": [],
        "no_target_class": [],
    }
    
    # Get all image and label files
    img_files = sorted([f for f in os.listdir(img_folder) if f.endswith(".png")]) if os.path.isdir(img_folder) else []
    lbl_files = sorted([f for f in os.listdir(lbl_folder) if f.endswith(".txt")]) if os.path.isdir(lbl_folder) else []
    
    result["total_images"] = len(img_files)
    result["total_labels"] = len(lbl_files)
    
    img_basenames = {os.path.splitext(f)[0] for f in img_files}
    lbl_basenames = {os.path.splitext(f)[0] for f in lbl_files}
    
    # Missing labels (image exists but no label)
    result["missing_labels"] = sorted(img_basenames - lbl_basenames)
    
    # Orphan labels (label exists but no image)
    result["orphan_labels"] = sorted(lbl_basenames - img_basenames)
    
    # Check images
    file_hashes = {}
    perceptual_hashes = {}
    
    for img_file in img_files:
        img_path = os.path.join(img_folder, img_file)
        
        # Check image opens
        try:
            img = Image.open(img_path)
            img.verify()  # Verify integrity
            # Re-open after verify (verify closes the file)
            img = Image.open(img_path)
            w, h = img.size
            if (w, h) != TARGET_SIZE:
                result["wrong_size"].append((img_file, w, h))
        except Exception as e:
            result["bad_images"].append((img_file, str(e)))
            continue
        
        # Exact duplicate check
        fhash = compute_file_hash(img_path)
        if fhash in file_hashes:
            result["exact_duplicates"].append((img_file, file_hashes[fhash]))
        else:
            file_hashes[fhash] = img_file
        
        # Perceptual duplicate check
        phash = compute_image_hash(img_path)
        if phash and phash in perceptual_hashes:
            result["perceptual_duplicates"].append((img_file, perceptual_hashes[phash]))
        elif phash:
            perceptual_hashes[phash] = img_file
    
    # Check labels
    for lbl_file in lbl_files:
        lbl_path = os.path.join(lbl_folder, lbl_file)
        basename = os.path.splitext(lbl_file)[0]
        
        lbl_result = validate_label_file(lbl_path, cid)
        
        if "empty_label" in lbl_result["issues"]:
            result["empty_labels"].append(basename)
        
        if lbl_result["issues"]:
            for issue in lbl_result["issues"]:
                if issue != "empty_label":
                    result["label_errors"].append((basename, issue))
        
        if lbl_result["wrong_class"]:
            result["wrong_class_ids"].append((basename, lbl_result["wrong_class"]))
        
        # Check if target class is present at all (non-empty label but no detection of expected class)
        if lbl_result["detections"] > 0 and lbl_result["detections"] == len(lbl_result["wrong_class"]):
            result["no_target_class"].append(basename)
    
    return result


def main():
    print("=" * 70)
    print("  DATASET VALIDATION — 20 Classes × 200 Images")
    print("=" * 70)
    
    all_results = []
    total_issues = 0
    
    for cid, cname in CLASSES.items():
        print(f"\n  Validating {cname} ({cid})...", end="", flush=True)
        r = validate_class(cid, cname)
        all_results.append(r)
        
        issues = (len(r["missing_labels"]) + len(r["orphan_labels"]) +
                  len(r["bad_images"]) + len(r["wrong_size"]) +
                  len(r["empty_labels"]) + len(r["label_errors"]) +
                  len(r["wrong_class_ids"]) + len(r["exact_duplicates"]) +
                  len(r["perceptual_duplicates"]) + len(r["no_target_class"]))
        total_issues += issues
        
        if issues == 0:
            print(f" ✓ {r['total_images']} imgs, {r['total_labels']} labels — OK")
        else:
            print(f" ⚠ {issues} issue(s)")
    
    # Detailed report
    print("\n" + "=" * 70)
    print("  DETAILED REPORT")
    print("=" * 70)
    
    # Summary table
    print(f"\n  {'Class':<15} {'Imgs':>4} {'Lbls':>4} {'Empty':>5} {'NoCls':>5} {'WrCls':>5} {'Dup':>4} {'PDup':>4} {'Bad':>4} {'Size':>4}")
    print("  " + "-" * 66)
    
    totals = defaultdict(int)
    for r in all_results:
        e = len(r["empty_labels"])
        nc = len(r["no_target_class"])
        wc = len(r["wrong_class_ids"])
        d = len(r["exact_duplicates"])
        pd = len(r["perceptual_duplicates"])
        b = len(r["bad_images"])
        s = len(r["wrong_size"])
        
        totals["imgs"] += r["total_images"]
        totals["lbls"] += r["total_labels"]
        totals["empty"] += e
        totals["no_cls"] += nc
        totals["wrong_cls"] += wc
        totals["dup"] += d
        totals["pdup"] += pd
        totals["bad"] += b
        totals["size"] += s
        
        flag = "  " if (e + nc + wc + d + pd + b + s) == 0 else "⚠ "
        print(f"  {flag}{r['class']:<13} {r['total_images']:>4} {r['total_labels']:>4} {e:>5} {nc:>5} {wc:>5} {d:>4} {pd:>4} {b:>4} {s:>4}")
    
    print("  " + "-" * 66)
    print(f"  {'TOTAL':<15} {totals['imgs']:>4} {totals['lbls']:>4} {totals['empty']:>5} {totals['no_cls']:>5} {totals['wrong_cls']:>5} {totals['dup']:>4} {totals['pdup']:>4} {totals['bad']:>4} {totals['size']:>4}")
    
    # Print details for each issue type
    for r in all_results:
        has_issues = (r["missing_labels"] or r["orphan_labels"] or r["bad_images"] or
                      r["wrong_size"] or r["empty_labels"] or r["label_errors"] or
                      r["wrong_class_ids"] or r["exact_duplicates"] or
                      r["perceptual_duplicates"] or r["no_target_class"])
        
        if not has_issues:
            continue
        
        print(f"\n  --- {r['class']} (class {r['cid']}) ---")
        
        if r["missing_labels"]:
            print(f"    Missing labels ({len(r['missing_labels'])}): {', '.join(r['missing_labels'][:10])}")
        
        if r["orphan_labels"]:
            print(f"    Orphan labels ({len(r['orphan_labels'])}): {', '.join(r['orphan_labels'][:10])}")
        
        if r["bad_images"]:
            for f, err in r["bad_images"][:5]:
                print(f"    Bad image: {f} — {err}")
        
        if r["wrong_size"]:
            for f, w, h in r["wrong_size"][:5]:
                print(f"    Wrong size: {f} — {w}x{h} (expected {TARGET_SIZE[0]}x{TARGET_SIZE[1]})")
            if len(r["wrong_size"]) > 5:
                print(f"    ... and {len(r['wrong_size']) - 5} more")
        
        if r["empty_labels"]:
            print(f"    Empty labels ({len(r['empty_labels'])}): {', '.join(r['empty_labels'][:10])}")
            if len(r["empty_labels"]) > 10:
                print(f"    ... and {len(r['empty_labels']) - 10} more")
        
        if r["no_target_class"]:
            print(f"    No target class in label ({len(r['no_target_class'])}): {', '.join(r['no_target_class'][:10])}")
            if len(r["no_target_class"]) > 10:
                print(f"    ... and {len(r['no_target_class']) - 10} more")
        
        if r["wrong_class_ids"]:
            for f, cids in r["wrong_class_ids"][:5]:
                print(f"    Wrong class IDs in {f}: found {cids}, expected {r['cid']}")
            if len(r["wrong_class_ids"]) > 5:
                print(f"    ... and {len(r['wrong_class_ids']) - 5} more with wrong class IDs")
        
        if r["exact_duplicates"]:
            for f1, f2 in r["exact_duplicates"]:
                print(f"    Exact duplicate: {f1} == {f2}")
        
        if r["perceptual_duplicates"]:
            for f1, f2 in r["perceptual_duplicates"][:10]:
                print(f"    Perceptual duplicate: {f1} ≈ {f2}")
            if len(r["perceptual_duplicates"]) > 10:
                print(f"    ... and {len(r['perceptual_duplicates']) - 10} more perceptual duplicates")
        
        if r["label_errors"]:
            for f, err in r["label_errors"][:5]:
                print(f"    Label error in {f}: {err}")
            if len(r["label_errors"]) > 5:
                print(f"    ... and {len(r['label_errors']) - 5} more label errors")
    
    # Legend
    print(f"\n  Legend: Imgs=images, Lbls=labels, Empty=empty label files,")
    print(f"         NoCls=label has detections but none of target class,")
    print(f"         WrCls=label contains wrong class IDs, Dup=exact duplicates,")
    print(f"         PDup=perceptual duplicates, Bad=corrupt images, Size=wrong dimensions")
    
    print(f"\n  Total issues: {total_issues}")
    if total_issues == 0:
        print("  ✓ Dataset is clean!")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
