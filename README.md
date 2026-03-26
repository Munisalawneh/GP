# SYN-COCO: Synthetic COCO Dataset Pipeline

A pipeline for generating photorealistic synthetic images using **Gemini 3 (Nano Banana)**, auto-labeling them with **Roboflow**, and producing a YOLO-compatible dataset that mirrors the 80-class COCO structure.

## Goal

Demonstrate that AI-generated images with carefully engineered prompts can reduce the performance gap in object detection tasks where real-world data is scarce, expensive, or impractical to obtain.

## Project Structure

```
GP/
├── coco.yaml                         # Original COCO 80-class reference
├── requirements.txt                  # Python dependencies
├── SYN_COCO.pdf                      # Research paper draft
│
├── pipeline/                         # All pipeline code
│   ├── .env                          # API keys (DO NOT commit — gitignored)
│   ├── config.py                     # Paths, class definitions, model settings
│   ├── prompt_engine.py              # Prompt generation with 5-dimension variation grid
│   ├── generate_images.py            # Stage 1: Gemini image generation
│   ├── label_images.py               # Stage 2: Roboflow auto-labeling + boxed previews
│   ├── tracker.py                    # Stage 3: Excel logging (tracking.xlsx)
│   ├── upload_to_roboflow.py         # Upload images + labels to Roboflow for adjustment
│   └── run_pipeline.py               # Master orchestrator
│
├── dataset/                          # Generated dataset (gitignored)
│   ├── data.yaml                     # YOLO dataset config
│   ├── tracking.xlsx                 # Auto-generated: image | prompt | labels | metadata
│   ├── images/{class_name}/          # Generated images organized by class
│   ├── labels/{class_name}/          # YOLO label files (class_id x_center y_center w h)
│   └── boxed_previews/{class_name}/  # Visual QA: images with bounding boxes drawn
│
└── COCO VS SYN/                      # Comparison experiments
```

## Quick Start

### 1. Create Conda Environment

```bash
conda create -n grad python=3.11 -y
conda activate grad
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys

Create the file `pipeline/.env` with your keys:

```
GEMINI_API_KEY=your_gemini_api_key_here
ROBOFLOW_API_KEY=your_roboflow_api_key_here
```

> **Important:** Never commit this file. It is already in `.gitignore`.

### 4. Configure Your Classes

Open `pipeline/config.py` and update the `CLASSES` dictionary to match **your assigned class range**. The current config uses classes 40–59:

```python
CLASSES = {
    40: "wine glass",
    41: "cup",
    ...
    59: "bed",
}
```

If you are working on a different range (e.g., 0–19 or 60–79), replace the dictionary with your classes from `coco.yaml`.

Also update `IMAGES_PER_CLASS_PHASE_B` if you want more or fewer images per class.

### 5. Update Prompt Engine

Open `pipeline/prompt_engine.py` and add entries for your classes in:

- **`OBJECT_STATES`** — 5 variations describing the object in different states, with other COCO objects co-occurring in the scene.
- **`CLASS_CATEGORY`** — Map each class name to a scene category (`"kitchen_item"`, `"food"`, `"furniture"`, etc.). Add new categories in `SCENES` if needed.

Each prompt is assembled from 5 dimensions:
| Dimension | Purpose |
|-----------|---------|
| Object state | What the object looks like + surrounding objects |
| Scene | Environment / location |
| Viewpoint | Camera angle |
| Lighting | Light source and quality |
| Disturbance | Noise, blur, occlusion, clutter |

### 6. Run the Pipeline

All commands are run from the `pipeline/` directory:

```bash
cd pipeline
```

**Phase A — Generate 1 image per class for verification:**
```bash
python run_pipeline.py A              # All your classes
python run_pipeline.py A 46 47 48     # Specific classes only
```

**Phase B — Generate 20 images per class with prompt variations:**
```bash
python run_pipeline.py B              # All your classes
python run_pipeline.py B 46           # One class at a time
```

**Label only (images already exist, re-run labeling):**
```bash
python run_pipeline.py label
python run_pipeline.py label 46 47
```

**Upload to Roboflow for manual annotation adjustment:**
```bash
python run_pipeline.py upload your-roboflow-project-id
python run_pipeline.py upload your-roboflow-project-id 46 47 48
```

### 7. Review Results

- **Boxed previews** — Check `dataset/boxed_previews/{class}/` to visually verify bounding boxes.
- **Excel tracking** — Open `dataset/tracking.xlsx` to see every image with its prompt, labels, detection count, and timestamp.
- **YOLO labels** — `dataset/labels/{class}/` contains one `.txt` per image in YOLO format.

## Pipeline Stages

| Stage | Script | What It Does |
|-------|--------|-------------|
| 1 | `generate_images.py` | Calls Gemini 3 to generate photorealistic images per class |
| 2 | `label_images.py` | Sends images to Roboflow COCO model, converts to YOLO labels, draws boxed previews |
| 3 | `tracker.py` | Logs image name, prompt, labels, and metadata to `tracking.xlsx` |

## Label Format

Labels use **COCO class IDs** (not 0-indexed per batch), so all team members' datasets are directly compatible:

```
<coco_class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized (0–1). Example for a banana image:
```
46 0.495739 0.406901 0.417614 0.454427
56 0.075994 0.186198 0.150568 0.226562
```

## Class Assignments

| Range | Classes | Assignee |
|-------|---------|----------|
| 0–19  | person → cow | TBD |
| 20–39 | elephant → bottle | TBD |
| 40–59 | wine glass → bed | Muni |
| 60–79 | dining table → toothbrush | TBD |

## Notes

- The pipeline **skips** images that already exist (safe to re-run).
- Rate limiting (2s delay) is built in to respect API quotas.
- Phase B uses a **deterministic seed** (`seed=42`) so prompts are reproducible.
- Prompts are designed for COCO-style realism: multi-object scenes, occlusion, noise, casual phone-photo quality — not clean studio shots.
