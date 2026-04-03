# Playwright Pipeline — Synthetic COCO Image Generation

Automated image generation pipeline using **Playwright** to interact with Google Gemini's web interface. Generates photorealistic synthetic images for COCO classes 40–59 (20 classes, 200 images each = 4,000 total), with automatic labeling via Roboflow.

## Why Playwright?

The Gemini API has strict daily quotas. This pipeline automates the Gemini **website** using a real browser session, supporting **multiple Google accounts** with automatic rotation when one hits its daily image generation limit.

---

## Environment Setup

### Prerequisites

- **Python 3.11+**
- **Conda** (recommended) or virtualenv
- **Google account(s)** with Gemini Pro access
- **Roboflow API key** (for auto-labeling)

### 1. Create Conda Environment

```bash
conda create -n grad python=3.11 -y
conda activate grad
```

### 2. Install Dependencies

```bash
pip install -r pipeline/requirements.txt
```

### 3. Install Playwright Browser

```bash
playwright install chromium
```

### 4. Configure Environment Variables

Create a `pipeline/.env` file:

```env
# Gemini API keys (optional — only needed for API-based generation)
GEMINI_API_KEY_1=your_key_here
GEMINI_API_KEY_2=your_key_here
GEMINI_API_KEY_3=your_key_here

# Roboflow (required for auto-labeling)
ROBOFLOW_API_KEY=your_roboflow_key
```

---

## Project Structure

```
pipeline/
├── gemini_web.py          # Playwright-based Gemini web automation (main entry)
├── config.py              # Configuration: API keys, paths, COCO classes 40-59
├── prompt_engine.py       # Prompt generation (97,020 combos per class)
├── generate_images.py     # API-based generation (fallback, quota-limited)
├── label_images.py        # Auto-labeling via Roboflow COCO model
├── run_pipeline.py        # CLI runner for API pipeline + labeling
├── tracker.py             # Excel tracking for generation progress
├── upload_to_roboflow.py  # Upload dataset to Roboflow
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not committed)
└── .playwright_profiles/  # Browser profiles per account (not committed)
    ├── account_1/
    ├── account_2/
    ├── account_3/
    └── account_4/
```

---

## Usage

### Step 1: Log In to Google Accounts

Run this once per account. A Chromium browser will open — log in to your Google account, then close the browser window.

```bash
conda activate grad

python pipeline/gemini_web.py --login 1
python pipeline/gemini_web.py --login 2
python pipeline/gemini_web.py --login 3
python pipeline/gemini_web.py --login 4
```

Each account gets its own persistent browser profile in `.playwright_profiles/account_N/`.

### Step 2: Generate Images

**Generate all remaining images across all classes:**

```bash
python pipeline/gemini_web.py --all-remaining
```

**Generate images for a specific class:**

```bash
python pipeline/gemini_web.py --class donut --start 32 --end 200
python pipeline/gemini_web.py --class bed --start 114 --end 200
python pipeline/gemini_web.py --class "potted plant" --start 21 --end 200
```

### Step 3: Label Images

After generating images, auto-label them using the Roboflow COCO detection model:

```bash
python pipeline/run_pipeline.py label
```

Or label a specific class by COCO ID:

```bash
python pipeline/run_pipeline.py label 54   # label donuts (class 54)
python pipeline/run_pipeline.py label 59   # label beds (class 59)
```

Labels are saved in YOLO format (`dataset/labels/<class>/`).

---

## How It Works

### Prompt Engine

Each image uses a unique prompt generated from 5 variation dimensions:

| Dimension       | Options |
|-----------------|---------|
| Object state    | 12      |
| Scene           | 15      |
| Viewpoint       | 7       |
| Lighting        | 7       |
| Disturbance     | 11      |

Total: **97,020 unique combinations per class**, deterministically sampled with `seed=42`.

### Multi-Account Rotation

1. Starts with account #1
2. When Gemini responds with "can't generate more images today", marks that account as exhausted
3. Automatically switches to the next available account
4. Retries the failed image with the new account
5. Stops when all accounts are exhausted

### Error Recovery

- **Page not ready**: Dismisses consent/welcome dialogs, retries with page refresh (up to 3 attempts)
- **Single failure**: Automatically retries once with a fresh page
- **5+ consecutive failures**: Full page refresh before next attempt
- **Rate limit**: Switches account and retries immediately

### Image Download Methods

Three fallback methods (tried in order):

1. **Download button** — clicks "Download full size image"
2. **URL fetch** — fetches the image `src` URL directly (supports base64 data URIs)
3. **Screenshot** — screenshots the image element as a last resort

All images are resized to **1408×768** (pipeline standard).

---

## COCO Classes (40–59)

| ID | Class        | ID | Class        |
|----|--------------|----|--------------|
| 40 | wine glass   | 50 | broccoli     |
| 41 | cup          | 51 | carrot       |
| 42 | fork         | 52 | hot dog      |
| 43 | knife        | 53 | pizza        |
| 44 | spoon        | 54 | donut        |
| 45 | bowl         | 55 | cake         |
| 46 | banana       | 56 | chair        |
| 47 | apple        | 57 | couch        |
| 48 | sandwich     | 58 | potted plant |
| 49 | orange       | 59 | bed          |

---

## Output

```
dataset/
├── images/
│   ├── wine_glass/     # 200 PNG images (1408×768)
│   ├── cup/
│   ├── ...
│   └── bed/
├── labels/
│   ├── wine_glass/     # 200 YOLO-format .txt label files
│   ├── cup/
│   └── ...
└── boxed_previews/     # images with bounding boxes drawn (for QA)
```

### Label Format (YOLO)

Each `.txt` file contains one line per detected object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized (0–1) relative to image dimensions.

---

## Notes

- Always activate the conda environment before running: `conda activate grad`
- Browser profiles in `.playwright_profiles/` contain login sessions — do **not** commit them
- The `.env` file contains API keys — do **not** commit it
- Images that already exist are automatically skipped (safe to re-run)
- The script uses `headless=False` (visible browser) so you can monitor progress
