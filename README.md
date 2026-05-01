# SYN-COCO

SYN-COCO is a computer vision research project for testing whether AI-generated images can reduce the performance gap when real object-detection data is hard, expensive, or impractical to collect.

The dataset follows the MS COCO object-detection label space and uses synthetic images generated with Nano Banana (Gemini 3). The public Roboflow dataset version is:

https://app.roboflow.com/munisdrafts/syn-coco-qdonc-hopnq/2

The experiment trains YOLO26 models on the synthetic dataset, then evaluates them on real COCO validation images to measure synthetic-to-real transfer.

## Project Layout

```text
GP/
  README.md
  results.txt
  coco.yaml
  requirements.txt
  SYN_YOLO26x_Traning.ipynb
  relabel_dataset.py
  pipeline/
    config.py
    prompt_engine.py
    generate_images.py
    gemini_web.py
    label_images.py
    run_pipeline.py
    tracker.py
    upload_to_roboflow.py
    upload_sdk.py
    validate_dataset.py
    redistribute.py
  SYN_COCO/
    data.yaml
    train/images/
    train/labels/
    valid/images/
    valid/labels/
  weights/
    SYN_YOLO26s.pt
    SYN_YOLO26m.pt
    SYN_YOLO26l.pt
    SYN_YOLO26x.pt
  SYN_COCO Overleaf/
  COCO VS SYN/
```

## Important Files

| File | Purpose |
| --- | --- |
| `SYN_YOLO26x_Traning.ipynb` | Main training and evaluation notebook. Downloads/locates SYN-COCO, trains YOLO26, builds real COCO validation labels, and records metrics. |
| `results.txt` | Current SYN validation and real COCO validation metrics for YOLO26n/s/m/l/x. |
| `SYN_COCO/data.yaml` | Roboflow YOLO26 dataset config used by training. |
| `coco.yaml` | COCO 2017 reference label map and Ultralytics dataset config. |
| `pipeline/gemini_web.py` | Playwright automation for Gemini web image generation with account rotation. |
| `pipeline/prompt_engine.py` | Prompt variation engine for realistic COCO-style scenes. |
| `pipeline/label_images.py` | Roboflow COCO model labeling and boxed preview generation. |
| `pipeline/upload_sdk.py` | Roboflow SDK upload path for relabeled datasets. |
| `pipeline/validate_dataset.py` | Dataset image/label validation helper. |
| `relabel_dataset.py` | Relabels local images with YOLO26x and writes YOLO labels. |

## Research Workflow

1. Generate diverse synthetic images using engineered prompts.
2. Annotate images in Roboflow and export YOLO26 format.
3. Train YOLO26 models on SYN-COCO.
4. Evaluate trained weights on SYN-COCO validation.
5. Convert real COCO val2017 annotations into the matching YOLO label space.
6. Evaluate the same trained weights on real COCO val2017.
7. Compare synthetic validation scores against real validation scores.

## Setup

```powershell
conda create -n grad python=3.11 -y
conda activate grad
pip install -r requirements.txt
pip install -r pipeline/requirements.txt
playwright install chromium
```

Create `pipeline/.env`:

```env
ROBOFLOW_API_KEY=your_roboflow_api_key_here
GEMINI_API_KEY_1=
GEMINI_API_KEY_2=
GEMINI_API_KEY_3=
```

For Gemini web generation, log in once per account:

```powershell
python pipeline/gemini_web.py --login 1
python pipeline/gemini_web.py --login 2
```

## Dataset Generation

Generate remaining images:

```powershell
python pipeline/gemini_web.py --all-remaining
```

Generate a class range manually:

```powershell
python pipeline/gemini_web.py --class banana --start 1 --end 200
python pipeline/gemini_web.py --class "potted plant" --start 1 --end 200
```

Label generated images:

```powershell
python pipeline/run_pipeline.py label
```

Upload image/label pairs to Roboflow:

```powershell
python pipeline/upload_sdk.py --dataset-root dataset --tag SYN-COCO
```

## Training And Evaluation

Use `SYN_YOLO26x_Traning.ipynb` for the full training/evaluation flow.

Main notebook stages:

1. Verify GPU and Python environment.
2. Locate or download the Roboflow SYN-COCO export.
3. Prepare `data_fixed.yaml` with absolute dataset paths.
4. Download COCO val2017 images and annotations.
5. Train YOLO26 on SYN-COCO.
6. Evaluate on SYN-COCO validation.
7. Convert COCO val2017 annotations to YOLO labels.
8. Evaluate on real COCO val2017.

Current results:

| SYN Model | SYN mAP50-95 | SYN mAP50 | SYN mAP75 | REAL mAP50-95 | REAL mAP50 | REAL mAP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26n | 0.539213 | 0.694938 | 0.596969 | 0.138968 | 0.212157 | 0.148705 |
| YOLO26s | 0.667051 | 0.778553 | 0.717442 | 0.250638 | 0.360981 | 0.269858 |
| YOLO26m | 0.704938 | 0.807951 | 0.754258 | 0.227839 | 0.327709 | 0.245538 |
| YOLO26l | 0.755635 | 0.845381 | 0.804426 | 0.352206 | 0.487931 | 0.384526 |
| YOLO26x | 0.766471 | 0.855739 | 0.822162 | 0.393776 | 0.546147 | 0.431982 |

## Notes

- Keep API keys in `pipeline/.env`.
- Keep generated datasets, trained weights, COCO downloads, and browser profiles local.
- Use `results.txt` as the quick metric summary.
- Use the notebook when exact training and real COCO evaluation parity matters.
