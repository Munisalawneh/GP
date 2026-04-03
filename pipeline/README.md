# Playwright Pipeline — Synthetic COCO Image Generation

Automated image generation pipeline using **Playwright** to control Google Gemini's web interface. Generates photorealistic synthetic images for COCO classes 40–59 (20 classes, 200 images each).

---

## Full Setup Guide (Start to Finish)

### 1. Install Conda

If you don't have Conda installed:
- Download **Miniconda** from: https://docs.conda.io/en/latest/miniconda.html
- Run the installer, check "Add to PATH" during setup
- Restart your terminal after installation

Verify it works:

```bash
conda --version
```

### 2. Clone the Repository

```bash
git clone https://github.com/Munisalawneh/GP.git
cd GP
git checkout Playwright-pipeline
```

### 3. Create the Python Environment

```bash
conda create -n grad python=3.11 -y
conda activate grad
```

> **IMPORTANT:** You must run `conda activate grad` every time you open a new terminal before running any pipeline commands.

### 4. Install Python Dependencies

```bash
pip install -r pipeline/requirements.txt
```

This installs:

| Package        | Version | Purpose                              |
|----------------|---------|--------------------------------------|
| playwright     | 1.58.0  | Browser automation for Gemini        |
| Pillow         | 12.1.1  | Image resizing                       |
| google-genai   | 1.68.0  | Gemini API client (fallback)         |
| python-dotenv  | 1.2.2   | Load `.env` config files             |
| requests       | 2.32.5  | HTTP requests for Roboflow labeling  |
| openpyxl       | 3.1.5   | Excel tracking spreadsheet           |

### 5. Install the Chromium Browser for Playwright

```bash
playwright install chromium
```

This downloads a Chromium browser that Playwright will control. It's separate from your regular Chrome — nothing changes on your system.

### 6. Set Up API Keys

Create a file called `.env` inside the `pipeline/` folder:

```bash
# On Windows:
notepad pipeline\.env

# Or on Mac/Linux:
nano pipeline/.env
```

Paste this into the file and fill in your keys:

```env
# Roboflow API key (REQUIRED for auto-labeling)
# Get yours at: https://app.roboflow.com → Settings → API Key
ROBOFLOW_API_KEY=your_roboflow_api_key_here

# Gemini API keys (OPTIONAL — only needed for API-based generation, not Playwright)
GEMINI_API_KEY_1=
GEMINI_API_KEY_2=
GEMINI_API_KEY_3=
```

Save and close.

### 7. Log In Your Google Account(s)

This is the most important step. You need to log in to at least **one** Google account that has Gemini Pro access.

```bash
conda activate grad
python pipeline/gemini_web.py --login 1
```

**What happens:**
1. A Chromium browser window opens and goes to gemini.google.com
2. You log in to your Google account manually (email + password + 2FA if enabled)
3. Wait until you see the Gemini chat interface
4. **Close the browser window** (just click the X button)
5. Your login session is saved in `pipeline/.playwright_profiles/account_1/`

**To add more accounts** (for faster generation with account rotation):

```bash
python pipeline/gemini_web.py --login 2
python pipeline/gemini_web.py --login 3
python pipeline/gemini_web.py --login 4
```

Each account needs a **different** Google account. Log in, see the chat, close the browser.

> **Note:** You only need to do this login step once per account. The session is saved and reused automatically.

---

## How to Generate Images

Always activate the environment first:

```bash
conda activate grad
```

### Generate ALL remaining images (recommended):

```bash
python pipeline/gemini_web.py --all-remaining
```

This will:
- Check which classes still need images
- Start generating from where it left off
- Automatically skip images that already exist
- Rotate between accounts when one hits the daily limit

### Generate a specific class:

```bash
python pipeline/gemini_web.py --class donut --start 32 --end 200
python pipeline/gemini_web.py --class bed --start 114 --end 200
python pipeline/gemini_web.py --class "potted plant" --start 21 --end 200
```

### Label images after generating:

```bash
python pipeline/run_pipeline.py label         # label all classes
python pipeline/run_pipeline.py label 54      # label only donuts (class 54)
```

---

## Troubleshooting

### "No accounts logged in!"
You haven't run the login step. Run `python pipeline/gemini_web.py --login 1` first.

### Browser opens but Gemini doesn't load
Your Google account might not have Gemini access, or there's a regional restriction. Try a different account.

### "can't generate more images today"
You've hit Gemini's daily limit on that account. The script will automatically switch to the next account. If all accounts are exhausted, wait until tomorrow or add more accounts.

### Timeout errors on input field
The page didn't load properly. The script will automatically retry with a page refresh. If it keeps happening, try:
1. Run `python pipeline/gemini_web.py --login N` again for that account number
2. Make sure you can access gemini.google.com manually in a regular browser

### `conda activate grad` doesn't work
Make sure Conda is installed and initialized:
```bash
conda init powershell   # Windows
conda init bash         # Mac/Linux
```
Then restart your terminal.

---

## Project Structure

```
pipeline/
├── gemini_web.py          # Main script — Playwright web automation
├── config.py              # Configuration: paths, COCO classes 40-59
├── prompt_engine.py       # Generates unique prompts (97,020 combos/class)
├── generate_images.py     # API-based generation (fallback)
├── label_images.py        # Auto-labeling via Roboflow
├── run_pipeline.py        # CLI for labeling
├── tracker.py             # Progress tracking (Excel)
├── upload_to_roboflow.py  # Upload dataset to Roboflow
├── requirements.txt       # Python dependencies
├── .env                   # API keys (DO NOT commit)
└── .playwright_profiles/  # Browser login sessions (DO NOT commit)
```

## Output

Generated images go to `dataset/images/<class_name>/` (1408×768 PNG).
Labels go to `dataset/labels/<class_name>/` (YOLO format `.txt` files).

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
