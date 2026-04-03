"""
Gemini Web Automation — Uses Playwright to generate images via Gemini website.
Supports multiple Google accounts with automatic rotation on daily limit.

Usage:
  1. Log in to each account (run once per account):
       python gemini_web.py --login 1
       python gemini_web.py --login 2
       python gemini_web.py --login 3
       python gemini_web.py --login 4

  2. Generate images for a class:
       python gemini_web.py --class bed --start 111 --end 200

  3. Or generate for all remaining classes:
       python gemini_web.py --all-remaining
"""

import os
import sys
import time
import argparse
import base64

from PIL import Image
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

from config import CLASSES, IMAGES_DIR, IMAGES_PER_CLASS_PHASE_B
from prompt_engine import get_variation_prompts
from generate_images import sanitize_name


# ─── Constants ───────────────────────────────────────────────────────────────

GEMINI_URL = "https://gemini.google.com/app"
PROFILES_DIR = os.path.join(os.path.dirname(__file__), ".playwright_profiles")
NUM_ACCOUNTS = 4  # total accounts available
TARGET_SIZE = (1408, 768)
DOWNLOAD_WAIT_SEC = 150  # max wait for image generation

# Rate-limit keywords from Gemini's response
RATE_LIMIT_KEYWORDS = [
    "can't generate more images",
    "come back tomorrow",
    "can't create more images",
    "reached your limit",
    "too many requests",
    "limit for today",
]


# ─── Selectors (verified against live Gemini DOM) ───────────────────────────

SEL_INPUT = '.ql-editor[aria-label="Enter a prompt for Gemini"]'
SEL_SEND = 'button[aria-label="Send message"]'
SEL_NEW_CHAT = 'a[href="/app"]'  # pen icon top-left
SEL_DOWNLOAD = 'button[aria-label="Download full size image"]'
SEL_IMAGE = 'img.image.loaded'  # generated image in response


# ─── JS helpers ──────────────────────────────────────────────────────────────

JS_CHECK_STATE = """() => {
    // Check for loading indicators
    const loading = document.querySelectorAll(
        '[class*="loading"], [class*="spinner"], [class*="progress"]'
    );
    const visibleLoading = Array.from(loading).filter(l => {
        const s = window.getComputedStyle(l);
        return s.display !== 'none' && s.visibility !== 'hidden' && l.offsetHeight > 0;
    });

    // Check for generated images (class "image loaded", size > 200px)
    const imgs = document.querySelectorAll('img.image');
    const bigImages = Array.from(imgs).filter(i => 
        (i.naturalWidth || i.width) > 200 && (i.naturalHeight || i.height) > 200
    );

    // Check for error/limit text in model response
    const turns = document.querySelectorAll('model-response, .model-response-text');
    let errorText = '';
    let isRateLimit = false;
    for (const t of turns) {
        const text = (t.innerText || '').toLowerCase();
        if (text.includes("can't generate more images") ||
            text.includes("come back tomorrow") ||
            text.includes("can't create more images") ||
            text.includes("reached your limit") ||
            text.includes("limit for today")) {
            errorText = t.innerText.substring(0, 200);
            isRateLimit = true;
            break;
        }
        if (text.match(/i('m| am) (not able|unable)|can't (create|generate)|sorry.*couldn't/)) {
            errorText = t.innerText.substring(0, 200);
            break;
        }
    }

    return {
        loading: visibleLoading.length,
        imageCount: bigImages.length,
        hasDownloadBtn: !!document.querySelector('button[aria-label="Download full size image"]'),
        errorText: errorText,
        isRateLimit: isRateLimit,
    };
}"""


# ─── Profile helpers ─────────────────────────────────────────────────────────

def get_profile_dir(account_num: int) -> str:
    """Get the browser profile directory for a given account number."""
    return os.path.join(PROFILES_DIR, f"account_{account_num}")


def get_logged_in_accounts() -> list[int]:
    """Return list of account numbers that have been logged in."""
    accounts = []
    for i in range(1, NUM_ACCOUNTS + 1):
        profile = get_profile_dir(i)
        if os.path.isdir(profile):
            accounts.append(i)
    # Also check legacy single profile
    legacy = os.path.join(os.path.dirname(__file__), ".playwright_profile")
    if os.path.isdir(legacy) and 1 not in accounts:
        # Migrate legacy profile to account_1
        import shutil
        dest = get_profile_dir(1)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(legacy, dest)
        accounts.insert(0, 1)
        print(f"[INFO] Migrated legacy profile → account_1")
    return sorted(accounts)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_existing_count(class_name: str) -> int:
    """Count existing images for a class."""
    folder = os.path.join(IMAGES_DIR, sanitize_name(class_name))
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(".png")])


def resize_image(path: str, target: tuple[int, int] = TARGET_SIZE):
    """Resize image to target dimensions if needed."""
    img = Image.open(path)
    if img.size != target:
        img = img.resize(target, Image.LANCZOS)
        img.save(path, optimize=True)


def login_account(account_num: int):
    """Open browser for manual Google login for a specific account."""
    profile_dir = get_profile_dir(account_num)
    os.makedirs(profile_dir, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            profile_dir,
            headless=False,
            viewport={"width": 1280, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(GEMINI_URL, wait_until="domcontentloaded")
        print(f"\n{'='*60}")
        print(f"  ACCOUNT #{account_num}")
        print(f"  Log in to your Google account in the browser window.")
        print(f"  Once you see the Gemini chat, close the browser.")
        print(f"{'='*60}\n")
        try:
            page.wait_for_event("close", timeout=600_000)
        except (PWTimeout, Exception):
            pass
        context.close()
    print(f"[OK] Account #{account_num} profile saved.")


def wait_for_ready(page, max_retries: int = 3) -> bool:
    """Wait until the Gemini input box is visible. Retries with page refresh."""
    for attempt in range(1, max_retries + 1):
        try:
            # Dismiss any overlay / consent dialogs
            try:
                for sel in ['button:has-text("Got it")', 'button:has-text("I agree")',
                            'button:has-text("Accept")', 'button:has-text("Continue")',
                            'button:has-text("No thanks")', 'button:has-text("Dismiss")']:
                    btn = page.locator(sel).first
                    if btn.is_visible(timeout=500):
                        btn.click()
                        page.wait_for_timeout(1000)
            except Exception:
                pass

            page.locator(SEL_INPUT).wait_for(state="visible", timeout=20000)
            return True
        except PWTimeout:
            if attempt < max_retries:
                print(f"    [WAIT] Input not found (attempt {attempt}/{max_retries}), refreshing...")
                page.goto(GEMINI_URL, wait_until="domcontentloaded")
                page.wait_for_timeout(5000)
            else:
                print(f"    [WARN] Input still not found after {max_retries} attempts")
    return False


def start_new_chat(page):
    """Navigate to a fresh Gemini chat and verify it's ready."""
    page.goto(GEMINI_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(3000)
    wait_for_ready(page, max_retries=2)


def send_prompt_and_download(page, prompt: str, output_path: str,
                              timeout_sec: int = DOWNLOAD_WAIT_SEC) -> str:
    """
    Type a prompt into Gemini, wait for image generation, and save it.
    Returns: "ok", "rate_limit", or "error"
    """
    try:
        # 1. Find and fill the input box
        input_box = page.locator(SEL_INPUT)
        input_box.wait_for(state="visible", timeout=15000)
        input_box.click()
        page.wait_for_timeout(500)

        # Use keyboard.type for contenteditable divs (more reliable than fill)
        page.keyboard.type(prompt, delay=5)
        page.wait_for_timeout(500)

        # 2. Click send
        send_btn = page.locator(SEL_SEND)
        send_btn.wait_for(state="visible", timeout=5000)
        send_btn.click()
        print(f"    Prompt sent, waiting for image...")

        # 3. Poll for image or error
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            page.wait_for_timeout(3000)

            state = page.evaluate(JS_CHECK_STATE)

            # Rate limit hit
            if state["isRateLimit"]:
                print(f"    [RATE LIMIT] {state['errorText'][:100]}")
                return "rate_limit"

            # Other error from Gemini
            if state["errorText"]:
                print(f"    [GEMINI] {state['errorText'][:120]}")
                return "error"

            # Image appeared
            if state["imageCount"] > 0 and state["loading"] == 0:
                break
        else:
            print(f"    [TIMEOUT] No image within {timeout_sec}s")
            return "error"

        # Extra wait for full render
        page.wait_for_timeout(2000)

        # 4. Download the image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        downloaded = False

        # Method A: Click "Download full size image" button
        try:
            dl_btn = page.locator(SEL_DOWNLOAD).first
            if dl_btn.is_visible(timeout=3000):
                with page.expect_download(timeout=30000) as dl_info:
                    dl_btn.click()
                download = dl_info.value
                download.save_as(output_path)
                downloaded = True
        except Exception:
            pass

        # Method B: Fetch the image src URL directly
        if not downloaded:
            try:
                img_el = page.locator(SEL_IMAGE).first
                src = img_el.get_attribute("src") or ""

                if src.startswith("data:image"):
                    _, b64data = src.split(",", 1)
                    img_bytes = base64.b64decode(b64data)
                    with open(output_path, "wb") as f:
                        f.write(img_bytes)
                    downloaded = True
                elif src.startswith("https://"):
                    resp = page.request.get(src)
                    if resp.ok:
                        with open(output_path, "wb") as f:
                            f.write(resp.body())
                        downloaded = True
            except Exception as e:
                print(f"    [WARN] URL fetch failed: {e}")

        # Method C: Screenshot the image element
        if not downloaded:
            try:
                img_el = page.locator(SEL_IMAGE).first
                img_el.screenshot(path=output_path)
                downloaded = True
                print(f"    [INFO] Used screenshot fallback")
            except Exception:
                pass

        if not downloaded:
            print(f"    [FAIL] Could not save image")
            return "error"

        # 5. Resize to target dimensions
        resize_image(output_path)
        return "ok"

    except PWTimeout as e:
        print(f"    [TIMEOUT] {e}")
        return "error"
    except Exception as e:
        print(f"    [ERROR] {e}")
        return "error"


class AccountRotator:
    """Manages multiple Gemini accounts with Playwright browser contexts."""

    def __init__(self, playwright):
        self._pw = playwright
        self._accounts = get_logged_in_accounts()
        if not self._accounts:
            print("[ERROR] No accounts logged in! Run: python gemini_web.py --login 1")
            sys.exit(1)
        self._exhausted = set()
        self._index = 0
        self._context = None
        self._page = None
        print(f"[ACCOUNTS] {len(self._accounts)} account(s) available: {self._accounts}")
        self._open_current()

    def _open_current(self):
        """Open browser for the current account."""
        if self._context:
            try:
                self._context.close()
            except Exception:
                pass

        acct = self._accounts[self._index]
        profile = get_profile_dir(acct)
        print(f"[ACCOUNT] Switching to account #{acct}")

        self._context = self._pw.chromium.launch_persistent_context(
            profile,
            headless=False,
            viewport={"width": 1280, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
            accept_downloads=True,
        )
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        self._page.goto(GEMINI_URL, wait_until="domcontentloaded")
        self._page.wait_for_timeout(5000)
        # Verify the page is actually ready (dismiss dialogs, wait for input)
        if not wait_for_ready(self._page):
            print(f"[WARN] Account #{acct} page not ready — will retry on first use")

    @property
    def page(self):
        return self._page

    def mark_exhausted(self) -> bool:
        """Mark current account as exhausted. Returns True if switched to a new one."""
        acct = self._accounts[self._index]
        self._exhausted.add(acct)
        print(f"[ACCOUNT] Account #{acct} exhausted ({len(self._exhausted)}/{len(self._accounts)})")

        # Find next non-exhausted account
        for i in range(len(self._accounts)):
            idx = (self._index + 1 + i) % len(self._accounts)
            if self._accounts[idx] not in self._exhausted:
                self._index = idx
                self._open_current()
                return True

        print("[STOP] All accounts exhausted!")
        return False

    @property
    def all_exhausted(self):
        return len(self._exhausted) >= len(self._accounts)

    def close(self):
        if self._context:
            try:
                self._context.close()
            except Exception:
                pass


def generate_all_with_rotation():
    """Generate images for all remaining classes, rotating accounts on limit."""
    # Build work list
    work = []
    for cid, cname in CLASSES.items():
        existing = get_existing_count(cname)
        if existing < IMAGES_PER_CLASS_PHASE_B:
            work.append((cid, cname, existing + 1, IMAGES_PER_CLASS_PHASE_B))

    if not work:
        print("All classes complete!")
        return

    print("\n" + "=" * 60)
    print("  Remaining classes to generate:")
    for cid, cname, start, end in work:
        print(f"    {cname} ({cid}): {start - 1}/{end} → need {end - start + 1} more")
    total_needed = sum(end - start + 1 for _, _, start, end in work)
    print(f"  Total: {total_needed} images")
    print("=" * 60)

    all_prompts_cache = {}
    total_success = 0
    total_fail = 0
    MAX_CONSECUTIVE_ERRORS = 5  # refresh page after this many errors in a row

    with sync_playwright() as p:
        rotator = AccountRotator(p)

        for cid, cname, start_idx, end_idx in work:
            safe_name = sanitize_name(cname)
            folder = os.path.join(IMAGES_DIR, safe_name)
            os.makedirs(folder, exist_ok=True)

            if cname not in all_prompts_cache:
                all_prompts_cache[cname] = get_variation_prompts(cname, count=IMAGES_PER_CLASS_PHASE_B)
            all_prompts = all_prompts_cache[cname]

            print(f"\n{'='*60}")
            print(f"  Generating {cname}: images {start_idx} to {end_idx}")
            print(f"{'='*60}\n")

            class_success = 0
            consecutive_errors = 0

            for i in range(start_idx, end_idx + 1):
                if rotator.all_exhausted:
                    print("\n[STOP] All accounts exhausted — stopping.")
                    break

                filename = f"{safe_name}_{i:03d}.png"
                output_path = os.path.join(folder, filename)

                if os.path.exists(output_path):
                    print(f"  [SKIP] {filename} — already exists")
                    continue

                prompt_idx = i - 1
                if prompt_idx >= len(all_prompts):
                    print(f"  [SKIP] No prompt for index {i}")
                    continue

                prompt = all_prompts[prompt_idx]
                print(f"  [{i}/{end_idx}] Generating {filename}...")

                # If too many consecutive errors, try refreshing the page
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"  [RECOVERY] {consecutive_errors} errors in a row — refreshing page...")
                    start_new_chat(rotator.page)
                    consecutive_errors = 0

                # New chat
                start_new_chat(rotator.page)

                result = send_prompt_and_download(rotator.page, prompt, output_path)

                if result == "ok":
                    size_kb = os.path.getsize(output_path) / 1024
                    print(f"    ✓ Saved ({size_kb:.0f}KB)")
                    total_success += 1
                    class_success += 1
                    consecutive_errors = 0
                elif result == "rate_limit":
                    consecutive_errors = 0
                    # Switch to next account and retry this same image
                    if rotator.mark_exhausted():
                        print(f"    Retrying {filename} with new account...")
                        start_new_chat(rotator.page)
                        result2 = send_prompt_and_download(rotator.page, prompt, output_path)
                        if result2 == "ok":
                            size_kb = os.path.getsize(output_path) / 1024
                            print(f"    ✓ Saved ({size_kb:.0f}KB)")
                            total_success += 1
                            class_success += 1
                        elif result2 == "rate_limit":
                            # This account is also exhausted
                            if not rotator.mark_exhausted():
                                break
                            total_fail += 1
                        else:
                            total_fail += 1
                    else:
                        break
                else:
                    consecutive_errors += 1
                    # On first error, retry once with a page refresh
                    if consecutive_errors == 1:
                        print(f"    ✗ Failed — retrying once...")
                        start_new_chat(rotator.page)
                        result2 = send_prompt_and_download(rotator.page, prompt, output_path)
                        if result2 == "ok":
                            size_kb = os.path.getsize(output_path) / 1024
                            print(f"    ✓ Saved on retry ({size_kb:.0f}KB)")
                            total_success += 1
                            class_success += 1
                            consecutive_errors = 0
                        elif result2 == "rate_limit":
                            consecutive_errors = 0
                            if rotator.mark_exhausted():
                                pass  # will retry on next iteration
                            else:
                                break
                        else:
                            total_fail += 1
                            print(f"    ✗ Failed again")
                    else:
                        total_fail += 1
                        print(f"    ✗ Failed ({consecutive_errors} in a row)")

                rotator.page.wait_for_timeout(2000)

            if rotator.all_exhausted:
                break

            print(f"  → {cname}: {class_success} saved this run")

        rotator.close()

    print(f"\n{'='*60}")
    print(f"  Session complete! {total_success} saved, {total_fail} failed")
    print(f"{'='*60}")


def generate_class(class_name: str, start_idx: int, end_idx: int):
    """Generate images for a single class (uses account rotation)."""
    # Temporarily override CLASSES to only include this class
    cid = None
    for k, v in CLASSES.items():
        if v == class_name:
            cid = k
            break
    if cid is None:
        print(f"Unknown class: {class_name}")
        return

    safe_name = sanitize_name(class_name)
    folder = os.path.join(IMAGES_DIR, safe_name)
    os.makedirs(folder, exist_ok=True)
    all_prompts = get_variation_prompts(class_name, count=IMAGES_PER_CLASS_PHASE_B)

    total = end_idx - start_idx + 1
    print(f"\n{'='*60}")
    print(f"  Generating {class_name}: images {start_idx} to {end_idx} ({total} images)")
    print(f"{'='*60}\n")

    total_success = 0
    total_fail = 0
    MAX_CONSECUTIVE_ERRORS = 5

    with sync_playwright() as p:
        rotator = AccountRotator(p)
        consecutive_errors = 0

        for i in range(start_idx, end_idx + 1):
            if rotator.all_exhausted:
                print("\n[STOP] All accounts exhausted — stopping.")
                break

            filename = f"{safe_name}_{i:03d}.png"
            output_path = os.path.join(folder, filename)

            if os.path.exists(output_path):
                print(f"  [SKIP] {filename} — already exists")
                continue

            prompt_idx = i - 1
            if prompt_idx >= len(all_prompts):
                print(f"  [SKIP] No prompt for index {i}")
                continue

            prompt = all_prompts[prompt_idx]
            print(f"  [{i}/{end_idx}] Generating {filename}...")

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"  [RECOVERY] {consecutive_errors} errors in a row — refreshing page...")
                start_new_chat(rotator.page)
                consecutive_errors = 0

            start_new_chat(rotator.page)
            result = send_prompt_and_download(rotator.page, prompt, output_path)

            if result == "ok":
                size_kb = os.path.getsize(output_path) / 1024
                print(f"    ✓ Saved ({size_kb:.0f}KB)")
                total_success += 1
                consecutive_errors = 0
            elif result == "rate_limit":
                consecutive_errors = 0
                if rotator.mark_exhausted():
                    print(f"    Retrying {filename} with new account...")
                    start_new_chat(rotator.page)
                    result2 = send_prompt_and_download(rotator.page, prompt, output_path)
                    if result2 == "ok":
                        size_kb = os.path.getsize(output_path) / 1024
                        print(f"    ✓ Saved ({size_kb:.0f}KB)")
                        total_success += 1
                    elif result2 == "rate_limit":
                        if not rotator.mark_exhausted():
                            break
                        total_fail += 1
                    else:
                        total_fail += 1
                else:
                    break
            else:
                consecutive_errors += 1
                if consecutive_errors == 1:
                    print(f"    ✗ Failed — retrying once...")
                    start_new_chat(rotator.page)
                    result2 = send_prompt_and_download(rotator.page, prompt, output_path)
                    if result2 == "ok":
                        size_kb = os.path.getsize(output_path) / 1024
                        print(f"    ✓ Saved on retry ({size_kb:.0f}KB)")
                        total_success += 1
                        consecutive_errors = 0
                    elif result2 == "rate_limit":
                        consecutive_errors = 0
                        if rotator.mark_exhausted():
                            pass
                        else:
                            break
                    else:
                        total_fail += 1
                        print(f"    ✗ Failed again")
                else:
                    total_fail += 1
                    print(f"    ✗ Failed ({consecutive_errors} in a row)")

            rotator.page.wait_for_timeout(2000)

        rotator.close()

    print(f"\n{'='*60}")
    print(f"  Done! {class_name}: {total_success} saved, {total_fail} failed")
    print(f"{'='*60}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate images via Gemini website")
    parser.add_argument("--login", type=int, metavar="N",
                        help="Log in to account N (1-4). Run once per account.")
    parser.add_argument("--class", dest="class_name", type=str,
                        help="Class name to generate (e.g., 'bed', 'donut')")
    parser.add_argument("--start", type=int,
                        help="Start image index (e.g., 111)")
    parser.add_argument("--end", type=int,
                        help="End image index (e.g., 200)")
    parser.add_argument("--all-remaining", action="store_true",
                        help="Generate all remaining images for all classes")

    args = parser.parse_args()

    if args.login:
        login_account(args.login)
        return

    if args.all_remaining:
        generate_all_with_rotation()
        return

    if args.class_name:
        cname = args.class_name.replace("_", " ").lower()
        if cname not in [v for v in CLASSES.values()]:
            print(f"Unknown class: {cname}")
            print(f"Available: {', '.join(CLASSES.values())}")
            sys.exit(1)

        existing = get_existing_count(cname)
        start = args.start or (existing + 1)
        end = args.end or IMAGES_PER_CLASS_PHASE_B

        generate_class(cname, start, end)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
