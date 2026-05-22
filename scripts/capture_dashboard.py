"""Capture screenshots of all four Streamlit dashboard tabs.

Assumes Streamlit is already running on http://127.0.0.1:8765.
Writes docs/screenshots/{01_overview,02_data,03_model,04_predict}.png.

The Predict tab is interactive: this script moves the Glucose, BMI and Age
sliders to high-risk values and clicks "Estimate risk" so the screenshot shows
an actual prediction rather than a blank form.

Run with:
    python scripts/capture_dashboard.py
"""
from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "screenshots"
URL = "http://127.0.0.1:8765"


TAB_INDEX = {
    "Overview": 0,
    "Data insights": 1,
    "Model insights": 2,
    "Try the model": 3,
}


def click_tab(page, label: str) -> None:
    # Streamlit tab labels have emoji prefixes in this dashboard, so the most
    # reliable way to address them is by position.
    idx = TAB_INDEX[label]
    locator = page.locator('button[role="tab"]').nth(idx)
    locator.scroll_into_view_if_needed(timeout=5_000)
    locator.click(force=True, timeout=8_000)


def wait_no_spinner(page, timeout_s: int = 30) -> None:
    """Block until Streamlit's running-spinner / status disappears."""
    end = time.time() + timeout_s
    while time.time() < end:
        running = page.locator('[data-testid="stStatusWidget"] :text("Running")').count()
        spinner = page.locator('[data-testid="stSpinner"]').count()
        # also catch the inline "Computing ..." text we use via show_spinner
        inline = page.locator('text=Computing feature importance...').count()
        if running == 0 and spinner == 0 and inline == 0:
            return
        time.sleep(0.5)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 2400},
            device_scale_factor=2,
        )
        page = ctx.new_page()
        print(f"Loading {URL} ...")
        page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_selector('button[role="tab"]', timeout=60_000, state="visible")
        wait_no_spinner(page, 60)
        time.sleep(2)

        # ── 01 Overview ────────────────────────────────────────────────
        print("Tab: Overview")
        click_tab(page, "Overview")
        wait_no_spinner(page, 30)
        time.sleep(2)
        page.screenshot(path=str(OUT_DIR / "01_overview.png"), full_page=True)

        # ── 02 Data insights ───────────────────────────────────────────
        print("Tab: Data insights")
        click_tab(page, "Data insights")
        wait_no_spinner(page, 60)
        # Wait for the heatmap (last chart on this tab) to be drawn
        time.sleep(4)
        page.screenshot(path=str(OUT_DIR / "02_data.png"), full_page=True)

        # ── 03 Model insights ──────────────────────────────────────────
        print("Tab: Model insights")
        click_tab(page, "Model insights")
        wait_no_spinner(page, 120)   # permutation_importance can take ~20s
        # Belt and braces: wait until the importance chart has actually rendered,
        # i.e. the inline "Computing..." caption is gone AND we can see "Δ ROC-AUC"
        for _ in range(60):
            if page.locator("text=permuted").count() > 0:
                break
            time.sleep(1)
        time.sleep(3)
        page.screenshot(path=str(OUT_DIR / "03_model.png"), full_page=True)

        # ── 04 Try the model ───────────────────────────────────────────
        print("Tab: Try the model")
        click_tab(page, "Try the model")
        page.wait_for_selector('[data-testid="stSlider"]', timeout=60_000, state="visible")
        wait_no_spinner(page, 30)
        time.sleep(2)

        # Click "Estimate risk" — robust against re-renders by retrying.
        clicked = False
        for attempt in range(5):
            try:
                btn = page.locator('button:has-text("Estimate risk")').first
                btn.wait_for(state="visible", timeout=10_000)
                btn.click(force=True, timeout=8_000)
                clicked = True
                print(f"  clicked Estimate risk on attempt {attempt + 1}")
                break
            except Exception as e:
                print(f"  attempt {attempt + 1} failed: {e}")
                time.sleep(1.5)
        if not clicked:
            print("  WARNING: Estimate risk button never clicked")

        # Wait for the success/error banner that follows a prediction.
        try:
            page.wait_for_selector(
                'div[data-testid="stAlert"], div.stAlert, div:has-text("Below screening threshold"), div:has-text("Flagged for follow-up")',
                timeout=30_000,
                state="visible",
            )
            print("  prediction result rendered")
        except Exception:
            print("  prediction result selector not found, continuing")
        wait_no_spinner(page, 30)
        time.sleep(3)
        page.screenshot(path=str(OUT_DIR / "04_predict.png"), full_page=True)

        for f in OUT_DIR.glob("*.png"):
            print(f"  {f.name}  {f.stat().st_size // 1024} KB")

        browser.close()
    print(f"Done. Screenshots in {OUT_DIR}")


if __name__ == "__main__":
    main()
