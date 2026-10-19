"""Real-browser end-to-end coverage (optional: pip install data-audit[browser])."""

import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlsplit
import pandas as pd
import pytest
from data_audit.cli import free_port

pw = pytest.importorskip("playwright.sync_api")


@pytest.fixture
def browser_page(tmp_path):
    port = free_port()
    log = (tmp_path / "browser-server.log").open("w")
    process = subprocess.Popen(
        [sys.executable, "-m", "data_audit.cli", "ui", "--no-browser", "--offline", "--port", str(port)],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=os.name != "nt",
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with opener.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=0.3) as response:
                    if response.read() == b"ok":
                        break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail("Browser test server did not become ready")
        with pw.sync_playwright() as p:
            executable = os.environ.get("DATA_AUDIT_CHROMIUM") or shutil.which("chromium")
            browser = p.chromium.launch(
                headless=True, executable_path=executable, args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            context = browser.new_context(accept_downloads=True, viewport={"width": 1440, "height": 1100})
            # Browser must be usable with external requests blocked.
            context.route(
                "**/*",
                lambda route: (
                    route.continue_()
                    if urlsplit(route.request.url).hostname in ("127.0.0.1", "localhost")
                    else route.abort()
                ),
            )
            page = context.new_page()
            page.set_default_timeout(30000)
            page.goto(f"http://127.0.0.1:{port}")
            yield page
            browser.close()
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT if os.name != "nt" else signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        log.close()


def test_real_browser_demo_selection_export_and_restore(browser_page, tmp_path):
    page = browser_page
    page.get_by_role("button", name="Use synthetic demonstration dataset", exact=True).click()
    page.get_by_role("button", name="Run audit", exact=True).click()
    pw.expect(page.get_by_text("3 · Investigate", exact=True)).to_be_visible(timeout=45000)
    pw.expect(page.get_by_text("4 · Export and save", exact=True)).to_be_visible(timeout=30000)
    assert page.get_by_test_id("stException").count() == 0
    # Select a real plot area through Plotly's selection tool.
    chart = page.locator(".js-plotly-plot").first
    chart.scroll_into_view_if_needed()
    chart.hover()
    select = chart.locator('[data-title="Box Select"]')
    select.click()
    box = chart.locator(".nsewdrag").bounding_box()
    assert box
    page.mouse.move(box["x"] + box["width"] * 0.1, box["y"] + box["height"] * 0.1)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.9, box["y"] + box["height"] * 0.9, steps=10)
    page.mouse.up()
    caption = page.get_by_text(re.compile(r"^Selection: [\d,]+ samples$"))
    pw.expect(caption).not_to_have_text("Selection: 0 samples", timeout=10000)
    selected_count = int(caption.inner_text().split()[1].replace(",", ""))
    assert 0 < selected_count < 240
    page.get_by_role("combobox", name="CSV export scope", exact=False).click()
    page.get_by_role("option", name="Selected samples", exact=True).click()
    pw.expect(page.get_by_text(f"Selected samples: {selected_count:,} rows", exact=True)).to_be_visible()
    with page.expect_download() as info:
        page.get_by_role("button", name="Download findings CSV", exact=True).click()
    selected_path = tmp_path / "selected.csv"
    info.value.save_as(selected_path)
    assert len(pd.read_csv(selected_path)) == selected_count
    # Export actual filtered rows, including IDs matching the search string.
    page.get_by_label("Search sample ID", exact=True).fill("sample_000")
    page.get_by_label("Search sample ID", exact=True).press("Enter")
    page.wait_for_timeout(1000)
    combo = page.get_by_role("combobox", name="CSV export scope", exact=False)
    combo.click()
    page.get_by_role("option", name="Filtered rows", exact=True).click()
    pw.expect(page.get_by_text("Filtered rows: 10 rows", exact=True)).to_be_visible()
    with page.expect_download() as info:
        page.get_by_role("button", name="Download findings CSV", exact=True).click()
    destination = tmp_path / "findings.csv"
    info.value.save_as(destination)
    frame = pd.read_csv(destination)
    assert len(frame) == 10 and frame.sample_id.str.startswith("sample_000").all()
    with page.expect_download() as info:
        page.get_by_role("button", name="Save complete run", exact=True).click()
    bundle = tmp_path / "run.audit.zip"
    info.value.save_as(bundle)
    assert bundle.stat().st_size > 1000
    page.get_by_text("Open an existing .audit.zip run", exact=True).click()
    page.locator("input[type=file]").nth(1).set_input_files(bundle)
    page.get_by_role("button", name="Open saved run", exact=True).click()
    pw.expect(
        page.get_by_text("Restored results without fitting or loading extension code.", exact=True)
    ).to_be_visible()
    assert page.get_by_test_id("stException").count() == 0
    page.screenshot(path=str(tmp_path / "investigation.png"), full_page=True)


@pytest.mark.parametrize("kind", ["csv", "xlsx", "xls"])
def test_real_browser_upload_shared_readers(browser_page, tmp_path, kind):
    page = browser_page
    f = pd.DataFrame({"id": [f"{i:04}" for i in range(40)], "feature": range(40), "label": ["A", "B"] * 20})
    path = tmp_path / ("source." + kind)
    if kind == "csv":
        f.to_csv(path, index=False)
    elif kind == "xlsx":
        f.to_excel(path, index=False)
    else:
        import xlwt

        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet("samples")
        for i, row in enumerate([list(f.columns), *f.values.tolist()]):
            for j, value in enumerate(row):
                sheet.write(i, j, value)
        workbook.save(str(path))
    page.locator("input[type=file]").first.set_input_files(path)
    id_input = page.get_by_label("ID column (optional; preserves textual IDs)", exact=True)
    id_input.fill("id")
    id_input.press("Enter")
    page.get_by_role("button", name="Load source data", exact=True).click()
    page.get_by_role("button", name="Run audit", exact=True).click()
    pw.expect(page.get_by_text("3 · Investigate", exact=True)).to_be_visible(timeout=45000)
    with page.expect_download() as info:
        page.get_by_role("button", name="Download findings CSV", exact=True).click()
    destination = tmp_path / "findings.csv"
    info.value.save_as(destination)
    out = pd.read_csv(destination, dtype={"sample_id": str})
    assert out.sample_id.tolist() == f.id.tolist()
    assert page.get_by_test_id("stException").count() == 0
