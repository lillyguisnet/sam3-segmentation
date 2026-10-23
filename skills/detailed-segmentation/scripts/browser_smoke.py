"""Optional isolated-browser smoke check; not collected by ordinary pytest."""

import argparse
import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--browser-executable",
        type=Path,
        help="Dedicated test browser, never a shared-profile launcher",
    )
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    args.out.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(QuietHandler, directory=str(args.export.resolve()))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors = []
    try:
        with sync_playwright() as p:
            options = dict(headless=True, args=["--disable-gpu"])
            if args.browser_executable:
                options["executable_path"] = str(args.browser_executable)
            browser = p.chromium.launch(**options)
            page = browser.new_page(viewport=dict(width=1320, height=1000))
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}")
            page.wait_for_function("window.__segmentationReady === true")
            assert page.locator("#count").inner_text() != "—"
            page.screenshot(path=str(args.out / "desktop.png"), full_page=True)
            page.locator("#nextflag").click()
            assert "mask-" in page.locator("#selection").inner_text()
            page.locator("#original").click()
            page.wait_for_function(
                "document.getElementById('original').classList.contains('active')"
            )
            page.locator("#original").click()
            page.locator("#fill").check()
            page.locator("#flags").check()
            page.locator("#layer").select_option(index=1)
            page.locator("#pixels").click()
            page.wait_for_function("document.getElementById('zoom').textContent.startsWith('6.0')")
            box = page.locator("#canvas").bounding_box()
            page.mouse.move(box["x"] + 80, box["y"] + 80)
            page.mouse.down()
            page.mouse.move(box["x"] + 120, box["y"] + 100)
            page.mouse.up()
            page.locator("#fit").click()
            page.locator("#fill").uncheck()
            async_download = page.expect_download()
            with async_download as info:
                page.locator('a[href="masks.zip"]').click()
            assert info.value.suggested_filename == "masks.zip"
            page.set_viewport_size(dict(width=430, height=900))
            page.wait_for_timeout(100)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(args.out / "mobile.png"), full_page=True)
            page.evaluate("""() => {const s=document.documentElement.style;
                s.setProperty('--color-background-primary','#111111');
                s.setProperty('--color-background-secondary','#222222');
                s.setProperty('--color-text-primary','#eeeeee');
                s.setProperty('--color-text-secondary','#bbbbbb');
                s.setProperty('--color-border-primary','#555555');
                s.setProperty('--color-text-info','#ffffff');} """)
            page.wait_for_timeout(100)
            assert (
                page.evaluate("getComputedStyle(document.body).backgroundColor")
                == "rgb(17, 17, 17)"
            )
            page.screenshot(path=str(args.out / "dark.png"), full_page=True)
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    result = dict(
        page_errors=errors,
        checks=[
            "desktop",
            "mobile-no-overflow",
            "theme variables",
            "selection",
            "original comparison",
            "fill",
            "flags",
            "category",
            "zoom",
            "pan",
            "download",
        ],
    )
    (args.out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
