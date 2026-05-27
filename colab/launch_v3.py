"""Launch training on Colab: connect GPU, inject code, run cell.

Uses Chrome CDP. Strategy:
1. Find Colab notebook tab
2. Click "Connect" button to allocate GPU runtime
3. Wait for runtime connection (when google.colab becomes available)
4. Inject training code into Monaco editor
5. Click "Run" to execute the training cell
"""

import asyncio
import json
import base64
import urllib.request
import websockets
import sys
import time


CDP_PORT = 40107
NOTEBOOK_ID = "1U3AYDmQJqR3RT2MzuIiONZ0ZunOGorGM"


class CDP:
    def __init__(self, url):
        self.url = url
        self.ws = None
        self._id = 0
        self._pending = {}

    async def connect(self):
        self.ws = await websockets.connect(self.url)
        asyncio.create_task(self._reader())

    async def _reader(self):
        while True:
            try:
                msg = json.loads(await self.ws.recv())
                if msg.get("id") in self._pending:
                    self._pending[msg["id"]].set_result(msg)
            except Exception:
                break

    async def send(self, method, params=None):
        if params is None:
            params = {}
        self._id += 1
        f = asyncio.get_event_loop().create_future()
        self._pending[self._id] = f
        await self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        return await f

    async def js(self, expr, timeout_sec=10):
        """Evaluate JS and return value or None."""
        r = await self.send("Runtime.evaluate", {
            "expression": expr,
            "returnByValue": True,
            "awaitPromise": True,
        })
        result = r.get("result", {})
        if "exceptionDetails" in result:
            return None
        return result.get("result", {}).get("value")

    async def click(self, selector):
        """Click element by CSS selector using DOM.click."""
        try:
            # Use Runtime.evaluate to find and click the element
            return await self.js(f"""
            (function() {{
                var el = document.querySelector({json.dumps(selector)});
                if (!el) return 'not found: {selector}';
                var rect = el.getBoundingClientRect();
                el.click({{bubbles: true, cancelable: true}});
                return 'clicked {selector}';
            }})();
            """)
        except Exception as e:
            return f"error: {e}"

    async def wait_for(self, js_cond, timeout=30, interval=1):
        """Wait for JS condition to be truthy."""
        for i in range(int(timeout / interval)):
            val = await self.js(js_cond)
            if val:
                return val
            await asyncio.sleep(interval)
        return None

    async def close(self):
        try:
            await self.ws.close()
        except Exception:
            pass


async def get_targets():
    resp = urllib.request.urlopen(f"http://localhost:{CDP_PORT}/json")
    return json.loads(resp.read().decode())


async def main():
    # Find notebook tab
    targets = await get_targets()
    tab = None
    for t in targets:
        if NOTEBOOK_ID in t.get("url", "") and t.get("type") == "page":
            tab = t
            break
    if not tab:
        for t in targets:
            if "colab.research.google.com/drive/" in t.get("url", "") and t.get("type") == "page":
                tab = t
                break
    if not tab:
        print("ERROR: No Colab notebook tab found!")
        return

    print(f"Using tab: {tab.get('title','?')}")
    print(f"URL: {tab.get('url','')[:100]}")

    cdp = CDP(tab["webSocketDebuggerUrl"])
    await cdp.connect()
    await cdp.send("Page.enable")
    await cdp.send("Runtime.enable")
    print("Connected to Chrome CDP\n")

    # ---- STEP 1: Ensure we're on the right notebook ----
    print("=" * 60)
    print("STEP 1: Navigate to notebook")
    print("=" * 60)
    await cdp.send("Page.navigate", {"url": f"https://colab.research.google.com/drive/{NOTEBOOK_ID}"})
    await asyncio.sleep(3)

    # ---- STEP 2: Click "Connect" button to allocate GPU ----
    print("\n" + "=" * 60)
    print("STEP 2: Connect to GPU runtime")
    print("=" * 60)

    # Find and click the Connect button
    click_result = await cdp.js("""
    (function() {
        // Try all possible connect button variations
        var selectors = [
            'button:has-text("Connect")',
            'button:has-text("connect")',
            'span:has-text("Connect")',
        ];
        // Broader search
        var buttons = document.querySelectorAll('button, span[role="button"], div[role="button"]');
        for (var b of buttons) {
            var text = (b.textContent || '').trim();
            if (text === 'Connect' || text.startsWith('Connect')) {
                if (b.offsetParent !== null) {
                    b.click();
                    return 'Clicked: "' + text + '" on ' + b.tagName;
                }
            }
        }
        // Try span elements too
        var spans = document.querySelectorAll('span');
        for (var s of spans) {
            if ((s.textContent || '').trim() === 'Connect') {
                var parent = s.closest('button, [role="button"]');
                if (parent && parent.offsetParent !== null) {
                    parent.click();
                    return 'Clicked parent of Connect span';
                }
            }
        }
        return 'No connect button found';
    })();
    """)
    print(f"Connect attempt: {click_result}")

    # Wait for the runtime to connect (google.colab becomes available)
    print("\nWaiting for runtime to connect...")
    runtime_connected = await cdp.wait_for(
        "typeof google !== 'undefined' && typeof google.colab !== 'undefined' && "
        "typeof google.colab.kernel !== 'undefined'",
        timeout=60,
        interval=2
    )
    if runtime_connected:
        print("Runtime connected! google.colab is available.")
    else:
        print("Timeout waiting for runtime. Checking current state...")
        # Check what we have
        colab_state = await cdp.js("typeof google !== 'undefined' ? Object.keys(google) : 'no google'")
        print(f"  google keys: {colab_state}")

        # Maybe auto-connect is happening. Let's check if there's an A100 selection dialog
        dialog = await cdp.js("""
        (function() {
            var dialogs = document.querySelectorAll('[role="dialog"], .dialog, [class*="modal"]');
            var info = [];
            for (var d of dialogs) {
                if (d.offsetParent !== null) {
                    info.push((d.textContent || '').trim().substring(0, 200));
                }
            }
            return info.length > 0 ? info : 'no visible dialogs';
        })();
        """)
        print(f"  Visible dialogs: {dialog}")

        # Check if we need to select runtime type
        runtime_ui = await cdp.js("""
        (function() {
            var items = document.querySelectorAll('[role="listbox"] option, [role="radio"], select option');
            var info = [];
            for (var i of items) {
                var t = (i.textContent || '').trim();
                if (t.includes('A100') || t.includes('T4') || t.includes('GPU') || t.includes('TPU')) {
                    info.push(t);
                }
            }
            return info;
        })();
        """)
        print(f"  GPU options found: {runtime_ui}")

    # ---- STEP 3: Inject training code into Monaco editor ----
    print("\n" + "=" * 60)
    print("STEP 3: Inject training code into cell")
    print("=" * 60)

    # Build the training code to inject
    training_code = """# === SWE-Agent MoE: Full Pretraining Pipeline ===
import os, sys, subprocess, time, json
from pathlib import Path

print("="*60)
print("SWE-Agent MoE Pretraining Pipeline")
print("="*60)

# Step 1: Install dependencies
print("\\n[1/5] Installing dependencies...")
deps = [
    "torch>=2.4.0", "torchvision>=0.19.0", "transformers>=4.44.0",
    "accelerate>=0.33.0", "datasets>=2.20.0", "wandb>=0.17.0",
    "sentencepiece>=0.2.0", "protobuf>=4.25.0", "einops>=0.7.0",
]
for dep in deps:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])
print("Dependencies installed")

# Step 2: Mount Drive
print("\\n[2/5] Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')

# Step 3: Copy project
print("\\n[3/5] Setting up project...")
import pathlib
project_path = pathlib.Path('/content/model-training-pipeline')
if not project_path.exists():
    !cp -r /content/drive/MyDrive/model-training-pipeline /content/
sys.path.insert(0, str(project_path))
py_files = len(list(project_path.rglob('*.py')))
print(f"Project ready: {py_files} .py files")

# Step 4: Hardware check
print("\\n[4/5] Hardware check...")
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu} ({mem:.1f} GB)")
    print(f"CUDA: {torch.version.cuda}")
else:
    print("WARNING: No GPU detected!")

# Step 5: Start pretraining
print("\\n[5/5] Starting pretraining...")
os.chdir(str(project_path))
!python train/pretrain.py 2>&1 | tee /content/drive/MyDrive/pretrain_log.txt
print("Pretraining complete!")
"""

    # Try injecting via Monaco API
    inject_result = await cdp.js(f"""
    (function() {{
        var code = {json.dumps(training_code)};

        // Method 1: Monaco editor API
        try {{
            if (typeof monaco !== 'undefined' && monaco.editor) {{
                var editors = monaco.editor.getEditors();
                if (editors && editors.length > 0) {{
                    editors[0].setValue(code);
                    return 'injected via Monaco API';
                }}
            }}
        }} catch(e) {{}}

        // Method 2: Find the textarea and use native setter
        try {{
            var textareas = document.querySelectorAll('textarea');
            for (var ta of textareas) {{
                if ((ta.getAttribute('aria-label') || '').includes('code') ||
                    (ta.getAttribute('aria-label') || '').includes('Code')) {{
                    var setter = Object.getOwnPropertyDescriptor(
                        window.HTMLTextAreaElement.prototype, 'value'
                    ).set;
                    setter.call(ta, code);
                    ta.dispatchEvent(new Event('input', {{bubbles: true}}));
                    ta.dispatchEvent(new Event('change', {{bubbles: true}}));
                    return 'injected via textarea';
                }}
            }}
        }} catch(e) {{}}

        // Method 3: Find the cell's editable area
        try {{
            var cell = document.querySelector('[role="code"]');
            if (cell) {{
                cell.focus();
                // Try pasting via clipboard API
                navigator.clipboard.writeText(code).then(function() {{
                    document.execCommand('paste');
                }});
                return 'focused';
            }}
        }} catch(e) {{}}

        return 'no injection method worked';
    }})();
    """)
    print(f"Injection result: {inject_result}")

    # Take screenshot
    screenshot = await cdp.send("Page.captureScreenshot", {"format": "png"})
    sd = screenshot.get("result", {}).get("data", "")
    if sd:
        with open("/tmp/colab_after_inject.png", "wb") as f:
            f.write(base64.b64decode(sd))
        print(f"Screenshot saved ({len(sd)} bytes)")

    # ---- STEP 4: Run the cell ----
    print("\n" + "=" * 60)
    print("STEP 4: Run the training cell")
    print("=" * 60)

    run_result = await cdp.js("""
    (function() {
        // Method 1: Ctrl+Enter keyboard shortcut
        var cell = document.querySelector('[role="code"]');
        if (cell) {
            cell.focus();
            cell.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'Enter',
                code: 'Enter',
                ctrlKey: true,
                bubbles: true
            }));
            return 'dispatched Ctrl+Enter';
        }

        // Method 2: Find and click play/run button in cell toolbar
        var buttons = document.querySelectorAll('button');
        for (var b of buttons) {
            var text = (b.textContent || '').trim();
            var label = (b.getAttribute('aria-label') || '').toLowerCase();
            if (text.includes('Run') || text.includes('Play') ||
                label.includes('run') || label.includes('play') ||
                b.getAttribute('title') === 'Run cell' ||
                b.getAttribute('title') === 'Play') {
                if (b.offsetParent !== null) {
                    b.click();
                    return 'clicked run button: ' + (text || label);
                }
            }
        }

        // Method 3: Look for the cell toolbar run button
        var icons = document.querySelectorAll('svg, i, span, img');
        for (var ic of icons) {
            var title = (ic.getAttribute('title') || ic.getAttribute('aria-label') || '').toLowerCase();
            if (title.includes('run') || title.includes('play') || title.includes('execute')) {
                var btn = ic.closest('button');
                if (btn && btn.offsetParent !== null) {
                    btn.click();
                    return 'clicked icon run button';
                }
            }
        }

        return 'no run method worked';
    })();
    """)
    print(f"Run result: {run_result}")

    await asyncio.sleep(2)

    # Final screenshot
    screenshot2 = await cdp.send("Page.captureScreenshot", {"format": "png"})
    sd2 = screenshot2.get("result", {}).get("data", "")
    if sd2:
        with open("/tmp/colab_after_run.png", "wb") as f:
            f.write(base64.b64decode(sd2))
        print(f"After-run screenshot saved ({len(sd2)} bytes)")

    await cdp.close()
    print("\nDone! Training should now be running in Colab.")


if __name__ == "__main__":
    asyncio.run(main())
