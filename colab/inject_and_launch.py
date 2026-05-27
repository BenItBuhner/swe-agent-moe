"""Inject training code into Colab notebook and launch training via CDP.

Usage: python colab/inject_and_launch.py

Connects to Chrome CDP at localhost:40107, finds the Colab tab,
injects the pretraining code, connects to GPU runtime, and runs it.
"""

import asyncio
import json
import os
import sys
import time

try:
    import websockets
except ImportError:
    os.system(f"{sys.executable} -m pip install -q websockets")
    import websockets


CDP_PORT = 40107
NOTEBOOK_ID = "1U3AYDmQJqR3RT2MzuIiONZ0ZunOGorGM"


class CDPClient:
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.msg_id = 0
        self.pending = {}
        self.ws = None
        self.recv_task = None

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url)
        self.recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self):
        while True:
            try:
                msg = json.loads(await self.ws.recv())
                if msg.get("id") and msg["id"] in self.pending:
                    self.pending[msg["id"]].set_result(msg)
            except Exception:
                break

    async def send(self, method, params=None):
        if params is None:
            params = {}
        self.msg_id += 1
        msg = {"id": self.msg_id, "method": method, "params": params}
        await self.ws.send(json.dumps(msg))
        self.pending[self.msg_id] = asyncio.get_event_loop().create_future()
        return await self.pending[self.msg_id]

    async def close(self):
        try:
            await self.ws.close()
        except Exception:
            pass


async def get_page_targets():
    """Get page targets from Chrome's /json endpoint."""
    import urllib.request
    resp = urllib.request.urlopen(f"http://localhost:{CDP_PORT}/json")
    return json.loads(resp.read().decode())


async def find_colab_tab():
    """Find the Colab notebook tab by iterating pages."""
    targets = await get_page_targets()
    for t in targets:
        url = t.get("url", "")
        if NOTEBOOK_ID in url and t.get("type") == "page":
            return t["id"], t["webSocketDebuggerUrl"]
    # fallback: any colab page
    for t in targets:
        url = t.get("url", "")
        if "colab.research.google.com/drive/" in url and t.get("type") == "page":
            return t["id"], t["webSocketDebuggerUrl"]
    return None, None


async def main():
    print("Looking for Colab notebook tab...")
    tab_id, ws_url = await find_colab_tab()
    if not tab_id:
        print("ERROR: Could not find Colab notebook tab!")
        sys.exit(1)
    print(f"Found Colab tab: {tab_id}")

    # Connect CDP
    client = CDPClient(ws_url)
    await client.connect()
    print("Connected to Chrome CDP")

    # Navigate to the notebook explicitly
    notebook_url = f"https://colab.research.google.com/drive/{NOTEBOOK_ID}"
    print(f"Ensuring we're on notebook: {notebook_url}")
    await client.send("Page.navigate", {"url": notebook_url})
    await asyncio.sleep(3)

    # Inject training code into the first Monaco Editor cell
    # Colab uses Monaco Editor - the code cell is a div with role="code"
    # The actual editor content is in a hidden textarea with class "lm-content"
    # or accessible via monaco API

    training_code = """# === SWE-Agent MoE: Phase 1 - Pretraining ===
# This cell installs deps, mounts drive, copies project, and starts pretraining

import os, sys, subprocess, time, json
from pathlib import Path

print("=" * 60)
print("SWE-Agent MoE Pretraining Pipeline")
print("=" * 60)

# Install dependencies
print("\\n[1/5] Installing dependencies...")
deps = [
    "torch>=2.4.0",
    "torchvision>=0.19.0",
    "transformers>=4.44.0",
    "accelerate>=0.33.0",
    "datasets>=2.20.0",
    "wandb>=0.17.0",
    "sentencepiece>=0.2.0",
    "protobuf>=4.25.0",
    "einops>=0.7.0",
]
for dep in deps:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])

# Mount Google Drive for persistent storage
print("\\n[2/5] Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')

# Copy project to Colab (fast since Drive is local)
print("\\n[3/5] Copying project files...")
!cp -r /content/drive/MyDrive/model-training-pipeline /content/ 2>/dev/null || \\
    git clone https://github.com/barnacle-agent/swe-agent-moe /content/model-training-pipeline 2>/dev/null

sys.path.insert(0, '/content/model-training-pipeline')
print(f"Project ready: {len(list(Path('/content/model-training-pipeline').rglob('*.py')))} .py files")

# Check hardware
print("\\n[4/5] Hardware check...")
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"CUDA: {torch.version.cuda}")
else:
    print("WARNING: No GPU detected! Go to Runtime > Change runtime type > A100 GPU")
    print("Will continue on CPU (extremely slow)")

# Start pretraining
print("\\n[5/5] Starting pretraining...")
os.chdir('/content/model-training-pipeline')
!python train/pretrain.py
"""

    # Try injecting via JavaScript evaluating monaco editor API
    js_code = """
    // Find the CodeMirror/Monaco editor instance in Colab
    // Colab stores notebook editor state in window.google.colab or similar
    function injectCode(code) {
        // Try monaco editor API first
        try {
            var editors = window.monaco && window.monaco.editor;
            if (editors) {
                var all = editors.getEditors();
                if (all && all.length > 0) {
                    all[0].setValue(code);
                    return true;
                }
            }
        } catch(e) {}

        // Try colab's internal API
        try {
            var colab = window.google && window.google.colab;
            if (colab && colab.injectCode) {
                colab.injectCode(code);
                return true;
            }
        } catch(e) {}

        // Try finding the hidden textarea
        try {
            var ta = document.querySelector('textarea.lm-content, textarea[aria-label="Code cell content"]');
            if (ta) {
                var nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeInputValueSetter.call(ta, code);
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                return true;
            }
        } catch(e) {}

        // Try Jupyter's CodeMirror
        try {
            var cm = document.querySelector('.CodeMirror');
            if (cm && cm.CodeMirror) {
                cm.CodeMirror.setValue(code);
                return true;
            }
        } catch(e) {}

        return false;
    }
    injectCode(INJECTED_CODE);
    """

    # Build the full JS with our code embedded
    escaped_code = training_code.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    full_js = js_code.replace("INJECTED_CODE", "`" + escaped_code + "`")

    print("Injecting training code into cell...")
    result = await client.send("Runtime.evaluate", {
        "expression": full_js,
        "returnByValue": True,
    })
    print(f"Injection result: {result.get('result', {}).get('value', 'check output')}")

    await asyncio.sleep(1)

    # Try to find and click the "Connect" button to allocate GPU
    print("Looking for runtime connect button...")
    find_connect_js = """
    // Look for various connect/run button states
    var buttons = document.querySelectorAll('button, span, div[role="button"]');
    var results = [];
    for (var b of buttons) {
        var text = (b.textContent || '').trim();
        if (text.includes('Connect') || text.includes('Runtime') || text.includes('Run')) {
            results.push({text: text, tag: b.tagName, class: b.className});
        }
    }
    results;
    """
    result = await client.send("Runtime.evaluate", {
        "expression": find_connect_js,
        "returnByValue": True,
    })
    buttons_found = result.get("result", {}).get("value", [])
    if buttons_found:
        print(f"Found buttons: {buttons_found[:5]}")

    # Try clicking the "Connect" button to allocate a GPU
    click_connect_js = """
    function clickConnect() {
        var buttons = document.querySelectorAll('button');
        for (var b of buttons) {
            if (b.textContent.trim().includes('Connect') && b.offsetParent !== null) {
                b.click();
                return 'Clicked Connect';
            }
        }
        // Try runtime menu
        var spans = document.querySelectorAll('span');
        for (var s of spans) {
            if (s.textContent.trim() === 'Runtime') {
                s.click();
                setTimeout(function() {
                    var items = document.querySelectorAll('li, div[role="menuitem"]');
                    for (var i of items) {
                        if (i.textContent.trim().includes('Change runtime type')) {
                            i.click();
                            return 'Opened runtime settings';
                        }
                    }
                }, 500);
                return 'Clicked Runtime menu';
            }
        }
        return 'No connect button found';
    }
    clickConnect();
    """
    result = await client.send("Runtime.evaluate", {
        "expression": click_connect_js,
        "returnByValue": True,
    })
    print(f"Connect attempt: {result.get('result', {}).get('value', 'N/A')}")

    # Take a screenshot to see current state
    screenshot = await client.send("Page.captureScreenshot", {"format": "png"})
    screenshot_data = screenshot.get("result", {}).get("data", "")
    if screenshot_data:
        import base64
        with open("/tmp/colab_state.png", "wb") as f:
            f.write(base64.b64decode(screenshot_data))
        print("Screenshot saved to /tmp/colab_state.png")

    await client.close()
    print("\\nDone! The training code has been injected into the Colab cell.")


if __name__ == "__main__":
    asyncio.run(main())
