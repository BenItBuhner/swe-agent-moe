"""Full Colab TPU setup: wait for load, connect TPU, inject code, run training.

Uses authenticated CDP Chrome (barnacle.agent@gmail.com).
"""

import asyncio, json, urllib.request, websockets, base64

NOTEBOOK_ID = "1U3AYDmQJqR3RT2MzuIiONZ0ZunOGorGM"

async def send(ws, mid, method, params=None):
    await ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    while True:
        r = json.loads(await ws.recv())
        if r.get("id") == mid: return r

async def js(ws, mid, expr):
    r = await send(ws, mid, "Runtime.evaluate", {
        "expression": expr, "returnByValue": True, "awaitPromise": True
    })
    result = r.get("result", {})
    if "exceptionDetails" in result: return None
    return result.get("result", {}).get("value")

async def get_tab():
    targets = json.loads(urllib.request.urlopen("http://localhost:40107/json").read().decode())
    for t in targets:
        if NOTEBOOK_ID in t.get("url", "") and t.get("type") == "page": return t
    for t in targets:
        if t.get("type") == "page" and "colab.research.google.com/drive/" in t.get("url",""): return t
    return None

async def main():
    tab = await get_tab()
    if not tab:
        print("No Colab tab!"); return

    print(f"Tab: {tab.get('title','?')}")
    ws = await websockets.connect(tab["webSocketDebuggerUrl"])
    await send(ws, 0, "Page.enable")
    await send(ws, 0, "Runtime.enable")

    # Navigate to notebook
    print("\n--- Navigating to notebook ---")
    await send(ws, 10, "Page.navigate", {"url": f"https://colab.research.google.com/drive/{NOTEBOOK_ID}"})

    # Wait for notebook editor to appear (monaco-editor or [role="code"])
    print("\n--- Waiting for notebook editor ---")
    mid = 20
    for attempt in range(30):
        await asyncio.sleep(2)
        monaco = await js(ws, mid, "document.querySelectorAll('.monaco-editor').length")
        if monaco and monaco > 0:
            print(f"  Monaco editor appeared! (attempt {attempt + 1})")
            break
        # Also check for [role="code"]
        code_role = await js(ws, mid + 1, "document.querySelectorAll('[role=\"code\"]').length")
        if code_role and code_role > 0:
            print(f"  Code cell appeared! (attempt {attempt + 1})")
            break
        mid += 1
    else:
        print("  Timeout waiting for editor. Taking screenshot.")
        ss = await send(ws, 99, "Page.captureScreenshot", {"format": "png"})
        data = ss.get("result", {}).get("data")
        if data:
            with open("/tmp/colab_timeout.png", "wb") as f:
                f.write(base64.b64decode(data))
            print(f"  Screenshot: /tmp/colab_timeout.png")
        # Continue anyway

    # Connect to runtime
    print("\n--- Connecting runtime ---")
    connected = await js(ws, 50, "typeof google !== 'undefined' && typeof google.colab !== 'undefined'")
    print(f"  Already connected: {connected}")

    if not connected:
        # Find Connect button and click it
        clicked = await js(ws, 51, """
        (() => {
            const btns = document.querySelectorAll('button');
            for (const b of btns) {
                const text = (b.textContent || '').trim();
                if ((text === 'Connect' || text.startsWith('Connect')) && b.offsetParent !== null) {
                    b.click();
                    return 'clicked Connect';
                }
            }
            return 'not found';
        })();
        """)
        print(f"  Connect: {clicked}")

        await asyncio.sleep(5)

        # Check again
        connected = await js(ws, 52, 
            "typeof google !== 'undefined' && typeof google.colab !== 'undefined'")
        print(f"  Runtime connected: {connected}")

        if not connected:
            # Try Runtime menu -> Change runtime type -> TPU
            print("\n--- Selecting TPU runtime ---")
            await js(ws, 60, """
            (() => {
                const menus = document.querySelectorAll('[role="button"], [role="menubar"] span, [role="menubar"] div');
                for (const m of menus) {
                    if ((m.textContent || '').trim() === 'Runtime' && m.offsetParent !== null) {
                        m.click();
                        return 'clicked Runtime';
                    }
                }
                return 'Runtime not found';
            })();
            """)
            await asyncio.sleep(2)

            # Click Change runtime type
            await js(ws, 61, """
            (() => {
                const items = document.querySelectorAll('[role="menuitem"], [role="option"], span, div');
                for (const item of items) {
                    const text = (item.textContent || '').trim();
                    if (text.includes('Change runtime type') && item.offsetParent !== null) {
                        item.click();
                        return 'clicked: ' + text.substring(0, 30);
                    }
                }
                return 'not found';
            })();
            """)
            await asyncio.sleep(2)

            # Select TPU
            await js(ws, 62, """
            (() => {
                const options = document.querySelectorAll('[role="radio"], [role="option"], ' +
                    'label, span, div');
                for (const o of options) {
                    const text = (o.textContent || '').trim();
                    if (text.includes('TPU') && o.offsetParent !== null) {
                        o.click();
                        return 'clicked TPU: ' + text.substring(0, 30);
                    }
                }
                // Try V100/A100 if no TPU
                for (const o of options) {
                    const text = (o.textContent || '').trim();
                    if ((text.includes('A100') || text.includes('V5E') || text.includes('V6E')) 
                        && o.offsetParent !== null) {
                        o.click();
                        return 'clicked: ' + text.substring(0, 30);
                    }
                }
                return 'TPU/A100 not found';
            })();
            """)
            await asyncio.sleep(2)

            # Click Connect/Select
            await js(ws, 63, """
            (() => {
                const btns = document.querySelectorAll('button');
                for (const b of btns) {
                    const text = (b.textContent || '').trim();
                    if (b.offsetParent !== null && 
                        (text.includes('Select') || text.includes('Connect') || 
                         text.includes('Save') || text.includes('OK') ||
                         text.includes('Confirm') || text.includes('Yes'))) {
                        b.click();
                        return 'clicked: ' + text.substring(0, 20);
                    }
                }
                return 'confirm button not found';
            })();
            """)
            await asyncio.sleep(5)

            connected = await js(ws, 64,
                "typeof google !== 'undefined' && typeof google.colab !== 'undefined'")
            print(f"  Runtime connected after TPU selection: {connected}")

    # Now inject training code into the cell and run it
    if connected:
        print("\n--- Runtime connected! Injecting and running training ---")
        
        # Try to clear current cell and set training code
        inject = await js(ws, 70, """
        (() => {
            const CODE = `# SWE-Agent MoE: Full Training Pipeline
import os, sys, subprocess, time
from pathlib import Path

print("=== SWE-Agent MoE Pretraining ===")

# Install deps
!pip install -q torch>=2.4.0 transformers>=4.44.0 accelerate>=0.33.0 datasets>=2.20.0 wandb>=0.17.0 sentencepiece>=0.2.0 protobuf>=4.25.0 einops>=0.7.0

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy project
!cp -r /content/drive/MyDrive/model-training-pipeline /content/
sys.path.insert(0, '/content/model-training-pipeline')

# Start pretraining
os.chdir('/content/model-training-pipeline')
!python train/pretrain.py
`;
            
            // Monaco API
            if (typeof monaco !== 'undefined' && monaco.editor) {
                const editors = monaco.editor.getEditors();
                if (editors.length > 0) {
                    editors[0].setValue(CODE);
                    return 'set via Monaco';
                }
            }
            return 'Monaco not available';
        })();
        """)
        print(f"  Inject: {inject}")

        # Run the cell (Ctrl+Enter)
        await js(ws, 71, """
        (() => {
            if (typeof google !== 'undefined' && google.colab && google.colab.kernel) {
                // Use kernel API
                google.colab.kernel.requestExecute({
                    code: document.querySelector('.monaco-editor textarea')?.value || '',
                });
                return 'executed via kernel';
            }
            // Try keyboard shortcut
            const cell = document.querySelector('[role="code"]');
            if (cell) {
                cell.dispatchEvent(new KeyboardEvent('keydown', {
                    key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true
                }));
                return 'dispatched Ctrl+Enter';
            }
            return 'could not run';
        })();
        """)
        print("  Run command dispatched!")

    # Final screenshot
    ss = await send(ws, 99, "Page.captureScreenshot", {"format": "png"})
    data = ss.get("result", {}).get("data")
    if data:
        with open("/tmp/colab_final2.png", "wb") as f:
            f.write(base64.b64decode(data))
        print(f"\nFinal screenshot: {len(data)} bytes")

    await ws.close()
    print("\nDone! Training should be running in Colab.")

asyncio.run(main())
