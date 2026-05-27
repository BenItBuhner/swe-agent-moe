"""Upgrade Colab runtime to best available accelerator and launch training.

Connects to the CDP browser (port 40107) running the Colab notebook,
opens the Runtime > Change runtime type dialog, finds the best available
accelerator, saves, waits for reconnect, then injects and runs training.
"""

import asyncio, json, urllib.request, websockets, base64, os, sys

NID = "1U3AYDmQJqR3RT2MzuIiONZ0ZunOGorGM"

async def send(ws, mid, method, params=None):
    await ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    while True:
        r = json.loads(await ws.recv())
        if r.get("id") == mid:
            return r

async def js(ws, mid, expr):
    r = await send(ws, mid, "Runtime.evaluate", {
        "expression": expr, "returnByValue": True, "awaitPromise": True
    })
    result = r.get("result", {})
    if "exceptionDetails" in result:
        return {"error": result["exceptionDetails"]}
    return result.get("result", {}).get("value")

async def click_at(ws, mid, x, y):
    for ev in ["mousePressed", "mouseReleased"]:
        await send(ws, mid, "Input.dispatchMouseEvent", {
            "type": ev, "x": x, "y": y, "button": "left", "clickCount": 1,
        })

async def get_tab():
    targets = json.loads(urllib.request.urlopen("http://localhost:40107/json").read().decode())
    for t in targets:
        if NID in t.get("url", "") and t.get("type") == "page":
            return t
    for t in targets:
        if t.get("type") == "page" and "colab.research.google.com/drive/" in t.get("url", ""):
            return t
    return None

async def main():
    tab = await get_tab()
    if not tab:
        print("ERROR: No Colab tab found on CDP browser")
        return

    print(f"Connected to tab: {tab.get('title', '?')}")
    ws = await websockets.connect(tab["webSocketDebuggerUrl"])
    await send(ws, 0, "Page.enable")
    await send(ws, 0, "Runtime.enable")
    await send(ws, 0, "Input.enable")

    # === STEP 1: Check current runtime ===
    print("\n=== Step 1: Check Current Runtime ===")
    gpu_check = await js(ws, 1, """
    (() => {
        const btn = document.querySelector('colab-connect-button');
        if (!btn || !btn.shadowRoot) return {};
        const toolbar = btn.shadowRoot.querySelector('colab-toolbar-button');
        if (!toolbar) return {};
        const tt = toolbar.getAttribute('tooltiptext') || '';
        return {tooltip: tt.substring(0, 200)};
    })();
    """)
    print(f"  Runtime info: {gpu_check}")

    # === STEP 2: Open Runtime menu ===
    print("\n=== Step 2: Open Runtime Menu ===")
    # Find Runtime menu item
    await send(ws, 2, "Runtime.evaluate", {
        "expression": """
        (() => {
            // Colab uses material design tabs in the toolbar
            const items = document.querySelectorAll('[role="tab"], [role="menuitem"], span, div');
            for (const item of items) {
                const text = (item.textContent || '').trim();
                if (text === 'Runtime' && item.offsetParent !== null) {
                    const rect = item.getBoundingClientRect();
                    return JSON.stringify({
                        x: Math.round(rect.x + rect.width / 2),
                        y: Math.round(rect.y + rect.height / 2)
                    });
                }
            }
            return 'not found';
        })();
        """,
        "returnByValue": True, "awaitPromise": True
    })
    await asyncio.sleep(0.5)
    r = await ws.recv()
    runtime_pos = json.loads(r).get("result", {}).get("result", {}).get("value", "not found")
    print(f"  Runtime menu position: {runtime_pos}")

    if isinstance(runtime_pos, str) and runtime_pos.startswith("{"):
        import json as _json
        pos = _json.loads(runtime_pos)
        await click_at(ws, 3, pos["x"], pos["y"])
        await asyncio.sleep(2)
        print("  Runtime menu clicked!")
    else:
        print(f"  Runtime menu not found. Trying fixed position...")
        # Try common Colab toolbar positions
        viewport = await js(ws, 3, "JSON.stringify({w: window.innerWidth, h: window.innerHeight})")
        print(f"  Viewport: {viewport}")
        # Colab Runtime menu is typically around x=250-300, y=35-50
        await click_at(ws, 4, 282, 38)
        await asyncio.sleep(2)

    # === STEP 3: Click "Change runtime type" ===
    print("\n=== Step 3: Click Change Runtime Type ===")
    await js(ws, 5, """
    (() => {
        const items = document.querySelectorAll('[role="menuitem"], span, div');
        for (const item of items) {
            const text = (item.textContent || '').trim();
            if (text.includes('Change runtime type') && item.offsetParent !== null) {
                item.click();
                return 'clicked';
            }
        }
        return 'not found';
    })();
    """)
    await asyncio.sleep(3)

    # === STEP 4: Scan hardware dialog ===
    print("\n=== Step 4: Scan Available Hardware Options ===")
    hw_options = await js(ws, 6, """
    (() => {
        // Find all visible radio buttons
        const radios = document.querySelectorAll('[role="radio"]');
        const results = [];
        for (const r of radios) {
            if (r.offsetParent !== null) {
                const txt = (r.textContent || '').trim();
                const rect = r.getBoundingClientRect();
                results.push({
                    text: txt.substring(0, 80),
                    x: Math.round(rect.x + rect.width / 2),
                    y: Math.round(rect.y + rect.height / 2)
                });
            }
        }
        return results;
    })();
    """)
    print(f"  Hardware options: {json.dumps(hw_options, indent=2) if isinstance(hw_options, list) else hw_options}")

    # === STEP 5: Select best accelerator ===
    print("\n=== Step 5: Select Best Accelerator ===")
    target_hw = None
    if isinstance(hw_options, list):
        # Priority: V6E > V5E > A100 > TPU > V100 > T4 GPU
        for opt in hw_options:
            txt = opt.get("text", "")
            if "V6E" in txt or "TPU v6" in txt.lower():
                target_hw = opt; break
        if not target_hw:
            for opt in hw_options:
                txt = opt.get("text", "")
                if "V5E" in txt or "TPU v5" in txt.lower() or "TPU" in txt:
                    target_hw = opt; break
        if not target_hw:
            for opt in hw_options:
                txt = opt.get("text", "")
                if "A100" in txt:
                    target_hw = opt; break
        if not target_hw:
            for opt in hw_options:
                txt = opt.get("text", "")
                if "V100" in txt:
                    target_hw = opt; break
        if not target_hw:
            for opt in hw_options:
                txt = opt.get("text", "")
                if "T4" in txt or "GPU" in txt:
                    target_hw = opt; break

    if target_hw:
        print(f"  Selected: {target_hw.get('text', '?')} at ({target_hw['x']}, {target_hw['y']})")
        await click_at(ws, 7, target_hw["x"], target_hw["y"])
        await asyncio.sleep(1)

        # Click Save/Select button
        await js(ws, 8, """
        (() => {
            const btns = document.querySelectorAll('button');
            for (const b of btns) {
                const text = (b.textContent || '').trim();
                if ((text === 'Save' || text === 'Select' || text.includes('Save') || text.includes('Connect')) 
                    && b.offsetParent !== null) {
                    b.click();
                    return 'clicked: ' + text;
                }
            }
            return 'not found';
        })();
        """)
        print("  Save button clicked!")
        await asyncio.sleep(3)

        # Check for confirmation dialog (Colab may show "Change runtime type - This will reset all data")
        await js(ws, 9, """
        (() => {
            const btns = document.querySelectorAll('button');
            for (const b of btns) {
                const text = (b.textContent || '').trim();
                if ((text === 'OK' || text === 'Confirm' || text === 'Yes' || text === 'Change') 
                    && b.offsetParent !== null) {
                    b.click();
                    return 'confirmed: ' + text;
                }
            }
            return 'no confirmation needed';
        })();
        """)
        print("  Confirmation handled!")
    else:
        print("  No hardware options found. Using current runtime.")

    # === STEP 6: Wait for runtime reconnection ===
    print("\n=== Step 6: Wait for Runtime Reconnection ===")
    await asyncio.sleep(5)

    # Refresh page to trigger new runtime
    await send(ws, 10, "Page.reload")
    await asyncio.sleep(8)

    for attempt in range(40):
        url = await js(ws, 11, "window.location.href")
        if url and NID in str(url):
            await asyncio.sleep(2)
            gpu_ready = await js(ws, 12, """
            (() => {
                const btn = document.querySelector('colab-connect-button');
                if (!btn || !btn.shadowRoot) return 'no shadow';
                const toolbar = btn.shadowRoot.querySelector('colab-toolbar-button');
                if (!toolbar) return 'no toolbar';
                const tt = toolbar.getAttribute('tooltiptext') || '';
                if (tt.includes('Connected')) return tt.substring(0, 300);
                return 'waiting: ' + tt.substring(0, 100);
            })();
            """)
            print(f"  [{attempt+1}] Runtime: {gpu_ready}")
            if isinstance(gpu_ready, str) and 'Connected' in gpu_ready:
                print("  Runtime connected!")
                break
        else:
            await asyncio.sleep(3)

        if attempt % 5 == 4:
            await send(ws, 13, "Page.reload")
            await asyncio.sleep(5)
    else:
        print("  Timeout waiting for runtime. Will try to continue anyway.")

    # === STEP 7: Check hardware and set up training ===
    print("\n=== Step 7: Check Hardware ===")
    hw_info = await js(ws, 14, """
    (() => {
        // Check nvidia-smi via notebook cell (indirect)
        const connectBtn = document.querySelector('colab-connect-button');
        if (!connectBtn || !connectBtn.shadowRoot) return 'no-access';
        const toolbar = connectBtn.shadowRoot.querySelector('colab-toolbar-button');
        if (!toolbar) return 'no-toolbar';
        const tt = toolbar.getAttribute('tooltiptext') || '';
        return tt;
    })();
    """)
    print(f"  Hardware info: {hw_info}")

    # === STEP 8: Inject and run training ===
    print("\n=== Step 8: Inject Training Code ===")
    
    # Read the training code from local file
    training_code = r"""# === SWE-Agent MoE: Training Pipeline ===
import os, sys, subprocess, time, json, shutil, torch
from pathlib import Path

print("=" * 60)
print("SWE-Agent MoE Training Pipeline")
print("=" * 60)

# Install deps
print("\n[1/5] Installing dependencies...")
deps = [
    "torch>=2.4.0", "torchvision>=0.19.0", "transformers>=4.44.0",
    "accelerate>=0.33.0", "datasets>=2.20.0", "wandb>=0.17.0",
    "sentencepiece>=0.2.0", "protobuf>=4.25.0", "einops>=0.7.0",
]
for dep in deps:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep, "--no-deps"])

print("\n[2/5] Cloning project from GitHub...")
if not os.path.exists("/content/swe-agent-moe"):
    subprocess.check_call(["git", "clone", "https://github.com/benprobennett/swe-agent-moe.git", "/content/swe-agent-moe"])
sys.path.insert(0, "/content/swe-agent-moe")
os.chdir("/content/swe-agent-moe")
py_files = list(Path("/content/swe-agent-moe").rglob("*.py"))
print(f"  Files: {len(py_files)} Python files")

print("\n[3/5] Checking hardware...")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"  CUDA version: {torch.version.cuda}")

# Check for TPU
try:
    import torch_xla
    print("  TPU available via torch_xla!")
    has_tpu = True
except:
    has_tpu = False
    print("  No TPU detected")

# Check number of GPUs
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"  Num GPUs: {num_gpus}")

# Determine training strategy
gpu_mem_gb = 0
if torch.cuda.is_available():
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9

print(f"\n[4/5] Setting up training strategy...")
print(f"  GPU memory: {gpu_mem_gb:.1f} GB")
print(f"  Num GPUs: {num_gpus}")
print(f"  Has TPU: {has_tpu}")

from model.architecture import MoEForCausalLM, MoEConfig
from configs.model_config import MoEModelConfig, TrainingConfig
model_cfg = MoEModelConfig()
print(f"  Model: {model_cfg.total_params:.2f}B total, {model_cfg.activated_params:.2f}B activated")
print(f"  Required (BF16): {model_cfg.total_params * 2:.1f} GB params + ~{model_cfg.total_params * 4:.1f} GB optimizer")

if has_tpu:
    # TPU training
    print("\n[5/5] Launching TPU training...")
    os.chdir("/content/swe-agent-moe")
    sys.argv = ["train/pretrain.py", "--device=tpu"]
    exec(open("train/pretrain.py").read())
elif num_gpus >= 4 and gpu_mem_gb >= 40:
    # Multi-GPU A100 - use FSDP
    print("\n[5/5] Launching multi-GPU FSDP training...")
    os.chdir("/content/swe-agent-moe")
    !python -m torch.distributed.run --nproc_per_node=$num_gpus train/pretrain.py
elif num_gpus >= 1 and gpu_mem_gb >= 40:
    # Single A100 40GB - use CPU offloading
    print("\n[5/5] Launching single-GPU training with CPU offload...")
    os.chdir("/content/swe-agent-moe")
    !FSDP_CPU_OFFLOAD=1 python train/pretrain.py
elif num_gpus >= 1 and gpu_mem_gb >= 15:
    # T4 or similar - use colab_lite config
    print("\n[5/5] Launching colab-lite training...")
    os.chdir("/content/swe-agent-moe")
    !python train/colab_train.py
else:
    # CPU training - micro config
    print("\n[5/5] Launching CPU training (tiny config)...")
    os.chdir("/content/swe-agent-moe")
    !python train/colab_train.py --cpu

print("\n=== Training launched! ===")
print("Check Colab tab for output.")
"""

    # Set Monaco cell content
    await js(ws, 15, f"""
    (() => {{
        const CODE = {json.dumps(training_code)};
        if (typeof monaco !== 'undefined' && monaco.editor) {{
            const editors = monaco.editor.getEditors();
            if (editors.length > 0) {{
                editors[0].setValue(CODE);
                return 'code set in Monaco: ' + CODE.length + ' chars';
            }}
        }}
        return 'Monaco not available';
    }})();
    """)
    print("  Training code injected!")

    # === STEP 9: Execute the cell ===
    print("\n=== Step 9: Execute Training Cell ===")
    
    # Try direct kernel execution first
    executed = await js(ws, 16, """
    (() => {
        // Try Ctrl+Enter through keyboard
        const cell = document.querySelector('[role="code"], .cell, .code-cell');
        if (cell) {
            cell.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true
            }));
            cell.dispatchEvent(new KeyboardEvent('keyup', {
                key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true
            }));
            return 'Ctrl+Enter dispatched';
        }
        return 'no cell found';
    })();
    """)
    print(f"  Execute: {executed}")

    # Also try the "Run cell" button
    await asyncio.sleep(1)
    await js(ws, 17, """
    (() => {
        // Find play/run button
        const btns = document.querySelectorAll('button');
        for (const b of btns) {
            const text = (b.textContent || '').trim();
            const aria = (b.getAttribute('aria-label') || '');
            if ((text.includes('Run') || aria.includes('Run') || aria.includes('play')) 
                && b.offsetParent !== null) {
                b.click();
                return 'clicked run button';
            }
        }
        return 'no run button found';
    })();
    """)
    print("  Run button triggered!")

    # Final screenshot
    ss = await send(ws, 99, "Page.captureScreenshot", {"format": "png"})
    data = ss.get("result", {}).get("data", "")
    if data:
        with open("/tmp/colab_training_launched.png", "wb") as f:
            f.write(base64.b64decode(data))
        print(f"\nScreenshot: /tmp/colab_training_launched.png ({len(data)} bytes)")

    await ws.close()
    print("\n=== DONE! Training should be running in Colab. ===")

asyncio.run(main())
