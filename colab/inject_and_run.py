"""Inject colab_runner.py into Colab T4 runtime and execute.

Uses google.colab.kernel.requestExecute via CDP.
"""

import asyncio, json, urllib.request, websockets, base64, os

NOTEBOOK_ID = "1U3AYDmQJqR3RT2MzuIiONZ0ZunOGorGM"


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
        return None
    return result.get("result", {}).get("value")


async def ss(ws, path):
    r = await send(ws, 999, "Page.captureScreenshot", {"format": "png"})
    d = r.get("result", {}).get("data")
    if d:
        with open(path, "wb") as f:
            f.write(base64.b64decode(d))
        return f"saved {path}"
    return "no screenshot"


async def main():
    targets = json.loads(urllib.request.urlopen("http://localhost:40107/json").read().decode())
    tab = None
    for t in targets:
        if NOTEBOOK_ID in t.get("url", "") and t.get("type") == "page":
            tab = t; break
    if not tab:
        for t in targets:
            if t.get("type") == "page" and "colab.research.google.com/drive/" in t.get("url", ""):
                tab = t; break
    if not tab:
        print("No Colab tab!"); return

    print(f"Tab: {tab.get('title','?')}")
    ws = await websockets.connect(tab["webSocketDebuggerUrl"])
    await send(ws, 0, "Page.enable")
    await send(ws, 0, "Runtime.enable")
    print("Connected to CDP\n")

    # Navigate to notebook
    print("1. Navigating to notebook...")
    await send(ws, 1, "Page.navigate", {
        "url": f"https://colab.research.google.com/drive/{NOTEBOOK_ID}"
    })
    await asyncio.sleep(4)

    # Wait for runtime
    for i in range(15):
        con = await js(ws, 2, "typeof google !== 'undefined' && typeof google.colab !== 'undefined' && typeof google.colab.kernel !== 'undefined'")
        if con:
            print(f"   Runtime connected! (attempt {i+1})")
            break
        await asyncio.sleep(2)
    else:
        print("   Runtime NOT connected. State:")
        st = await js(ws, 3, "typeof google !== 'undefined' ? Object.keys(google).join(',') : 'no google'")
        print(f"   google state: {st}")
        await ss(ws, "/tmp/colab_no_runtime.png")
        await ws.close()
        return

    # Read the runner script
    runner_path = "/mnt/project-drive/model-training-pipeline/colab/colab_runner.py"
    with open(runner_path) as f:
        runner_code = f.read()

    # Escape for JSON embedding
    escaped = runner_code.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    escaped = escaped.replace("'''", "\\'\\'\\'")

    # The full code to execute in Colab
    setup_code = f"""
import os, sys, subprocess, json, base64

# Install minimal deps first
deps = ["torch>=2.4.0", "transformers>=4.44.0", "accelerate>=0.33.0",
        "datasets>=2.20.0", "sentencepiece>=0.2.0", "protobuf>=4.25.0",
        "einops>=0.7.0", "psutil"]
for d in deps:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", d])
print("Dependencies installed")

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Execute runner
exec(open('/content/runner.py').read())
"""

    # Write runner to Colab via CDP
    print(f"\n2. Writing runner script to Colab...")
    write_result = await js(ws, 10, f"""
    (function() {{
        var code = {json.dumps(runner_code)};
        if (typeof google !== 'undefined' && google.colab && google.colab.kernel) {{
            google.colab.kernel.requestExecute({{
                code: `import os; os.makedirs('/content', exist_ok=True); 
with open('/content/runner.py', 'w') as f: f.write({json.dumps(runner_code)})
print("Runner script written to /content/runner.py")`
            }});
            return 'write queued';
        }}
        // Monaco fallback
        try {{
            if (typeof monaco !== 'undefined' && monaco.editor) {{
                var eds = monaco.editor.getEditors();
                if (eds.length > 0) {{ eds[0].setValue(code); return 'set in Monaco'; }}
            }}
        }} catch(e) {{}}
        return 'could not inject';
    }})();
    """)
    print(f"   Write: {write_result}")
    await asyncio.sleep(3)

    # Take screenshot after write
    await ss(ws, "/tmp/colab_after_write.png")

    # Now execute the setup and runner
    print(f"\n3. Launching setup + pretraining...")
    run_result = await js(ws, 20, f"""
    (function() {{
        var code = {json.dumps(setup_code)};
        if (typeof google !== 'undefined' && google.colab && google.colab.kernel) {{
            google.colab.kernel.requestExecute({{
                code: code
            }});
            return 'training launched via kernel';
        }}
        return 'kernel not available';
    }})();
    """)
    print(f"   Launch: {run_result}")

    await asyncio.sleep(5)
    await ss(ws, "/tmp/colab_training_launched.png")
    print("\n4. Training launched! Check Colab tab for progress.")
    await ws.close()


if __name__ == "__main__":
    asyncio.run(main())
