"""Robust CDP v2: Connect to Colab, inject code, and launch training.

Key improvements: proper page navigation, iframe detection, execution context handling.
"""

import asyncio
import json
import base64
import urllib.request
import websockets
import sys


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

    async def eval(self, expr, ctx_id=None):
        params = {"expression": expr, "returnByValue": True, "awaitPromise": True}
        if ctx_id is not None:
            params["contextId"] = ctx_id
        r = await self.send("Runtime.evaluate", params)
        result = r.get("result", {})
        if "exceptionDetails" in result:
            return {"error": result["exceptionDetails"]["text"]}
        val = result.get("result", {}).get("value")
        return {"value": val}

    async def js(self, expr, ctx_id=None):
        r = await self.eval(expr, ctx_id)
        if "error" in r:
            print(f"  JS error: {r['error']}")
            return None
        return r.get("value")

    async def close(self):
        try:
            await self.ws.close()
        except Exception:
            pass


async def get_targets():
    resp = urllib.request.urlopen(f"http://localhost:{CDP_PORT}/json")
    return json.loads(resp.read().decode())


async def main():
    targets = await get_targets()

    # Find the notebook tab
    tab = None
    for t in targets:
        url = t.get("url", "")
        if NOTEBOOK_ID in url and t.get("type") == "page":
            tab = t
            break

    if not tab:
        for t in targets:
            url = t.get("url", "")
            if "colab.research.google.com/drive/" in url and t.get("type") == "page":
                tab = t
                break

    if not tab:
        print("ERROR: No Colab notebook tab found!")
        print("Available tabs:")
        for t in targets:
            if t.get("type") == "page":
                print(f"  - {t.get('title','?')[:60]}: {t.get('url','')[:80]}")
        return

    print(f"Tab: {tab.get('title','?')}")
    print(f"URL: {tab.get('url','')[:100]}")

    cdp = CDP(tab["webSocketDebuggerUrl"])
    await cdp.connect()
    print("Connected to CDP\n")

    # Enable domains
    await cdp.send("Page.enable")
    await cdp.send("Runtime.enable")
    await cdp.send("DOM.enable")

    # Navigate to notebook URL explicitly
    notebook_url = f"https://colab.research.google.com/drive/{NOTEBOOK_ID}"
    print(f"Navigating to: {notebook_url}")
    nav = await cdp.send("Page.navigate", {"url": notebook_url})
    print(f"Navigation result: {nav.get('result', {}).get('text', 'OK')}")

    # Wait for page to load
    await asyncio.sleep(3)

    # Try explicit frame tree to understand structure
    ft = await cdp.send("Page.getFrameTree")
    ft = ft.get("result", {}).get("frameTree", {})
    print(f"\nFrame tree root: {ft.get('frame', {}).get('url','?')[:80]}")

    frames = [ft.get("frame", {})]
    for child in ft.get("childFrames", []):
        frames.append(child.get("frame", {}))
        for gc in child.get("childFrames", []):
            frames.append(gc.get("frame", {}))

    print(f"Total frames in tree: {len(frames)}")
    for f in frames:
        furl = f.get("url", "")
        if furl and furl != "about:blank":
            print(f"  Frame: {f.get('id','')[:20]}... -> {furl[:100]}")

    # Find the main frame (colab.research.google.com)
    main_frame_id = None
    output_frame_id = None
    for f in frames:
        furl = f.get("url", "")
        if "colab.research.google.com/drive/" in furl:
            main_frame_id = f.get("id")
        if "outputframe.html" in furl:
            output_frame_id = f.get("id")

    # Get execution contexts
    ctxs = await cdp.send("Runtime.executionContextsCreated")  # Not a real command
    # Actually, on connect Runtime.executionContextsCreated is sent as event
    # Let's get them via Runtime.evaluate in different frames

    print(f"\nMain frame (drive): {main_frame_id[:30] if main_frame_id else 'N/A'}...")
    print(f"Output frame: {output_frame_id[:30] if output_frame_id else 'N/A'}...")

    # Try evaluating in main page context
    if main_frame_id:
        await cdp.send("Runtime.evaluate", {
            "expression": "1+1",
            "contextId": 1,
            "returnByValue": True,
        })

    # Use Page.navigate to find the execution context via frames
    target_id = tab["id"]

    # Get document URL from main frame
    doc_url = await cdp.js("window.location.href", ctx_id=1)
    doc_title = await cdp.js("document.title", ctx_id=1)
    print(f"\nDoc URL: {doc_url}")
    print(f"Doc title: {doc_title}")

    # Try context 1 (default) and log all available contexts
    for ctx_id in [1, 2, 3, 4, 5]:
        for expr in ["window.location.href", "typeof google", "typeof monaco", "typeof CodeMirror"]:
            try:
                result = await cdp.eval(expr, ctx_id)
                if result.get("value") is not None and str(result["value"]) != "undefined":
                    print(f"  ctx {ctx_id}: {expr} = {result['value'][:80]}")
            except Exception:
                pass

    # Inspect the page DOM
    print("\n--- Page Inspection ---")
    insp = await cdp.js("""
    (function(){
        var r = {};
        r.url = window.location.href;
        r.title = document.title;
        r.bodyChildren = document.body ? document.body.children.length : -1;
        r.selectors = {};
        var checks = ['iframe','colab-editor','colab-notebook',
                      '.cell','[role="code"]','textarea','button',
                      '.monaco-editor','#notebook-container',
                      '.code-cell','jp-Notebook','colab-cell'];
        checks.forEach(function(s) {
            var n = document.querySelectorAll(s).length;
            if(n > 0) r.selectors[s] = n;
        });
        // colab API
        r.hasColab = typeof google !== 'undefined' && typeof google.colab !== 'undefined';
        if(r.hasColab) {
            r.colabKeys = Object.keys(google.colab);
        }
        return r;
    })();
    """)
    print(json.dumps(insp, indent=2) if insp else "null")

    await cdp.close()
    print("\nDone")


if __name__ == "__main__":
    asyncio.run(main())
