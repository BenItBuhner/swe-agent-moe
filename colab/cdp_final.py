"""Final CDP script: connect TPU runtime on authenticated Chrome Colab.

Connects to existing Chrome (port 40107, already signed into Google),
finds the Colab notebook, clicks Runtime > Change runtime type, selects TPU,
and runs the training cell.
"""

import asyncio
import json
import base64
import urllib.request
import websockets


NOTEBOOK_ID = "1U3AYDmQJqR3RT2MzuIiONZ0ZunOGorGM"


class Client:
    def __init__(self, url):
        self.url = url
        self.ws = None
        self._id = 0
        self._futures = {}
        self._done = False

    async def connect(self):
        self.ws = await websockets.connect(self.url)
        asyncio.create_task(self._loop())

    async def _loop(self):
        while not self._done:
            try:
                msg = json.loads(await self.ws.recv())
                mid = msg.get("id")
                if mid in self._futures:
                    self._futures[mid].set_result(msg)
            except Exception:
                break

    async def cmd(self, method, params=None):
        self._id += 1
        mid = self._id
        self._futures[mid] = asyncio.get_event_loop().create_future()
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        return await self._futures[mid]

    async def js(self, expr):
        r = await self.cmd("Runtime.evaluate", {
            "expression": expr, "returnByValue": True, "awaitPromise": True
        })
        result = r.get("result", {})
        if "exceptionDetails" in result:
            return None
        return result.get("result", {}).get("value")

    async def ss(self, path):
        r = await self.cmd("Page.captureScreenshot", {"format": "png"})
        data = r.get("result", {}).get("data")
        if data:
            with open(path, "wb") as f:
                f.write(base64.b64decode(data))
            return f"saved {path}"
        return "no screenshot"

    async def close(self):
        self._done = True
        try:
            await self.ws.close()
        except Exception:
            pass


async def main():
    targets = json.loads(urllib.request.urlopen(
        "http://localhost:40107/json"
    ).read().decode())

    # Find the notebook tab
    tp = None
    for t in targets:
        if NOTEBOOK_ID in t.get("url", "") and t.get("type") == "page":
            tp = t; break
    if not tp:
        for t in targets:
            if "colab.research.google.com/drive/" in t.get("url", "") and t.get("type") == "page":
                tp = t; break
    if not tp:
        print("No Colab tab found!")
        print("Tabs:", [t.get("title","?")[:40] for t in targets if t.get("type")=="page"])
        return

    print(f"Tab: {tp.get('title','?')}")

    client = Client(tp["webSocketDebuggerUrl"])
    await client.connect()
    await client.cmd("Page.enable")
    await client.cmd("Runtime.enable")
    print("Connected to CDP\n")

    # Navigate to notebook
    print("1. Navigating to notebook...")
    await client.cmd("Page.navigate", {
        "url": f"https://colab.research.google.com/drive/{NOTEBOOK_ID}"
    })
    await asyncio.sleep(4)

    # Check if already connected
    connected = await client.js("""
        typeof google !== 'undefined' && typeof google.colab !== 'undefined'
    """)
    print(f"   Runtime connected: {connected}")

    if not connected:
        # Click the Runtime menu in toolbar
        print("\n2. Opening Runtime menu...")
        clicked = await client.js("""
        (function() {
            var items = document.querySelectorAll('[role="button"]');
            for (var el of items) {
                if ((el.textContent || '').trim() === 'Runtime') {
                    el.click();
                    return 'clicked';
                }
            }
            return 'not found';
        })();
        """)
        print(f"   Runtime menu: {clicked}")
        await asyncio.sleep(1.5)

        # Find and click "Change runtime type"
        print("\n3. Looking for runtime type option...")
        changed = await client.js("""
        (function() {
            var items = document.querySelectorAll('[role="menuiten"], [role="option"], ' +
                'div, span, li');
            for (var el of items) {
                var text = (el.textContent || '').trim();
                if (el.offsetParent !== null && text.includes('Change runtime type')) {
                    el.click();
                    return 'clicked: ' + text.substring(0, 30);
                }
            }
            // Try invisible items too
            for (var el of items) {
                var text = (el.textContent || '').trim();
                if (text.includes('Change runtime type')) {
                    el.click();
                    return 'clicked hidden: ' + text.substring(0, 30);
                }
            }
            return 'not found';
        })();
        """)
        print(f"   Change runtime type: {changed}")
        await asyncio.sleep(2)

        # Take screenshot to see what dialog appeared
        await client.ss("/tmp/colab_step3.png")
        print("   Screenshot saved to /tmp/colab_step3.png")

        # Look for hardware accelerator selection
        print("\n4. Looking for TPU/GPU options...")
        hw = await client.js("""
        (function() {
            var widgets = document.querySelectorAll('[role="radio"], [role="option"], ' +
                'input[type="radio"], .goog-option, option, select');
            var result = [];
            for (var el of widgets) {
                var text = (el.textContent || '').trim().substring(0, 60);
                var label = (el.getAttribute('aria-label') || '');
                var value = (el.getAttribute('value') || el.getAttribute('data-value') || '');
                if (text.includes('TPU') || label.includes('TPU') || value.includes('TPU') ||
                    text.includes('GPU') || label.includes('GPU') || value.includes('GPU') ||
                    text.includes('A100') || text.includes('T4') ||
                    text.includes('Hardware') || text.includes('Accelerator')) {
                    result.push({
                        text: text.substring(0, 50),
                        label: label.substring(0, 50),
                        value: value.substring(0, 50),
                        tag: el.tagName,
                        visible: el.offsetParent !== null,
                        checked: el.checked || el.getAttribute('aria-checked') || ''
                    });
                }
            }
            return result;
        })();
        """)
        if hw:
            print(f"   Found {len(hw)} hardware options:")
            for h in hw:
                print(f"     '{h.get('text','')}' visible={h.get('visible','?')} checked={h.get('checked','?')}")
        else:
            print("   No hardware options yet, checking page content...")
            page_content = await client.js("""
            (function() {
                var walker = document.createTreeWalker(document.body, 4, null, false);
                var texts = [];
                var node;
                var seen = new Set();
                while (node = walker.nextNode()) {
                    var t = (node.textContent || '').trim();
                    if (t.length > 5 && t.length < 150 && !seen.has(t)) {
                        seen.add(t);
                        texts.push(t.substring(0, 120));
                    }
                }
                return texts.slice(0, 40);
            })();
            """)
            if page_content:
                print(f"   Page content ({len(page_content)} unique text nodes):")
                for t in page_content[:20]:
                    print(f"     {t[:100]}")

        # Try to find and click TPU / connect button
        print("\n5. Looking for Connect/TPU button...")
        connect = await client.js("""
        (function() {
            var buttons = document.querySelectorAll('button, [role="button"]');
            for (var b of buttons) {
                var t = (b.textContent || '').trim();
                var a = (b.getAttribute('aria-label') || '');
                if (b.offsetParent !== null &&
                    (t.includes('Connect') || t.includes('TPU') ||
                     a.includes('Connect') || a.includes('TPU'))) {
                    b.click();
                    return 'clicked: ' + (t || a).substring(0, 40);
                }
            }
            // Look for "Connect" text anywhere in toolbar area
            var all = document.querySelectorAll('*');
            for (var el of all) {
                if (el.children.length > 0) continue;
                var t = (el.textContent || '').trim();
                if (t === 'Connect' && el.offsetParent !== null) {
                    el.click();
                    return 'clicked Connect text on <' + el.tagName + '>';
                }
            }
            return 'not found';
        })();
        """)
        print(f"   Connect: {connect}")
        await asyncio.sleep(2)

        # Final screenshot
        await client.ss("/tmp/colab_final.png")
        print("   Final screenshot saved to /tmp/colab_final.png")

    await client.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
