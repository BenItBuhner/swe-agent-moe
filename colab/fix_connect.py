"""Inspect and connect Colab GPU runtime, then run the injected cell."""

import asyncio
import json
import base64
import urllib.request
import websockets


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

    async def js(self, expr):
        r = await self.send("Runtime.evaluate", {
            "expression": expr,
            "returnByValue": True,
            "awaitPromise": True,
        })
        result = r.get("result", {})
        if "exceptionDetails" in result:
            return {"error": result["exceptionDetails"]["text"]}
        return result.get("result", {}).get("value")

    async def close(self):
        try:
            await self.ws.close()
        except Exception:
            pass


async def main():
    targets = json.loads(urllib.request.urlopen(
        f"http://localhost:{CDP_PORT}/json"
    ).read().decode())

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
        print("ERROR: No Colab tab!")
        return

    cdp = CDP(tab["webSocketDebuggerUrl"])
    await cdp.connect()
    await cdp.send("Page.enable")
    await cdp.send("Runtime.enable")
    print("Connected to CDP\n")

    # Navigate
    await cdp.send("Page.navigate", {
        "url": f"https://colab.research.google.com/drive/{NOTEBOOK_ID}"
    })
    await asyncio.sleep(4)

    # === DEEP INSPECTION ===
    print("=" * 60)
    print("DEEP PAGE INSPECTION")
    print("=" * 60)

    # Get ALL text content of buttons
    buttons = await cdp.js("""
    (function() {
        var all = document.querySelectorAll('button');
        var result = [];
        for (var b of all) {
            var text = (b.textContent || '').trim().substring(0, 80);
            var html = b.innerHTML.substring(0, 150);
            var visible = b.offsetParent !== null;
            var rect = b.getBoundingClientRect();
            result.push({
                text: text,
                html: html,
                visible: visible,
                rect: {x: Math.round(rect.x), y: Math.round(rect.y), w: Math.round(rect.width), h: Math.round(rect.height)},
                id: b.id,
                class: (b.className || '').substring(0, 100),
                type: b.type
            });
        }
        return result;
    })();
    """)
    if buttons:
        print(f"\nFound {len(buttons)} buttons:")
        for b in buttons:
            print(f"  visible={b.get('visible','?')} text='{b.get('text','')[:50]}' "
                  f"rect=({b.get('rect',{}).get('x','?')},{b.get('rect',{}).get('y','?')}) "
                  f"id='{b.get('id','')}'")
    else:
        print("No buttons found")

    # Get spans with text
    spans = await cdp.js("""
    (function() {
        var all = document.querySelectorAll('span, div[role="button"]');
        var result = [];
        for (var s of all) {
            var text = (s.textContent || '').trim().substring(0, 80);
            if (text && text.length < 60) {
                result.push({
                    text: text,
                    tag: s.tagName,
                    visible: s.offsetParent !== null,
                    role: s.getAttribute('role') || ''
                });
            }
        }
        return result;
    })();
    """)
    if buttons:
        # Check what's at the top-right toolbar area (where Connect usually is)
        toolbar = await cdp.js("""
        (function() {
            // Get elements in the top area
            var all = document.querySelectorAll('*');
            var results = [];
            for (var el of all) {
                var rect = el.getBoundingClientRect();
                if (rect.top < 70 && rect.width > 20 && rect.width < 400) {
                    var text = (el.textContent || '').trim().substring(0, 50);
                    if (text && text.length > 0) {
                        var tag = el.tagName;
                        results.push({
                            tag: tag,
                            text: text,
                            x: Math.round(rect.x), y: Math.round(rect.y),
                            w: Math.round(rect.width), h: Math.round(rect.height)
                        });
                    }
                }
            }
            // Deduplicate
            var seen = new Set();
            var unique = [];
            for (var r of results) {
                var key = r.tag + '|' + r.text;
                if (!seen.has(key)) {
                    seen.add(key);
                    unique.push(r);
                }
            }
            return unique;
        })();
        """)
        if toolbar:
            print(f"\nToolbar elements (y < 70):")
            for t in toolbar:
                print(f"  <{t.get('tag','?')}> '{t.get('text','')[:50]}' "
                      f"({t.get('x','?')},{t.get('y','?')}) {t.get('w','?')}x{t.get('h','?')}")

    # Try to find the runtime state indicator
    runtime_state = await cdp.js("""
    (function() {
        // Look for runtime status indicators
        var all = document.querySelectorAll('[class*="runtime"], [class*="status"], '
            + '[class*="connect"], [aria-label*="runtime"], [aria-label*="Runtime"]');
        var result = [];
        for (var el of all) {
            result.push({
                text: (el.textContent || '').trim().substring(0, 80),
                cls: (el.className || '').substring(0, 80),
                tag: el.tagName,
                visible: el.offsetParent !== null
            });
        }
        return result;
    })();
    """)
    if runtime_state:
        print(f"\nRuntime state elements:")
        for r in runtime_state:
            print(f"  <{r.get('tag','?')}> text='{r.get('text','')[:60]}' class='{r.get('cls','')[:60]}'")

    # Try to find elements with: Connect, Sign in, Runtime, etc.
    for keyword in ["Connect", "Runtime", "Sign in", "Run cell", "RAM", "Disk", "GPU"]:
        found = await cdp.js(f"""
        (function() {{
            var all = document.querySelectorAll('*');
            var results = [];
            for (var el of all) {{
                if (el.children.length > 0) continue;
                var text = (el.textContent || '').trim();
                if (text.includes('{keyword}')) {{
                    var rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {{
                        results.push({{
                            tag: el.tagName,
                            text: text.substring(0, 80),
                            visible: el.offsetParent !== null,
                            x: Math.round(rect.x), y: Math.round(rect.y),
                        }});
                    }}
                }}
            }}
            return results.slice(0, 5);
        }})();
        """)
        if found:
            print(f"\nElements containing '{keyword}':")
            for f in found:
                print(f"  <{f.get('tag','?')}> '{f.get('text','')[:60]}' visible={f.get('visible','?')}")

    # Take screenshot
    ss = await cdp.send("Page.captureScreenshot", {"format": "png"})
    sd = ss.get("result", {}).get("data", "")
    if sd:
        with open("/tmp/colab_inspect2.png", "wb") as f:
            f.write(base64.b64decode(sd))
        print(f"\nScreenshot saved ({len(sd)} bytes)")

    await cdp.close()
    print("\nDone")


if __name__ == "__main__":
    asyncio.run(main())
