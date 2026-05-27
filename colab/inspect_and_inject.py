"""Inspect Colab page DOM and inject training code via CDP.

Usage: python3 colab/inspect_and_inject.py
"""

import asyncio
import json
import base64
import urllib.request
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

    async def js(self, expr):
        result = await self.send("Runtime.evaluate", {
            "expression": expr,
            "returnByValue": True,
        })
        return result.get("result", {}).get("value")

    async def close(self):
        try:
            await self.ws.close()
        except Exception:
            pass


async def get_page_targets():
    resp = urllib.request.urlopen(f"http://localhost:{CDP_PORT}/json")
    return json.loads(resp.read().decode())


async def main():
    # Find the specific Colab notebook tab
    targets = await get_page_targets()
    tab_id, ws_url = None, None
    for t in targets:
        url = t.get("url", "")
        if NOTEBOOK_ID in url and t.get("type") == "page":
            tab_id, ws_url = t["id"], t["webSocketDebuggerUrl"]
            title = t.get("title", "?")
            print(f"Found notebook tab: {title}")
            break

    if not tab_id:
        # fallback: any colab drive page
        for t in targets:
            url = t.get("url", "")
            if "colab.research.google.com/drive/" in url and t.get("type") == "page":
                tab_id, ws_url = t["id"], t["webSocketDebuggerUrl"]
                print(f"Fallback to colab tab: {t.get('title', '?')}")
                break

    if not tab_id:
        print("ERROR: No Colab notebook tab found!")
        return

    client = CDPClient(ws_url)
    await client.connect()
    print("Connected to Chrome CDP\n")

    # --- STEP 1: Get page URL and title ---
    url = await client.js("window.location.href")
    title = await client.js("document.title")
    print(f"URL: {url}")
    print(f"Title: {title}\n")

    # --- STEP 2: Inspect page structure ---
    print("=" * 60)
    print("PAGE STRUCTURE")
    print("=" * 60)

    page_info = await client.js("""
    (function() {
        var info = {};
        info.bodyChildren = document.body ? document.body.children.length : 0;
        info.hasColab = typeof google !== 'undefined' && typeof google.colab !== 'undefined';
        info.hasMonaco = typeof monaco !== 'undefined';
        info.hasCodeMirror = typeof CodeMirror !== 'undefined';

        // Look for cells
        info.importantSelectors = {};
        var selectors = [
            '.cell', '.code-cell', '.notebook-cell', '[role="code"]',
            '.CodeMirror', '.monaco-editor', 'textarea.lm-content',
            'textarea[aria-label*="code"]', 'textarea[aria-label*="Code"]',
            '[id*="cell"]', '.jp-Cell', '.cell-input',
            'colab-editor', 'colab-cell', 'colab-notebook'
        ];
        for (var s of selectors) {
            var els = document.querySelectorAll(s);
            if (els.length > 0) {
                info.importantSelectors[s] = els.length;
            }
        }

        // Find visible textareas
        var textareas = document.querySelectorAll('textarea');
        info.textareaCount = textareas.length;
        info.textareaInfo = [];
        for (var ta of textareas) {
            if (ta.offsetParent !== null) {
                info.textareaInfo.push({
                    id: ta.id,
                    className: ta.className.substring(0, 80),
                    placeholder: (ta.placeholder || '').substring(0, 50),
                    ariaLabel: (ta.getAttribute('aria-label') || '').substring(0, 50),
                    rows: ta.rows,
                    cols: ta.cols,
                    valueLen: (ta.value || '').length,
                    visible: true
                });
            }
        }
        // Add hidden textareas too
        for (var ta of textareas) {
            if (ta.offsetParent === null) {
                info.textareaInfo.push({
                    id: ta.id,
                    className: ta.className.substring(0, 80),
                    placeholder: (ta.placeholder || '').substring(0, 50),
                    ariaLabel: (ta.getAttribute('aria-label') || '').substring(0, 50),
                    rows: ta.rows,
                    cols: ta.cols,
                    valueLen: (ta.value || '').length,
                    visible: false
                });
            }
        }

        // Find buttons
        var buttons = document.querySelectorAll('button, [role="button"]');
        info.buttons = [];
        var seen = new Set();
        for (var b of buttons) {
            var text = (b.textContent || '').trim().substring(0, 40);
            if (text && !seen.has(text)) {
                seen.add(text);
                if (text.toLowerCase().includes('connect') ||
                    text.toLowerCase().includes('runtime') ||
                    text.toLowerCase().includes('run') ||
                    text.toLowerCase().includes('play') ||
                    text.toLowerCase().includes('cell') ||
                    text.toLowerCase().includes('code') ||
                    text.toLowerCase().includes('edit')) {
                    info.buttons.push({
                        text: text,
                        visible: b.offsetParent !== null,
                        tag: b.tagName
                    });
                }
            }
        }

        // Check if the page has colab-frame (iframe)
        var iframes = document.querySelectorAll('iframe');
        info.iframeCount = iframes.length;
        info.iframeInfo = [];
        for (var f of iframes) {
            info.iframeInfo.push({
                src: (f.src || '').substring(0, 100),
                id: f.id,
                visible: f.offsetParent !== null
            });
        }

        return info;
    })();
    """)
    print(json.dumps(page_info, indent=2) if page_info else "null")

    # --- STEP 3: Try to interact with Colab's API ---
    print("\n" + "=" * 60)
    print("COLAB NOTEBOOK INTERACTION")
    print("=" * 60)

    # Try using Colab's internal API to add a code cell and set its content
    # Colab has a colab.notebook.setCellText API
    injection_result = await client.js("""
    (function() {
        var results = [];

        // Method 1: Try colab.notebook.setCellText if available
        try {
            if (typeof google !== 'undefined' && google.colab && google.colab.notebook) {
                var nb = google.colab.notebook;
                results.push('colab.notebook found');
                // List methods
                var methods = [];
                for (var k in nb) {
                    if (typeof nb[k] === 'function') methods.push(k);
                }
                results.push('methods: ' + methods.join(', '));
            }
        } catch(e) {
            results.push('colab.notebook error: ' + e.message);
        }

        // Method 2: Try the prompt API for code injection
        try {
            if (typeof google !== 'undefined' && google.colab && google.colab.injectCode) {
                results.push('colab.injectCode exists');
            }
        } catch(e) {}

        // Method 3: Check for colab's IPython kernel API
        try {
            if (typeof google !== 'undefined' && google.colab && google.colab.kernel) {
                results.push('colab.kernel found');
                var kmethods = [];
                for (var k in google.colab.kernel) {
                    if (typeof google.colab.kernel[k] === 'function') kmethods.push(k);
                }
                results.push('kernel methods: ' + kmethods.join(', '));
            }
        } catch(e) {}

        // Method 4: Check for Jupyter API
        try {
            if (typeof Jupyter !== 'undefined') {
                results.push('Jupyter global found');
            }
        } catch(e) {}

        // Method 5: Check for __colab_extension__
        try {
            if (typeof __colab_extension__ !== 'undefined') {
                results.push('colab extension found');
            }
        } catch(e) {}

        // Method 6: Look at all window properties for colab-related ones
        var colabProps = [];
        for (var k in window) {
            if (k.toLowerCase().includes('colab') || k.toLowerCase().includes('notebook')) {
                colabProps.push(k);
            }
        }
        results.push('colab-related window props: ' + colabProps.join(', '));

        return results.join('\\n');
    })();
    """)
    print(injection_result)

    # --- STEP 4: Take screenshot ---
    screenshot = await client.send("Page.captureScreenshot", {"format": "png"})
    screenshot_data = screenshot.get("result", {}).get("data", "")
    if screenshot_data:
        with open("/tmp/colab_inspect.png", "wb") as f:
            f.write(base64.b64decode(screenshot_data))
        print(f"\nScreenshot saved to /tmp/colab_inspect.png ({len(screenshot_data)} bytes)")

    await client.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
