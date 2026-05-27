"""Simple, robust CDP runner for Colab TPU setup and training launch."""

import asyncio, json, urllib.request, websockets, base64


class CDP:
    def __init__(self):
        self._id = 0
        self._pending = {}
        self.ws = None

    async def connect(self, url):
        self.ws = await websockets.connect(url)
        asyncio.create_task(self._recv())
        await self._cmd("Page.enable")
        await self._cmd("Runtime.enable")

    async def _recv(self):
        while True:
            try:
                msg = json.loads(await self.ws.recv())
                mid = msg.get("id")
                if mid in self._pending:
                    self._pending[mid].set_result(msg)
            except Exception:
                break

    async def _cmd(self, method, params=None):
        self._id += 1
        mid = self._id
        self._pending[mid] = asyncio.get_event_loop().create_future()
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        return await self._pending[mid]

    async def eval(self, expr):
        r = await self._cmd("Runtime.evaluate", {
            "expression": expr, "returnByValue": True, "awaitPromise": True
        })
        result = r.get("result", {})
        if "exceptionDetails" in result:
            return None
        return result.get("result", {}).get("value")

    async def ss(self, path):
        r = await self._cmd("Page.captureScreenshot", {"format": "png"})
        d = r.get("result", {}).get("data")
        if d:
            with open(path, "wb") as f:
                f.write(base64.b64decode(d))
            return f"Saved {path}"
        return "No screenshot data"

    async def close(self):
        try:
            await self.ws.close()
        except Exception:
            pass


async def main():
    targets = json.loads(urllib.request.urlopen("http://localhost:40107/json").read().decode())
    tab = None
    for t in targets:
        if "1U3AYDmQJqR3RT2MzuIiONZ0ZunOGorGM" in t.get("url","") and t.get("type") == "page":
            tab = t; break
    if not tab:
        for t in targets:
            if t.get("type") == "page" and "colab.research.google.com/drive/" in t.get("url",""):
                tab = t; break
    if not tab:
        print("No Colab tab found"); return

    print(f"Tab: {tab.get('title','?')}")
    print(f"URL: {tab.get('url','')[:80]}")

    cdp = CDP()
    await cdp.connect(tab["webSocketDebuggerUrl"])
    print("Connected!\n")

    # --- STEP 1: Page state ---
    title = await cdp.eval("document.title")
    url = await cdp.eval("window.location.href")
    print(f"Title: {title}")
    print(f"URL: {url[:80]}\n")

    # --- STEP 2: Navigate to notebook if needed ---
    if "1U3AYDmQJqR3RT2MzuIiONZ0ZunOGorGM" not in (url or ""):
        print("Navigating to notebook...")
        await cdp._cmd("Page.navigate", {
            "url": "https://colab.research.google.com/drive/1U3AYDmQJqR3RT2MzuIiONZ0ZunOGorGM"
        })
        await asyncio.sleep(4)
        title = await cdp.eval("document.title")
        print(f"After navigate: {title}")

    # --- STEP 3: Check if runtime is connected ---
    connected = await cdp.eval("typeof google !== 'undefined' && typeof google.colab !== 'undefined'")
    print(f"Runtime connected: {connected}")

    if not connected:
        # --- STEP 4: Click "Runtime" in the toolbar ---
        print("\n--- Opening Runtime menu ---")
        clicked = await cdp.eval("""
            (() => {
                const els = document.querySelectorAll('[role="button"]');
                for (const el of els) {
                    if ((el.textContent || '').trim() === 'Runtime') {
                        el.click();
                        return true;
                    }
                }
                return false;
            })();
        """)
        print(f"Runtime clicked: {clicked}")
        await asyncio.sleep(1.5)

        # --- STEP 5: Find and click "Change runtime type" ---
        print("\n--- Looking for 'Change runtime type' ---")
        menu_items = await cdp.eval("""
            (() => {
                const items = document.querySelectorAll('[role="menuitem"], [role="option"]');
                return Array.from(items).map(i => ({
                    text: (i.textContent || '').trim().substring(0, 40),
                    visible: i.offsetParent !== null
                }));
            })();
        """)
        print(f"Menu items: {json.dumps(menu_items[:5])}")

        # Click the option
        changed = await cdp.eval("""
            (() => {
                const items = document.querySelectorAll('[role="menuitem"], [role="option"]');
                for (const item of items) {
                    const text = (item.textContent || '').trim();
                    if (text.includes('Change runtime type') || text.includes('change runtime type')) {
                        if (item.offsetParent !== null) {
                            item.click();
                            return 'visible: ' + text.substring(0, 30);
                        }
                    }
                }
                // Try hidden items
                for (const item of items) {
                    const text = (item.textContent || '').trim();
                    if (text.includes('Change runtime type') || text.includes('change runtime type')) {
                        item.click();
                        return 'hidden: ' + text.substring(0, 30);
                    }
                }
                return 'not found';
            })();
        """)
        print(f"Change runtime type: {changed}")
        await asyncio.sleep(2)

        # --- STEP 6: Screenshot to see what dialog appeared ---
        await cdp.ss("/tmp/colab_dialog_state.png")
        print("Screenshot: /tmp/colab_dialog_state.png")

        # --- STEP 7: Look for TPU/hardware accelerator selection ---
        print("\n--- Looking for TPU options ---")
        hw = await cdp.eval("""
            (() => {
                const all = document.querySelectorAll('[role="radio"], [role="option"], ' +
                    'input[type="radio"], label, select option, .goog-option');
                const results = [];
                for (const el of all) {
                    const text = (el.textContent || '').trim().substring(0, 60);
                    const label = (el.getAttribute('aria-label') || '').substring(0, 40);
                    const val = (el.getAttribute('value') || el.getAttribute('data-value') || '');
                    if (text.toLowerCase().includes('tpu') || label.toLowerCase().includes('tpu') ||
                        val.toLowerCase().includes('tpu') || text.toLowerCase().includes('accelerator') ||
                        text.toLowerCase().includes('hardware') || text.toLowerCase().includes('gpu') ||
                        label.toLowerCase().includes('gpu')) {
                        results.push({
                            text: text, label: label, value: val,
                            tag: el.tagName,
                            visible: el.offsetParent !== null,
                            checked: el.checked || el.getAttribute('aria-checked') || ''
                        });
                    }
                }
                return results;
            })();
        """)
        if hw:
            print(f"Hardware options ({len(hw)}):")
            for h in hw:
                print(f"  '{h.get('text','')}' visible={h.get('visible','')} checked={h.get('checked','')}")
        else:
            print("No hardware options found - checking page text...")
            dialog_text = await cdp.eval("""
                (() => {
                    const dialogs = document.querySelectorAll('[role="dialog"]');
                    for (const d of dialogs) {
                        if (d.offsetParent !== null) {
                            return (d.textContent || '').trim().substring(0, 500);
                        }
                    }
                    return 'no dialog visible';
                })();
            """)
            print(f"Dialog: {dialog_text[:300]}")

    # --- STEP 8: Re-check runtime status ---
    await asyncio.sleep(2)
    connected = await cdp.eval("typeof google !== 'undefined' && typeof google.colab !== 'undefined'")
    print(f"\nRuntime connected after changes: {connected}")

    await cdp.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
