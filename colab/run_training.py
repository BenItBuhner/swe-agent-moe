"""FIXED version: Connect TPU runtime and start training on Colab.

Uses CDP with authenticated Chrome (barnacle.agent@gmail.com).
"""

import asyncio, json, urllib.request, websockets, base64

NID = "1U3AYDmQJqR3RT2MzuIiONZ0ZunOGorGM"

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

async def ss(ws, path):
    r = await send(ws, 999, "Page.captureScreenshot", {"format": "png"})
    d = r.get("result", {}).get("data")
    if d:
        with open(path, "wb") as f:
            f.write(base64.b64decode(d))
        return f"saved {path}"

async def main():
    targets = json.loads(urllib.request.urlopen("http://localhost:40107/json").read().decode())
    tab = None
    for t in targets:
        if NID in t.get("url","") and t.get("type") == "page":
            tab = t; break
    if not tab:
        for t in targets:
            if t.get("type") == "page" and "colab.research.google.com/drive/" in t.get("url",""):
                tab = t; break
    if not tab:
        print("No Colab tab!"); return

    print(f"Tab: {tab.get('title','?')}")
    ws = await websockets.connect(tab["webSocketDebuggerUrl"])
    await send(ws, 0, "Page.enable")
    await send(ws, 0, "Runtime.enable")
    print("Connected!\n")

    # Navigate
    print("--- Loading notebook ---")
    await send(ws, 1, "Page.navigate", {"url": f"https://colab.research.google.com/drive/{NID}"})

    # Wait for notebook editor
    for i in range(20):
        await asyncio.sleep(2)
        m = await js(ws, 10 + i, "document.querySelectorAll('#runtime-menu-button').length")
        if m and m > 0:
            print(f"  Notebook loaded (attempt {i+1})")
            break
    else:
        print("  Timeout loading notebook")

    # Check if runtime already connected
    connected = await js(ws, 50, "typeof google != 'undefined' && typeof google.colab != 'undefined'")
    print(f"  Runtime connected: {connected}")

    if connected:
        print("  Runtime already connected! Skipping setup.")
    else:
        # Click Runtime menu
        print("\n--- Clicking Runtime menu ---")
        r = await js(ws, 51, """
            (function() {
                var el = document.getElementById('runtime-menu-button');
                if (el) { el.click(); return 'clicked'; }
                return 'not found';
            })();
        """)
        print(f"  Runtime menu: {r}")
        await asyncio.sleep(1.5)

        # Click "Change runtime type" in the dropdown
        r = await js(ws, 52, """
            (function() {
                var items = document.querySelectorAll('[role="menuitem"]');
                for (var i = 0; i < items.length; i++) {
                    var txt = (items[i].textContent || '').trim();
                    if (txt.indexOf('Change runtime type') >= 0) {
                        if (items[i].offsetParent !== null) {
                            items[i].click();
                            return 'clicked visible';
                        }
                    }
                }
                for (var i = 0; i < items.length; i++) {
                    var txt = (items[i].textContent || '').trim();
                    if (txt.indexOf('Change runtime type') >= 0) {
                        items[i].click();
                        return 'clicked hidden';
                    }
                }
                return 'not found';
            })();
        """)
        print(f"  Change runtime type: {r}")
        await asyncio.sleep(2)

        # Take screenshot of dialog
        print(f"  Screenshot: {await ss(ws, '/tmp/colab_dialog.png')}")

        # Look for TPU option in the dialog
        print("\n--- Selecting TPU runtime ---")
        r = await js(ws, 53, """
            (function() {
                var result = {found: []};
                var options = document.querySelectorAll('[role="radio"], [role="option"], label, span, div');
                for (var i = 0; i < options.length; i++) {
                    var txt = (options[i].textContent || '').trim();
                    if (txt.indexOf('TPU') >= 0 || txt.indexOf('tpu') >= 0) {
                        result.found.push({
                            text: txt.substring(0, 30),
                            tag: options[i].tagName,
                            visible: options[i].offsetParent !== null
                        });
                    }
                }
                return JSON.stringify(result);
            })();
        """)
        print(f"  TPU options: {r}")

        # If TPU options found, click one
        r = await js(ws, 54, """
            (function() {
                var options = document.querySelectorAll('[role="radio"], [role="option"], label');
                for (var i = 0; i < options.length; i++) {
                    var txt = (options[i].textContent || '').trim();
                    if (txt.indexOf('TPU') >= 0 && options[i].offsetParent !== null) {
                        options[i].click();
                        return 'clicked TPU';
                    }
                }
                for (var i = 0; i < options.length; i++) {
                    var txt = (options[i].textContent || '').trim();
                    if ((txt.indexOf('V5E') >= 0 || txt.indexOf('V6E') >= 0 || txt.indexOf('A100') >= 0)
                        && options[i].offsetParent !== null) {
                        options[i].click();
                        return 'clicked ' + txt.substring(0, 20);
                    }
                }
                return 'no TPU/GPU option';
            })();
        """)
        print(f"  Selection: {r}")

        # Look for and click confirm button (Save/Select/Connect)
        await asyncio.sleep(1)
        r = await js(ws, 55, """
            (function() {
                var btns = document.querySelectorAll('button');
                for (var i = 0; i < btns.length; i++) {
                    var txt = (btns[i].textContent || '').trim();
                    if (btns[i].offsetParent !== null) {
                        if (txt.indexOf('Save') >= 0 || txt.indexOf('Select') >= 0 || 
                            txt.indexOf('Change') >= 0 || txt.indexOf('Connect') >= 0) {
                            btns[i].click();
                            return 'clicked: ' + txt.substring(0, 20);
                        }
                    }
                }
                return 'no confirm button';
            })();
        """)
        print(f"  Confirm: {r}")

        # Wait for runtime to connect
        await asyncio.sleep(5)
        connected = await js(ws, 56, 
            "typeof google != 'undefined' && typeof google.colab != 'undefined'")
        print(f"  Runtime connected: {connected}")

    # Inject training code and run
    if connected:
        print("\n--- Runtime is connected! ---")
        print("  Training code is already in the Monaco cell (from earlier injection).")
        print("  Re-running the cell with Ctrl+Enter...")
        
        r = await js(ws, 60, """
            var cells = document.querySelectorAll('[role="code"]');
            if (cells.length > 0) {
                cells[0].dispatchEvent(new KeyboardEvent('keydown', {
                    key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true
                }));
                return 'dispatched Ctrl+Enter';
            }
            return 'no code cell found';
        """)
        print(f"  Run: {r}")
        print("\n  Training should now be running!")
    else:
        print("\n  Runtime still not connected.")
        print("  Training code is already in the Monaco cell.")
        r = await js(ws, 57, 
            "document.querySelectorAll('.monaco-editor').length")
        print(f"  Monaco editors: {r}")

    # Final screenshot
    print(f"\n  Final: {await ss(ws, '/tmp/colab_final3.png')}")
    await ws.close()
    print("Done!")

asyncio.run(main())
