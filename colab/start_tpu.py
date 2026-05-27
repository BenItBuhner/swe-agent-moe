"""Connect TPU runtime on Colab and start training.

Workflow:
1. Click Runtime > Change runtime type > TPU v4
2. Click Connect 
3. Verify runtime is connected
4. Run the training cell
"""

import asyncio
import json
import base64
import urllib.request
import websockets


CDP_PORT = 40107
NOTEBOOK_ID = "1U3AYDmQJqR3RT2MzuIiONZ0ZunOGorGM"


async def resolve(ws, msg):
    """Send CDP message and get response."""
    await ws.send(json.dumps(msg))
    while True:
        resp = json.loads(await ws.recv())
        if resp.get("id") == msg["id"]:
            return resp


async def js(ws, expr):
    """Evaluate JS with proper ID sequencing."""
    r = await resolve(ws, {
        "id": 1, "method": "Runtime.evaluate",
        "params": {"expression": expr, "returnByValue": True, "awaitPromise": True}
    })
    result = r.get("result", {})
    if "exceptionDetails" in result:
        return None
    return result.get("result", {}).get("value")


async def main():
    # Find tab
    targets = json.loads(urllib.request.urlopen(
        f"http://localhost:{CDP_PORT}/json"
    ).read().decode())
    tab = None
    for t in targets:
        if NOTEBOOK_ID in t.get("url", "") and t.get("type") == "page":
            tab = t; break
    if not tab:
        for t in targets:
            if "colab.research.google.com/drive/" in t.get("url", "") and t.get("type") == "page":
                tab = t; break
    if not tab:
        print("No Colab tab!"); return

    ws = await websockets.connect(tab["webSocketDebuggerUrl"])
    await resolve(ws, {"id": 0, "method": "Page.enable", "params": {}})
    await resolve(ws, {"id": 0, "method": "Runtime.enable", "params": {}})
    await asyncio.sleep(0.3)

    # Navigate
    await resolve(ws, {"id": 0, "method": "Page.navigate",
        "params": {"url": f"https://colab.research.google.com/drive/{NOTEBOOK_ID}"}})
    await asyncio.sleep(4)

    print("=" * 60)
    print("COLAB TPU RUNTIME SETUP")
    print("=" * 60)

    # ===== STEP 1: Click Runtime menu =====
    print("\n1. Opening Runtime menu...")
    await js(ws, """
    (function() {
        var menus = document.querySelectorAll('[role="button"]');
        for (var m of menus) {
            if ((m.textContent || '').trim() === 'Runtime') {
                m.click();
                return;
            }
        }
    })();
    """)
    await asyncio.sleep(1)

    # ===== STEP 2: Verify dropdown appeared =====
    menu_visible = await js(ws, """
    (function() {
        var items = document.querySelectorAll('[role="menuitem"], [class*="menu-item"]');
        var visible = [];
        for (var i of items) {
            if (i.offsetParent !== null) {
                visible.push((i.textContent||'').trim().substring(0,40));
            }
        }
        return visible;
    })();
    """)
    print(f"  Visible menu items: {menu_visible[:5] if menu_visible else 'none'}")

    # ===== STEP 3: Click "Change runtime type" =====
    print("\n2. Clicking 'Change runtime type'...")
    await js(ws, """
    (function() {
        var items = document.querySelectorAll('[role="menuitem"], [class*="menu-item"], ' +
            'div[role="button"], span');
        for (var i of items) {
            var text = (i.textContent || '').trim();
            if (text.includes('Change runtime type') && i.offsetParent !== null) {
                i.click();
                return;
            }
        }
        // Try hidden items too
        for (var i of items) {
            var text = (i.textContent || '').trim();
            if (text.includes('Change runtime type')) {
                i.click();
                return;
            }
        }
    })();
    """)
    await asyncio.sleep(2)

    # ===== STEP 4: Select TPU in dialog =====
    print("3. Looking for runtime type dialog...")
    dialog = await js(ws, """
    (function() {
        var dialogs = document.querySelectorAll('[role="dialog"], [role="listbox"], ' +
            '[class*="dialog"], [class*="Dialog"], [class*="modal"]');
        var result = [];
        for (var d of dialogs) {
            if (d.offsetParent !== null) {
                result.push({
                    text: (d.textContent || '').trim().substring(0, 300),
                    rect: {
                        x: Math.round(d.getBoundingClientRect().x),
                        y: Math.round(d.getBoundingClientRect().y),
                        w: Math.round(d.getBoundingClientRect().width),
                        h: Math.round(d.getBoundingClientRect().height)
                    }
                });
            }
        }
        return result;
    })();
    """)
    if dialog:
        print(f"  Dialogs found: {len(dialog)}")
        for d in dialog:
            print(f"  Text: {d.get('text','')[:200]}")
            print(f"  Rect: {d.get('rect',{})}")
    else:
        print("  No dialogs found")

    # Check for hardware accelerator options
    print("\n4. Looking for TPU/GPU options...")
    hw = await js(ws, """
    (function() {
        var all = document.querySelectorAll('[role="radio"], [role="option"], ' +
            'input[type="radio"], label, .goog-option, option');
        var result = [];
        for (var el of all) {
            var text = (el.textContent || '').trim().substring(0, 80);
            var label = el.getAttribute('aria-label') || '';
            var val = el.getAttribute('value') || '';
            if (text.includes('TPU') || text.includes('A100') || text.includes('T4') ||
                text.includes('GPU') || text.includes('Hardware') ||
                label.includes('TPU') || label.includes('GPU') ||
                val.includes('TPU') || val.includes('GPU') || val.includes('tpu')) {
                result.push({
                    text: text.substring(0, 60),
                    label: label.substring(0, 60),
                    value: val.substring(0, 60),
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
        print(f"  Hardware options:")
        for h in hw:
            print(f"    text='{h.get('text','')}' label='{h.get('label','')}' "
                  f"value='{h.get('value','')}' visible={h.get('visible','?')} "
                  f"checked={h.get('checked','?')}")
    else:
        print("  No hardware options found")

    # Also search DOM text
    print("\n5. Full text scan for accelerator options...")
    text_nodes = await js(ws, """
    (function() {
        var result = [];
        var walker = document.createTreeWalker(document.body, 4, null, false);
        var node;
        var seen = new Set();
        while (node = walker.nextNode()) {
            var text = (node.textContent || '').trim();
            if (text && text.length < 100 && text.length > 1) {
                var parent = node.parentElement;
                if (parent && parent.offsetParent !== null &&
                    (text.includes('TPU') || text.includes('A100') ||
                     text.includes('T4') || text.includes('GPU') ||
                     text.includes('Accelerator') || text.includes('Hardware') ||
                     text.includes('Runtime') || text.includes('None'))) {
                    var key = text.substring(0, 50);
                    if (!seen.has(key)) {
                        seen.add(key);
                        result.push({
                            text: text.substring(0, 80),
                            tag: parent.tagName,
                            y: Math.round(parent.getBoundingClientRect().y)
                        });
                    }
                }
            }
        }
        return result;
    })();
    """)
    if text_nodes:
        print(f"  Text nodes with accelerator keywords:")
        for t in text_nodes:
            print(f"    '{t.get('text','')[:60]}' <{t.get('tag','?')}> y={t.get('y','?')}")

    # Take screenshot
    ss = await resolve(ws, {"id": 100, "method": "Page.captureScreenshot", "params": {"format": "png"}})
    sd = ss.get("result", {}).get("data", "")
    if sd:
        with open("/tmp/colab_tpu_setup.png", "wb") as f:
            f.write(base64.b64decode(sd))
        print(f"\nScreenshot saved ({len(sd)} bytes)")

    await ws.close()
    print("\nDone")


if __name__ == "__main__":
    asyncio.run(main())
