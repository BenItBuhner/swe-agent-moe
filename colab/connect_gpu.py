"""Connect to A100 GPU runtime in Colab via Runtime menu."""

import asyncio
import json
import base64
import urllib.request
import websockets


CDP_PORT = 40107
NOTEBOOK_ID = "1U3AYDmQJqR3RT2MzuIiONZ0ZunOGorGM"


async def cdp_eval(ws, expr):
    """Evaluate JS and return value."""
    await ws.send(json.dumps({
        "id": 1, "method": "Runtime.evaluate",
        "params": {"expression": expr, "returnByValue": True, "awaitPromise": True}
    }))
    resp = json.loads(await ws.recv())
    while resp.get("id") != 1:
        resp = json.loads(await ws.recv())
    result = resp.get("result", {})
    if "exceptionDetails" in result:
        return {"error": str(result["exceptionDetails"])}
    return result.get("result", {}).get("value")


async def click_element(ws, selector, description):
    """Find element and click it."""
    result = await cdp_eval(ws, f"""
    (function() {{
        var el = document.querySelector({json.dumps(selector)});
        if (!el) return {{error: 'not found'}};
        el.click({{bubbles: true, cancelable: true}});
        return {{ok: true, text: (el.textContent || '').trim().substring(0, 50)}};
    }})();
    """)
    if isinstance(result, dict) and result.get("ok"):
        print(f"  Clicked {description}: '{result.get('text','')}'")
    else:
        print(f"  FAILED to click {description}: {result}")
    return result


async def main():
    # Get page targets
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
        print("ERROR: No Colab tab!")
        return

    ws = await websockets.connect(tab["webSocketDebuggerUrl"])
    # Enable domains
    await ws.send(json.dumps({"id": 0, "method": "Page.enable", "params": {}}))
    await ws.recv()
    await ws.send(json.dumps({"id": 0, "method": "Runtime.enable", "params": {}}))
    await ws.recv()
    # Consume any additional events
    await asyncio.sleep(0.5)

    # Navigate to ensure fresh state
    await ws.send(json.dumps({
        "id": 0, "method": "Page.navigate",
        "params": {"url": f"https://colab.research.google.com/drive/{NOTEBOOK_ID}"}
    }))
    await asyncio.sleep(3)

    print("=" * 60)
    print("CONNECTING GPU RUNTIME")
    print("=" * 60)

    # Step 1: Click "Runtime" menu
    print("\n1. Opening Runtime menu...")
    # Try different selectors for Runtime menu
    for sel in [
        'span:has-text("Runtime")', 'div:has-text("Runtime")',
        '[aria-label="Runtime"]', 'span:has-text("Runtime")',
        '#runtime-menu-button', '[data-action="runtime"]'
    ]:
        result = await cdp_eval(ws, f"""
        (function() {{
            var el = document.querySelector({json.dumps(sel)});
            return el ? {{found: true, tag: el.tagName, text: (el.textContent||'').trim().substring(0,40)}} : null;
        }})();
        """)
        if result:
            print(f"  Found via '{sel}': {result}")

    # Direct approach: find Runtime text element and click it
    click_runtime = await cdp_eval(ws, """
    (function() {
        var all = document.querySelectorAll('*');
        for (var el of all) {
            var text = (el.textContent || '').trim();
            if (text === 'Runtime' && el.children.length === 0) {
                el.click();
                return 'Clicked Runtime text on <' + el.tagName + '>';
            }
        }
        return 'Runtime element not found';
    })();
    """)
    print(f"  Runtime menu: {click_runtime}")
    await asyncio.sleep(1)

    # Step 2: In the Runtime dropdown, find "Change runtime type"
    print("\n2. Looking for runtime type options...")
    menu_items = await cdp_eval(ws, """
    (function() {
        var items = document.querySelectorAll(
            '[role="menuitem"], [role="option"], .goog-menuitem, ' +
            '[class*="menu-item"], [class*="MenuItem"], ' +
            'li[class*="menu"], div[role="listbox"] > div');
        var result = [];
        for (var item of items) {
            var text = (item.textContent || '').trim().substring(0, 80);
            if (text && text.length > 1) {
                result.push({
                    text: text,
                    tag: item.tagName,
                    role: item.getAttribute('role') || '',
                    visible: item.offsetParent !== null,
                    rect: {
                        x: Math.round(item.getBoundingClientRect().x),
                        y: Math.round(item.getBoundingClientRect().y)
                    }
                });
            }
        }
        return result;
    })();
    """)
    if menu_items:
        print(f"  Found {len(menu_items)} menu items:")
        for item in menu_items:
            if item.get('text',''):
                print(f"    '{item.get('text','')[:60]}' "
                      f"visible={item.get('visible',False)} "
                      f"y={item.get('rect',{}).get('y','?')}")
    else:
        print("  No menu items found")

    # Step 3: Try clicking "Change runtime type"
    change_runtime = await cdp_eval(ws, """
    (function() {
        var all = document.querySelectorAll('*');
        for (var el of all) {
            var text = (el.textContent || '').trim();
            if (text.includes('Change runtime type') || text.includes('change runtime type')) {
                el.click();
                return 'Clicked: ' + text.substring(0, 60);
            }
        }
        return 'Change runtime type not found';
    })();
    """)
    print(f"  Change runtime: {change_runtime}")
    await asyncio.sleep(1)

    # Step 4: If a dialog opened, find GPU options and select A100
    print("\n3. Looking for runtime type selection dialog...")
    dialog_content = await cdp_eval(ws, """
    (function() {
        var dialogs = document.querySelectorAll('[role="dialog"], [role="listbox"], ' +
            '[class*="dialog"], [class*="Dialog"], [class*="modal"], ' +
            '.goog-menu, [id*="menu"]');
        var result = [];
        for (var d of dialogs) {
            if (d.offsetParent !== null) {
                result.push({
                    tag: d.tagName,
                    role: d.getAttribute('role') || '',
                    text: (d.textContent || '').trim().substring(0, 200),
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
    if dialog_content:
        print(f"  Visible dialogs/menus: {len(dialog_content)}")
        for d in dialog_content:
            print(f"    <{d.get('tag','?')}> role={d.get('role','')}")
            print(f"    text: {d.get('text','')[:150]}...")
            print(f"    rect: {d.get('rect',{})}")
    else:
        print("  No visible dialogs")

    # Step 5: Find GPU/TPU options
    print("\n4. Looking for hardware accelerator options...")
    gpu_options = await cdp_eval(ws, """
    (function() {
        var all = document.querySelectorAll('[role="radio"], [role="option"], ' +
            'input[type="radio"], label, option');
        var result = [];
        // Also search text
        var walker = document.createTreeWalker(document.body, 4, null, false);
        var node;
        while (node = walker.nextNode()) {
            var text = (node.textContent || '').trim();
            if ((text.includes('A100') || text.includes('T4') || text.includes('GPU') ||
                 text.includes('TPU') || text.includes('Hardware')) && text.length < 100) {
                result.push({
                    text: text,
                    tag: node.parentElement.tagName
                });
            }
        }
        return result;
    })();
    """)
    if gpu_options:
        print(f"  Hardware options found:")
        for opt in gpu_options:
            print(f"    '{opt.get('text','')[:80]}'")
    else:
        print("  No hardware options found")

    # Take screenshot
    ss = await cdp_eval(ws, "null")  # dummy to get screenshot via send
    await ws.send(json.dumps({
        "id": 100, "method": "Page.captureScreenshot", "params": {"format": "png"}
    }))
    resp = json.loads(await ws.recv())
    while resp.get("id") != 100:
        resp = json.loads(await ws.recv())
    sd = resp.get("result", {}).get("data", "")
    if sd:
        with open("/tmp/colab_dialog.png", "wb") as f:
            f.write(base64.b64decode(sd))
        print(f"\nScreenshot saved ({len(sd)} bytes)")

    await ws.close()
    print("\nDone")


if __name__ == "__main__":
    asyncio.run(main())
