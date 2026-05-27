"""Simple step-by-step Colab setup. Connects, navigates, inspects, and sets up TPU."""

import asyncio, json, urllib.request, websockets, base64, sys

NOTES = "1U3AYDmQJqR3RT2MzuIiONZ0ZunOGorGM"

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

async def find_tab():
    targets = json.loads(urllib.request.urlopen("http://localhost:40107/json").read().decode())
    for t in targets:
        if NOTES in t.get("url", "") and t.get("type") == "page":
            return t
    for t in targets:
        if t.get("type") == "page" and "colab.research.google.com/drive/" in t.get("url",""):
            return t
    return None

async def step(mid, label, expr, ws):
    val = await js(ws, mid, expr)
    print(f"  {label}: {val}")
    return val

async def main():
    tab = await find_tab()
    if not tab:
        print("No Colab tab found!")
        return

    print(f"Tab: {tab.get('title','?')}")
    ws = await websockets.connect(tab["webSocketDebuggerUrl"])
    await send(ws, 0, "Page.enable")
    await send(ws, 0, "Runtime.enable")
    print("Connected!\n")

    # Navigate to ensure fresh state
    print("--- Navigating to notebook ---")
    await send(ws, 10, "Page.navigate", {
        "url": f"https://colab.research.google.com/drive/{NOTES}"
    })
    await asyncio.sleep(5)

    # Inspect page
    print("\n--- Page State ---")
    await step(20, "URL", "window.location.href", ws)
    await step(21, "Title", "document.title", ws)
    
    # Check for various elements
    print("\n--- Element Check ---")
    for el_name, selector in [
        ("role=button", "'[role=\"button\"]'"),
        ("button", "'button'"),
        ("menubar", "'[role=\"menubar\"]'"),
        ("cell", "'.cell'"),
        ("monaco", "'.monaco-editor'"),
        ("code", "'[role=\"code\"]'"),
        ("textarea", "'textarea'"),
        ("colab-notebook", "'colab-notebook'"),
        ("colab-editor", "'colab-editor'"),
    ]:
        mid = 30 + len(el_name)
        val = await js(ws, mid, f"document.querySelectorAll({selector}).length")
        if val and val > 0:
            print(f"  {el_name}: {val}")

    # Check runtime status
    print("\n--- Runtime ---")
    await step(50, "google.colab available", 
        "typeof google !== 'undefined' && typeof google.colab !== 'undefined'", ws)
    
    # Get page HTML head for loading state
    loading = await js(ws, 51, """
        (() => {
            const texts = [];
            for (const el of document.querySelectorAll('[class*="load"], [aria-busy="true"], .loading, .spinner, [class*="spinner"], [class*="progress"]')) {
                if (el.offsetParent !== null) {
                    texts.push((el.className || el.tagName || '').substring(0, 40));
                }
            }
            return texts.length > 0 ? texts : 'no loading indicators';
        })();
    """)
    print(f"  Loading: {json.dumps(loading)}")

    # Take screenshot
    await asyncio.sleep(1)
    ss = await send(ws, 99, "Page.captureScreenshot", {"format": "png"})
    data = ss.get("result", {}).get("data")
    if data:
        with open("/tmp/colab_state_now.png", "wb") as f:
            f.write(base64.b64decode(data))
        print(f"\nScreenshot: {len(data)} bytes")

    await ws.close()
    print("\nDone!")

asyncio.run(main())
