"""Verified step-by-step Colab TPU setup with checks between each step."""

import asyncio, json, urllib.request, websockets, base64

NID = "1U3AYDmQJqR3RT2MzuIiONZ0ZunOGorGM"

async def run():
    targets = json.loads(urllib.request.urlopen("http://localhost:40107/json").read().decode())
    tab = None
    for t in targets:
        if NID in t.get("url","") and t.get("type") == "page": tab = t; break
    if not tab:
        for t in targets:
            if t.get("type") == "page" and "colab.research.google.com/drive/" in t.get("url",""): tab = t; break
    if not tab: print("No Colab tab!"); return

    ws = await websockets.connect(tab["webSocketDebuggerUrl"])

    async def cmd(mid, m, p=None):
        await ws.send(json.dumps({"id": mid, "method": m, "params": p or {}}))
        while True:
            r = json.loads(await ws.recv())
            if r.get("id") == mid: return r

    async def js(mid, expr):
        r = await cmd(mid, "Runtime.evaluate", {
            "expression": expr, "returnByValue": True, "awaitPromise": True
        })
        result = r.get("result", {})
        if "exceptionDetails" in result: return None
        return result.get("result", {}).get("value")

    async def click_at(mid, x, y):
        for ev in ["mousePressed", "mouseReleased"]:
            await cmd(mid, "Input.dispatchMouseEvent", {
                "type": ev, "x": x, "y": y, "button": "left", "clickCount": 1,
            })

    async def ss(mid, path):
        r = await cmd(mid, "Page.captureScreenshot", {"format": "png"})
        d = r.get("result", {}).get("data")
        if d:
            with open(path, "wb") as f:
                f.write(base64.b64decode(d))
            return f"ok({len(d)}b)"

    await cmd(0, "Page.enable")
    await cmd(0, "Runtime.enable")
    await cmd(0, "Input.enable")
    print("Connected to authenticated Chrome\n")

    # Navigate
    print("1. Loading notebook...")
    await cmd(1, "Page.navigate", {"url": f"https://colab.research.google.com/drive/{NID}"})
    await asyncio.sleep(5)

    # Check runtime
    con = await js(2, "typeof google != 'undefined' && typeof google.colab != 'undefined' ? 'yes' : 'no'")
    print(f"   Runtime: {con}")
    if con == "yes":
        print("   Already connected!")

    if con != "yes":
        # STEP 2: Open Runtime menu
        print("\n2. Clicking Runtime menu at (252, 38)...")
        await click_at(3, 252, 38)
        await asyncio.sleep(1.5)

        # Verify dropdown appeared
        has_menu = await js(4, """
            (function() {
                var items = document.querySelectorAll('[role="menuitem"]');
                var visible = [];
                for (var i = 0; i < items.length; i++) {
                    if (items[i].offsetParent !== null) {
                        visible.push((items[i].textContent || '').trim().substring(0, 30));
                    }
                }
                return JSON.stringify(visible);
            })();
        """)
        print(f"   Menu items visible: {str(has_menu)[:200] if has_menu else 'NONE'}")

        if has_menu and "Change runtime" in str(has_menu):
            # Find exact "Change runtime type" coordinates
            coords = await js(5, """
                (function() {
                    var items = document.querySelectorAll('[role="menuitem"]');
                    for (var i = 0; i < items.length; i++) {
                        var txt = (items[i].textContent || '').trim();
                        if (txt.indexOf('Change runtime type') >= 0) {
                            var r = items[i].getBoundingClientRect();
                            return JSON.stringify({
                                x: Math.round(r.x + r.width/2),
                                y: Math.round(r.y + r.height/2)
                            });
                        }
                    }
                    return 'not found';
                })();
            """)
            print(f"   Change runtime type at: {coords}")

            if coords and coords != "not found":
                try:
                    c = json.loads(coords)
                    cx, cy = c["x"], c["y"]

                    # Click Change runtime type
                    print(f"3. Clicking at ({cx}, {cy})...")
                    await click_at(6, cx, cy)
                    await asyncio.sleep(3)

                    # Verify dialog
                    dialog = await js(7, """
                        (function() {
                            var dialogs = document.querySelectorAll(
                                '[role="dialog"], [class*="dialog"], [class*="Dialog"], ' +
                                '[class*="modal"], [class*="overlay"]');
                            for (var d = 0; d < dialogs.length; d++) {
                                if (dialogs[d].offsetParent !== null) {
                                    return (dialogs[d].textContent || '').trim().substring(0, 1000);
                                }
                            }
                            return 'no dialog';
                        })();
                    """)
                    print(f"   Dialog: {str(dialog)[:400] if dialog else 'None'}")

                    if dialog and dialog != "no dialog":
                        # Find and click TPU radio button
                        print("\n4. Looking for TPU/GPU options...")
                        
                        tpu = await js(8, """
                            (function() {
                                var result = [];
                                var walker = document.createTreeWalker(document.body, 4, null, false);
                                var node;
                                while (node = walker.nextNode()) {
                                    var text = (node.textContent || '').trim();
                                    if (text.length > 0 && text.length < 200) {
                                        var p = node.parentElement;
                                        if (p && p.offsetParent !== null &&
                                            (text.indexOf('V5E') >= 0 || text.indexOf('V6E') >= 0 ||
                                             text.indexOf('A100') >= 0 || text.indexOf('T4') >= 0 ||
                                             text.indexOf('TPU') >= 0 || text.indexOf('GPU') >= 0)) {
                                            result.push({text: text.substring(0, 60)});
                                        }
                                    }
                                }
                                return JSON.stringify(result);
                            })();
                        """)
                        print(f"   Options: {str(tpu)[:500] if tpu else 'None'}")

                        # Find radio buttons in dialog
                        radios = await js(9, """
                            (function() {
                                var dialogs = document.querySelectorAll('[role="dialog"], [class*="dialog"]');
                                for (var d = 0; d < dialogs.length; d++) {
                                    if (dialogs[d].offsetParent !== null) {
                                        var radios = dialogs[d].querySelectorAll('[role="radio"]');
                                        var result = [];
                                        for (var i = 0; i < radios.length; i++) {
                                            var txt = (radios[i].textContent || '').trim();
                                            var label = radios[i].getAttribute('aria-label') || '';
                                            var r = radios[i].getBoundingClientRect();
                                            result.push({
                                                text: txt.substring(0, 50),
                                                label: label.substring(0, 30),
                                                y: Math.round(r.y + r.height/2)
                                            });
                                        }
                                        // Also check for inputs with types
                                        var inputs = dialogs[d].querySelectorAll('input[type="radio"]');
                                        for (var i = 0; i < inputs.length; i++) {
                                            var label = dialogs[d].querySelector('label[for="' + inputs[i].id + '"]');
                                            var txt = label ? (label.textContent || '').trim() : '';
                                            var r = inputs[i].getBoundingClientRect();
                                            if (txt) {
                                                result.push({
                                                    text: txt.substring(0, 50),
                                                    id: inputs[i].id,
                                                    y: Math.round(r.y + r.height/2)
                                                });
                                            }
                                        }
                                        return JSON.stringify(result);
                                    }
                                }
                                return 'no dialog';
                            })();
                        """)
                        print(f"   Radios: {str(radios)[:500] if radios else 'None'}")
                        
                        # Also scan all text nodes near center
                        center_text = await js(10, """
                            (function() {
                                var dialogs = document.querySelectorAll('[role="dialog"], [class*="dialog"]');
                                for (var d = 0; d < dialogs.length; d++) {
                                    if (dialogs[d].offsetParent !== null) {
                                        return dialogs[d].outerHTML.substring(0, 3000);
                                    }
                                }
                                return 'no dialog';
                            })();
                        """)
                        print(f"   Dialog HTML ({len(str(center_text))} chars):")
                        print(str(center_text)[:2000])
                        
                        await ss(11, "/tmp/colab_dialog_radios.png")
                    else:
                        # No dialog appeared - try clicking the Runtime menu item coordinates again
                        print("   Dialog not found. Trying again...")
                        await click_at(12, 252, 38)
                        await asyncio.sleep(1.5)
                        await click_at(13, cx, cy)
                        await asyncio.sleep(3)
                        
                        await ss(14, "/tmp/colab_retry2.png")
                        dialog = await js(15, """
                            (function() {
                                var dialogs = document.querySelectorAll('[class*="dialog"]');
                                for (var d = 0; d < dialogs.length; d++) {
                                    if (dialogs[d].offsetParent !== null) {
                                        return (dialogs[d].textContent || '').trim().substring(0, 500);
                                    }
                                }
                                // Check for any visible overlay
                                var overlays = document.querySelectorAll('[class*="overlay"], [class*="modal"]');
                                for (var o = 0; o < overlays.length; o++) {
                                    if (overlays[o].offsetParent !== null) {
                                        return 'overlay: ' + (overlays[o].textContent || '').trim().substring(0, 200);
                                    }
                                }
                                return 'still no dialog';
                            })();
                        """)
                        print(f"   After retry: {str(dialog)[:300] if dialog else 'None'}")
                except Exception as e:
                    print(f"   Error: {e}")
            else:
                print("   Could not find Change runtime type coordinates")
                await ss(16, "/tmp/colab_no_menu.png")
        else:
            print("   Runtime dropdown didn't appear!")
            await ss(17, "/tmp/colab_no_dropdown.png")

    await ws.close()
    print("\nDone!")

asyncio.run(run())
