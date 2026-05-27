"""Final Colab setup: Connect TPU runtime & start training.

Uses authenticated Chrome CDP. MouseEvent-based clicks for reliability.
"""

import asyncio, json, urllib.request, websockets, base64, time

NID = "1U3AYDmQJqR3RT2MzuIiONZ0ZunOGorGM"

async def run():
    targets = json.loads(urllib.request.urlopen("http://localhost:40107/json").read().decode())
    tab = None
    for t in targets:
        if NID in t.get("url","") and t.get("type") == "page": tab = t; break
    if not tab:
        for t in targets:
            if t.get("type") == "page" and "colab.research.google.com/drive/" in t.get("url",""):
                tab = t; break
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
        if "exceptionDetails" in result:
            detail = result["exceptionDetails"]
            return f"ERR: {detail.get('text','')}"
        return result.get("result", {}).get("value")
    
    async def click_el(mid, selector):
        """Click element by CSS selector using MouseEvent dispatch."""
        return await js(mid, f"""
            (function() {{
                var el = document.querySelector({json.dumps(selector)});
                if (!el) return 'not found: {selector}';
                ['mousedown','mouseup','click'].forEach(function(ev) {{
                    el.dispatchEvent(new MouseEvent(ev, {{bubbles:true, cancelable:true, view:window}}));
                }});
                return 'clicked: {selector}';
            }})();
        """)
    
    async def ss(mid, path):
        r = await cmd(mid, "Page.captureScreenshot", {"format": "png"})
        d = r.get("result", {}).get("data")
        if d:
            with open(path, "wb") as f:
                f.write(base64.b64decode(d))
            return f"ok ({len(d)}b)"

    await cmd(0, "Page.enable")
    await cmd(0, "Runtime.enable")
    
    print("=== SWE-Agent MoE: Colab TPU Setup ===\n")

    # Navigate
    print("1. Loading notebook...")
    await cmd(1, "Page.navigate", {"url": f"https://colab.research.google.com/drive/{NID}"})
    await asyncio.sleep(5)

    # Check runtime status
    connected = await js(2, "typeof google != 'undefined' && typeof google.colab != 'undefined' ? 'yes' : 'no'")
    print(f"   Runtime connected: {connected}")

    if connected == "yes":
        print("   Runtime already connected! Skipping to training injection.")
    else:
        # Open Runtime menu via MouseEvent on runtime-menu-button
        print("\n2. Opening Runtime menu...")
        r1 = await js(3, """
            (function() {
                var el = document.getElementById('runtime-menu-button');
                if (!el) return 'no runtime-menu-button';
                ['mousedown','mouseup','click'].forEach(function(ev) {
                    el.dispatchEvent(new MouseEvent(ev, {bubbles:true, cancelable:true, view:window}));
                });
                return 'clicked';
            })();
        """)
        print(f"   {r1}")
        await asyncio.sleep(1.5)

        # Click "Change runtime type" in the visible dropdown
        print("3. Clicking 'Change runtime type'...")
        r2 = await js(4, """
            (function() {
                var items = document.querySelectorAll('[role="menuitem"]');
                for (var i = 0; i < items.length; i++) {
                    var txt = (items[i].textContent || '').trim();
                    if (txt.indexOf('Change runtime type') >= 0) {
                        ['mousedown','mouseup','click'].forEach(function(ev) {
                            items[i].dispatchEvent(new MouseEvent(ev, {bubbles:true, cancelable:true, view:window}));
                        });
                        return 'clicked';
                    }
                }
                return 'not found';
            })();
        """)
        print(f"   {r2}")
        await asyncio.sleep(3)

        # Screenshot to see if dialog appeared
        print(f"   Screenshot: {await ss(5, '/tmp/colab_setup.png')}")

        # Look for the runtime type dialog using multiple approaches
        print("\n4. Looking for hardware accelerator dialog...")
        
        # Approach A: Check [role="dialog"]
        dialog = await js(6, """
            (function() {
                var dialogs = document.querySelectorAll('[role="dialog"]');
                for (var d = 0; d < dialogs.length; d++) {
                    if (dialogs[d].offsetParent !== null) {
                        var text = (dialogs[d].textContent || '').trim();
                        var html = dialogs[d].innerHTML.substring(0, 1000);
                        return JSON.stringify({
                            text: text.substring(0, 500), 
                            html: html.substring(0, 500)
                        });
                    }
                }
                return 'no dialog';
            })();
        """)
        print(f"   Dialog: {str(dialog)[:500] if dialog else 'None'}")

        # Approach B: Check ALL elements visible near center of page
        center_content = await js(7, """
            (function() {
                var result = [];
                var all = document.querySelectorAll('div, span, section');
                for (var i = 0; i < all.length; i++) {
                    var rect = all[i].getBoundingClientRect();
                    if (rect.width > 200 && rect.height > 100 && rect.top > 100 && rect.top < 600) {
                        var text = (all[i].textContent || '').trim().substring(0, 100);
                        if (text.length > 10) {
                            result.push({
                                text: text.substring(0, 80),
                                y: Math.round(rect.top),
                                h: Math.round(rect.height),
                                visible: all[i].offsetParent !== null
                            });
                        }
                    }
                }
                // Dedup
                var seen = {};
                return JSON.stringify(result.filter(function(r) {
                    if (seen[r.text]) return false;
                    seen[r.text] = true;
                    return true;
                }).slice(0, 20));
            })();
        """)
        print(f"   Center content: {str(center_content)[:1000] if center_content else 'None'}")

        # Approach C: If no dialog appeared, try clicking the menu item another way
        if dialog == "no dialog" or dialog is None:
            print("\n5. Dialog didn't appear. Trying alternative approaches...")
            
            # Try Runtime > Change runtime type again (maybe it closed)
            print("   Re-opening Runtime menu...")
            await click_el(8, "#runtime-menu-button")
            await asyncio.sleep(1.5)
            
            # Try clicking the specific span inside the menu item
            r3 = await js(9, """
                (function() {
                    var items = document.querySelectorAll('[role="menuitem"]');
                    for (var i = 0; i < items.length; i++) {
                        var txt = (items[i].textContent || '').trim();
                        if (txt.indexOf('Change runtime type') >= 0) {
                            // Try clicking the inner span
                            var span = items[i].querySelector('span');
                            if (span) {
                                ['mousedown','mouseup','click'].forEach(function(ev) {
                                    span.dispatchEvent(new MouseEvent(ev, {bubbles:true, cancelable:true, view:window}));
                                });
                                return 'clicked span';
                            }
                            // Fallback to regular click
                            items[i].click();
                            return 'clicked element';
                        }
                    }
                    return 'not found';
                })();
            """)
            print(f"   Alternative click: {r3}")
            await asyncio.sleep(3)
            
            # Check for dialog again
            dialog2 = await js(10, """
                (function() {
                    var dialogs = document.querySelectorAll('[role="dialog"]');
                    for (var d = 0; d < dialogs.length; d++) {
                        if (dialogs[d].offsetParent !== null) {
                            return (dialogs[d].textContent || '').trim().substring(0, 800);
                        }
                    }
                    return 'no dialog';
                })();
            """)
            print(f"   Dialog after retry: {str(dialog2)[:500] if dialog2 else 'None'}")
            
            await ss(11, "/tmp/colab_retry.png")

        # If dialog appeared, select TPU
        if dialog and dialog != "no dialog" and dialog != "None":
            print("\n6. Dialog found! Selecting TPU...")
            # Find TPU radio button
            select_tpu = await js(12, """
                (function() {
                    var radios = document.querySelectorAll('[role="radio"]');
                    for (var i = 0; i < radios.length; i++) {
                        var txt = (radios[i].textContent || '').trim();
                        var label = (radios[i].getAttribute('aria-label') || '');
                        if ((txt.indexOf('TPU') >= 0 || label.indexOf('TPU') >= 0) &&
                            radios[i].offsetParent !== null) {
                            ['mousedown','mouseup','click'].forEach(function(ev) {
                                radios[i].dispatchEvent(new MouseEvent(ev, {bubbles:true, cancelable:true, view:window}));
                            });
                            return 'clicked TPU: ' + txt.substring(0, 30);
                        }
                    }
                    return 'no TPU option';
                })();
            """)
            print(f"   TPU selection: {select_tpu}")
            
            if select_tpu and "clicked" in str(select_tpu):
                # Click Save button
                await asyncio.sleep(1)
                save = await click_el(13, "button:has-text('Save'), button:has-text('Select')")
                print(f"   Save: {save}")
                await asyncio.sleep(5)
                
                # Wait for runtime to connect
                for i in range(15):
                    await asyncio.sleep(2)
                    connected = await js(14, 
                        "typeof google != 'undefined' && typeof google.colab != 'undefined' ? 'yes' : 'no'")
                    if connected == "yes":
                        print(f"   Runtime connected! (attempt {i+1})")
                        break
                else:
                    print("   Timeout waiting for runtime")

    # If runtime is connected, inject and run training
    if connected == "yes":
        print("\n=== RUNTIME CONNECTED! ===")
        print("\n7. Injecting full training code...")
        
        # The training code might already be there from earlier. Let's use kernel API
        run_code = await js(15, """
            (function() {
                if (typeof google != 'undefined' && google.colab && google.colab.kernel) {
                    var code = `
# ===== SWE-Agent MoE: Full Pretraining Pipeline =====
import os, sys
os.chdir('/content')
if not os.path.exists('/content/model-training-pipeline'):
    from google.colab import drive
    drive.mount('/content/drive')
    import shutil
    shutil.copytree('/content/drive/MyDrive/model-training-pipeline', '/content/model-training-pipeline')
sys.path.insert(0, '/content/model-training-pipeline')

print("Installing dependencies...")
!pip install -q torch>=2.4.0 transformers>=4.44.0 accelerate>=0.33.0 datasets>=2.20.0 wandb>=0.17.0 sentencepiece>=0.2.0

print("Starting pretraining...")
os.chdir('/content/model-training-pipeline')
!python train/pretrain.py
`;
                    google.colab.kernel.requestExecute({code: code});
                    return 'executed via kernel API';
                }
                return 'kernel not available';
            })();
        """)
        print(f"   {run_code}")
        
        print("\n=== TRAINING LAUNCHED! ===")
        print("   Check Colab tab for progress.")
    else:
        print("\n=== RUNTIME NOT CONNECTED ===")
        print("   Please click 'Connect' or Runtime > Change runtime type manually.")
        print("   Training code is in the Monaco cell (Ctrl+Enter to run).")

    await ss(99, "/tmp/colab_final_state.png")
    await ws.close()

asyncio.run(run())
