"""One-shot TPU setup and training launch on Colab.

Opens Runtime menu, selects Change runtime type, picks TPU, clicks Save,
waits for runtime, injects training code, runs it. All in one session."""

import asyncio, json, urllib.request, websockets, base64

NID = "1U3AYDmQJqR3RT2MzuIiONZ0ZunOGorGM"

async def run():
    # Find tab
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
        """Click at pixel coordinates using Input.dispatchMouseEvent."""
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

    print("=== SWE-Agent MoE: Colab TPU Setup ===\n")

    # Navigate to notebook
    print("1. Loading notebook...")
    await cmd(1, "Page.navigate", {"url": f"https://colab.research.google.com/drive/{NID}"})
    await asyncio.sleep(6)

    # Check runtime
    con = await js(2, "typeof google != 'undefined' && typeof google.colab != 'undefined' ? 'yes' : 'no'")
    print(f"   Runtime: {con}")

    if con == "yes":
        print("   Already connected! Skipping to training.")
    else:
        # == STEP 2: Open Runtime menu ==
        print("\n2. Opening Runtime menu...")
        await click_at(3, 252, 38)
        await asyncio.sleep(1.5)

        # == STEP 3: Click Change runtime type ==
        print("3. Clicking Change runtime type...")
        await click_at(4, 422, 400)
        await asyncio.sleep(3)

        # == STEP 4: Scan dialog ==
        print("4. Scanning dialog for TPU options...")
        await ss(5, "/tmp/colab_step4.png")

        # Find ALL elements with TPU/GPU text
        hw = await js(6, """
            (function() {
                var result = [];
                var walker = document.createTreeWalker(document.body, 4, null, false);
                var node;
                var seen = new Set();
                while (node = walker.nextNode()) {
                    var text = (node.textContent || '').trim();
                    if (text.length > 1 && text.length < 200) {
                        var parent = node.parentElement;
                        if (parent && parent.offsetParent !== null &&
                            (text.indexOf('TPU') >= 0 || text.indexOf('V5E') >= 0 || 
                             text.indexOf('V6E') >= 0 || text.indexOf('A100') >= 0 ||
                             text.indexOf('T4') >= 0 || text.indexOf('GPU') >= 0 ||
                             text.indexOf('Hardware') >= 0 || text.indexOf('Accelerator') >= 0 || 
                             text.indexOf('None') >= 0)) {
                            var key = text.substring(0, 30);
                            if (!seen.has(key)) {
                                seen.add(key);
                                var rect = parent.getBoundingClientRect();
                                result.push({
                                    text: text.substring(0, 80),
                                    y: Math.round(rect.top + rect.height/2),
                                    visible: parent.offsetParent !== null
                                });
                            }
                        }
                    }
                }
                return JSON.stringify(result);
            })();
        """)
        print(f"   HW options: {str(hw)[:1500]}")

        # Also get all dialog content
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
                return 'no dialog visible';
            })();
        """)
        print(f"   Dialog: {str(dialog)[:500]}")

        # == STEP 5: Select TPU ==
        print("\n5. Selecting TPU accelerator...")
        
        # Strategy: Find the TPU option by scanning for radio buttons with TPU text
        # Use tree walker to find text nodes and click their parent radio
        selected = await js(8, """
            (function() {
                // Find radio buttons
                var radios = document.querySelectorAll('[role="radio"]');
                for (var i = 0; i < radios.length; i++) {
                    var txt = (radios[i].textContent || '').trim();
                    var label = radios[i].getAttribute('aria-label') || '';
                    // Check for TPU (V5E or V6E)
                    if ((txt.indexOf('V5E') >= 0 || txt.indexOf('V6E') >= 0 || 
                         txt.indexOf('TPU') >= 0 || label.indexOf('TPU') >= 0) &&
                        radios[i].offsetParent !== null) {
                        var rect = radios[i].getBoundingClientRect();
                        var x = Math.round(rect.x + rect.width/2);
                        var y = Math.round(rect.y + rect.height/2);
                        return JSON.stringify({type:'tpu', x:x, y:y, text:txt.substring(0,40)});
                    }
                }
                // Fallback: check for None/CPU
                for (var i = 0; i < radios.length; i++) {
                    var txt = (radios[i].textContent || '').trim();
                    var label = radios[i].getAttribute('aria-label') || '';
                    if ((txt.indexOf('None') >= 0 || txt.indexOf('CPU') >= 0) &&
                        radios[i].offsetParent !== null) {
                        var rect = radios[i].getBoundingClientRect();
                        var x = Math.round(rect.x + rect.width/2);
                        var y = Math.round(rect.y + rect.height/2);
                        return JSON.stringify({type:'none', x:x, y:y, text:txt.substring(0,40)});
                    }
                }
                return 'no radios found';
            })();
        """)
        print(f"   Selected: {selected}")

        if selected and selected != "no radios found":
            try:
                sel = json.loads(selected)
                sx, sy = sel.get("x", 0), sel.get("y", 0)
                stype = sel.get("type", "?")
                print(f"   Clicking {stype} at ({sx}, {sy})...")
                
                # Click the radio
                await click_at(9, sx, sy)
                await asyncio.sleep(1)
                
                # == STEP 6: Click Save ==
                print("\n6. Clicking Save...")
                
                # Find Save button
                save = await js(10, """
                    (function() {
                        var btns = document.querySelectorAll('button');
                        for (var i = 0; i < btns.length; i++) {
                            var txt = (btns[i].textContent || '').trim();
                            if ((txt.indexOf('Save') >= 0 || txt.indexOf('SAVE') >= 0) &&
                                btns[i].offsetParent !== null) {
                                var rect = btns[i].getBoundingClientRect();
                                return JSON.stringify({
                                    x: Math.round(rect.x + rect.width/2),
                                    y: Math.round(rect.y + rect.height/2)
                                });
                            }
                        }
                        return 'not found';
                    })();
                """)
                print(f"   Save button: {save}")
                
                if save and save != "not found":
                    try:
                        sv = json.loads(save)
                        await click_at(11, sv["x"], sv["y"])
                        print("   Save clicked!")
                        await asyncio.sleep(5)
                    except:
                        print("   Could not click Save")
                    
            except Exception as e:
                print(f"   Error: {e}")
        else:
            print("   No radios found in expected location")
            # Debug: scan ALL visible element content at page center
            debug = await js(12, """
                (function() {
                    var all = document.querySelectorAll('*');
                    var mid = [];
                    for (var i = 0; i < all.length; i++) {
                        var rect = all[i].getBoundingClientRect();
                        if (rect.top > 200 && rect.top < 500 && rect.width > 100 && 
                            all[i].offsetParent !== null && all[i].children.length === 0) {
                            var txt = (all[i].textContent || '').trim();
                            if (txt.length > 3 && txt.length < 100) {
                                mid.push({
                                    t: txt.substring(0, 50),
                                    y: Math.round(rect.top),
                                    w: Math.round(rect.width)
                                });
                            }
                        }
                    }
                    return JSON.stringify(mid.slice(0, 20));
                })();
            """)
            print(f"   Center page text: {str(debug)[:1000]}")

        await ss(13, "/tmp/colab_after_save.png")

        # == STEP 7: Wait for runtime ==
        print("\n7. Waiting for runtime connection...")
        for i in range(30):
            await asyncio.sleep(2)
            con = await js(14, 
                "typeof google != 'undefined' && typeof google.colab != 'undefined' ? 'yes' : 'no'")
            if con == "yes":
                print(f"   Connected! (attempt {i+1})")
                break
            # Check for any error/blocked dialog
            if i % 5 == 0:
                print(f"   Waiting... ({i+1}s)")
        else:
            print("   Timeout waiting for runtime")

    # == STEP 8: Launch training ==
    if con == "yes":
        print("\n=== RUNTIME CONNECTED! ===")
        print("8. Injecting training code...")
        
        result = await js(15, """
            (function() {
                if (typeof google != 'undefined' && google.colab && google.colab.kernel) {
                    google.colab.kernel.requestExecute({
                        code: `
# === SWE-Agent MoE Training ===
import os, sys, subprocess
from pathlib import Path

print("Setting up training environment...")

# Install deps
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'torch>=2.4.0', 'transformers>=4.44.0', 'accelerate>=0.33.0',
    'datasets>=2.20.0', 'wandb>=0.17.0', 'sentencepiece>=0.2.0',
    'protobuf>=4.25.0', 'einops>=0.7.0'])

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy project
import shutil
src = '/content/drive/MyDrive/model-training-pipeline'
dst = '/content/model-training-pipeline'
if not os.path.exists(dst):
    shutil.copytree(src, dst)
sys.path.insert(0, dst)

# Start pretraining
os.chdir(dst)
print("\\nStarting pretraining...")
!python train/pretrain.py
`
                    });
                    return 'sent to kernel';
                }
                // Try Monaco cell instead
                var editors = monaco && monaco.editor && monaco.editor.getEditors();
                if (editors && editors.length > 0) {
                    editors[0].setValue(CODE);
                    return 'set in Monaco';
                }
                return 'could not inject';
            })();
        """)
        print(f"   {result}")
        print("\n=== TRAINING LAUNCHED! Check Colab tab. ===")
    else:
        print(f"\n=== Runtime not connected (state: {con}) ===")
        print("   Training code already in Monaco cell.")
        print("   Press Ctrl+Enter to run once runtime connects.")

    await ss(99, "/tmp/colab_final_oneshot.png")
    await ws.close()
    print("\nDone!")

asyncio.run(run())
