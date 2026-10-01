#!/usr/bin/env python3
"""One bounded instrumented render used by ``profile_renderer.py``.

The worker keeps the frozen renderer's automatic p5 startup order: p5.js and
p5.brush load first, the saved source is appended to the body, and the canvas
contract bootstrap follows the source.  It never calls ``new p5()`` itself.
The parent owns the hard process-group deadline; this process only writes
phase/call events as they become observable.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping


MARKER = "__BRUSH_PROFILE__"
CANVAS = (600, 600)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_script(text: str) -> str:
    return re.sub(r"</script", r"<\\/script", text, flags=re.IGNORECASE)


BOOTSTRAP = """if(typeof window.setup==='function'){const original=window.setup;window.setup=function(){try{const make=window.createCanvas;window.createCanvas=function(w,h,...rest){if(w!==600||h!==600)throw Error('Expected a 600x600 canvas');return make.call(this,w,h,...rest)};original.apply(this,arguments)}finally{window.__setupDone=true}}}else{window.__setupDone=true}"""


def instrumentation_script() -> str:
    """Wrap brush methods without changing this, arguments, returns, or RNG."""

    return f"""<script>(function(){{
const marker={json.dumps(MARKER)};
const names=['fill','fillBleed','fillTexture','line','stroke','strokeWeight','polygon','rect','spline','flowLine','hatch','hatchArray','mass','massArray','circle','ellipse','beginShape','vertex','endShape','beginStroke','endStroke','move','set','noFill','noStroke','noWash'];
const stats={{calls:{{}},duration_ms:{{}},active:null,next_id:0}};
window.__brushProfile=stats;
function emit(event){{try{{console.log(marker+JSON.stringify(event));}}catch(_error){{}}}}
for(const name of names){{
  const original=window.brush && window.brush[name];
  if(typeof original!=='function') continue;
  window.brush[name]=function(...args){{
    const id=++stats.next_id, start=performance.now();
    stats.active={{id:id,name:name,start_ms:start}};
    emit({{event:'start',id:id,name:name,start_ms:start}});
    try{{return original.apply(this,args);}}
    finally{{
      const end=performance.now(), duration=end-start;
      stats.active=null;
      stats.calls[name]=(stats.calls[name]||0)+1;
      stats.duration_ms[name]=(stats.duration_ms[name]||0)+duration;
      emit({{event:'end',id:id,name:name,start_ms:start,end_ms:end,duration_ms:duration}});
    }}
  }};
}}
window.__brushProfileInstalled=true;
}})();</script>"""


def build_html(p5: str, brush: str, source: str, seed: int, instrument: bool = True) -> str:
    """Build the original automatic-startup HTML with optional instrumentation."""

    html = '''<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; frame-src 'none'; worker-src 'none'; form-action 'none'; base-uri 'none'">
<script>let s=SEED||1;Math.random=()=>{s=(s*1664525+1013904223)>>>0;return s/4294967296;};</script>
<script>P5</script><script>BRUSH</script></head><body>
<script>window.__errors=[];window.__setupDone=false;
window.onerror=(m)=>{window.__errors.push(String(m));return true;};
window.addEventListener('unhandledrejection',e=>window.__errors.push(String(e.reason)));
window.addEventListener('securitypolicyviolation',e=>window.__errors.push('Blocked by CSP: '+e.violatedDirective));
for(const key of ['RTCPeerConnection','webkitRTCPeerConnection']){Object.defineProperty(window,key,{value:undefined,writable:false,configurable:false});}
</script></body></html>'''
    html = html.replace("SEED", str(seed & 0xFFFFFFFF), 1).replace("P5", _safe_script(p5), 1).replace("BRUSH", _safe_script(brush), 1)
    suffix = "</body></html>"
    diagnostic = instrumentation_script() if instrument else ""
    # This ordering is intentionally byte-shaped like the frozen renderer:
    # source is before the contract bootstrap, and no manual p5 constructor is
    # introduced.  Instrumentation is the only script inserted before source.
    return html[:-len(suffix)] + diagnostic + "<script>" + _safe_script(source) + "</script><script>" + BOOTSTRAP + "</script>" + suffix


def browser_args(backend: str) -> list[str]:
    if backend != "swiftshader":
        raise ValueError("profile worker supports only swiftshader")
    return ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--disable-background-networking", "--disable-component-update"]


def append_journal(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(event), sort_keys=True) + "\n")
        stream.flush()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_worker(args: argparse.Namespace) -> int:
    if args.backend != "swiftshader":
        raise ValueError("profile worker supports only swiftshader")
    runtime = Path(args.runtime_root)
    source_path = Path(args.source)
    output = Path(args.output)
    journal = Path(args.journal)
    source = source_path.read_text(encoding="utf-8")
    p5_path, brush_path = runtime / "assets" / "p5.min.js", runtime / "assets" / "p5.brush.js"
    p5, brush = p5_path.read_text(encoding="utf-8"), brush_path.read_text(encoding="utf-8")
    result: dict[str, Any] = {
        "schema": "watercolour-render-profile-v2",
        "profile_mode": "instrumented_original_startup",
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "seed": args.seed,
        "renderer_sha256": sha256(runtime / "renderer.py"),
        "assets_sha256": {"p5.min.js": sha256(p5_path), "p5.brush.js": sha256(brush_path)},
        "canvas_contract": list(CANVAS),
        "network_enabled": False,
        "browser_sandbox_enabled": True,
        "valid": False,
        "finished": False,
        "errors": [],
        "blocked_requests": [],
    }
    append_journal(journal, {"event": "worker_start", "monotonic": time.monotonic()})
    try:
        from playwright.sync_api import sync_playwright
        from PIL import Image
        import io

        errors: list[str] = []
        blocked: list[str] = []
        with sync_playwright() as playwright:
            append_journal(journal, {"event": "playwright_ready", "monotonic": time.monotonic()})
            launch_started = time.monotonic()
            append_journal(journal, {"event": "phase_start", "phase": "browser_launch", "monotonic": launch_started})
            browser = playwright.chromium.launch(headless=True, chromium_sandbox=True, args=browser_args(args.backend))
            append_journal(journal, {"event": "phase_end", "phase": "browser_launch", "monotonic": time.monotonic(), "duration_seconds": time.monotonic() - launch_started})
            result["chromium_version"] = browser.version
            context_started = time.monotonic()
            append_journal(journal, {"event": "phase_start", "phase": "context_setup", "monotonic": context_started})
            context = browser.new_context(viewport={"width": 700, "height": 700}, device_scale_factor=1, accept_downloads=False, service_workers="block", offline=True)

            def deny(route: Any) -> None:
                blocked.append(route.request.url[:200])
                route.abort()

            context.route("**/*", deny)
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("dialog", lambda dialog: dialog.dismiss())

            def on_console(message: Any) -> None:
                text = message.text
                if text.startswith(MARKER):
                    try:
                        event = json.loads(text[len(MARKER):])
                        event["event"] = "brush_" + str(event.get("event"))
                        event["monotonic"] = time.monotonic()
                        append_journal(journal, event)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        append_journal(journal, {"event": "brush_profile_parse_error", "text": text[:500], "monotonic": time.monotonic()})

            page.on("console", on_console)
            context_duration = time.monotonic() - context_started
            append_journal(journal, {"event": "phase_end", "phase": "context_setup", "monotonic": time.monotonic(), "duration_seconds": context_duration})
            html = build_html(p5, brush, source, args.seed, instrument=True)
            painting_started = time.monotonic()
            append_journal(journal, {"event": "phase_start", "phase": "automatic_startup_and_drawing", "monotonic": painting_started})
            page.set_content(html, wait_until="commit", timeout=max(1000, int(max(1.0, args.timeout - 2.0) * 1000)))
            deadline = time.monotonic() + args.timeout
            finished = False
            while time.monotonic() < deadline:
                done = page.evaluate("!!window.__setupDone && (typeof draw!=='function'||(typeof frameCount!=='undefined'&&frameCount>=1&&(typeof isLooping!=='function'||!isLooping())))")
                if done:
                    finished = True
                    break
                if errors or page.evaluate("(window.__errors || []).length>0"):
                    break
                time.sleep(.04)
            append_journal(journal, {"event": "phase_end", "phase": "automatic_startup_and_drawing", "monotonic": time.monotonic(), "duration_seconds": time.monotonic() - painting_started, "finished": finished})
            payload_started = time.monotonic()
            append_journal(journal, {"event": "phase_start", "phase": "png_encode_readback", "monotonic": payload_started})
            payload = page.evaluate("({png:document.querySelector('canvas')?.toDataURL('image/png'), errors:window.__errors})")
            append_journal(journal, {"event": "phase_end", "phase": "png_encode_readback", "monotonic": time.monotonic(), "duration_seconds": time.monotonic() - payload_started})
            payload_errors = payload.get("errors") or []
            errors.extend(str(value) for value in payload_errors)
            result.update({"finished": finished, "errors": errors, "blocked_requests": blocked, "timed_out": not finished and not errors, "brush_profile_installed": True})
            if payload.get("png"):
                png_started = time.monotonic()
                png = base64.b64decode(payload["png"].split(",", 1)[1])
                image = Image.open(io.BytesIO(png))
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(png)
                result.update({"dimensions": list(image.size), "png_sha256": hashlib.sha256(png).hexdigest(), "png_decode_write_seconds": time.monotonic() - png_started})
                result["valid"] = finished and not errors and not blocked and image.size == CANVAS
            else:
                result["errors"].append("No canvas produced")
            result["painting_seconds"] = time.monotonic() - painting_started
            context.close()
            browser.close()
        append_journal(journal, {"event": "worker_complete", "monotonic": time.monotonic()})
        write_json(Path(args.receipt), result)
        return 0 if result.get("valid") else 1
    except Exception as error:
        result.update({"error_code": "renderer_error", "error": f"{type(error).__name__}: {error}"})
        append_journal(journal, {"event": "worker_error", "error": result["error"], "monotonic": time.monotonic()})
        write_json(Path(args.receipt), result)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--backend", choices=["swiftshader"], default="swiftshader")
    return run_worker(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
