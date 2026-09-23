#!/usr/bin/env python3
"""Render pinned p5.brush source in a bounded, fresh Chromium sandbox.

Single: renderer.py source.js --output painting.png
Batch: renderer.py a.js b.js c.js --output-dir renders --workers 2
Requires playwright==1.58.0 and its Chromium installed. Uses the project's
work/playwright-browsers if present, otherwise PLAYWRIGHT_BROWSERS_PATH/default.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
ASSETS = HERE / 'assets'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_backend(backend=None):
    value = backend or os.environ.get('BRUSH_RENDER_BACKEND', 'swiftshader')
    if value not in ('swiftshader', 'metal'):
        raise ValueError('Renderer backend must be swiftshader or metal')
    if value == 'metal' and sys.platform != 'darwin':
        raise ValueError('Metal rendering requires macOS')
    return value


def browser_args(backend):
    args = ['--use-gl=angle', '--use-angle=' + backend]
    if backend == 'swiftshader':
        args.append('--enable-unsafe-swiftshader')
    return args + ['--disable-background-networking', '--disable-component-update']


def render(source, output, seed=0, timeout=90, telemetry=False, backend=None):
    backend = resolve_backend(backend)
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.stat().st_size > 100_000:
        raise ValueError('Sketch exceeds 100KB bound')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Refusing to overwrite existing render/validation')
    started = time.monotonic()
    result = {'source': str(source), 'source_sha256': sha(source), 'seed': seed,
              'canvas_contract': [600, 600], 'valid': False,
              'assets': {p.name: sha(p) for p in ASSETS.glob('*.js')},
              'renderer_sha256': sha(Path(__file__)),
              'telemetry_enabled': telemetry, 'requested_angle_backend': backend,
              'telemetry_sha256': sha(HERE / 'telemetry.py') if telemetry else None,
              'network_enabled': False, 'browser_sandbox_enabled': True}
    with tempfile.TemporaryDirectory(prefix='painting-render-') as folder:
        scratch = Path(folder)
        # Child/browser gets no inherited API keys, SSH agent, cookies or profile.
        env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': folder,
               'TMPDIR': folder, 'LANG': 'en_US.UTF-8'}
        bundled = PROJECT / 'work/playwright-browsers'
        if os.environ.get('PLAYWRIGHT_BROWSERS_PATH'):
            env['PLAYWRIGHT_BROWSERS_PATH'] = str(Path(os.environ['PLAYWRIGHT_BROWSERS_PATH']).resolve())
        elif bundled.exists():
            env['PLAYWRIGHT_BROWSERS_PATH'] = str(bundled)
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(source),
                   '--output', str(scratch / 'image.png'), '--seed', str(seed), '--timeout', str(timeout), '--backend', backend]
        if telemetry:
            command.append('--telemetry')
        with (scratch / 'worker.log').open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
                metadata = scratch / 'image.json'
                if code == 0 and metadata.exists():
                    result.update(json.loads(metadata.read_text()))
                    if (scratch / 'image.png').exists():
                        output.write_bytes((scratch / 'image.png').read_bytes())
                        result['png_sha256'] = sha(output)
                else:
                    result['error'] = (scratch / 'worker.log').read_text(errors='replace')[-4000:]
                    result['error_code'] = 'worker_failure'
            except subprocess.TimeoutExpired:
                result['error'] = f'Hard render deadline exceeded ({timeout}s)'
                result.update(timed_out=True, error_code='render_timeout')
            finally:
                # Kill the complete owned process group, including stuck renderer/GPU children.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def page_errors(payload):
    values = payload.get('errors') or []
    return [str(value) for value in values] if isinstance(values, list) else [str(values)]


def worker(source, output, seed, timeout=90, telemetry_enabled=False, backend=None):
    backend = resolve_backend(backend)
    from playwright.sync_api import sync_playwright
    if telemetry_enabled:
        from telemetry import INSTALL as TELEMETRY_INSTALL
    from PIL import Image
    import io
    import re
    source = Path(source).read_text()
    p5 = (ASSETS / 'p5.min.js').read_text()
    brush = (ASSETS / 'p5.brush.js').read_text()
    # Prevent source/asset text from escaping its inline script element.
    safe = lambda text: re.sub(r'</script', r'<\\/script', text, flags=re.I)
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
    # Sequential substitution avoids replacement inside user source.
    html = html.replace('SEED', str(seed & 0xffffffff), 1).replace('P5', safe(p5), 1).replace('BRUSH', safe(brush), 1)
    bootstrap = """if(typeof window.setup==='function'){const original=window.setup;window.setup=function(){try{const make=window.createCanvas;window.createCanvas=function(w,h,...rest){if(w!==600||h!==600)throw Error('Expected a 600x600 canvas');return make.call(this,w,h,...rest)};original.apply(this,arguments)}finally{window.__setupDone=true}}}else{window.__setupDone=true}"""
    if not telemetry_enabled:
        # Keep the frozen automatic p5 startup path unless diagnostics are chosen.
        suffix='</body></html>'
        html=html[:-len(suffix)]+'<script>'+safe(source)+'</script><script>'+bootstrap+'</script>'+suffix
    errors = []
    blocked = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, chromium_sandbox=True,
            args=browser_args(backend))
        context = browser.new_context(viewport={'width': 700, 'height': 700},
                                      device_scale_factor=1, accept_downloads=False,
                                      service_workers='block', offline=True)
        def deny(route):
            blocked.append(route.request.url[:200]); route.abort()
        context.route('**/*', deny)
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('dialog', lambda dialog: dialog.dismiss())
        painting_started = time.monotonic()
        page.set_content(html, wait_until='commit', timeout=max(1000, (timeout - 2) * 1000))
        telemetry_handle = None
        if telemetry_enabled:
            telemetry_handle = page.evaluate_handle(TELEMETRY_INSTALL)
            try:
                page.add_script_tag(content=source)
            except Exception:
                if not errors and not page.evaluate('(window.__errors || []).length>0'):
                    raise
            page.evaluate('() => {'+bootstrap+";if(typeof window.setup==='function'){new p5();}}")
        deadline = time.monotonic() + timeout
        finished = False
        while time.monotonic() < deadline:
            done = page.evaluate("!!window.__setupDone && (typeof draw!=='function'||(typeof frameCount!=='undefined'&&frameCount>=1&&(typeof isLooping!=='function'||!isLooping())))")
            if done:
                finished = True
                break
            if errors or page.evaluate('(window.__errors || []).length>0'):
                break
            time.sleep(.04)
        payload = page.evaluate("({png:document.querySelector('canvas')?.toDataURL('image/png'), errors:window.__errors})")
        painting_seconds = time.monotonic() - painting_started
        gpu = page.evaluate("""() => {
            const canvas = document.querySelector('canvas');
            const gl = canvas && (canvas.getContext('webgl2') || canvas.getContext('webgl'));
            if (!gl) return {error:'No existing WebGL context'};
            const ext = gl.getExtension('WEBGL_debug_renderer_info');
            return {vendor:gl.getParameter(gl.VENDOR),renderer:gl.getParameter(gl.RENDERER),
                    version:gl.getParameter(gl.VERSION),
                    unmasked_renderer:ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):null};
        }""")
        if backend == 'metal' and 'metal' not in str(gpu.get('unmasked_renderer', '')).lower():
            errors.append('Requested Metal backend was not confirmed; refusing a silent software fallback')
        errors.extend(page_errors(payload))
        result = {'finished': finished, 'errors': errors, 'blocked_requests': blocked, 'valid': False, 'chromium_version': browser.version,
                  'requested_angle_backend': backend, 'gpu': gpu, 'browser_args': browser_args(backend),
                  'painting_seconds': painting_seconds,
                  'brush_telemetry': telemetry_handle.evaluate('(controller) => controller.snapshot()') if telemetry_handle else None}
        if payload.get('png'):
            png = base64.b64decode(payload['png'].split(',', 1)[1])
            image = Image.open(io.BytesIO(png))
            result['dimensions'] = list(image.size)
            result['valid'] = finished and not errors and not blocked and image.size == (600, 600)
            Path(output).write_bytes(png)
        else:
            result['errors'].append('No canvas produced')
        Path(output).with_suffix('.json').write_text(json.dumps(result))
        context.close()
        browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--telemetry', action='store_true', help='Opt in to runtime diagnostics; never enables an efficiency reward')
    parser.add_argument('--backend', choices=['swiftshader', 'metal'], help='Explicit ANGLE backend; defaults to BRUSH_RENDER_BACKEND or swiftshader')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.sources[0], args.output, args.seed, args.timeout, args.telemetry, args.backend)
        except Exception as error:
            from playwright.sync_api import TimeoutError as BrowserTimeout
            timed_out = isinstance(error, BrowserTimeout)
            args.output.with_suffix('.json').write_text(json.dumps({
                'valid': False, 'timed_out': timed_out,
                'error_code': 'render_timeout' if timed_out else 'renderer_error',
                'error': f'{type(error).__name__}: {error}'}))
        return
    if not 1 <= args.workers <= 4 or not 1 <= args.timeout <= 900:
        parser.error('Use 1..4 workers and a 1..900 second deadline')
    if bool(args.output) == bool(args.output_dir) or (args.output and len(args.sources) != 1):
        parser.error('Use --output for one source or --output-dir for a batch')
    outputs = [args.output or args.output_dir / (p.stem + '.png') for p in args.sources]
    if len(set(outputs)) != len(outputs):
        parser.error('Batch source names must be unique')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda pair: render(*pair, args.seed, args.timeout, args.telemetry, args.backend), zip(args.sources, outputs)))
    print(json.dumps(results, indent=2))
    if not all(r['valid'] for r in results):
        sys.exit(1)


if __name__ == '__main__':
    main()
