# Building a Thingy as a Gradio-Lite page: read this first

Date: 2026-09-10.  Written after the mbirtorch geometry viewer's static Space
failed on its first upload, for the reasons below, and cost a day of debugging.
Give this file to the Claude session that builds your page, and read the first
two sections yourself.  Everything here was verified on 2026-09-10 in a
headless Chromium; the page it describes is
https://huggingface.co/spaces/gbuzzard/mbirtorch-geometry-viewer.

## The one problem you will hit, and its fix

A Gradio-Lite page runs a Gradio app inside the visitor's browser, with the
Python interpreter compiled to WebAssembly (Pyodide).  The runtime that does
this, `@gradio/lite`, was published for the last time on 2025-09-10, as
version 5.45.0, and its source has been removed from Gradio's repository.
The runtime is therefore frozen, but it is not self-contained.  When a page
loads, the runtime installs gradio's own dependencies from PyPI at whatever
versions are current that day.  Every dependency release since September 2025
can break every Gradio-Lite page in the world, and three releases have.  A
page built from the documentation shows a red "Error" box with a Python
traceback before the app starts.  The traceback names `micropip`, `filelock`,
or `typing_extensions`, and none of it is your code.

The fix is one `<script>` element, given in full below, pasted into the
page's `<head>` before the runtime's own script.  It patches the runtime as it
loads, so that gradio's dependencies are installed at the versions that were
current when the runtime was published, and it repairs a second defect of the
runtime, which stops the app for good after the first error in an event
handler.  Nothing else about the runtime changes, and the page needs no
server and no paid plan.  You cannot get the same effect from the page's own
`<gradio-requirements>` element, because the runtime reads that element after
the failure has happened.

## The checklist

1. Pin the runtime to release 5.45.0: the page loads
   `https://cdn.jsdelivr.net/npm/@gradio/lite@5.45.0/dist/lite.js` and
   `https://cdn.jsdelivr.net/npm/@gradio/lite@5.45.0/dist/lite.css`.
2. Paste the shim from the section below into `<head>`, before the `lite.js`
   script element.
3. Develop and test the app under `gradio==5.45.0`, the release inside the
   runtime.  The documentation at gradio.app describes Gradio 6, and a
   feature added since 5.45 does not exist in the page.
4. Use only packages that Pyodide 0.27.3 ships, at the versions it ships (see
   the appendix), or packages with a pure Python wheel on PyPI.  A compiled
   package that Pyodide does not ship cannot be installed.  torch, jax, and
   tensorflow are not available.
5. Follow the app rules in the section "Writing the app for the browser":
   no exception may escape an event handler, no thread may be started, and a
   matplotlib figure is sent as PNG.
6. Serve the page over HTTP and read the browser's console while it loads.
   Expect ten to fifteen seconds before the app appears.
7. Upload three files to a static Space: `index.html`, `README.md` with the
   front matter the site reads, and `icon.png`, a square tile of about 400 by
   400 pixels.

## The page

The page is one HTML file.  The runtime finds the app's Python in
`<gradio-file>` elements, runs the one marked `entrypoint`, and installs the
packages named in `<gradio-requirements>` before it does.  The Python is the
element's text content, so the three characters `<`, `>`, and `&` in the
source must be written as `&lt;`, `&gt;`, and `&amp;`, and nothing else
changes.  Write a small script that assembles the page from the Python files,
rather than pasting by hand, so that the copies inside the page cannot drift
from the files you test.

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>your thingy</title>
    SHIM GOES HERE, BEFORE THE RUNTIME'S SCRIPT
    <script type="module" crossorigin src="https://cdn.jsdelivr.net/npm/@gradio/lite@5.45.0/dist/lite.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@gradio/lite@5.45.0/dist/lite.css" />
  </head>
  <body>
    <gradio-lite>
<gradio-requirements>matplotlib</gradio-requirements>
<gradio-file name="app.py" entrypoint>
...the app, with &lt; &gt; &amp; escaped...
</gradio-file>
<gradio-file name="helper.py">
...any other module the app imports...
</gradio-file>
</gradio-lite>
  </body>
</html>
```

## The shim

Paste this element as it stands.  It is the element the geometry viewer's
page carries, and a test in that project keeps this copy identical to the
one the page is built with.

```html
<script>
  // Gradio-Lite 5.45.0 is the last release of the Lite runtime.  This
  // script patches the runtime's worker as it loads, in two places.  The
  // runtime resolves gradio's dependencies from PyPI at whatever versions
  // are current when the page loads, and three of them had moved on by
  // 2026-09-10, each stopping the page before the app started; the first
  // patch holds gradio's dependencies at the releases that were current
  // when the runtime was published, at the versions written into the
  // patch below.  gradio's queue
  // keeps a thread runner that Pyodide cannot run, so one error in an
  // event handler stopped the queue for good; the second patch gives the
  // queue the runtime's own thread-free runner.  Nothing else changes.
  (function () {
    var patches = [["await i.add_mock_package(\"ffmpy\",\"0.3.0\"),await E(s,i,r)", "await i.add_mock_package(\"ffmpy\",\"0.3.0\"),await s.runPythonAsync(\"# Hold gradio's PyPI dependencies at the releases that were current when\\n# Gradio-Lite 5.45.0 was published, on 2025-09-10.  The resolver keeps its\\n# own logic; only the version each name may take is narrowed.\\nimport micropip.transaction as _transaction\\nfrom packaging.specifiers import SpecifierSet as _SpecifierSet\\n_PINS = {\\n    \\\"aiofiles\\\": \\\"==24.1.0\\\",\\n    \\\"annotated-doc\\\": \\\"==0.0.1\\\",\\n    \\\"anyio\\\": \\\"==4.10.0\\\",\\n    \\\"fastapi\\\": \\\"==0.116.1\\\",\\n    \\\"filelock\\\": \\\"==3.19.1\\\",\\n    \\\"groovy\\\": \\\"==0.1.2\\\",\\n    \\\"huggingface-hub\\\": \\\"==0.34.4\\\",\\n    \\\"pydub\\\": \\\"==0.25.1\\\",\\n    \\\"python-multipart\\\": \\\"==0.0.20\\\",\\n    \\\"safehttpx\\\": \\\"==0.1.6\\\",\\n    \\\"semantic-version\\\": \\\"==2.10.0\\\",\\n    \\\"starlette\\\": \\\"==0.47.3\\\",\\n    \\\"tomlkit\\\": \\\"==0.13.3\\\",\\n    \\\"typing-inspection\\\": \\\"==0.4.1\\\",\\n    \\\"websockets\\\": \\\"==15.0.1\\\"\\n}\\n_find_wheel = _transaction.find_wheel\\ndef _pinned_find_wheel(metadata, req):\\n    pin = _PINS.get(req.name)\\n    if pin:\\n        req.specifier &= _SpecifierSet(pin)\\n    return _find_wheel(metadata, req)\\n_transaction.find_wheel = _pinned_find_wheel\\n\"),await E(s,i,r)"], ["anyio.to_thread.run_sync = mocked_anyio_to_thread_run_sync", "anyio.to_thread.run_sync = mocked_anyio_to_thread_run_sync\nimport gradio.queueing\ngradio.queueing.run_sync = mocked_anyio_to_thread_run_sync"]];
    var OriginalWorker = window.Worker;
    function bootstrapSource(stubUrl) {
      // This text runs inside the worker.  The runtime loads its worker
      // through a one-line script that imports the real worker from the
      // CDN, and this bootstrap reads that line, fetches the worker's
      // code, patches it, and runs it.  The requests are synchronous so
      // that the runtime's message handler is installed before its first
      // message arrives.
      return [
        '(function () {',
        '  function text(url) { var request = new XMLHttpRequest(); request.open("GET", url, false); request.send(); return request.responseText; }',
        '  var stub = text(' + JSON.stringify(stubUrl) + ');',
        '  var match = stub.match(/importScripts\\("([^"]+)"\\)/);',
        '  if (!match) { console.warn("Lite runtime shim: the worker stub was not recognized, so the worker runs unpatched"); importScripts(' + JSON.stringify(stubUrl) + '); return; }',
        '  var code = text(match[1]);',
        '  var patches = ' + JSON.stringify(patches) + ';',
        '  patches.forEach(function (pair) {',
        '    var pieces = code.split(pair[0]);',
        '    if (pieces.length === 2) { code = pieces.join(pair[1]); }',
        '    else { console.warn("Lite runtime shim: an anchor was not found, so the worker runs without that patch: " + pair[0]); }',
        '  });',
        '  console.debug("Lite runtime shim: the worker holds 15 PyPI packages at their releases of 2025-09-10 and the queue has a thread-free runner");',
        '  importScripts(URL.createObjectURL(new Blob([code], { type: "text/javascript" })));',
        '})();'
      ].join('\n');
    }
    window.Worker = class extends OriginalWorker {
      constructor(url, options) {
        if (String(url).indexOf('blob:') === 0) {
          var blob = new Blob([bootstrapSource(String(url))], { type: 'text/javascript' });
          super(URL.createObjectURL(blob), options);
        } else {
          super(url, options);
        }
      }
    };
  })();
</script>
```

What it does.  The runtime runs its Python in a web worker, and it loads that
worker through a one-line script that imports the worker's code from the
CDN.  The shim replaces the page's `Worker` constructor with one that reads
the one-line script, fetches the worker's code, changes two exact pieces of
its text, and runs the changed code.  The first piece is the step that
installs the gradio wheels.  A few lines of Python are run before it, which
narrow the versions that Pyodide's installer may choose for fifteen packages
to the newest release each had before 2025-09-10T17:06:22Z, the moment the
gradio 5.45.0 wheel reached PyPI.  The second piece is the runtime's own line
that replaces anyio's thread runner with a thread-free one.  gradio's queue
module binds the runner before that line runs and keeps the original, so the
shim rebinds the queue's copy as well.

What it cannot do.  The two pieces of text are the runtime's minified code,
and the shim finds them by exact match.  Release 5.45.0 will not change, so
the match holds as long as the page pins that release.  If the browser
console ever says that an anchor was not found, the runtime being loaded is
not 5.45.0.  If your app's own requirements ask for a newer release of one of
the fifteen pinned packages, the installer reports that it cannot find a
wheel; that has not happened for any package the geometry viewer uses.

## Writing the app for the browser

The app is ordinary Gradio code with a few rules, each with a reason.

Use Gradio 5.45 features only, and test under that release.  A virtual
environment with `pip install gradio==5.45.0` runs the app on your machine
exactly as the page does, apart from the interpreter underneath.  One
difference between releases matters for forms: Gradio 5.45 sends 0 for an
empty integer `Number` box, where Gradio 6 sends nothing, so treat 0 and
blank alike.

Never let an exception escape an event handler.  The runtime shows every
exception a handler raises in a window with the Python traceback, in front of
the page, and it does this for `gr.Error` as well, which on a server is a
short message in a corner.  Validate the inputs yourself and return the
message in a `gr.Markdown` component, keeping the other outputs unchanged
with `gr.update()`.  Without the shim, one such exception also stopped the
app for good; with the shim it does not, but the traceback window remains.

Keep a dependent control's value inside its range when another control
changes that range.  Gradio checks a `Slider` value against its maximum
before your function runs, and a value past the new maximum is refused with
an error, which the runtime shows as a traceback window.  When one control
sets another's maximum, return both `maximum` and a clamped `value` in the
update, and trigger the redraw with `.then()` after that update rather than
from the same change event.

Start no thread and no process, and open no socket.  Pyodide has none of
them.  `threading.Thread.start()` raises `RuntimeError`, and any library
that starts a thread for you does the same.  Network access exists only
through the browser's `fetch`, which the page's origin and the other site's
CORS policy constrain.

For matplotlib, call `matplotlib.use('Agg')` before any pyplot import, and
give the plot component `gr.Plot(format='png')`, because the default format is
webp and the matplotlib release inside Pyodide cannot write it.  Pyodide
0.27.3 carries matplotlib 3.8.4, released in April 2024.  An argument added
since then, such as `axlim_clip` from 3.10, fails with "got an unexpected
keyword argument".  The offline check is an isolated virtual environment with
`pip install "numpy<2" "matplotlib==3.8.4" "pillow<11" gradio==5.45.0
pytest`, in which the app's tests are run; the geometry viewer found its one
incompatible argument that way in four seconds.

Call `demo.launch()` when the module is the entry point and also when it
runs under Pyodide, because the runtime shows the app that `launch`
registers:

```python
def running_in_pyodide():
    return sys.platform == 'emscripten' or 'pyodide' in sys.modules

if __name__ == '__main__' or running_in_pyodide():
    demo.launch()
```

Budget the time.  The first load downloads the interpreter, gradio, and the
requirements, about thirty megabytes, and then matplotlib builds its font
cache.  The geometry viewer's page shows its first figure ten to fourteen
seconds after the page loads on a fast connection.  Each later control change
is a full call of your function; a matplotlib figure with five panels takes
about 0.4 s to draw and send.  Print the time your function took in a status
line, so that a visitor sees that the page is working.

## Testing the page

Serve the directory over HTTP and open the page in a browser; a `file://`
URL does not work.  `python -m http.server 8000` in the directory is enough.
Open the browser's developer console before the page loads.  The runtime
prints one line per step, and the step that is running when a traceback
appears names the cause:

```
step printed in the console     what a failure there means
Loading Pyodide                 the CDN is unreachable, or the browser blocks WebAssembly
Loading Gradio wheels           a dependency of gradio: the shim is missing or was edited
Importing gradio package        a dependency of gradio that installed but does not import
Installing packages             a package in <gradio-requirements> has no wheel for Pyodide
Setting matplotlib backend      (never seen to fail)
App is now loaded               the runtime is done; anything after this is your app
```

A traceback after "App is now loaded" is your code, or Gradio's handling of
it, and the two windows above explain the two kinds.  A traceback that
mentions `micropip` before the app loads is a dependency problem and not your
code.  Check first that the shim is present and that the console shows its
line, "Lite runtime shim: the worker holds 15 PyPI packages at their releases
of 2025-09-10 and the queue has a thread-free runner".

A headless browser makes the check repeatable.  The script below opens the
page with Playwright, prints every console line, waits for the app, and
saves a screenshot; `pip install playwright` and `playwright install
chromium` set it up.  Extend it with the controls your page has.

```python
"""Open a Gradio-Lite page, print its console, wait for the app, screenshot."""
import time
from playwright.sync_api import sync_playwright

URL = 'http://127.0.0.1:8000/index.html'
TIMEOUT_S = 300

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1500, 'height': 1100})
    page.on('console', lambda m: print(f'[{m.type}] {m.text[:400]}', flush=True))
    page.goto(URL, wait_until='domcontentloaded')
    started = time.perf_counter()
    while time.perf_counter() - started < TIMEOUT_S:
        # The runtime's loading screen says "Loading ..." until the app is up.
        text = page.evaluate('() => document.body.innerText')
        if page.locator('.gradio-container').count() and 'Loading' not in text:
            break
        if 'Traceback' in text:
            break
        time.sleep(1.0)
    print(f'app or error after {time.perf_counter() - started:.0f} s')
    page.screenshot(path='page.png', full_page=True)
    browser.close()
```

Two notes for a Claude Code session.  The Bash tool's sandbox keeps a
headless browser it launches from reaching a local server, and the failure
looks like a timeout rather than a refusal; run the Playwright script with
the sandbox disabled for that one command.  The in-app browser pane opens
only the root of the served directory, keeps serving a cached copy after the
page is rebuilt, and does not show the worker's network requests, so it
cannot verify a Gradio-Lite page; use Playwright.

## The Space

Create a Space on huggingface.co with the SDK set to "static" and upload the
three files at the repository root.  The README's front matter is what the
site reads; the geometry viewer's is

```yaml
---
title: mbirtorch geometry viewer
emoji: 🩻
colorFrom: indigo
colorTo: purple
sdk: static
app_file: index.html
pinned: false
license: bsd-3-clause
short_description: "Draw a CT scan geometry: source, detector, volume, offsets."
---
```

The Thingy Repository's acceptance list adds nothing to those files: the
Space runs when it is opened, its content is appropriate for a public site,
the front matter has `license` and a one-line `short_description`, and
`icon.png` is a square tile of about 400 by 400 pixels at the repository
root.  If the site shows the tile with something cut off at an edge, the
drawing's window was sized to the points and not to the marks drawn around
them; leave a margin for the marker sizes.

## If it still breaks

Read the console and match the step, as in the table above.  Then:

- A `micropip` error during "Loading Gradio wheels" with the shim present and
  its line printed is a new dependency release outside the fifteen pins.
  Find the package in the traceback, take its newest release with a pure
  Python wheel uploaded before 2025-09-10T17:06:22Z, and add it to the
  `_PINS` dictionary inside the shim.  The geometry viewer's repository has
  the script that computes the list (`gv5_lite_pins.py` in
  `plans/geometry_viewer/experiments` of mbirtorch_plans).
- An `ImportError` or `AttributeError` inside a library during "Importing
  gradio package" is the same problem one step later; treat it the same way.
- "Can't find a pure Python 3 wheel" during "Installing packages" is one of
  your requirements.  Check the appendix and PyPI; a package without a pure
  wheel cannot be used, and its job needs a pure Python replacement.
- A traceback window after the app loads is your handler; return messages
  instead of raising, as described above.
- A page that draws once and never again after an error is the queue's
  thread runner; the shim's second patch is missing.

## Appendix: what Pyodide 0.27.3 ships

The runtime uses Pyodide 0.27.3 with Python 3.12.7.  A package below is
installed at this version whenever it is required, so an app must work with
these releases.  A package not shipped is installed from PyPI only if it has
a pure Python wheel.

```
numpy 2.0.2         scipy 1.14.1        matplotlib 3.8.4      pandas 2.2.3
pillow 10.2.0       scikit-learn 1.6.1  scikit-image 0.25.0   sympy 1.13.3
networkx 3.4.2      opencv-python 4.10  statsmodels 0.14.4    xarray 2024.11.0
h5py 3.12.1         pywavelets 1.7.0    shapely 2.0.6         lxml 5.2.1
pyyaml 6.0.2        requests 2.31.0     pydantic 2.10.5       bokeh 3.6.0
altair 5.4.1        typing-extensions 4.11.0
not shipped: torch, jax, tensorflow, numba, plotly (pure wheel on PyPI)
```

The complete list is `https://cdn.jsdelivr.net/pyodide/v0.27.3/full/pyodide-lock.json`.
