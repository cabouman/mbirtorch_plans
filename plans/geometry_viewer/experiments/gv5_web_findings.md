# gv5: the geometry viewer as a web page

Date: 2026-09-10.  Files: `geometry_defaults.py`, `web/app.py`,
`web/requirements.txt`, `web/README.md`, `gv5_build_web.py`,
`test_geometry_defaults.py`, `test_web_app.py`.

## What was built

One Python app now draws the geometry viewer's figure in a web page, and one
script packages that app two ways.  The app is `web/app.py`, a Gradio page
whose controls set the scan and whose output is the viewer's five-panel
figure.  The script is `gv5_build_web.py`, and it writes two directories.
`web/lite/` is a static Hugging Face Space that runs the app in the browser.
`web/space/` is a Hugging Face Space with the Gradio SDK, which runs the app
on a server.  The static packaging is the one to deploy, because a free
Hugging Face account can host it.

The app needs no mbirtorch and no torch.  A scene needs the reconstruction
shape and the voxel pitch that an mbirtorch model constructor chooses, so
`geometry_defaults.py` computes them.  That file is a copy of mbirtorch's
rules, and `test_geometry_defaults.py` is what keeps the copy honest; the last
section of this record says how.

### How the page's controls map to the viewer's widgets

The desktop viewer owns four widgets, and the page carries all four.  The
view slider becomes a Gradio slider whose maximum follows the view count.  The
three toggles become three checkboxes: "source path", "3D zoom to volume", and
"angle-0 reference".  Each one calls the same `GeometryFigure` argument the
desktop widget calls: `show_trajectory`, `zoom`, and `show_reference`.

The page adds the controls that the desktop example holds as constants.  A
dropdown chooses one of the six geometries.  Numbers set the view count, the
detector shape, the two detector pitches, the two detector offsets, and the
angular range.  Three geometries add one control of their own: the two
distances for cone and translation, the elevation for multiaxis, and the
helical travel for the helical cone scan.  The translation geometry replaces
the view count with its grid of positions, so its two counts and two spacings
appear and the view count and angular range are hidden.

Two sections hold the rest.  A comparison section takes `name=value` lines,
one per parameter to override, and a checkbox draws the comparison; this is
the `compare` argument of `GeometryFigure`.  An advanced section takes a
reconstruction shape by hand, as three numbers, and leaving them blank keeps
the automatic shape.

The figure is a static image, so the app builds it with `widgets=False` and
`blit=False`.  The `blit=False` argument is required and not a preference.
With blitting allowed, the viewer marks every artist that a view change moves
as animated, and a full draw skips an animated artist.  The source, the
detector, and the volume would then be missing from the image.

## One change to geometry_viewer.py

`_full_redraw` now settles the text panel's blocks in both of its paths.  The
old code took the settling passes only when the partial-redraw path was
usable, and returned early otherwise.  A figure built with `blit=False`
therefore kept the places its three text blocks were created with.  The
derived quantities, the comparison list, and the view line then printed on top
of one another.  The change is small.  The draw call is chosen once, and the
same settling loop then follows in both cases.  Only the background caching
still differs between the paths, because a figure that cannot blit has no
background to cache.  This was unavoidable for the web use, since `blit=False`
is required there.  The 187 existing tests still pass.

## The Gradio-Lite status finding, and the pin

Gradio-Lite has no release for Gradio 6.  The npm package `@gradio/lite` was
last published on 2025-09-10, as version 5.45.0, and its source has been
removed from the Gradio main branch.  The static packaging is therefore pinned
to that release:

```
https://cdn.jsdelivr.net/npm/@gradio/lite@5.45.0/dist/lite.js
https://cdn.jsdelivr.net/npm/@gradio/lite@5.45.0/dist/lite.css
```

The app is written to the older API for that reason.  It uses only components
and arguments that exist in both 5.45 and 6.x: `Blocks`, `Row`, `Column`,
`Accordion`, `Dropdown`, `Slider`, `Number`, `Checkbox`, `Textbox`,
`Markdown`, `Plot`, the `.change` and `.click` events, `gr.update`, and
`gr.Error`.  No Gradio 6 feature is used anywhere.

The app was run against Gradio 5.45.0 itself, which is the strongest check
available here.  That release was installed from PyPI into a second virtual
environment, and the app's page built and rendered under it.  The whole test
file then passed under it as well, 20 tests of 20, including the end-to-end
browser test.  The Lite runtime carries the same Gradio Python and the same
JavaScript frontend as that release.  What it changes is the interpreter
underneath.  So this check covers every component and every argument the app
uses.
`figures/gv5_gradio545_screenshot.png` is the page under 5.45.

That run found one difference between the two Gradio releases.  Gradio 5.45
sends zero for an empty integer number box, and Gradio 6 sends nothing at all.
The advanced section's three reconstruction counts start empty, so under 5.45
they arrived as zeros.  The render then refused a reconstruction of zero rows,
and the page showed an error instead of a figure when it opened.  A zero count
now means the automatic shape, as a blank box does, and a test covers both
releases' behavior.  No user means a reconstruction of zero rows, so nothing
is lost.

Three details of the app exist for the browser.  The first is
`matplotlib.use('Agg')`, at the top of the file and before any pyplot import,
because Pyodide has no GUI toolkit.  The second is `gr.Plot(format='png')`,
because the default format is webp and the matplotlib release inside Pyodide
cannot write webp.  The third is the figure size: 12 by 7.5 inches at 90 dots
per inch, which is 1080 by 675 pixels, against the desktop viewer's 15 by 9
inches.  A smaller image is drawn and sent faster.

## How to deploy each packaging

The static Space is the packaging to deploy.  These steps follow the target
site's builder's guide, and a free account can complete them:

1. Create a new Space on the site, with the SDK set to "static" and the Blank
   template.
2. Upload the three files in `web/lite/`: `index.html`, `README.md`, and
   `icon.png`.
3. Open the Space.  The first load takes a while, because the browser
   downloads Python, numpy, and matplotlib before the app starts.

The Gradio Space is the alternative, and it needs a paid plan:

1. Create a new Space with the SDK set to "gradio".
2. Upload the contents of `web/space/`: `app.py`, `geometry_defaults.py`,
   `geometry_scene.py`, `geometry_viewer.py`, `requirements.txt`,
   `README.md`, and `icon.png`.
3. Open the Space.  The server installs the requirements and starts the app.

Both READMEs carry the front matter the site reads.  Each one names the title,
the emoji, `colorFrom`, `colorTo`, the SDK, the app file, `pinned: false`,
`license: bsd-3-clause`, and a `short_description` of 59 characters.  The
acceptance list the guide gives is met by these files: the Space runs when it
is opened, its content is a scan geometry and is appropriate for a public
site, the license field names mbirtorch's BSD 3-clause license, `icon.png` is
a 400 by 400 pixel tile, and `short_description` is set.

One thing needs the lead's decision.  This repository ignores `*.png`, so both
copies of `icon.png` are ignored, as is the screenshot in `figures/`.  The
Space needs the icon in the uploaded directory, so either the lead force-adds
the two icon files, or Greg runs `gv5_build_web.py` before uploading and takes
the icon from his own run.

To run the app on a machine instead of in a Space:

```
cd plans/geometry_viewer/experiments/web
pip install -r requirements.txt
python app.py
```

## Tests

The 187 existing tests still pass.  The command is unchanged:

```
MPLBACKEND=Agg PYTHONPATH=<mbirtorch clone> python -m pytest -q \
    test_geometry_scene.py test_geometry_viewer.py test_geometry_interaction.py
```

Two new test files hold 37 tests.  `test_geometry_defaults.py` holds 17 and
needs mbirtorch on the path.  `test_web_app.py` holds 20 and needs neither
mbirtorch nor torch, because the app needs neither:

```
MPLBACKEND=Agg PYTHONPATH=<mbirtorch clone> python -m pytest -q \
    test_geometry_defaults.py test_web_app.py
```

`test_web_app.py` covers three things.  The render function must return a
matplotlib figure for each of the six geometries, for a comparison, and for a
reconstruction shape given by hand, and it must raise `gr.Error` with a plain
sentence for a view count of zero, for a partial reconstruction shape, and for
three kinds of bad override line.  The build script's output must round-trip:
the test parses `index.html` with `html.parser`, extracts the text of each
`<gradio-file>` element, and asserts that it equals the source file, then
compiles it.  Both Space directories must hold their files, both READMEs must
carry their front matter, and both icons must be 400 by 400 pixels.

### The end-to-end result

The whole page was driven in a browser, and it works under both Gradio
releases.  The test launches the app with
`demo.launch(prevent_thread_lock=True)`, opens it in a
headless Chromium through Playwright, waits for the plot image, presses the
right arrow key on the view slider, and waits for the image to change.  The
image changed, which means the control reached the server and a new figure
came back.  The test writes a full-page screenshot to
`figures/gv5_space_screenshot.png`, which is ignored by git like every image
here.  The same test passes under Gradio 5.45.0, as the section above
reports.

The screenshot showed two layout problems, and both were fixed.  The text
panel's blocks printed on top of one another, which is the
`geometry_viewer.py` change above.  The control column was twice as tall as it
needs to be.  Gradio's minimum width for a number box is wider than half of
that column, so every pair of boxes wrapped onto two lines.  The paired boxes
now carry a smaller minimum width and sit side by side.

Three more paths were driven in the browser by hand, outside the test.
Choosing "translation" and "multiaxis" from the dropdown redrew the figure and
showed each geometry's own controls.  Turning the comparison on drew the
second geometry dashed and listed the difference in the text panel.

## Render times

A render takes about three quarters of a second in this container.  The
numbers below are the median of five renders after one warm-up render, on 4
CPUs with Python 3.11.15, matplotlib 3.11.1, numpy 2.4.6, and Gradio 6.26.0,
under the Agg backend:

```
default cone, 180 views                       740 ms
helical, 1800 views, 30 ALU travel            861 ms
default cone, source path on                  734 ms
default cone, with a comparison             1171 ms
```

The view count costs little and the comparison costs the most.  A render
builds the whole figure, including the settling passes over the text panel.
The fixed cost of the figure is most of the time, so 1800 views cost 16
percent more than 180.  These results indicate that the page stays usable at
any scan size a scanner produces.  They also indicate that a comparison makes
each control change take about half a second longer.

Expect a few times these times in the browser.  The static packaging runs the
same Python under Pyodide, which is slower than a native interpreter, and the
first load also downloads the interpreter, numpy, and matplotlib.

## Limitations

The 3D panel cannot be rotated.  On the desktop the panel accepts a mouse
drag, and matplotlib rotates the camera.  On the web the figure is an image,
so the camera is fixed at the viewer's default: 25 degrees above the drawing's
top and 40 degrees off the +x axis.  A page could offer two numbers for the
camera angles, which `GeometryFigure` already takes as `elevation_deg` and
`azimuth_deg`; this app does not.

Every control change is a full render.  On the Gradio Space each change is
also a round trip to the server.  The desktop viewer's partial-redraw path
cannot be used, for the reason given above, so there is no faster path to
take.

The static packaging was not opened in a browser here.  The build environment
cannot reach a content delivery network, so the Gradio-Lite runtime could not
be downloaded.  Three things were tested in its place: the app under Gradio
5.45.0 on a server, the round-trip of every Python file inside `index.html`,
and the markup of the page itself.  What was not tested is the Lite runtime,
which is the wasm build of that same Gradio release running under Pyodide.

Two risks remain in that untested part.  The first is the runtime itself.  A
release frozen in September 2025 may fail to start in a browser of 2026, and
this environment cannot check that.  The second is Pyodide's matplotlib, which
is older than the one used here; the app already avoids the one
incompatibility that is known, the webp image format, by asking the plot
component for png.  The browser's developer console names either failure.  A
third and smaller risk is the launch.  Gradio-Lite shows the app that `launch`
registers.  `app.py` calls `launch` both when it is run as the program and
when it detects Pyodide, so either way of running the file registers the app.

If the Lite page does fail, the Gradio Space is the packaging that was tested
end to end, and it needs a paid plan.

## What could drift from mbirtorch, and how the test catches it

`geometry_defaults.py` copies four groups of rules, and each one can change in
mbirtorch without changing anything here.  The four are the parameter defaults
of `_utils.py`, what each model's `__init__` sets, each model's
`auto_set_recon_geometry`, and the two utilities the translation geometry uses,
`calc_tct_recon_params` and `gen_translation_vectors`.  A change to any of
them makes this copy wrong.

The test builds the real model and compares every parameter a scene reads.
The names are `required_parameter_names(kind)`, which is the set
`GeometryScene.from_model` reads, so nothing a scene uses is left unchecked.
Shapes, integers, booleans, and strings must be equal; floats and per-view
arrays must agree to a relative 1e-6.  That tolerance is loose enough for the
float32 storage the models use for their view arrays and little more, so a
changed rule fails the test.  The cases are the six scan geometries of
`gv1_conventions_probe.py` and twelve randomized parameter sets per geometry
kind, from a generator with a fixed seed.

One case needed a second test.  A model constructor runs
`auto_set_recon_geometry` while the detector still holds its defaults of one
ALU pitch and zero offset, so a case built through a constructor never
exercises the copied formulas at other detector values.  Two pieces of the
rules are invisible at the default detector: the parallel rule's ratio of the
row pitch to the voxel pitch, which is then one, and the multiaxis rule's
floor on the cosine of the elevation, which binds only above 84 degrees.  The
second test replaces the module's detector defaults with the case's own
detector and calls `auto_set_recon_geometry` again on the model, which is the
call a user makes after changing a detector parameter.  Two deliberately
broken rules were caught by that test and by neither the probe cases nor the
randomized ones.

One mbirtorch behavior is copied on purpose and is worth stating.  Setting a
detector parameter with `set_params` does not re-run
`auto_set_recon_geometry`, so the reconstruction shape a model reports after a
detector change is still the shape of the default detector.
`default_parameters` does the same, `test_geometry_defaults.py` asserts it in
both places, and the module's docstring says so.  A user who wants the
automatic geometry of the new detector calls `auto_set_recon_geometry` by
hand, and the page's advanced section is where a hand-set shape goes.

## Tile update, 2026-09-10

The tile now carries the MBIRTorch wordmark (Greg's request).  The geometry
drawing fills the upper 70 percent of the 400 by 400 pixel tile, and the
wordmark runs across the lower part.  The wordmark is a copy of
`mbirtorch/docs/source/_static/logo.png`, kept at `web/assets/logo.png`; the
build cuts away the reflection below the letters and trims the transparent
margin before placing it.  Without that file the build still writes the tile,
without the wordmark, and prints a line saying so.

Later the same day, the drawing's window was widened by the radius of the
source marker (Greg's request).  The window had been sized to the drawn
points, and the marker is a disk around its point, so the lower part of the
disk was cut at the edge of the drawing, just above the wordmark.

## The static Space did not start, and what fixed it (2026-09-10)

The static Space failed before the app started, and the failure had four
causes.  Three of them are in the way the Gradio-Lite runtime installs
gradio's dependencies, and one is in the viewer's use of matplotlib.  Two more
problems appeared while the fix was verified: one error message froze the
page, and an invalid number showed a Python traceback.  All six are fixed, and
the page now runs in a browser.  The figure appears, the view slider and the
geometry dropdown redraw it, a render takes about 0.4 s inside Pyodide, and an
invalid number is reported in the status line.  The sections below give each
cause, the fix, and the evidence.

### How the runtime starts, and why the page's own requirements come too late

The Lite runtime runs its Python in a web worker, and the worker takes these
steps in order:

1. it loads Pyodide and micropip;
2. it installs the gradio and gradio_client wheels that the runtime carries,
   and micropip resolves their dependencies from PyPI at the newest versions
   that satisfy the wheels' bounds;
3. it replaces ``os.link`` with a two-argument lambda;
4. it imports gradio;
5. it replaces anyio's thread runner with one that calls the function
   directly, because Pyodide has no threads;
6. it writes the app's files, installs the page's ``<gradio-requirements>``,
   and runs the app.

Every failure below happened before step 6.  The
``<gradio-requirements>`` element therefore cannot fix any of them, and the fix
had to reach the worker's own steps.

### The three dependency failures

The first failure is a race in dependency resolution.  gradio 5.45 requires
``huggingface-hub<1.0`` and gradio_client 1.13 requires it with no upper bound.
micropip resolves the two wheels' requirements concurrently and has no
resolver.  The first requirement to reach a package fixes its version.  Since
huggingface_hub 1.0 was released, the unbounded requirement resolves to a 1.x
release, and if it wins the race the bounded one fails with "already
installed".  If the bounded one wins, resolution succeeds.  Three browsers
gave three different results from this one race.  Greg's browser passed it.
Playwright's Chromium failed with the "already installed" message.  The
sandboxed browser of the coding tool failed with "Can't find a pure Python 3
wheel".  micropip gives that message when a requirement fails inside another
package's requirements, so the third result was most likely the same conflict
met one level down.

The second failure is in filelock, which huggingface_hub imports.  filelock
3.30.0, released on 2026-07-16, probes ``os.link(source, target,
follow_symlinks=False)`` when it is imported.  The runtime's lambda has no
``follow_symlinks`` argument, so the probe raises ``TypeError``, which the
probe does not catch, and ``import gradio`` dies inside filelock.  This is the
traceback Greg saw.  Every Gradio-Lite page has been broken this way since
mid-July 2026.

The third failure is a double installation of typing-extensions.  Pyodide's
own package set holds typing-extensions 4.11.0, and micropip takes a package
from that set when its version satisfies the requirement.  The newest anyio,
fastapi, and starlette require typing-extensions 4.12 or later, which micropip
downloads from PyPI as well.  Both copies are installed and the older copy
writes its files last, so anyio's ``from typing_extensions import sentinel``
fails during the import of fastapi.  In September 2025 no dependency asked
for more than 4.11.0, so this failure did not exist when the runtime was
released.

These three failures have one cause.  The runtime is frozen at its last
release, from 2025-09-10, but it resolves its dependencies at load time, so
every dependency release since then can break it.  The fix holds each
dependency that comes from PyPI at its newest release before the runtime was
published.

### The fix: a shim that patches the worker as it loads

``index.html`` now carries a short script before the runtime's own script.
The runtime loads its worker through a one-line script that imports the real
worker from the CDN, because a browser cannot load a worker from another
origin directly.  The shim replaces the page's ``Worker`` constructor with one
that reads that one-line script, fetches the worker's code, changes two pieces
of its text, and runs the changed code from a blob.  The requests are
synchronous, so the worker's message handler is in place before the runtime's
first message arrives.  The two changes are these.

The first change runs a few lines of Python before the worker installs the
gradio wheels.  The Python narrows micropip's wheel search.  For each pinned
name, the requirement's specifier is combined with ``==version`` inside
micropip's ``find_wheel``.  micropip's own logic is untouched, and only the
version a pinned name may take changes.  The versions come from
``web/lite_pins.txt``, one ``name==version`` per line, fifteen packages.
``gv5_lite_pins.py`` writes that file.  Its list of names is the set of
packages that micropip reported with source ``pypi`` after a successful
installation.  For each name it takes the newest release on PyPI with a pure
Python wheel uploaded before 2025-09-10T17:06:22Z, which is the moment the
gradio 5.45.0 wheel reached PyPI.  Packages that Pyodide ships
are not pinned, because micropip takes them from Pyodide's fixed set.  With
these pins the resolution is the same on every load: the race has one
outcome, filelock is 3.19.1, and no package asks for a typing-extensions
newer than 4.11.0.  A pre-installation of the pinned packages was tried first
and rejected, because it installs Pyodide's typing-extensions for real before
the wheels resolve, and requirements for a newer one then fail outright.

The second change concerns gradio's queue.  ``gradio/queueing.py`` binds
anyio's thread runner with ``from anyio.to_thread import run_sync`` when
gradio is imported.  The runtime replaces ``anyio.to_thread.run_sync`` after
that import, so the queue keeps the original.  The queue calls the runner
when an event handler raises, after it has sent the error message to the
page, and Pyodide cannot start a thread.  The call raises ``RuntimeError``,
and the queue's slot for the function is never released.  The page then shows
the error message and never draws again.  In this app every invalid number,
such as a view count of zero, raises ``gr.Error`` on purpose, so one typing
mistake froze the page.  The shim's second change appends two lines to the
worker's own replacement step, so that ``gradio.queueing.run_sync`` is the
thread-free runner as well.  With that change the queue survives an error.

Both changes look for an exact piece of the worker's minified code.  The
runtime is frozen, so the text will not change, and a worker that lacks a
piece runs without that change and says so in the browser console.  Two other
fixes were considered and set aside.  A copy of the runtime's files on the
Space, with the worker edited, would work, but it is about twenty megabytes
of files and a fork of the runtime.  A ``<gradio-requirements>`` pin cannot
work, for the reason given above.

### Two behaviors of the page under Lite, and the app changes for them

The Lite runtime shows every exception a handler raises in a window with the
Python traceback, in front of the page, and the window stays until it is
closed.  The runtime sends the traceback to the page for ``gr.Error`` as well,
which the app raises on purpose for an invalid number such as a view count of
zero.  On a server the same ``gr.Error`` is a short message in the corner of
the page.  Under Pyodide, ``render`` now catches the invalid set of values and
returns the message in the status block, as "Nothing drawn.  The number of
views must be at least one; got 0.  The figure above is the last one drawn.",
and the figure keeps its last value.  The server behavior is unchanged, and a
test covers the Pyodide branch by placing a ``pyodide`` module in
``sys.modules``, which is how the app tells where it runs.

The second behavior was an error of the app's own, present on the server too.
A change of the view count moved the slider's maximum.  A slider value past
the new maximum was then refused by Gradio's own bounds check before
``render`` ran, with the message "Value 1 is greater than maximum value 0."
On the server that was a short message.  Under Lite it was the traceback
window.  Three changes remove it.  ``view_maximum`` now takes the slider's
value as well and clamps it into the new range.  A count below one leaves the
slider alone.  The four controls that set the count draw the figure through a
chained event, after the slider update, so that the value the drawing reads is
inside the new range.

### The matplotlib failure, and the offline check that finds it

With the dependencies resolved, the first render failed with
``Line2D.set() got an unexpected keyword argument 'axlim_clip'``.  The viewer
passed ``axlim_clip=True`` to every 3D drawing call, which asks matplotlib to
hide the parts of a 3D line outside the axes box.  matplotlib added the
argument in release 3.10, and Pyodide 0.27.3 carries matplotlib 3.8.4.
``geometry_viewer.py`` now holds the argument in one dictionary,
``AXLIM_CLIP``, which is empty under a matplotlib older than 3.10, and every
3D drawing call spreads that dictionary into its keywords.  Under the older
release the figure is drawn without the clipping and is otherwise the same.

An isolated virtual environment with matplotlib 3.8.4 and numpy 1.26 is the
offline check for Pyodide's matplotlib.  ``test_web_app.py`` needs neither
torch nor mbirtorch, and it renders all six geometries, a comparison, and a
hand-set reconstruction shape, so it exercises the drawing under the old
release.  Before the fix ten of its tests failed there with the
``axlim_clip`` error, and after the fix all twenty pass.  The three viewer
test files import torch for the projector checks and were not run there.

### Evidence that the page works

The page was driven in a headless Chromium 151 through Playwright, served
from a local directory with the same files the Space holds, on this Mac.
The runtime comes from the same CDN addresses either way, so the local page
and the Space differ only in the address they are served from.  The
measurements of the final page:

```
page load to the first figure              10 to 14 s
render, default cone, 180 views            350 to 420 ms, inside Pyodide
one slider step to a new image             0.5 to 0.8 s
geometry dropdown to the helical scan      0.8 to 1.3 s
view count set to 0, then to 200           "Nothing drawn" in the status, then a new image in 0.8 s
```

The ranges are over the last five runs of the page.  The time to the first
figure includes the download of Pyodide, the wheels, and matplotlib, and
matplotlib's build of its font cache.  The render time is the one the app
prints in its status line, so it is measured inside the worker.  These
results indicate that the page is usable: a control change shows a new figure
in well under a second once the page has loaded.  The evidence is from one
machine and one browser.  The sandboxed browser pane of the coding tool could
not be used for the final check, because it kept serving an earlier copy of
the page.  Greg's own browser is the acceptance check that remains.

The test suite passes: 228 tests, which are the 226 of the handoff and two
new ones.  One checks the pin file, the shim, and the Python the shim runs;
the other checks the status-line report under Pyodide.  The suite ran with
gradio 5.45.0, the release the runtime carries, from a virtual environment on
top of the ``mbirtorch`` conda environment, which lacks gradio and Playwright.

### The Space files, and the acceptance list

``web/lite/`` was rebuilt, and its three files are the Space's files.  They
were copied into the Space checkout and committed there, and Greg pushed them
on 2026-09-10.  The live Space was then driven in the same headless Chromium
as the local page, with the same result: the first figure 11 s after the page
loads, renders of 360 to 435 ms, the slider and the dropdown at 0.8 s, and
the "Nothing drawn" message for a view count of zero followed by a new figure
0.8 s after the count is set again.  ``index.html`` carries the shim, the
pinned versions, and the four Python modules with the ``AXLIM_CLIP`` change.  The README no longer says that the
page was not tested, and it describes the pins.  The acceptance list of the
Thingy Repository stands as before: the README metadata has ``license:
bsd-3-clause`` and a one-line ``short_description``, and ``icon.png`` is a
400 by 400 pixel tile at the Space root.  The remaining item, that the Space
runs when opened, is what the evidence above establishes for a browser on this
machine, for the local page and for the live Space.

### If the page breaks again

The browser console names the step that failed.  A failure during "Loading
Gradio wheels" is a dependency that the pins do not cover.  The fix is a line
in ``web/lite_pins.txt``; ``gv5_lite_pins.py`` regenerates the file once the
new package name is added to its list.  A Python traceback after the app starts
is a difference between matplotlib 3.8.4 and the release the viewer was
developed with, and the isolated environment above reproduces it offline.  A
line saying that an anchor was not found means the runtime's worker changed,
which the pin to release 5.45.0 should prevent.

## The data overlays on the web page (2026-09-11)

Greg asked for the sinogram and the phantom, with their toggles, on the web page
too.  The page has no data of its own and cannot run the projector, because
Pyodide has no torch, so the two overlays reach it differently.

The phantom is generated in the browser.  `geometry_defaults.cube_phantom` is a
numpy copy of mbirtorch's `gen_cube_phantom`, built for whatever reconstruction
shape the current scan has, and a test compares it with mbirtorch's own at nine
shapes.  With the "cube phantom" box on, the page draws the phantom as the
desktop does: the filled silhouette and outline in the top and side views, the
wire outline in the 3D panel, and that outline projected onto the detector face.

The sinogram is the projector's own, computed when the page is built.  The
build script forward-projects the cube phantom with mbirtorch for each of the
six geometry choices at the page's default controls, quantizes each sinogram to
8 bits with its scale, and stores the six with the exact scan parameters they
came from as base64 text inside the page, `default_sinograms.b64`.  Eight bits
are for size.  The six sinograms compress to 10 MB at float32 and to 1.3 MB at
8 bits, because the projector's footprint weights give a float32 sinogram more
than 600,000 distinct values; as text the file is 1.85 MB, and the page grew from
0.37 MB to 2.23 MB.  The picture is the same to the eye.  With the "sinogram of
the cube phantom" box on, the page paints the stored sinogram while the current
scan's parameters equal the stored ones, compared name by name with
`values_are_equal`; when a control differs, it paints nothing and the status
block names the parameters that differ and says that the sinogram is for the
default scan of each geometry only.  A build on a machine without mbirtorch
keeps the file an earlier build left.

The projector is not reproducible to the last bit on this machine.  Three
projections of one phantom differ by one float32 step, and 20 of the 2.2 million
8-bit values then differ, so a new projection changes the whole file.  The build
therefore keeps the file while its stored parameters match the page's defaults,
and projects again only when the defaults change, the file is missing, or a
constant asks for it.  A test checks the stored parameters of the cone geometry
against the page's defaults, so a changed default breaks the build's data
visibly.

The build takes 5.5 s, of which the six projections take 5.1 s including the
imports.  On the server a render with both overlays costs 85 to 110 ms on top of
the 315 ms of the default cone scan.  In headless Chromium the page shows its
first figure 10 s after load, the phantom box redraws the figure in 0.8 s, and
leaving the default scan with the sinogram box on redraws without the sinogram
and names the differing parameters.  At the default cone scan the sinogram box
paints the phantom's shadow in 0.8 s, inside the projected outlines and under
the phantom's own projected outline.  The suite grew to 309 tests, and the web
tests pass under matplotlib 3.8.4.
