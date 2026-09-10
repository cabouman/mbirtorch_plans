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
cd plans/experiments/geometry_viewer/web
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
