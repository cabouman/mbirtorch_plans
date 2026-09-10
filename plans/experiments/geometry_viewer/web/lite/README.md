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

# mbirtorch geometry viewer, in the browser

This Space draws the scan geometry of an mbirtorch tomography model: where the
source and the detector sit, where the reconstruction volume sits, which way
the gantry turns, and whether the volume projects inside the detector.  Six
geometries are offered: a cone beam scan with a flat detector, the same scan
with a curved detector, a helical cone beam scan, a parallel beam scan, a
multiaxis parallel scan, and a translation scan.  Nothing is reconstructed and
no data is read.  The page draws parameters.

Everything runs in the browser.  The page loads Gradio-Lite, which runs the
app's Python under Pyodide, so there is no server and no session state.  The
first load takes a while, because the browser downloads Python, numpy, and
matplotlib before the app starts.

The files.  ``index.html`` carries four Python modules inside it:
``app.py`` builds the page, ``geometry_defaults.py`` computes the automatic
reconstruction geometry that an mbirtorch model constructor would choose,
``geometry_scene.py`` turns the parameters into drawable primitives, and
``geometry_viewer.py`` draws them with matplotlib.  All four are copies made by
``gv5_build_web.py`` in the ``mbirtorch_plans`` repository, under
``plans/experiments/geometry_viewer``; that directory holds the sources and the
tests.

This packaging was not tested where it was built.  The build environment
cannot reach a content delivery network, so the Gradio-Lite runtime could not
be downloaded there and the page could not be opened.  The script and the
stylesheet are pinned to @gradio/lite 5.45.0, the last release of that package,
and the app uses no Gradio feature newer than that release.  A server-run
version of the same app was tested, in a Space with the Gradio SDK.

The viewer and its geometry rules are part of mbirtorch, which carries the
BSD 3-clause license.
