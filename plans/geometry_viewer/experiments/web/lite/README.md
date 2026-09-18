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

Two check boxes add data to the drawing.  One draws a cube phantom, which the
page builds for the reconstruction shape of the current scan.  The other draws
that phantom's sinogram on the detector face.  The sinogram is mbirtorch's own
forward projection, computed when this page was built, stored at 8 bits, and
kept for the default scan of each geometry; a scan whose controls have been
changed is drawn without one, and the status block says so.

The files.  ``index.html`` carries four Python modules inside it:
``app.py`` builds the page, ``geometry_defaults.py`` computes the automatic
reconstruction geometry that an mbirtorch model constructor would choose,
``geometry_scene.py`` turns the parameters into drawable primitives, and
``geometry_viewer.py`` draws them with matplotlib.  It carries the six stored
sinograms as well, in ``default_sinograms.b64``.  All of these are copies made
by ``gv5_build_web.py`` in the ``mbirtorch_plans`` repository, under
``plans/experiments/geometry_viewer``; that directory holds the sources and the
tests.

The page holds the runtime's dependencies at fixed versions.  The script and
the stylesheet are pinned to @gradio/lite 5.45.0, the last release of that
package, and the app uses no Gradio feature newer than that release.  The
runtime itself installs gradio's dependencies from PyPI at whatever versions
are current when the page loads, and by 2026-09-10 three of them had moved on,
each stopping the page before the app started.  A short script in
``index.html`` therefore patches the runtime's worker as it loads, so that
gradio's dependencies are installed at their newest releases before 2025-09-10,
the day the runtime was published.  ``lite_pins.txt`` in the ``web`` directory
of the source repository lists them.

The viewer and its geometry rules are part of mbirtorch, which carries the
BSD 3-clause license.
