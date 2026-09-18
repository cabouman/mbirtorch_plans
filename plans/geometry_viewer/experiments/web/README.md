---
title: mbirtorch geometry viewer
emoji: 🩻
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: bsd-3-clause
short_description: "Draw a CT scan geometry: source, detector, volume, offsets."
---

# mbirtorch geometry viewer

This Space draws the scan geometry of an mbirtorch tomography model: where the
source and the detector sit, where the reconstruction volume sits, which way
the gantry turns, and whether the volume projects inside the detector.  Six
geometries are offered: a cone beam scan with a flat detector, the same scan
with a curved detector, a helical cone beam scan, a parallel beam scan, a
multiaxis parallel scan, and a translation scan.  Nothing is reconstructed and
no data is read.  The app draws parameters.

The figure has five panels: a 3D view, a top view of the xy plane, a side view
of the yz plane, the detector face in row and channel index, and a panel of
derived numbers.  A slider steps through the views.  Three checkboxes turn the
source's path over all views, the 3D zoom to the volume, and the angle-0
reference on and off.  Two more draw a cube phantom, which the app builds for
the current scan, and that phantom's sinogram, which is mbirtorch's own forward
projection, computed when this app was built, stored at 8 bits, and kept for
the default scan of each geometry only.  A comparison section draws a second
geometry over the first from a few parameter overrides, which is the
calibration use: a vendor geometry against the same geometry with an estimated
offset.

Where the files come from.  Every file in this directory is a copy, made by
``gv5_build_web.py`` in the ``mbirtorch_plans`` repository, under
``plans/geometry_viewer/experiments``.  The sources live one directory up from
the app: ``app.py``, ``requirements.txt``, and this README come from
``web/``, and ``geometry_defaults.py``, ``geometry_scene.py``, and
``geometry_viewer.py`` come from the plan directory that holds ``web/``.  Edit
them there and run the build script again; an edit made here is overwritten.
``default_sinograms.b64`` holds the six stored sinograms and is written by the
same build script, which needs mbirtorch and torch to run the projector.

What each module does.  ``geometry_defaults.py`` computes the reconstruction
shape and voxel pitch that an mbirtorch model constructor would choose, so the
app needs neither mbirtorch nor torch.  ``geometry_scene.py`` turns the
parameters into drawable primitives in plain numpy.  ``geometry_viewer.py``
draws them with matplotlib.  The tests of all three, including the test that
pins the defaults against the real mbirtorch models, live in the plan
directory.

To run it outside a Space:

    pip install -r requirements.txt
    python app.py

The viewer and its geometry rules are part of mbirtorch, which carries the
BSD 3-clause license.
