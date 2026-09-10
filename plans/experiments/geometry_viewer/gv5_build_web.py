"""Assemble the two web packagings of the geometry viewer.

What this script writes.  It takes the Python of the viewer and the web app and
produces two directories, each one a Hugging Face Space that can be uploaded as
it stands:

``web/lite/`` is a static Space.  ``index.html`` carries the whole app: the
Gradio-Lite script and stylesheet, pinned to one release, and one
``<gradio-file>`` element per Python module with the source HTML-escaped inside
it.  Gradio-Lite reads the element's text content, so the entities decode back
to the source.  The page runs the app in the browser under Pyodide, with no
server, which is the packaging a free Hugging Face account can host.

``web/space/`` is a Space with the Gradio SDK.  It holds a plain copy of every
Python file, the requirements, and the README.  A server runs the app there,
which needs a paid Hugging Face plan for a private Space and is the alternative
packaging.

Both directories also get ``icon.png``, the 400 by 400 pixel tile the site
shows.  :func:`write_icon` draws it from a cone geometry's own scene, so the
tile is a picture of what the app draws and not an illustration.

There are no command-line arguments.  The inputs are the files named in
:data:`MODULE_FILES` and :data:`WEB_FILES`, and every output is overwritten.

Run:
    cd plans/experiments/geometry_viewer
    MPLBACKEND=Agg python gv5_build_web.py

The script prints one line per file it wrote.  ``test_web_app.py`` runs it and
checks that every Python file inside ``index.html`` decodes back to its source
byte for byte.
"""

import os
import shutil
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')   # the icon is written to a file; no window is opened

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import geometry_defaults  # noqa: E402
from geometry_scene import GeometryScene  # noqa: E402
from geometry_viewer import COLORS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, 'web')
SPACE = os.path.join(WEB, 'space')
LITE = os.path.join(WEB, 'lite')

#: The Python modules the app needs, in the order the page lists them.  The
#: entry point comes first.
MODULE_FILES = ('app.py', 'geometry_defaults.py', 'geometry_scene.py',
                'geometry_viewer.py')

#: Where each module is read from.  The app lives under ``web/`` and the three
#: viewer modules live in the plan directory above it.
SOURCE_OF_MODULE = dict(
    (('app.py', os.path.join(WEB, 'app.py')),
     ('geometry_defaults.py', os.path.join(HERE, 'geometry_defaults.py')),
     ('geometry_scene.py', os.path.join(HERE, 'geometry_scene.py')),
     ('geometry_viewer.py', os.path.join(HERE, 'geometry_viewer.py'))))

#: The files copied into the Gradio Space beside the modules.
WEB_FILES = ('requirements.txt', 'README.md')

#: The Gradio-Lite release the static page is pinned to, and the two files it
#: loads.  Version 5.45.0 was published on 2025-09-10 and is the last release
#: of the package: there is no Gradio 6 release of Gradio-Lite, and the Lite
#: source has been removed from the Gradio main branch.  So the pin is exact,
#: and the app uses no Gradio feature newer than 5.45.
LITE_VERSION = '5.45.0'
LITE_SCRIPT = f'https://cdn.jsdelivr.net/npm/@gradio/lite@{LITE_VERSION}/dist/lite.js'
LITE_STYLESHEET = f'https://cdn.jsdelivr.net/npm/@gradio/lite@{LITE_VERSION}/dist/lite.css'

#: The packages Gradio-Lite installs into Pyodide before running the app.
#: numpy comes with Pyodide and needs no entry.
LITE_REQUIREMENTS = 'matplotlib'

#: The tile's size in pixels, as the figure size in inches and the resolution
#: that multiply to it.
ICON_PIXELS = 400
ICON_FIGSIZE = (4.0, 4.0)
ICON_DPI = 100

#: The view the tile draws, as a fraction of a full turn.  A quarter turn puts
#: the drawn gantry at a right angle to the angle-0 reference, so the tile
#: shows two positions of one scan and the two do not overlap.
ICON_VIEW_FRACTION = 0.25


# ── the static Space ─────────────────────────────────────────────────────────

def read_source(name):
    """The text of one module, as it sits on disk."""
    with open(SOURCE_OF_MODULE[name], 'r', encoding='utf-8') as handle:
        return handle.read()


def gradio_file_element(name, source, entrypoint=False):
    """One ``<gradio-file>`` element holding one module's source.

    The three characters that end an HTML element or start an entity are
    escaped, and nothing else is changed: no line is re-indented and no line is
    dropped, so the element's text content is the source with one newline in
    front of it.  That newline is what separates the source from the opening
    tag.

    Args:
        name (str): the file's name inside the app's working directory.
        source (str): the file's text.
        entrypoint (bool, optional): whether this is the file Gradio-Lite runs.

    Returns:
        str: the element, ending in a newline.
    """
    escaped = (source.replace('&', '&amp;').replace('<', '&lt;')
               .replace('>', '&gt;'))
    attribute = ' entrypoint' if entrypoint else ''
    return (f'<gradio-file name="{name}"{attribute}>\n{escaped}'
            f'</gradio-file>\n')


def build_index_html():
    """The text of ``web/lite/index.html``."""
    elements = [gradio_file_element(name, read_source(name),
                                    entrypoint=(name == 'app.py'))
                for name in MODULE_FILES]
    return f'''<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>mbirtorch geometry viewer</title>
    <script type="module" crossorigin src="{LITE_SCRIPT}"></script>
    <link rel="stylesheet" href="{LITE_STYLESHEET}" />
  </head>
  <body>
    <gradio-lite>
<gradio-requirements>{LITE_REQUIREMENTS}</gradio-requirements>
{''.join(elements)}</gradio-lite>
  </body>
</html>
'''


LITE_README = '''---
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
'''


# ── the tile ─────────────────────────────────────────────────────────────────

def icon_scene():
    """The scene the tile is drawn from: a flat cone beam scan.

    The detector is wide and the source is close to it, so the fan angle is
    large and the tile reads as a cone beam rather than as three parallel
    lines.  The volume then fills a good part of the fan.
    """
    shape = (60, 24, 96)
    params = geometry_defaults.default_parameters(
        'cone', shape, angles=geometry_defaults.view_angles(60),
        source_detector_dist=130.0, source_iso_dist=65.0)
    return GeometryScene(params, 'cone')


def write_icon(path):
    """Draw the tile and write it to ``path``.

    What the tile shows.  It is the top view of one view of a cone beam scan:
    the source, the four rays to the detector's corners, the detector, and the
    reconstruction volume, with the angle-0 position of the source and the
    detector behind them as dotted outlines.  The screen convention is the
    viewer's top view: y increases to the left and x increases downward, so the
    source of the angle-0 reference sits on the left.

    The drawing carries no axes, no ticks, and no text, because the tile is
    about 100 pixels wide where the site shows it.  The lines are thick for the
    same reason.
    """
    import matplotlib.pyplot as plt

    scene = icon_scene()
    index = int(round(ICON_VIEW_FRACTION * scene.num_views))
    view = scene.view(index)
    reference = scene.reference_view()

    def screen(points):
        """Object points as (across, down) screen coordinates."""
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return -points[:, 1], -points[:, 0]

    figure = plt.figure(figsize=ICON_FIGSIZE, dpi=ICON_DPI)
    figure.patch.set_facecolor('#fbfbfd')
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.set_facecolor('#fbfbfd')
    axes.set_axis_off()

    # The angle-0 reference, behind everything else: the source, the detector,
    # and the central ray between them, all dotted and faded.  The viewer's
    # own drawing marks the reference the same way.
    across, down = screen(reference.detector_outline)
    axes.plot(across, down, linestyle=':', linewidth=3.0,
              color=COLORS['detector'], alpha=0.45, zorder=1)
    across, down = screen(np.stack([reference.source_draw,
                                    reference.detector_origin]))
    axes.plot(across, down, linestyle=':', linewidth=2.0,
              color=COLORS['central_ray'], alpha=0.35, zorder=1)
    across, down = screen(reference.source_draw)
    axes.plot(across, down, marker='o', markersize=12,
              color=COLORS['source'], alpha=0.45, zorder=1)

    # The volume, as the rectangle its box covers in the xy plane.
    half_x, half_y, _ = scene.volume_half_extents()
    box = np.array([[half_y, half_x], [-half_y, half_x],
                    [-half_y, -half_x], [half_y, -half_x],
                    [half_y, half_x]])
    axes.fill(-box[:, 0], -box[:, 1], color=COLORS['volume'], alpha=0.22,
              zorder=2)
    axes.plot(-box[:, 0], -box[:, 1], linewidth=3.5, color=COLORS['volume'],
              zorder=3)

    # The four rays to the detector's corners, then the central ray.
    for ray in view.corner_rays:
        across, down = screen(ray)
        axes.plot(across, down, linewidth=3.0, color=COLORS['rays'], zorder=4)
    across, down = screen(np.stack([view.source_draw, view.detector_origin]))
    axes.plot(across, down, linewidth=3.0, color=COLORS['central_ray'],
              zorder=5)

    # The detector and the source of the view drawn.
    across, down = screen(view.detector_outline)
    axes.plot(across, down, linewidth=6.0, color=COLORS['detector'], zorder=6)
    across, down = screen(view.source_draw)
    axes.plot(across, down, marker='o', markersize=20,
              color=COLORS['source'], zorder=7)

    # A square window centered on the origin, which is where the volume and
    # the rotation axis are.  Centering there keeps the tile balanced: the
    # source of one view and the source of the reference sit on a circle about
    # that point, so no choice of view pushes the drawing into a corner.
    points = np.vstack([view.corner_rays.reshape(-1, 3),
                        view.detector_outline, reference.detector_outline,
                        np.atleast_2d(reference.source_draw)])
    across, down = screen(points)
    half = 1.07 * max(float(np.max(np.abs(across))),
                      float(np.max(np.abs(down))))
    axes.set_xlim(-half, half)
    axes.set_ylim(-half, half)
    axes.set_aspect('equal')

    figure.savefig(path, dpi=ICON_DPI, facecolor=figure.get_facecolor())
    plt.close(figure)
    return path


# ── assembling both directories ──────────────────────────────────────────────

def build(verbose=True):
    """Write both packagings and return the paths written.

    Returns:
        list of str: every file written, in the order it was written.
    """
    written = []

    os.makedirs(SPACE, exist_ok=True)
    for name in MODULE_FILES:
        target = os.path.join(SPACE, name)
        shutil.copyfile(SOURCE_OF_MODULE[name], target)
        written.append(target)
    for name in WEB_FILES:
        target = os.path.join(SPACE, name)
        shutil.copyfile(os.path.join(WEB, name), target)
        written.append(target)

    os.makedirs(LITE, exist_ok=True)
    index = os.path.join(LITE, 'index.html')
    with open(index, 'w', encoding='utf-8') as handle:
        handle.write(build_index_html())
    written.append(index)
    lite_readme = os.path.join(LITE, 'README.md')
    with open(lite_readme, 'w', encoding='utf-8') as handle:
        handle.write(LITE_README)
    written.append(lite_readme)

    for directory in (LITE, SPACE):
        written.append(write_icon(os.path.join(directory, 'icon.png')))

    if verbose:
        for path in written:
            print(os.path.relpath(path, HERE))
    return written


if __name__ == '__main__':
    build()
