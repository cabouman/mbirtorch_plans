"""Render the three figures of Increment 4: the comparison, the zoom, the helix.

What this script does.  It writes three PNG files under ``figures/``, one for
each thing Increment 4 added that a picture can show.

- ``gv4_compare_offset.png``: the flat cone geometry at view 2, with a
  comparison whose ``det_channel_offset`` is ten channels larger.  The
  comparison's projected volume outline on the detector face sits ten channels
  along the channel axis from the primary's, and the text panel lists the
  changed parameter with both values.
- ``gv4_zoom_volume.png``: the same geometry with the 3D panel zoomed to the
  volume, which is the control the Increment 3 review asked for.  Drawn to the
  scale of the whole scan the volume is about twenty pixels wide.
- ``gv4_helical_1800_trajectory.png``: the 1800-view helical scan of
  ``gv4_timing.py`` with the source path on.  The path is one polyline per
  panel, and 1800 views of it draw as a smooth helix.

There are no command-line arguments.  The parameters are the constants below
and, for the helical scan, the constants of ``gv4_timing.py``.

Run:
    cd plans/geometry_viewer/experiments
    MPLBACKEND=Agg PYTHONPATH=<mbirtorch clone> python gv4_render_figures.py

The run takes about half a minute and prints the path of each figure it wrote,
with the numbers the review of that figure quotes.
"""

import os
import sys

import numpy as np
import matplotlib

# The figures are written to files, so no window is needed.  This has to
# happen before pyplot resolves a backend, which the viewer defers to its
# first figure.
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gv1_conventions_probe as probe  # noqa: E402
import gv4_timing  # noqa: E402
from geometry_scene import GeometryScene  # noqa: E402
from geometry_viewer import GeometryFigure  # noqa: E402

# ── run parameters ───────────────────────────────────────────────────────────

#: Where the figures go, relative to this file.
FIGURE_DIRECTORY = 'figures'

#: The geometry the first two figures use, by name in
#: ``gv1_conventions_probe.CONFIGS``, and the view drawn.  View 2 has a nonzero
#: view angle, so the source is off the +y axis.
BASE_CONFIG = 'cone flat'
VIEW_INDEX = 2

#: How many channels the comparison's detector offset is moved by.  The plan's
#: gate names ten channels.
COMPARISON_CHANNEL_SHIFT = 10

#: The view drawn of the 1800-view helical scan.  A quarter of the way through
#: puts the source a quarter turn around and a quarter of the way up.
HELICAL_VIEW_FRACTION = 0.25

#: Dots per inch of the saved PNG files, and the figure size in inches.
DPI = 110
FIGSIZE = (15.0, 9.0)


def figure_path(name):
    """The output path of the figure named ``name``."""
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             FIGURE_DIRECTORY)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f'gv4_{name}.png')


def base_scene():
    """The scene of the flat cone geometry the first two figures use."""
    cfg = {config['name']: config for config in probe.CONFIGS}[BASE_CONFIG]
    return GeometryScene.from_model(probe.build_model(cfg))


def save_and_close(figure, name):
    """Write one figure and close it, returning the path written.

    A figure with a comparison also owns a comparison window, which ``save``
    writes beside the main file and which is closed here as well.
    """
    import matplotlib.pyplot as plt
    path = figure.save(figure_path(name), dpi=DPI)
    plt.close(figure.figure)
    if figure.compare_figure is not None:
        plt.close(figure.compare_figure)
    return path


def render_comparison():
    """The comparison figure, and the numbers its review quotes."""
    scene = base_scene()
    shift = COMPARISON_CHANNEL_SHIFT * scene.delta_det_channel
    overrides = dict(det_channel_offset=scene.det_channel_offset + shift)
    figure = GeometryFigure(scene, view_index=VIEW_INDEX, figsize=FIGSIZE,
                            compare=overrides)

    primary = scene.view(VIEW_INDEX).volume_outline_on_detector
    comparison = figure.compare_scene.view(
        VIEW_INDEX).volume_outline_on_detector
    channel_shift = float(np.max(np.abs(comparison[:, 1] - primary[:, 1])))
    print(f'  det_channel_offset {scene.det_channel_offset:.2f} -> '
          f'{overrides["det_channel_offset"]:.2f} ALU, which is '
          f'{COMPARISON_CHANNEL_SHIFT} channels of '
          f'{scene.delta_det_channel:.2f} ALU')
    print(f'  the projected volume outline moves {channel_shift:.2f} channels')
    return save_and_close(figure, 'compare_offset')


def render_zoom():
    """The zoomed figure, and the two cube widths its review quotes."""
    scene = base_scene()
    figure = GeometryFigure(scene, view_index=VIEW_INDEX, figsize=FIGSIZE)
    scan_width = float(np.ptp(figure.ax_3d.get_xlim()))
    figure.set_zoom('volume')
    volume_width = float(np.ptp(figure.ax_3d.get_xlim()))
    corners = scene.volume_corners()
    extent = float(np.max(corners.max(axis=0) - corners.min(axis=0)))
    print(f'  the 3D cube is {volume_width:.1f} ALU wide zoomed to the '
          f'volume and {scan_width:.1f} ALU wide over the scan')
    print(f'  the volume box is {extent:.1f} ALU across')
    return save_and_close(figure, 'zoom_volume')


def render_helical():
    """The 1800-view helical figure, and the numbers its review quotes."""
    scene = GeometryScene.from_model(gv4_timing.build_model())
    view_index = int(HELICAL_VIEW_FRACTION * scene.num_views)
    figure = GeometryFigure(scene, view_index=view_index, figsize=FIGSIZE,
                            show_trajectory=True)
    sources, _ = scene.trajectory()
    print(f'  {scene.num_views} views, source path from z = '
          f'{sources[0, 2]:.1f} to z = {sources[-1, 2]:.1f} ALU')
    print(f'  drawn at view {view_index}, angle '
          f'{scene.angles[view_index]:.2f} rad')
    return save_and_close(figure, 'helical_1800_trajectory')


def main():
    """Render the three figures and print what each one shows."""
    written = []
    for label, render in (('comparison, ten channels of offset',
                           render_comparison),
                          ('3D zoom to the volume', render_zoom),
                          ('1800-view helical scan with the source path',
                           render_helical)):
        print(label + ':')
        written.append(render())
    print('')
    for path in written:
        print(path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
