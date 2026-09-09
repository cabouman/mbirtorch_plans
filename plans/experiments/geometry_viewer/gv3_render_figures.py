"""Render one geometry figure for each of the six probe geometries.

What this script does.  For every configuration of ``gv1_conventions_probe.py``
it builds the model, builds a ``GeometryScene`` from it, draws a
``GeometryFigure``, and writes a PNG under ``figures/``.  Two extra figures are
written as well: the helical cone scan with the source's path over all views,
and the flat cone geometry at view 0, whose view angle is negative.

Why these six.  They are the geometries Increment 1 confirmed against the
projector: parallel, flat cone, curved cone, helical cone, multiaxis, and
translation.  Each carries nonzero detector offsets of both signs, unequal
detector pitches, and a voxel row aspect that is not one, so a sign error in
the drawing shows up as a marker in the wrong corner.

There are no command-line arguments.  The parameters are the constants below.

Run:
    cd plans/experiments/geometry_viewer
    MPLBACKEND=Agg PYTHONPATH=<mbirtorch clone> python gv3_render_figures.py

The run takes a few seconds and prints the path of each figure it wrote.
"""

import os
import sys

import matplotlib

# The figures are written to files, so no window is needed.  This has to happen
# before pyplot resolves a backend, which the viewer defers to its first
# figure.
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gv1_conventions_probe as probe  # noqa: E402
from geometry_scene import GeometryScene  # noqa: E402
from geometry_viewer import GeometryFigure  # noqa: E402

# ── run parameters ───────────────────────────────────────────────────────────

#: Where the figures go, relative to this file.
FIGURE_DIRECTORY = 'figures'

#: The view drawn for each of the six geometries.  View 2 has a nonzero angle
#: of 0.21 radians, so the source is off the +y axis and a rotation error is
#: visible.
VIEW_INDEX = 2

#: The view drawn for the extra flat cone figure.  View 0 has a negative angle.
EXTRA_CONE_VIEW_INDEX = 0

#: Dots per inch of the saved PNG files.
DPI = 110

#: The figure size in inches, passed to every figure.
FIGSIZE = (15.0, 9.0)


def figure_path(name):
    """The output path of the figure named ``name``.

    Spaces in a configuration's name become underscores, so that
    ``'cone flat'`` is written as ``gv3_cone_flat.png``.
    """
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             FIGURE_DIRECTORY)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f'gv3_{name.replace(" ", "_")}.png')


def render(cfg, view_index, show_trajectory, name):
    """Build the model, the scene, and the figure of one configuration.

    Args:
        cfg (dict): one entry of ``gv1_conventions_probe.CONFIGS``.
        view_index (int): the view to draw.
        show_trajectory (bool): whether to draw the source's path.
        name (str): the figure's name, used for its file name.

    Returns:
        str: the path written.
    """
    model = probe.build_model(cfg)
    scene = GeometryScene.from_model(model)
    figure = GeometryFigure(scene, view_index=view_index,
                            show_trajectory=show_trajectory, figsize=FIGSIZE)
    path = figure.save(figure_path(name), dpi=DPI)
    figure.figure.clf()
    import matplotlib.pyplot as plt
    plt.close(figure.figure)
    return path


def main():
    """Render the six figures and the two extra ones."""
    written = []
    for cfg in probe.CONFIGS:
        written.append(render(cfg, VIEW_INDEX, False, cfg['name']))

    configs_by_name = {cfg['name']: cfg for cfg in probe.CONFIGS}
    # The helical scan's source rises with the view index, which only a drawing
    # of every view shows.
    written.append(render(configs_by_name['cone helical'], VIEW_INDEX, True,
                          'cone helical trajectory'))
    # View 0 of the flat cone geometry has a negative view angle, which puts
    # the source on the other side of the +y axis.
    written.append(render(configs_by_name['cone flat'],
                          EXTRA_CONE_VIEW_INDEX, False, 'cone flat view0'))

    for path in written:
        print(path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
