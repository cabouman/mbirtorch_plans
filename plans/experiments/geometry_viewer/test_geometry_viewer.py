"""Headless tests for geometry_viewer.py.

The tests cover three things.  The first is the import discipline: importing
the viewer must not import ``matplotlib.pyplot``, which is checked in a
separate interpreter because this one has pyplot loaded already.  The second is
that a figure builds, redraws, and saves for each of the six scan geometries of
``gv1_conventions_probe.py``.  The third is that the detector-face panel draws
the scene's own projected volume outline and not a recomputed one.

Run:
    cd plans/experiments/geometry_viewer
    MPLBACKEND=Agg PYTHONPATH=<mbirtorch clone> python -m pytest -q \
        test_geometry_viewer.py
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')  # tests write files and open no window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gv1_conventions_probe as probe  # noqa: E402
from geometry_scene import GeometryScene  # noqa: E402
from geometry_viewer import (COLORS, GeometryFigure,  # noqa: E402
                             VOLUME_BOX_EDGES)

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIGS_BY_NAME = {cfg['name']: cfg for cfg in probe.CONFIGS}

#: The five panels a figure builds: the 3D view, the top view, the side view,
#: the detector face, and the text panel.
EXPECTED_PANEL_COUNT = 5

#: The three widget axes a figure builds beside the panels: the view slider and
#: the two toggles.  The figure's axes list holds these as well as the panels.
EXPECTED_WIDGET_AXES = 3

#: Smallest acceptable size of a saved figure, in bytes.  A PNG of an empty
#: figure of this size is a few kilobytes, so a file above this holds a
#: drawing.
MIN_PNG_BYTES = 10 * 1024


def build_figure(name, **kwargs):
    """Build the model, the scene, and a figure for one configuration."""
    cfg = CONFIGS_BY_NAME[name]
    model = probe.build_model(cfg)
    scene = GeometryScene.from_model(model)
    return scene, GeometryFigure(scene, **kwargs)


def close(figure):
    """Close a figure's matplotlib window so that the tests do not pile up."""
    import matplotlib.pyplot as plt
    plt.close(figure.figure)


# ── the import discipline ────────────────────────────────────────────────────

def test_import_does_not_load_pyplot():
    """Importing the viewer must not import pyplot.

    The check runs in a separate interpreter, because this test process has
    already imported pyplot itself.  A module that imported pyplot at import
    time would resolve a matplotlib backend, and on a machine with no display
    that can fail or open a window nobody asked for.  ``mbirtorch/viewer.py``
    follows the same rule.
    """
    program = ('import sys; import geometry_viewer; '
               "print('matplotlib.pyplot' in sys.modules); "
               "print('mpl_toolkits.mplot3d' in sys.modules)")
    result = subprocess.run([sys.executable, '-c', program], cwd=HERE,
                            capture_output=True, text=True, check=True)
    assert result.stdout.split() == ['False', 'False'], result.stdout


def test_pyplot_loads_on_first_figure():
    """The first figure built does import pyplot."""
    program = ('import sys; import geometry_viewer; '
               "import matplotlib; matplotlib.use('Agg'); "
               'import gv1_conventions_probe as probe; '
               'from geometry_scene import GeometryScene; '
               "cfg = [c for c in probe.CONFIGS "
               "if c['name'] == 'parallel'][0]; "
               'scene = GeometryScene.from_model(probe.build_model(cfg)); '
               'figure = geometry_viewer.GeometryFigure(scene); '
               "print('matplotlib.pyplot' in sys.modules)")
    environment = dict(os.environ)
    result = subprocess.run([sys.executable, '-c', program], cwd=HERE,
                            capture_output=True, text=True, check=True,
                            env=environment)
    assert result.stdout.split()[-1] == 'True', result.stdout


# ── one figure per geometry ──────────────────────────────────────────────────

@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_figure_builds_with_five_panels(name):
    """A figure builds for every geometry and holds the five panels."""
    _, figure = build_figure(name)
    try:
        assert len(figure.panel_axes) == EXPECTED_PANEL_COUNT
        assert (len(figure.figure.axes)
                == EXPECTED_PANEL_COUNT + EXPECTED_WIDGET_AXES)
        # The first panel is the 3D view, which counts as one axes.
        assert hasattr(figure.ax_3d, 'get_zlim')
        for axes in figure.panel_axes:
            assert axes.figure is figure.figure
            assert axes in figure.figure.axes
    finally:
        close(figure)


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_set_view_and_trajectory_redraw(name):
    """Changing the view and the trajectory toggle redraws without error."""
    _, figure = build_figure(name)
    try:
        figure.set_view(1)
        assert figure.view_index == 1
        figure.set_show_trajectory(True)
        assert figure.show_trajectory is True
        figure.set_view(0)
        figure.set_show_trajectory(False)
        assert (len(figure.figure.axes)
                == EXPECTED_PANEL_COUNT + EXPECTED_WIDGET_AXES)
    finally:
        close(figure)


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_save_writes_a_png(name, tmp_path):
    """Saving writes a PNG file that holds a drawing."""
    _, figure = build_figure(name, view_index=2)
    try:
        path = str(tmp_path / f'{name.replace(" ", "_")}.png')
        figure.save(path, dpi=100)
        assert os.path.exists(path)
        assert os.path.getsize(path) > MIN_PNG_BYTES
    finally:
        close(figure)


def test_view_index_outside_the_range_raises():
    """A view index outside the scan raises rather than drawing something."""
    scene, figure = build_figure('cone flat')
    try:
        with pytest.raises(IndexError):
            figure.set_view(scene.num_views)
        with pytest.raises(IndexError):
            figure.set_view(-1)
    finally:
        close(figure)


def test_figure_accepts_a_model():
    """The figure can be built from a model as well as from a scene."""
    cfg = CONFIGS_BY_NAME['cone flat']
    model = probe.build_model(cfg)
    figure = GeometryFigure.from_model(model, view_index=1)
    try:
        assert figure.scene.kind == 'cone'
        assert figure.view_index == 1
    finally:
        close(figure)


# ── the detector-face panel uses the scene's numbers ─────────────────────────

@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_detector_edges_are_the_scene_outline(name):
    """The projected volume edges are the scene's, corner for corner.

    The panel draws the twelve edges of the volume box in detector index
    coordinates.  Each edge's endpoints must be two entries of the scene's
    ``volume_outline_on_detector`` and nothing else, because a viewer that
    recomputed the projection could disagree with the projector.
    """
    scene, figure = build_figure(name, view_index=3)
    try:
        outline = scene.view(3).volume_outline_on_detector
        drawn = figure.detector_volume_edges
        assert drawn.shape == (12, 2, 2)
        for index, (first, second) in enumerate(VOLUME_BOX_EDGES):
            assert np.array_equal(drawn[index, 0], outline[first])
            assert np.array_equal(drawn[index, 1], outline[second])
    finally:
        close(figure)


def test_detector_panel_line_data_matches_the_scene():
    """Every projected corner appears in the lines the panel actually drew.

    This reads the line data back out of the axes, so it checks the drawing and
    not only the array the figure kept.  The lines are in (channel, row) order,
    because that is the panel's horizontal and vertical axis, while the scene
    reports (row, channel).  The twelve edges are drawn as one polyline with a
    row of NaN between edges, so the non-finite separators are dropped before
    the comparison.
    """
    scene, figure = build_figure('cone curved', view_index=2)
    try:
        outline = scene.view(2).volume_outline_on_detector
        drawn_points = []
        for line in figure.ax_detector.get_lines():
            if line.get_color() != COLORS['volume']:
                continue
            xdata, ydata = line.get_xdata(), line.get_ydata()
            drawn_points.extend(zip(np.asarray(ydata), np.asarray(xdata)))
        drawn_points = np.asarray(drawn_points, dtype=np.float64)
        drawn_points = drawn_points[np.isfinite(drawn_points).all(axis=1)]
        assert drawn_points.size > 0
        for corner in outline:
            distance = np.min(np.hypot(drawn_points[:, 0] - corner[0],
                                       drawn_points[:, 1] - corner[1]))
            assert distance < 1e-9, f'corner {corner} was not drawn'
    finally:
        close(figure)


def test_overshoot_is_drawn_in_the_overshoot_color():
    """A volume too large for the detector gets a red part on the face.

    The volume is enlarged until it projects past the detector's edge.  The
    panel must then hold a line in the overshoot color that carries points, and
    the scene must agree that the volume no longer fits.  The line exists in
    every view and is empty when the volume fits, so the test checks its data
    and not only its presence.
    """
    cfg = CONFIGS_BY_NAME['cone flat']
    model = probe.build_model(cfg)
    scene = GeometryScene.from_model(model)
    # A volume this wide projects well past the detector's channel range.
    scene.params['recon_shape'] = (10, 400, 8)
    wide = GeometryScene(scene.params, scene.kind)
    fits, overshoot = wide.volume_fits_detector()
    assert not fits and overshoot > 0.0

    figure = GeometryFigure(wide, view_index=2)
    try:
        drawn = [line for line in figure.ax_detector.get_lines()
                 if line.get_color() == COLORS['overshoot']
                 and np.isfinite(np.asarray(line.get_xdata(),
                                            dtype=np.float64)).any()]
        assert drawn, 'no line was drawn in the overshoot color'
    finally:
        close(figure)

    # The same panel holds no overshoot points when the volume fits.
    figure = GeometryFigure(scene, view_index=2)
    try:
        for line in figure.ax_detector.get_lines():
            if line.get_color() != COLORS['overshoot']:
                continue
            data = np.asarray(line.get_xdata(), dtype=np.float64)
            assert not np.isfinite(data).any()
    finally:
        close(figure)


# ── the text panel ───────────────────────────────────────────────────────────

@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_text_panel_reports_the_derived_quantities(name):
    """The text panel prints the scene's numbers, to three figures."""
    scene, figure = build_figure(name)
    try:
        texts = [artist.get_text()
                 for artist in figure.ax_text.texts]
        body = '\n'.join(texts)
        quantities = scene.derived_quantities()
        assert quantities['geometry_kind'] in body
        assert quantities['sinogram_shape_text'] in body
        assert quantities['recon_shape_text'] in body
        assert f'{quantities["magnification"]:.3g}' in body
        assert 'volume fits det' in body
        # The drawing note names every position that is a drawing choice, so
        # its first few words must reach the panel.
        first_words = ' '.join(quantities['drawing_note'].split()[:4])
        assert first_words in ' '.join(body.split())
        assert '-0 ' not in body, 'a negative zero reached the panel'
    finally:
        close(figure)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
