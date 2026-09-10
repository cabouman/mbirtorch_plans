"""Headless tests for geometry_viewer.py.

The tests cover four things.  The first is the import discipline: importing
the viewer must not import ``matplotlib.pyplot``, which is checked in a
separate interpreter because this one has pyplot loaded already.  The second is
that a figure builds, redraws, and saves for each of the six scan geometries of
``gv1_conventions_probe.py``.  The third is that the detector-face panel draws
the scene's own projected volume outline and not a recomputed one.  The fourth
is the display convention of 2026-09-10: every panel draws negative z at the
top, the top and side views draw y to the left, and the detector face puts row
0 at the top.  Those tests read display coordinates through ``ax.transData``,
so they check where a point is drawn on the screen and not only what the axes
limits say.

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
import geometry_viewer  # noqa: E402
from geometry_viewer import (COLORS, GeometryFigure,  # noqa: E402
                             VOLUME_BOX_EDGES, TOP_PANEL_COLUMNS,
                             SIDE_PANEL_COLUMNS)

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIGS_BY_NAME = {cfg['name']: cfg for cfg in probe.CONFIGS}

#: The five panels a figure builds: the 3D view, the top view, the side view,
#: the detector face, and the text panel.
EXPECTED_PANEL_COUNT = 5

#: The four widget axes a figure builds beside the panels: the view slider and
#: the three toggles.  The figure's axes list holds these as well as the
#: panels.
EXPECTED_WIDGET_AXES = 4

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


# ── the display convention: negative z is up ─────────────────────────────────

def display_point(axes, horizontal, vertical):
    """Where a 2D panel draws one point, in display coordinates.

    Display coordinates run from the bottom left of the figure, so a larger
    second coordinate is higher on the screen.

    Args:
        axes: the panel.
        horizontal, vertical (float): the point, in the panel's own two
            coordinates.

    Returns:
        ndarray: the display position, (2,).
    """
    return np.asarray(axes.transData.transform((horizontal, vertical)),
                      dtype=np.float64)


def display_point_3d(axes, point):
    """Where the 3D panel draws one object-frame point, in display coordinates.

    The 3D panel projects a point with its own camera matrix, which the first
    draw builds, so the caller must have drawn the figure.
    """
    from mpl_toolkits.mplot3d import proj3d
    flat = proj3d.proj_transform(float(point[0]), float(point[1]),
                                 float(point[2]), axes.M)
    return np.asarray(axes.transData.transform(flat[:2]), dtype=np.float64)


def test_the_panels_invert_the_axes_the_convention_turns_around():
    """Each panel's limits come in the order its screen orientation asks for.

    The top view draws y to the left and x downward, the side view draws y to
    the left and z downward, and the detector face draws the channel index to
    the right and the row index downward.  An axis that increases to the left
    or downward holds its limits in decreasing order.
    """
    _, figure = build_figure('cone flat', view_index=2)
    try:
        top = figure.ax_top
        assert top.get_xlim()[0] > top.get_xlim()[1], 'y must run left'
        assert top.get_ylim()[0] > top.get_ylim()[1], 'x must run down'
        side = figure.ax_side
        assert side.get_xlim()[0] > side.get_xlim()[1], 'y must run left'
        assert side.get_ylim()[0] > side.get_ylim()[1], 'z must run down'
        detector = figure.ax_detector
        assert detector.get_xlim()[0] < detector.get_xlim()[1]
        assert detector.get_ylim()[0] > detector.get_ylim()[1], 'row 0 on top'
    finally:
        close(figure)


def test_larger_z_is_drawn_lower_on_the_screen():
    """A point at larger z is drawn lower in the side view and the 3D view.

    The check is in display coordinates, so it is the drawn position and not
    the axes limits.  The 3D panel is checked in both zoom states, because the
    zoom replaces that panel's limits.
    """
    _, figure = build_figure('cone flat', view_index=2)
    try:
        low = display_point(figure.ax_side, 0.0, -10.0)
        high = display_point(figure.ax_side, 0.0, 10.0)
        assert high[1] < low[1], 'z must increase downward in the side view'

        figure.figure.canvas.draw()
        for zoom in ('scan', 'volume'):
            figure.set_zoom(zoom)
            figure.figure.canvas.draw()
            low = display_point_3d(figure.ax_3d, (0.0, 0.0, -3.0))
            high = display_point_3d(figure.ax_3d, (0.0, 0.0, 3.0))
            assert high[1] < low[1], f'z must increase downward, zoom {zoom}'
    finally:
        close(figure)


def test_row_0_is_at_the_top_of_the_detector_face():
    """A larger row index is drawn lower on the detector face.

    This is the view from the source toward the detector with -z up, and it is
    how ``imshow`` shows one view of a sinogram.
    """
    scene, figure = build_figure('cone flat', view_index=2)
    try:
        first = display_point(figure.ax_detector, 0.0, 0.0)
        last = display_point(figure.ax_detector, 0.0,
                             scene.num_det_rows - 1.0)
        assert last[1] < first[1], 'row 0 must be at the top'
        right = display_point(figure.ax_detector,
                              scene.num_det_channels - 1.0, 0.0)
        assert right[0] > first[0], 'the channel index must run right'
        assert 'row 0 at the top' in figure.ax_detector.get_title()
    finally:
        close(figure)


def test_larger_y_is_drawn_further_left():
    """A point at larger y is drawn further left in the top and side views.

    The source of a view at angle 0 sits at positive y, so this is what puts
    the source on the left of both panels and the detector on the right, as in
    the group's reference slide.
    """
    _, figure = build_figure('cone flat', view_index=2)
    try:
        for axes in (figure.ax_top, figure.ax_side):
            near = display_point(axes, 50.0, 0.0)
            far = display_point(axes, -50.0, 0.0)
            assert near[0] < far[0], 'y must increase to the left'
        # The top view's other axis is x, which increases downward.
        above = display_point(figure.ax_top, 0.0, -50.0)
        below = display_point(figure.ax_top, 0.0, 50.0)
        assert below[1] < above[1], 'x must increase downward'
    finally:
        close(figure)


def test_the_source_is_drawn_left_of_the_detector():
    """The drawn source sits left of the point where the central ray lands.

    The two panels that carry the beam draw it from left to right, which is
    the orientation of the reference slide.
    """
    scene, figure = build_figure('cone flat', view_index=2)
    try:
        view = scene.view(2)
        for axes, columns in ((figure.ax_top, list(TOP_PANEL_COLUMNS)),
                              (figure.ax_side, list(SIDE_PANEL_COLUMNS))):
            source = display_point(axes, *view.source_draw[columns])
            iso = display_point(axes, *view.detector_origin[columns])
            assert source[0] < iso[0]
    finally:
        close(figure)


def test_the_titles_name_the_side_each_panel_is_seen_from():
    """Each panel's title states where it is seen from and which way y runs."""
    _, figure = build_figure('cone flat', view_index=2)
    try:
        top = figure.ax_top.get_title()
        assert 'seen from -z' in top and 'y increases to the left' in top
        assert 'seen from +x' in figure.ax_side.get_title()
        detector = figure.ax_detector.get_title()
        assert 'seen from the source' in detector
        assert 'row 0 at the top' in detector
    finally:
        close(figure)


def test_the_text_panel_names_the_convention_and_the_detector_iso():
    """The text panel says which way z is drawn and what (du, dv) measures."""
    _, figure = build_figure('cone flat', view_index=2)
    try:
        body = '\n'.join(artist.get_text() for artist in figure.ax_text.texts)
        assert 'Drawn with -z up.' in body
        assert '(du, dv) from detector iso to detector center.' in body
    finally:
        close(figure)


def test_the_detector_face_names_its_three_markers():
    """The detector face's legend uses the reference slide's names."""
    _, figure = build_figure('cone flat', view_index=2)
    try:
        labels = [text.get_text()
                  for text in figure.ax_detector.get_legend().get_texts()]
        assert 'detector iso' in labels
        assert 'detector center' in labels
        assert 'pixel (0,0) = sino[v, 0, 0]' in labels
    finally:
        close(figure)


def test_the_source_travels_counterclockwise_in_the_top_view():
    """The source's travel arc turns counterclockwise on the screen.

    The object turns counterclockwise about z seen from +z, so in a drawing
    that holds the object fixed the source turns the other way, and the top
    view sees that from -z.  The two reversals cancel, so the source's arc
    reads counterclockwise on the screen and the object's own rotation reads
    clockwise.  The test measures the arc's signed area about the rotation
    axis in display coordinates, which is positive for a counterclockwise
    turn.
    """
    scene, figure = build_figure('cone flat', view_index=2)
    try:
        arc = scene.view(2).rotation_direction_arc
        columns = list(TOP_PANEL_COLUMNS)
        drawn = np.stack([display_point(figure.ax_top, *point[columns])
                          for point in arc])
        axis = display_point(figure.ax_top, 0.0, 0.0)
        spokes = drawn - axis[None, :]
        area = float(np.sum(spokes[:-1, 0] * spokes[1:, 1]
                            - spokes[1:, 0] * spokes[:-1, 1]))
        assert area > 0.0, 'the source travel must read counterclockwise'
    finally:
        close(figure)


def test_setting_z_up_sign_to_one_restores_the_old_presentation(monkeypatch):
    """Z_UP_SIGN = +1 inverts no axis and draws +z at the top.

    The constant is the one place each panel reads, so this checks that the
    display convention is a presentation choice and not something the drawing
    carries in its geometry.
    """
    monkeypatch.setattr(geometry_viewer, 'Z_UP_SIGN', 1)
    _, figure = build_figure('cone flat', view_index=2)
    try:
        for axes in (figure.ax_top, figure.ax_side, figure.ax_detector):
            assert axes.get_xlim()[0] < axes.get_xlim()[1]
            assert axes.get_ylim()[0] < axes.get_ylim()[1]
        low = display_point(figure.ax_side, 0.0, -10.0)
        high = display_point(figure.ax_side, 0.0, 10.0)
        assert high[1] > low[1], 'z must increase upward'
        first = display_point(figure.ax_detector, 0.0, 0.0)
        last = display_point(figure.ax_detector, 0.0, 20.0)
        assert last[1] > first[1], 'row 0 must be at the bottom'
        figure.figure.canvas.draw()
        for zoom in ('scan', 'volume'):
            figure.set_zoom(zoom)
            figure.figure.canvas.draw()
            low = display_point_3d(figure.ax_3d, (0.0, 0.0, -3.0))
            high = display_point_3d(figure.ax_3d, (0.0, 0.0, 3.0))
            assert high[1] > low[1], f'z must increase upward, zoom {zoom}'
        assert 'row 0 at the bottom' in figure.ax_detector.get_title()
        assert 'y increases to the right' in figure.ax_top.get_title()
        body = '\n'.join(artist.get_text() for artist in figure.ax_text.texts)
        assert 'Drawn with +z up.' in body
    finally:
        close(figure)


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_no_label_overlaps_another_or_leaves_its_panel(name):
    """Every label of the three 2D panels has its own place.

    The inversions of 2026-09-10 moved the source to the left of each panel
    and the detector to the right, which moved every label with them.  This
    measures the drawn size of each label with the renderer and checks that no
    two labels of one panel overlap and that none reaches outside its panel.
    The 3D panel is left out: its text artists are placed by the camera, and
    ``_clip_3d_artists`` hides the ones whose point leaves the panel.
    """
    _, figure = build_figure(name, view_index=2)
    try:
        canvas = figure.figure.canvas
        canvas.draw()
        renderer = canvas.get_renderer()
        for axes in (figure.ax_top, figure.ax_side, figure.ax_detector):
            panel = axes.get_window_extent(renderer)
            boxes = [(text.get_text(), text.get_window_extent(renderer))
                     for text in axes.texts
                     if text.get_visible() and text.get_text().strip()]
            for text, box in boxes:
                assert box.x0 >= panel.x0 - 2 and box.x1 <= panel.x1 + 2, (
                    f'{text!r} leaves its panel sideways')
                assert box.y0 >= panel.y0 - 2 and box.y1 <= panel.y1 + 2, (
                    f'{text!r} leaves its panel vertically')
            for first in range(len(boxes)):
                for second in range(first + 1, len(boxes)):
                    one, other = boxes[first][1], boxes[second][1]
                    across = min(one.x1, other.x1) - max(one.x0, other.x0)
                    down = min(one.y1, other.y1) - max(one.y0, other.y0)
                    assert across <= 1 or down <= 1, (
                        f'{boxes[first][0]!r} overlaps {boxes[second][0]!r}')
    finally:
        close(figure)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
