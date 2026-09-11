"""Headless tests for geometry_viewer.py.

The tests cover five things.  The first is the import discipline: importing
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

The fifth is the two data overlays of Increment 5: a sinogram painted on the
detector face and a reconstruction drawn as a silhouette in the volume box.
Those tests render the figure and read the pixels back, because where an image
lands on the screen is what they are about.  One of them forward-projects a
phantom with the real projector and compares the painted sinogram with the
projected outline the panel draws over it.

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
                             SOURCE_MARKER_SIZE, VOLUME_BOX_EDGES,
                             TOP_PANEL_COLUMNS, SIDE_PANEL_COLUMNS)

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


def build_scene(name):
    """Build the model and the scene of one configuration.

    A test that has to make an array of the scan's own shape needs the scene
    before it can build the figure, so the two steps are separate.
    """
    return GeometryScene.from_model(probe.build_model(CONFIGS_BY_NAME[name]))


def build_figure(name, **kwargs):
    """Build the model, the scene, and a figure for one configuration."""
    scene = build_scene(name)
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


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_text_panel_reports_the_fit_statement_in_three_rows(name):
    """The fit rows name the shape, count the views, and sweep the axis.

    The count of views is printed for every scan, because it is what separates
    a scan whose volume is in the wrong place from a helical scan whose volume
    leaves the detector in every view by design.  The swept z range is printed
    for a helical scan alone, because only a helical scan is judged by it.
    """
    scene, figure = build_figure(name)
    try:
        body = '\n'.join(artist.get_text() for artist in figure.ax_text.texts)
        quantities = scene.derived_quantities()
        assert 'volume fits det' in body
        assert 'leaves det in views' in body
        assert (f'{quantities["views_leaving_detector"]} of '
                f'{quantities["num_views"]}') in body
        assert quantities['fit_shape'] in ('box', 'cylinder')
        if quantities['helical_fit_rule']:
            assert 'swept z at axis' in body
        else:
            assert 'swept z at axis' not in body
            # A scan judged without the helical rule names the shape tested.
            assert quantities['fit_shape'] in body
    finally:
        close(figure)


def test_the_text_panel_reports_the_swept_z_range_of_a_helical_scan():
    """A helical scan prints the z range its detector sweeps, to three figures.

    The probe's helical configuration is the one with a nonzero travel, so it
    is the one the row appears for.
    """
    scene, figure = build_figure('cone helical')
    try:
        body = '\n'.join(artist.get_text() for artist in figure.ax_text.texts)
        quantities = scene.derived_quantities()
        assert quantities['helical_fit_rule'] is True
        assert 'swept z at axis' in body
        assert (f'{quantities["swept_z_min"]:.3g} to '
                f'{quantities["swept_z_max"]:.3g}') in body
        # This scan's volume does fit under the helical rule.
        assert 'volume fits det     : yes (helical rule)' in body
    finally:
        close(figure)


def test_the_detector_face_draws_the_region_of_reconstruction():
    """The panel draws the shape the fit statement is about, and names it.

    The fit statement tests the region-of-reconstruction cylinder when the mask
    is on, so the detector face draws that cylinder's two projected rims.  A
    scan with no mask has no cylinder and gets no such line.
    """
    scene, figure = build_figure('cone flat', view_index=2)
    try:
        labels = [text.get_text()
                  for text in figure.ax_detector.get_legend().get_texts()]
        assert 'region of reconstruction' in labels
        drawn = [line for line in figure.ax_detector.get_lines()
                 if line.get_label() == 'region of reconstruction']
        assert len(drawn) == 1
        # The line holds both rims, joined by a row of NaN.
        points = np.stack([np.asarray(drawn[0].get_ydata(), dtype=np.float64),
                           np.asarray(drawn[0].get_xdata(),
                                      dtype=np.float64)], axis=1)
        points = points[np.isfinite(points).all(axis=1)]
        outline = scene.view(2).ror_outline_on_detector.reshape(-1, 2)
        for rim_point in outline:
            distance = np.min(np.hypot(points[:, 0] - rim_point[0],
                                       points[:, 1] - rim_point[1]))
            assert distance < 1e-9, f'rim point {rim_point} was not drawn'
    finally:
        close(figure)

    # The translation model turns the mask off, so it has no region to draw.
    _, figure = build_figure('translation', view_index=2)
    try:
        labels = [text.get_text()
                  for text in figure.ax_detector.get_legend().get_texts()]
        assert 'region of reconstruction' not in labels
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


def source_marker_box(figure, axes, artist):
    """The box the source's marker covers in one 2D panel, in display pixels.

    A marker is drawn at a size in points, and matplotlib reports no extent for
    it, so the box is built here: a square of ``SOURCE_MARKER_SIZE`` points
    centered on the marker's data point.  The square is the marker's outer
    bound, because a star of that size is drawn inside it.

    Args:
        figure (GeometryFigure): the figure, for its dots per inch.
        axes: the panel the marker is drawn in.
        artist: the marker's line artist.

    Returns:
        Bbox: the square, or None when the artist carries no point.
    """
    from matplotlib.transforms import Bbox
    x, y = artist.get_xdata(), artist.get_ydata()
    if len(x) == 0:
        return None
    center = axes.transData.transform((float(x[0]), float(y[0])))
    half = 0.5 * SOURCE_MARKER_SIZE * figure.figure.dpi / 72.0
    return Bbox.from_extents(center[0] - half, center[1] - half,
                             center[0] + half, center[1] + half)


def assert_boxes_are_clear(first_name, first_box, second_name, second_box):
    """Fail unless two display boxes miss each other in x or in y.

    An overlap of a pixel is allowed, because a label placed a fixed number of
    points from a marker lands within a pixel of that marker's own box.
    """
    across = (min(first_box.x1, second_box.x1)
              - max(first_box.x0, second_box.x0))
    down = min(first_box.y1, second_box.y1) - max(first_box.y0, second_box.y0)
    assert across <= 1 or down <= 1, (
        f'{first_name!r} overlaps {second_name!r}')


def assert_panel_labels_have_their_own_place(figure, axes, marker_artist,
                                             where):
    """Fail unless every label of one panel has a place of its own.

    Three things are checked: no label reaches outside the panel, no two
    labels overlap, and no label is under the source's marker.

    Args:
        figure (GeometryFigure): the figure being measured.
        axes: the panel.
        marker_artist: the panel's source marker, or None where it draws none.
        where (str): what to name in a failure, such as ``'view 3'``.
    """
    renderer = figure.figure.canvas.get_renderer()
    panel = axes.get_window_extent(renderer)
    boxes = [(text.get_text(), text.get_window_extent(renderer))
             for text in axes.texts
             if text.get_visible() and text.get_text().strip()]
    for text, box in boxes:
        assert box.x0 >= panel.x0 - 2 and box.x1 <= panel.x1 + 2, (
            f'{text!r} leaves its panel sideways, {where}')
        assert box.y0 >= panel.y0 - 2 and box.y1 <= panel.y1 + 2, (
            f'{text!r} leaves its panel vertically, {where}')
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            assert_boxes_are_clear(f'{boxes[first][0]} ({where})',
                                   boxes[first][1],
                                   boxes[second][0], boxes[second][1])
    if marker_artist is None:
        return
    marker = source_marker_box(figure, axes, marker_artist)
    if marker is None:
        return
    for text, box in boxes:
        assert_boxes_are_clear(f'the source marker ({where})', marker,
                               text, box)


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_no_label_overlaps_another_or_leaves_its_panel(name):
    """Every label of the three 2D panels has its own place.

    The inversions of 2026-09-10 moved the source to the left of each panel
    and the detector to the right, which moved every label with them.  This
    measures the drawn size of each label with the renderer and checks that no
    two labels of one panel overlap and that none reaches outside its panel.
    The 3D panel is left out: its text artists are placed by the camera, and
    ``_clip_3d_artists`` hides the ones whose point leaves the panel.

    The source's marker is checked against the labels as well, in the two
    panels that draw it.  A marker is not a label, so it was not measured
    before, and the marker was found drawn over two characters of the
    ``recon_slice_offset`` label in the side view of the multiaxis example
    (``gv4_interaction_findings.md``, "A marker can cover a label").

    The side view is measured at three views and not at one, because its
    ``recon_slice_offset`` label is placed away from the source and so moves
    when the source does.  The multiaxis source rises and falls with the
    per-view elevation and the helical source rises through the scan.  The top
    view and the detector face keep the single view this test has always used.
    Measuring the top view at three views finds label crowding that predates
    the moving label: its channel-offset label runs into the angle-0 caption
    in three geometries, its travel label runs into the source's label in two,
    and in the multiaxis geometry the channel-offset label is long enough to
    reach the source's marker across the panel.  Those are the top view's own
    layout and are left for their own piece of work.
    """
    _, figure = build_figure(name, view_index=2)
    try:
        canvas = figure.figure.canvas
        canvas.draw()
        for axes, marker in ((figure.ax_top, figure._top['source']),
                             (figure.ax_side, figure._side['source']),
                             (figure.ax_detector, None)):
            assert_panel_labels_have_their_own_place(figure, axes, marker,
                                                     'view 2')
        num_views = figure.scene.num_views
        for view_index in (0, num_views // 2, num_views - 1):
            figure.set_view(view_index)
            canvas.draw()
            assert_panel_labels_have_their_own_place(
                figure, figure.ax_side, figure._side['source'],
                f'side view, view {view_index}')
    finally:
        close(figure)


# ── the data overlays ────────────────────────────────────────────────────────

#: The configuration the sinogram tests paint on, and the view, the row, and
#: the channel of the one bright pixel they paint.  The pixel sits away from
#: the middle of the detector in both directions, so a placement that mirrored
#: or transposed the array could not land on it by accident.
SINOGRAM_CONFIG = 'cone flat'
BRIGHT_VIEW = 5
BRIGHT_ROW = 7
BRIGHT_CHANNEL = 30

#: How close to the brightest pixel a pixel must be, in summed red, green, and
#: blue out of 765, to count as part of the bright block.  The block is drawn
#: in one flat color, so the tolerance only absorbs rounding.
BRIGHTNESS_TOLERANCE = 6

#: A pixel of a projected phantom counts as lit above this fraction of the
#: sinogram's largest value.  The bound separates a pixel the projector gave
#: some mass from a pixel it left at zero, so any small fraction serves.
LIT_FRACTION = 1e-3

#: How far the lit part of a painted sinogram may sit from the projected rims
#: of the region of reconstruction, in detector pixels.  The two numbers
#: differ, and ``test_the_painted_sinogram_lands_inside_the_projected_rims``
#: says why.
RIM_ROW_TOLERANCE = 1.0
RIM_CHANNEL_TOLERANCE = 2.0

#: The configuration the silhouette tests draw.  Its reconstruction has a
#: nonzero ``recon_slice_offset`` and three different voxel pitches, so a
#: drawing that ignored any of them would put the patch somewhere else.  It is
#: also a parallel-type geometry, whose panels span a few volume widths instead
#: of a whole source-detector distance, so a patch of a few voxels covers
#: several pixels on the screen.
SILHOUETTE_CONFIG = 'multiaxis'

#: How many voxels on a side the silhouette tests fill at a corner of the
#: volume.  One voxel is not enough to measure.  A corner voxel is drawn under
#: the volume box's own outline and, at voxel (0, 0, 0), under the marker that
#: names that voxel; in this configuration those two artists leave one pixel of
#: a single voxel showing in the top view, and in a cone geometry, whose panels
#: span the whole scan, one voxel is not a pixel wide to begin with.
SILHOUETTE_BLOCK = 2

#: How far apart two renderings of one pixel must be, in red, green, or blue,
#: for that pixel to count as changed.  Only rounding separates two renderings
#: of the same picture, so anything above a few counts is the overlay.
PIXEL_CHANGE_FLOOR = 4


def bright_sinogram(scene):
    """A sinogram of zeros with one bright pixel, for the placement tests."""
    values = np.zeros(scene.sinogram_shape, dtype=np.float32)
    values[BRIGHT_VIEW, BRIGHT_ROW, BRIGHT_CHANNEL] = 1.0
    return values


def rendered_rgb(figure):
    """The figure drawn, as an array of red, green, and blue values.

    Returns:
        ndarray: (height, width, 3) of int16.  Its first row is the top of the
        figure, which is the opposite of the display coordinates the axes use.
    """
    canvas = figure.figure.canvas
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].astype(np.int16)


def box_mask(shape, box):
    """The rendered pixels inside one display box, as a boolean array.

    The rendered array's first row is the top of the figure while a display box
    measures its height from the bottom, so the rows are flipped here.  A box
    whose limits come from an inverted axis holds them in decreasing order, so
    each pair is read as its smaller and its larger value.

    Args:
        shape (tuple): the rendered array's shape.
        box (Bbox): the display box.

    Returns:
        ndarray: (height, width) of bool.
    """
    height, width = shape[:2]
    horizontal = np.arange(width)[None, :] + 0.5
    vertical = height - np.arange(height)[:, None] - 0.5
    inside_x = ((horizontal >= min(box.x0, box.x1))
                & (horizontal <= max(box.x0, box.x1)))
    inside_y = ((vertical >= min(box.y0, box.y1))
                & (vertical <= max(box.y0, box.y1)))
    return inside_x & inside_y


def pixels_to_data(axes, height, rows, columns):
    """Where rendered pixels sit in one panel's own two coordinates.

    Half a pixel is added in each direction, so the position is the center of
    the pixel and not its corner.

    Args:
        axes: the panel.
        height (int): the rendered array's height.
        rows, columns (ndarray): the pixels' indices into that array.

    Returns:
        ndarray: (N, 2), each row one pixel's position in data coordinates.
    """
    display = np.stack([columns + 0.5, height - rows - 0.5], axis=1)
    return np.asarray(axes.transData.inverted().transform(display))


def changed_in_panel(figure, axes, before, after):
    """Where two renderings of the figure differ inside one panel.

    Returns:
        (ndarray, ndarray): the rows and the columns of the pixels that differ.
    """
    differs = np.abs(after - before).max(axis=2) > PIXEL_CHANGE_FLOOR
    window = axes.get_window_extent(figure.figure.canvas.get_renderer())
    inside = differs & box_mask(differs.shape, window)
    return np.nonzero(inside)


def test_the_sinogram_image_is_placed_and_updated_in_place():
    """The painted sinogram is the array, in the panel's own index axes.

    Three things are checked.  The image's extent puts array element (r, c) at
    data coordinates channel c and row r, which is what makes the panel's
    inverted row axis draw row 0 at the top.  A view change replaces the data,
    so after ``set_view`` the image holds the bright pixel at the row and
    channel it was painted at.  And the color scale is the whole array's, so it
    does not move from one view to the next.
    """
    scene = build_scene(SINOGRAM_CONFIG)
    values = bright_sinogram(scene)
    figure = GeometryFigure(scene, view_index=1, sinogram=values)
    try:
        image = figure._sinogram_image
        assert image.origin == 'upper'
        assert list(image.get_extent()) == [-0.5,
                                            scene.num_det_channels - 0.5,
                                            scene.num_det_rows - 0.5, -0.5]
        # The view built is not the bright one, so this view is all zeros.
        assert float(np.max(image.get_array())) == 0.0

        figure.set_view(BRIGHT_VIEW)
        drawn = np.asarray(image.get_array())
        assert drawn.shape == (scene.num_det_rows, scene.num_det_channels)
        assert (np.unravel_index(int(np.argmax(drawn)), drawn.shape)
                == (BRIGHT_ROW, BRIGHT_CHANNEL))
        assert image.get_clim() == (float(np.min(values)),
                                    float(np.max(values)))
    finally:
        close(figure)


def test_the_bright_sinogram_pixel_is_drawn_at_its_row_and_channel():
    """The painted pixel lands on the screen where its row and channel are.

    This is the orientation gate.  The figure is rendered, the brightest block
    of pixels is found in the image's own area, and its display position is
    converted back to data coordinates through the panel's inverted
    ``transData``.  Those coordinates must be the channel and the row the array
    holds the bright value at, so the panel's inverted row axis is part of what
    is measured.

    Two areas are left out of the search.  The panel's background is white and
    so is the bright pixel, so only the image's own area is searched; the
    legend's box is white as well, so it is taken out of that area.
    """
    scene = build_scene(SINOGRAM_CONFIG)
    figure = GeometryFigure(scene, view_index=1,
                            sinogram=bright_sinogram(scene))
    try:
        figure.set_view(BRIGHT_VIEW)
        rendered = rendered_rgb(figure)
        renderer = figure.figure.canvas.get_renderer()
        image_box = figure._sinogram_image.get_window_extent(renderer)
        legend = figure.ax_detector.get_legend()
        searched = (box_mask(rendered.shape, image_box)
                    & ~box_mask(rendered.shape,
                                legend.get_window_extent(renderer)))
        brightness = np.where(searched, rendered.sum(axis=2), -1)
        rows, columns = np.nonzero(
            brightness >= brightness.max() - BRIGHTNESS_TOLERANCE)
        assert rows.size > 0, 'no bright pixel was drawn'
        data = pixels_to_data(figure.ax_detector, rendered.shape[0],
                              rows, columns)
        channel, row = data.mean(axis=0)
        assert round(float(channel)) == BRIGHT_CHANNEL
        assert round(float(row)) == BRIGHT_ROW
    finally:
        close(figure)


def test_the_painted_sinogram_lands_inside_the_projected_rims():
    """A forward-projected phantom lights the detector where the rims are.

    The phantom is one inside the region of reconstruction and zero outside,
    and ``model.forward_project`` turns it into the sinogram the panel paints.
    The panel draws the projected rims of that same region over the image, so
    the lit part of the image and the rims are two accounts of one shape: one
    from the projector and one from the scene.  The test compares their extents
    in rows and in channels, in every view.

    The extents are compared edge to edge.  A lit pixel covers the half pixel
    on each side of its index, so the lit extent runs from the lowest lit index
    less a half to the highest plus a half, and the rims are already continuous
    coordinates.

    The two tolerances differ, and the reason is where the mass sits relative
    to the rims.  The rims run through the centers of the outermost voxels,
    which is the ellipse mbirtorch's own mask uses, and the material of those
    voxels reaches half a voxel further, which is 0.91 of a detector channel in
    this scan.  The projector's footprint then spreads that mass by up to half
    a channel more, and any pixel with some mass counts as lit here, so the lit
    channel edge lies outside the rims by up to about 1.4 channels.  Along the
    rows the cylinder already spans the outer faces of the voxels, so only the
    footprint's spread remains.  At a threshold of a tenth of the largest value
    the lit channel edge and the rims agree to 0.1 channel on average, which is
    the measurement that fixed the rims' semi-axes (2026-09-11).
    """
    import torch
    import mbirtorch

    model = probe.build_model(CONFIGS_BY_NAME[SINOGRAM_CONFIG])
    scene = GeometryScene.from_model(model)
    phantom = np.zeros(scene.recon_shape, dtype=np.float32)
    phantom[mbirtorch.get_2d_ror_mask(scene.recon_shape)] = 1.0
    sinogram = np.asarray(model.forward_project(torch.tensor(phantom)))
    floor = LIT_FRACTION * float(np.max(sinogram))

    figure = GeometryFigure(scene, sinogram=sinogram)
    try:
        for view_index in range(scene.num_views):
            figure.set_view(view_index)
            drawn = np.asarray(figure._sinogram_image.get_array())
            lit = drawn > floor
            assert lit.any(), f'view {view_index} painted nothing'
            rims = np.asarray(scene.view(view_index).ror_outline_on_detector)
            rims = rims.reshape(-1, 2)
            for axis, tolerance, name in ((0, RIM_ROW_TOLERANCE, 'row'),
                                          (1, RIM_CHANNEL_TOLERANCE,
                                           'channel')):
                indices = np.flatnonzero(lit.any(axis=1 - axis))
                for edge, expected in ((indices.min() - 0.5,
                                        rims[:, axis].min()),
                                       (indices.max() + 0.5,
                                        rims[:, axis].max())):
                    assert abs(float(edge) - float(expected)) <= tolerance, (
                        f'view {view_index}: the lit {name} edge is at '
                        f'{edge}, and the rims reach {expected}')
    finally:
        close(figure)


@pytest.mark.parametrize('corner', ('low', 'high'))
def test_the_silhouette_is_drawn_where_its_voxels_are(corner):
    """The silhouette of a corner block lands on that corner of the box.

    A block of voxels is filled at one corner of the reconstruction, and the
    top view and the side view must each draw it where the scene puts that
    block.  The check is on the rendered figure, so it covers both panels'
    inverted axes, the three voxel pitches, and the volume's slice offset: the
    patch's position on the screen is converted back to data coordinates and
    compared with the block's center, which ``GeometryScene.voxel_centers``
    gives at the block's fractional index.  At the low corner it is also
    compared with the marker of voxel (0, 0, 0), which each panel already
    draws.

    The patch is found by rendering the figure twice, once without the
    silhouette and once with it, and taking the pixels that differ.  Matching
    the fill color instead does not work here: the fill is the volume's color
    over the panel's white background, and the antialiased edge of the volume
    box's own outline, which is that color drawn solid, produces the same
    pixels along every edge of the box.
    """
    scene = build_scene(SILHOUETTE_CONFIG)
    rows, cols, slices = scene.recon_shape
    block = SILHOUETTE_BLOCK
    recon = np.zeros(scene.recon_shape, dtype=np.float32)
    if corner == 'low':
        recon[:block, :block, :block] = 1.0
        center = [(block - 1) / 2.0] * 3
    else:
        recon[rows - block:, cols - block:, slices - block:] = 1.0
        center = [rows - (block + 1) / 2.0, cols - (block + 1) / 2.0,
                  slices - (block + 1) / 2.0]
    expected = scene.voxel_centers([center])[0]
    # The pitches are indexed the way a panel's columns are, as (x, y, z).
    pitches = (scene.delta_voxel, scene.delta_voxel_row,
               scene.delta_voxel_slice)

    figure = GeometryFigure(scene, view_index=2)
    try:
        before = rendered_rgb(figure)
        figure.set_recon(recon)
        after = rendered_rgb(figure)
        view = scene.view(figure.view_index)
        for axes, columns, name in ((figure.ax_top, TOP_PANEL_COLUMNS, 'top'),
                                    (figure.ax_side, SIDE_PANEL_COLUMNS,
                                     'side')):
            pixel_rows, pixel_columns = changed_in_panel(figure, axes,
                                                         before, after)
            assert pixel_rows.size > 0, f'nothing was drawn in the {name} view'
            data = pixels_to_data(axes, after.shape[0], pixel_rows,
                                  pixel_columns)
            drawn = data.mean(axis=0)
            allowed = np.asarray([pitches[column] for column in columns])
            wanted = expected[list(columns)]
            assert np.all(np.abs(drawn - wanted) <= allowed), (
                f'the {name} view drew the patch at {drawn}, and the block '
                f'sits at {wanted}')
            if corner == 'low':
                marker = np.asarray(view.voxel0_center)[list(columns)]
                assert np.all(np.abs(drawn - marker) <= allowed)
    finally:
        close(figure)


def test_the_silhouette_takes_the_threshold_it_is_given():
    """The support is the voxels above the threshold, default or given.

    The default threshold is a tenth of the largest absolute value, so a voxel
    at half the largest value belongs to the support.  A threshold given by the
    caller is an absolute value, so a threshold above that voxel's value drops
    it.  The sign of a voxel does not matter, because the support is about the
    absolute value.
    """
    scene = build_scene(SILHOUETTE_CONFIG)
    recon = np.zeros(scene.recon_shape, dtype=np.float32)
    recon[0, 0, 0] = 1.0
    recon[1, 1, 1] = -0.5
    figure = GeometryFigure(scene, recon=recon)
    try:
        assert figure._recon_threshold == pytest.approx(0.1)
        assert int(figure._recon_support.sum()) == 2

        figure.set_recon(recon, threshold=0.7)
        assert figure._recon_threshold == pytest.approx(0.7)
        assert int(figure._recon_support.sum()) == 1
        assert figure._recon_support[0, 0, 0]
    finally:
        close(figure)


def test_an_overlay_of_the_wrong_shape_is_refused():
    """A mismatched array raises, and the message names both shapes."""
    scene, figure = build_figure(SINOGRAM_CONFIG)
    try:
        with pytest.raises(ValueError) as problem:
            figure.set_sinogram(np.zeros((3, 4, 5)))
        message = str(problem.value)
        assert '(3, 4, 5)' in message
        assert str(tuple(scene.sinogram_shape)) in message

        with pytest.raises(ValueError) as problem:
            figure.set_recon(np.zeros((3, 4, 5)))
        message = str(problem.value)
        assert '(3, 4, 5)' in message
        assert str(tuple(scene.recon_shape)) in message
    finally:
        close(figure)


def test_the_overlays_are_named_in_the_title_and_the_footer():
    """The detector title says a sinogram is painted, and the footer lists
    both overlays.

    Neither overlay has a legend entry, so these two lines are where the figure
    says what is drawn.  The threshold is named the way it was chosen: the
    default as the fraction of the largest value that it is, and a caller's own
    as the number it is.
    """
    scene = build_scene(SINOGRAM_CONFIG)
    recon = np.zeros(scene.recon_shape, dtype=np.float32)
    recon[0, 0, 0] = 1.0
    figure = GeometryFigure(scene, sinogram=bright_sinogram(scene),
                            recon=recon)
    try:
        assert 'with sinogram' in figure.ax_detector.get_title()
        body = '\n'.join(artist.get_text() for artist in figure.ax_text.texts)
        assert 'overlays : sinogram; recon above 0.1 max' in body

        figure.set_recon(recon, threshold=0.25)
        body = '\n'.join(artist.get_text() for artist in figure.ax_text.texts)
        assert 'overlays : sinogram; recon above 0.25' in body

        # Removing both takes the note out of the title and the footer.
        figure.set_sinogram(None)
        figure.set_recon(None)
        assert 'with sinogram' not in figure.ax_detector.get_title()
        body = '\n'.join(artist.get_text() for artist in figure.ax_text.texts)
        assert 'overlays' not in body
    finally:
        close(figure)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
