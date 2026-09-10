"""Headless tests for the widgets, the zoom, and the comparison overlay.

These are the Increment 4 additions to `geometry_viewer.py`.  The tests check
five things.  The view slider exists with the range the scan has, and moving it
changes what is drawn.  The trajectory toggle draws one polyline per panel and
not one marker per view.  The zoom toggle puts a cube around the volume on the
3D panel.  A comparison whose channel offset is ten channels larger draws its
projected volume outline ten channels over from the primary's and lists the
change in the text panel.  Removing the comparison removes its artists.

Later tests cover the display-test follow-up of 2026-09-09.  The labels on the
source, the detector, and the pixel-0 marker exist in the panels that carry
them, they follow the source when the view changes, and they are animated
exactly when the other moving artists are.  The angle-0 reference is drawn in
the 3D view and the top view, it does not move with the view, it is never
animated, its toggle hides and shows it, and the panel limits hold it whether
or not it is drawn.

The tests run under the Agg backend and open no window.

Run:
    cd plans/experiments/geometry_viewer
    MPLBACKEND=Agg PYTHONPATH=<mbirtorch clone> python -m pytest -q \
        test_geometry_interaction.py
"""

import os
import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')  # the tests draw into a buffer and open no window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gv1_conventions_probe as probe  # noqa: E402
from geometry_scene import GeometryScene  # noqa: E402
import geometry_viewer  # noqa: E402
from geometry_viewer import (COLORS, GeometryFigure,  # noqa: E402
                             VOLUME_BOX_EDGES, ZOOM_VOLUME_WIDTH_FACTOR,
                             TOP_PANEL_COLUMNS, SIDE_PANEL_COLUMNS,
                             show_geometry)

#: The object coordinates the top and side panels put on their two axes, as
#: lists for indexing.  The top view is the xy plane seen from -z, so it draws
#: y across the screen and x down it; see the display convention in
#: `geometry_viewer`.
TOP_COLUMNS = list(TOP_PANEL_COLUMNS)
SIDE_COLUMNS = list(SIDE_PANEL_COLUMNS)

CONFIGS_BY_NAME = {cfg['name']: cfg for cfg in probe.CONFIGS}

#: How many channels the comparison's detector offset is moved by.  The plan's
#: gate names ten channels.
COMPARISON_CHANNEL_SHIFT = 10


def build_figure(name='cone flat', **kwargs):
    """Build the scene and a figure for one probe configuration."""
    cfg = CONFIGS_BY_NAME[name]
    scene = GeometryScene.from_model(probe.build_model(cfg))
    return scene, GeometryFigure(scene, **kwargs)


def close(figure):
    """Close a figure's matplotlib figure so that the tests do not pile up."""
    import matplotlib.pyplot as plt
    plt.close(figure.figure)


def lines_of_color(axes, color):
    """The visible lines of one color in a panel, in drawing order."""
    return [line for line in axes.get_lines()
            if line.get_color() == color and line.get_visible()]


def polylines_of_color(axes, color):
    """The visible polylines of one color, leaving out the point markers.

    A marker is a line artist too, so a panel that marks a point in the
    comparison color holds more lines of that color than it draws polylines.
    """
    return [line for line in lines_of_color(axes, color)
            if line.get_marker() in ('None', '', None)]


def finite_points(line):
    """One line's data as an (N, 2) array, with the NaN separators dropped."""
    points = np.stack([np.asarray(line.get_xdata(), dtype=np.float64),
                       np.asarray(line.get_ydata(), dtype=np.float64)],
                      axis=1)
    return points[np.isfinite(points).all(axis=1)]


# ── the view slider ─────────────────────────────────────────────────────────

def test_slider_has_the_range_of_the_scan():
    """The slider covers every view and steps by one whole view."""
    scene, figure = build_figure()
    try:
        slider = figure.view_slider
        assert slider is not None
        assert slider.valmin == 0
        assert slider.valmax == scene.num_views - 1
        assert slider.valstep == 1
        assert int(round(float(slider.val))) == figure.view_index
    finally:
        close(figure)


def test_moving_the_slider_changes_the_drawn_view():
    """Setting the slider's value redraws another view.

    The check reads the drawn data and not only the view index, because a
    slider that moved the index without redrawing would leave the figure
    showing the old view.
    """
    scene, figure = build_figure(view_index=0)
    try:
        before = finite_points(figure._top['detector']).copy()
        figure.view_slider.set_val(3)
        assert figure.view_index == 3
        after = finite_points(figure._top['detector'])
        assert not np.allclose(before, after)
        # The drawn outline is the scene's outline for the new view.
        expected = scene.view(3).detector_outline
        assert np.allclose(after, expected[:, TOP_COLUMNS])
        # The title names the view drawn.
        assert 'view 3' in figure.ax_detector.get_title()
    finally:
        close(figure)


def test_a_single_view_scan_hides_the_slider():
    """One view has nothing to slide, so no slider is built."""
    cfg = CONFIGS_BY_NAME['cone flat']
    scene = GeometryScene.from_model(probe.build_model(cfg))
    params = dict(scene.params)
    params['sinogram_shape'] = (1, scene.num_det_rows, scene.num_det_channels)
    params['view_params_array'] = np.asarray(
        params['view_params_array'], dtype=np.float64)[:1]
    figure = GeometryFigure(GeometryScene(params, 'cone'))
    try:
        assert figure.view_slider is None
        assert figure._slider_axes.get_visible() is False
    finally:
        close(figure)


# ── the trajectory toggle ───────────────────────────────────────────────────

def test_trajectory_is_one_polyline_per_panel():
    """The source path is one line per panel, whatever the number of views.

    A path drawn as one marker per view would cost 1800 artists on a helical
    scan, which is what the plan's gate rules out.  The test therefore counts
    the lines and checks that they carry no marker.
    """
    scene, figure = build_figure('cone helical', show_trajectory=True)
    try:
        assert figure.show_trajectory is True
        for axes in (figure.ax_top, figure.ax_side):
            drawn = lines_of_color(axes, COLORS['trajectory'])
            assert len(drawn) == 1, axes.get_title()
            line = drawn[0]
            assert line.get_marker() in ('None', '', None)
            assert len(line.get_xdata()) == scene.num_views
        # The 3D panel draws it too, as one 3D line.
        path_3d = [line for line in figure.ax_3d.get_lines()
                   if line.get_color() == COLORS['trajectory']
                   and line.get_visible()]
        assert len(path_3d) == 1
        assert len(path_3d[0].get_data_3d()[0]) == scene.num_views
    finally:
        close(figure)


def test_the_trajectory_toggle_turns_the_path_on_and_off():
    """The toggle widget and set_show_trajectory agree on the state."""
    _, figure = build_figure('cone helical')
    try:
        assert figure.show_trajectory is False
        assert not lines_of_color(figure.ax_top, COLORS['trajectory'])
        assert figure.trajectory_check.get_status()[0] is False

        # Clicking the widget calls its callback with the label.
        figure.trajectory_check.set_active(0)
        assert figure.show_trajectory is True
        assert len(lines_of_color(figure.ax_top, COLORS['trajectory'])) == 1

        figure.set_show_trajectory(False)
        assert figure.trajectory_check.get_status()[0] is False
        assert not lines_of_color(figure.ax_top, COLORS['trajectory'])
    finally:
        close(figure)


# ── the 3D zoom ─────────────────────────────────────────────────────────────

def test_zoom_to_volume_puts_a_cube_around_the_volume():
    """The volume zoom is a cube on the volume, smaller than the scan cube."""
    scene, figure = build_figure()
    try:
        assert figure.zoom == 'scan'
        scan_width = float(np.ptp(figure.ax_3d.get_xlim()))

        figure.set_zoom('volume')
        assert figure.zoom == 'volume'
        limits = np.array([figure.ax_3d.get_xlim(), figure.ax_3d.get_ylim(),
                           figure.ax_3d.get_zlim()])
        widths = limits[:, 1] - limits[:, 0]
        # A cube: the three axes have one width.
        assert np.allclose(widths, widths[0])
        # About three volume extents wide.
        corners = scene.volume_corners()
        extent = float(np.max(corners.max(axis=0) - corners.min(axis=0)))
        assert widths[0] == pytest.approx(ZOOM_VOLUME_WIDTH_FACTOR * extent)
        # It holds the whole volume box, and it is closer in than the scan.
        for axis in range(3):
            assert limits[axis, 0] < corners[:, axis].min()
            assert limits[axis, 1] > corners[:, axis].max()
        assert widths[0] < scan_width

        figure.set_zoom('scan')
        assert float(np.ptp(figure.ax_3d.get_xlim())) == pytest.approx(
            scan_width)
    finally:
        close(figure)


def test_the_zoom_toggle_switches_the_two_states():
    """The toggle widget and set_zoom agree on the state."""
    _, figure = build_figure()
    try:
        assert figure.zoom_check.get_status()[0] is False
        figure.zoom_check.set_active(0)
        assert figure.zoom == 'volume'
        figure.set_zoom('scan')
        assert figure.zoom_check.get_status()[0] is False
        with pytest.raises(ValueError):
            figure.set_zoom('everything')
    finally:
        close(figure)


# ── the comparison overlay ──────────────────────────────────────────────────

def compare_overrides(scene, channels=COMPARISON_CHANNEL_SHIFT):
    """The override dictionary that moves the detector by whole channels."""
    return dict(det_channel_offset=(scene.det_channel_offset
                                    + channels * scene.delta_det_channel))


def test_comparison_outline_is_shifted_by_ten_channels():
    """Ten channels of offset move the comparison outline ten channels.

    The primary and the comparison differ in `det_channel_offset` alone, so
    their projected volume outlines have the same shape and the comparison's
    sits ten channels along the channel axis.  The check reads the two lines'
    data out of the detector-face panel.
    """
    scene, figure = build_figure(view_index=2)
    try:
        figure.set_compare(compare_overrides(scene))
        primary = polylines_of_color(figure.ax_detector, COLORS['volume'])
        comparison = polylines_of_color(figure.ax_detector,
                                        COLORS['compare'])
        assert len(primary) == 1 and len(comparison) == 1

        first = finite_points(primary[0])
        second = finite_points(comparison[0])
        assert first.shape == (2 * len(VOLUME_BOX_EDGES), 2)
        assert second.shape == first.shape
        # The panel's axes are (channel, row).
        assert np.allclose(second[:, 0] - first[:, 0],
                           COMPARISON_CHANNEL_SHIFT)
        assert np.allclose(second[:, 1], first[:, 1])
    finally:
        close(figure)


def test_comparison_is_listed_in_the_text_panel():
    """The text panel names the changed parameter with both values."""
    scene, figure = build_figure(view_index=2)
    try:
        overrides = compare_overrides(scene)
        figure.set_compare(overrides)
        body = '\n'.join(artist.get_text() for artist in figure.ax_text.texts)
        assert 'Comparison' in body
        assert 'det_channel_offset' in body
        line = [text for text in body.splitlines()
                if 'det_channel_offset' in text][0]
        assert '->' in line
        assert f'{scene.det_channel_offset:.3g}' in line
        assert f'{overrides["det_channel_offset"]:.3g}' in line
    finally:
        close(figure)


def test_comparison_follows_the_slider():
    """The comparison is drawn for the view the slider is on."""
    scene, figure = build_figure(view_index=0)
    try:
        figure.set_compare(compare_overrides(scene))
        figure.view_slider.set_val(4)
        assert figure.view_index == 4
        drawn = finite_points(figure._compare['detector_top'])
        expected = figure.compare_scene.view(4).detector_outline
        assert np.allclose(drawn, expected[:, TOP_COLUMNS])
    finally:
        close(figure)


def test_comparison_accepts_a_scene_and_a_model():
    """A comparison can be a scene, a model, or a dictionary of overrides."""
    scene, figure = build_figure()
    try:
        other = scene.with_parameters(compare_overrides(scene))
        figure.set_compare(other)
        assert figure.compare_scene is other

        model = probe.build_model(CONFIGS_BY_NAME['parallel'])
        figure.set_compare(model)
        assert figure.compare_scene.kind == 'parallel'

        figure.set_compare(compare_overrides(scene))
        assert figure.compare_scene.kind == scene.kind
    finally:
        close(figure)


def test_removing_the_comparison_removes_its_artists():
    """set_compare(None) leaves no comparison artist behind."""
    scene, figure = build_figure()
    try:
        figure.set_compare(compare_overrides(scene))
        assert figure.compare_scene is not None
        for axes in (figure.ax_3d, figure.ax_top, figure.ax_side,
                     figure.ax_detector):
            assert lines_of_color(axes, COLORS['compare'])

        figure.set_compare(None)
        assert figure.compare_scene is None
        for axes in (figure.ax_3d, figure.ax_top, figure.ax_side,
                     figure.ax_detector):
            assert not [line for line in axes.get_lines()
                        if line.get_color() == COLORS['compare']]
        body = '\n'.join(artist.get_text() for artist in figure.ax_text.texts)
        assert 'Comparison' not in body
        # The figure still redraws with no comparison.
        figure.set_view(1)
    finally:
        close(figure)


def test_a_comparison_at_construction_is_drawn():
    """The compare argument of the constructor draws the same overlay."""
    cfg = CONFIGS_BY_NAME['cone flat']
    scene = GeometryScene.from_model(probe.build_model(cfg))
    figure = GeometryFigure(scene, view_index=2,
                            compare=compare_overrides(scene))
    try:
        assert figure.compare_scene is not None
        assert lines_of_color(figure.ax_detector, COLORS['compare'])
    finally:
        close(figure)


# ── construction never opens a window ───────────────────────────────────────

def test_construction_does_not_call_show(monkeypatch):
    """Building, redrawing, and saving a figure never calls show.

    The slice viewer keeps window handling out of construction for the same
    reason: a class that opened a window could not be used in a script or a
    test.
    """
    _, figure = build_figure()
    close(figure)
    import matplotlib.pyplot as plt
    calls = []
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: calls.append(1))

    scene, figure = build_figure(view_index=1)
    try:
        figure.set_view(2)
        figure.set_show_trajectory(True)
        figure.set_zoom('volume')
        figure.set_compare(compare_overrides(scene))
        assert calls == []
    finally:
        close(figure)


def test_show_geometry_passes_block_through(monkeypatch, tmp_path):
    """show_geometry builds a figure and shows it once, with the block flag.

    Under Agg there is no window, so the backend name is replaced for the
    duration of the test.  What is checked is that ``show`` is called once and
    that ``block`` reaches it.
    """
    import matplotlib.pyplot as plt
    calls = []
    monkeypatch.setattr(plt, 'show', lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(matplotlib, 'get_backend', lambda: 'qtagg')

    cfg = CONFIGS_BY_NAME['cone flat']
    figure = show_geometry(probe.build_model(cfg), view_index=1, block=False)
    try:
        assert isinstance(figure, GeometryFigure)
        assert calls == [dict(block=False)]
    finally:
        close(figure)

    calls.clear()
    figure = show_geometry(probe.build_model(cfg))
    try:
        assert calls == [dict(block=True)]
    finally:
        close(figure)


def test_show_prints_and_returns_under_a_backend_with_no_window(capsys):
    """Under Agg, show says there is no window and returns."""
    _, figure = build_figure()
    try:
        figure.show()
        printed = capsys.readouterr().out
        assert 'no window' in printed
    finally:
        close(figure)


# ── the partial redraw draws the same picture as a full repaint ─────────────

def test_the_partial_redraw_matches_a_full_repaint(tmp_path):
    """A view reached by stepping looks like the same view drawn from new.

    The partial redraw restores a cached background and draws the artists that
    moved.  A stale background or a missing artist would show up here as two
    different images.  The widget row is left out of the comparison, because a
    slider draws a mark at the value it was built with and the two figures were
    built at different views.
    """
    scene, stepped = build_figure(view_index=0)
    _, direct = build_figure(view_index=4)
    try:
        stepped.set_view(4)
        first = str(tmp_path / 'stepped.png')
        second = str(tmp_path / 'direct.png')
        stepped.save(first, dpi=80)
        direct.save(second, dpi=80)
        import matplotlib.image as mpimg
        left = mpimg.imread(first)
        right = mpimg.imread(second)
        assert left.shape == right.shape
        panels = slice(0, int(0.86 * left.shape[0]))
        assert float(np.abs(left[panels] - right[panels]).max()) == 0.0
    finally:
        close(stepped)
        close(direct)


@pytest.mark.parametrize('name', list(CONFIGS_BY_NAME))
def test_every_geometry_takes_every_control(name, tmp_path):
    """Each of the six geometries survives the slider, the toggles, and a
    comparison.

    The controls touch different code for different geometries: the
    translation geometry has no rotation arc, the parallel and multiaxis
    geometries have no source position, and a curved detector has no filled
    face.  The test walks every geometry through every control and saves the
    result, so an artist that one geometry does not create cannot break a
    redraw.
    """
    scene, figure = build_figure(name, view_index=1)
    try:
        figure.set_zoom('volume')
        figure.set_view(3)
        figure.set_show_trajectory(True)
        figure.set_compare(compare_overrides(scene))
        figure.set_view(4)
        figure.set_zoom('scan')
        figure.set_show_trajectory(False)
        figure.set_compare(None)
        path = str(tmp_path / f'{name.replace(" ", "_")}.png')
        figure.save(path, dpi=80)
        assert os.path.getsize(path) > 10 * 1024
    finally:
        close(figure)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))


def test_plain_draw_paints_the_moving_artists_without_blitting():
    """Without the fast path, a plain full draw must paint the source and the
    detector.

    A backend outside BLIT_BACKENDS repaints the whole figure on every view
    change.  A full draw skips animated artists, so the moving artists must
    not be animated there.  Before this rule the source, the detector, and
    everything the slider moves were invisible on the macosx backend, while a
    saved file, which unmarks the artists, looked right.  The test drives the
    figure through the same plain draw and checks that it paints exactly what
    a draw with every artist unmarked paints.
    """
    import numpy as np
    import mbirtorch
    from geometry_viewer import GeometryFigure
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    model = mbirtorch.ConeBeamModel((12, 16, 24), angles,
                                    source_detector_dist=200.0,
                                    source_iso_dist=100.0, compile_mode='off')
    figure = GeometryFigure(model, view_index=0, blit=False, widgets=True)
    assert all(not artist.get_animated() for _, artist in figure._moving)

    def plain_draw():
        figure.figure.canvas.draw()
        return np.asarray(figure.figure.canvas.buffer_rgba()).copy()

    painted = plain_draw()
    figure._set_animated(False)
    reference = plain_draw()
    # Both draws paint the same artists, so the two buffers are identical.
    assert np.array_equal(painted, reference)

    # And a view change through the plain path moves the source on screen.
    before = plain_draw()
    figure.set_view(3)
    after = plain_draw()
    assert (before != after).any(axis=2).sum() > 500


def test_moving_artists_are_animated_only_on_a_blit_backend():
    """Under Agg with blitting enabled the fast path runs, so the moving
    artists are animated; with blitting disabled they are not."""
    import numpy as np
    import mbirtorch
    from geometry_viewer import GeometryFigure
    angles = np.linspace(0.0, np.pi, 4, endpoint=False)
    model = mbirtorch.ParallelBeamModel((4, 6, 16), angles, compile_mode='off')
    with_blit = GeometryFigure(model, blit=True, widgets=False)
    without = GeometryFigure(model, blit=False, widgets=False)
    assert all(artist.get_animated() for _, artist in with_blit._moving)
    assert all(not artist.get_animated() for _, artist in without._moving)


# ── the labels of the source, the detector, and the pixel-0 marker ──────────

def moving_artists(figure):
    """The artists a view change updates, without their axes."""
    return [artist for _, artist in figure._moving]


def all_labels(figure):
    """The seven labels the three drawing panels carry, as (label, text)."""
    return [(figure._source_text_3d, 'source'),
            (figure._detector_text_3d, 'detector'),
            (figure._top['source_label'], 'source'),
            (figure._top['detector_label'], 'detector'),
            (figure._top['pixel0_label'], 'pixel (0,0)'),
            (figure._side['source_label'], 'source'),
            (figure._side['detector_label'], 'detector')]


def test_the_source_and_the_detector_are_labeled_in_three_panels():
    """Each of the three drawing panels names the source and the detector.

    Greg's display run reported that nothing said which panel was the
    detector, so the elements carry their own labels now.  The pixel-0 marker
    is named in the top view alone.
    """
    scene, figure = build_figure(view_index=2)
    try:
        moving = moving_artists(figure)
        for label, text in all_labels(figure):
            assert label.get_text().strip() == text
            assert label in moving, text
        view = scene.view(2)
        # The labels sit where the things they name are.
        assert np.allclose(figure._source_text_3d.get_position_3d(),
                           view.source_draw)
        assert figure._top['source_label'].xy == pytest.approx(
            tuple(view.source_draw[TOP_COLUMNS]))
        assert figure._side['source_label'].xy == pytest.approx(
            tuple(view.source_draw[SIDE_COLUMNS]))
        assert figure._top['pixel0_label'].xy == pytest.approx(
            tuple(view.detector_pixel0[TOP_COLUMNS]))
        # The side view names no pixel-0 marker.
        assert 'pixel0_label' not in figure._side
        side_texts = [text.get_text() for text in figure.ax_side.texts]
        assert 'pixel (0,0)' not in side_texts
    finally:
        close(figure)


def test_the_labels_follow_the_source_and_the_detector():
    """A view change moves each label to the new position of its element."""
    scene, figure = build_figure(view_index=0)
    try:
        before = figure._top['source_label'].xy
        figure.set_view(3)
        view = scene.view(3)
        after = figure._top['source_label'].xy
        assert after != before
        assert after == pytest.approx(tuple(view.source_draw[TOP_COLUMNS]))
        assert np.allclose(figure._source_text_3d.get_position_3d(),
                           view.source_draw)
        # The detector's label sits on one of the detector's corners.
        for label, columns in ((figure._top['detector_label'], TOP_COLUMNS),
                               (figure._side['detector_label'],
                                SIDE_COLUMNS)):
            corners = view.detector_corners[:, columns]
            gap = np.linalg.norm(corners - np.asarray(label.xy)[None, :],
                                 axis=1)
            assert float(np.min(gap)) < 1e-9
        gap = np.linalg.norm(
            view.detector_corners
            - np.asarray(figure._detector_text_3d.get_position_3d())[None, :],
            axis=1)
        assert float(np.min(gap)) < 1e-9
    finally:
        close(figure)


def test_the_labels_are_animated_with_the_other_moving_artists():
    """A label follows the rule of every moving artist.

    The rule is `GeometryFigure._animate_moving`: a moving artist is animated
    only where the partial redraw runs.  A label that stayed animated on a
    backend without that path would be invisible there, which is the failure
    Greg saw for the source and the detector.
    """
    _, with_blit = build_figure(blit=True)
    _, without = build_figure(blit=False)
    try:
        assert with_blit._animate_moving() is True
        for label, _ in all_labels(with_blit):
            assert label.get_animated() is True
        assert without._animate_moving() is False
        for label, _ in all_labels(without):
            assert label.get_animated() is False
        # And every moving artist agrees with its labels.
        assert all(artist.get_animated()
                   for artist in moving_artists(with_blit))
        assert not any(artist.get_animated()
                       for artist in moving_artists(without))
    finally:
        close(with_blit)
        close(without)


# ── the angle-0 reference ───────────────────────────────────────────────────

def reference_positions(figure):
    """What each reference artist is drawn at, for the artists that can say.

    A line reports its data and a text its position.  The arrowheads, one
    ``quiver`` and one ``FancyArrowPatch``, report neither in a form worth
    comparing, so they are left out and only counted.
    """
    positions = []
    for _, artist in figure._reference_artists:
        if hasattr(artist, 'get_data_3d'):
            positions.append(np.concatenate(artist.get_data_3d()))
        elif hasattr(artist, 'get_position_3d'):
            positions.append(np.asarray(artist.get_position_3d()))
        elif isinstance(artist.get_visible(), bool) and hasattr(artist, 'xy'):
            positions.append(np.asarray(artist.xy, dtype=np.float64))
        elif hasattr(artist, 'get_data'):
            positions.append(np.concatenate(artist.get_data()))
    return positions


def test_the_reference_is_drawn_in_the_3d_and_top_panels():
    """The angle-0 reference is drawn where the record says it is.

    The 3D view and the top view get it.  The side view and the detector face
    do not: the side view is the yz plane, in which a rotation about z moves
    nothing.  The drawn source and detector are the scene's own
    ``reference_view``.
    """
    scene, figure = build_figure(view_index=2)
    try:
        assert figure.show_reference is True
        panels = [axes for axes, _ in figure._reference_artists]
        assert panels.count(figure.ax_3d) == 5
        assert panels.count(figure.ax_top) == 5
        assert figure.ax_side not in panels
        assert figure.ax_detector not in panels

        reference = scene.reference_view()
        stars = [artist for axes, artist in figure._reference_artists
                 if axes is figure.ax_top
                 and getattr(artist, 'get_marker', lambda: '')() == '*']
        assert len(stars) == 1
        assert np.allclose(np.concatenate(stars[0].get_data()),
                           reference.source_draw[TOP_COLUMNS])
        # The reference is not the view drawn, whose angle is not zero.
        assert not np.allclose(reference.source_draw,
                               scene.view(2).source_draw, atol=1e-3)
    finally:
        close(figure)


def test_the_reference_does_not_move_with_the_view():
    """The slider moves the geometry and leaves the reference where it is."""
    scene, figure = build_figure(view_index=0)
    try:
        before = reference_positions(figure)
        assert len(before) == 8
        figure.set_view(4)
        after = reference_positions(figure)
        assert len(after) == len(before)
        for first, second in zip(before, after):
            assert np.array_equal(first, second)
        # Meanwhile the source did move.
        assert not np.allclose(scene.view(0).source_draw,
                               scene.view(4).source_draw, atol=1e-3)
    finally:
        close(figure)


def test_the_reference_artists_are_not_animated():
    """The reference belongs to the background, so it is never animated.

    A view change does not redraw it, and a full repaint has to paint it.
    """
    _, figure = build_figure(blit=True)
    try:
        assert figure._animate_moving() is True
        for _, artist in figure._reference_artists:
            assert artist.get_animated() is False
        moving = moving_artists(figure)
        for _, artist in figure._reference_artists:
            assert artist not in moving
    finally:
        close(figure)


def test_the_reference_toggle_hides_and_shows_it():
    """The toggle widget and set_show_reference agree on the state."""
    _, figure = build_figure()
    try:
        assert figure.reference_check.get_status()[0] is True
        assert all(artist.get_visible()
                   for _, artist in figure._reference_artists)

        figure.set_show_reference(False)
        assert figure.show_reference is False
        assert figure.reference_check.get_status()[0] is False
        assert not any(artist.get_visible()
                       for _, artist in figure._reference_artists)

        # Clicking the widget turns it back on.
        figure.reference_check.set_active(0)
        assert figure.show_reference is True
        assert all(artist.get_visible()
                   for _, artist in figure._reference_artists)
    finally:
        close(figure)


def test_the_reference_is_off_when_the_constructor_says_so():
    """show_reference=False builds the artists and leaves them hidden."""
    _, figure = build_figure(show_reference=False)
    try:
        assert figure.show_reference is False
        assert figure.reference_check.get_status()[0] is False
        assert not any(artist.get_visible()
                       for _, artist in figure._reference_artists)
    finally:
        close(figure)


def test_the_panel_limits_hold_the_reference():
    """The reference is inside the panel limits, whether or not it is drawn.

    The limits are computed once, so a reference outside them would be cut off
    when the toggle turned it on.  The check is made with the reference off,
    because that is the case a limit computation could leave out.
    """
    scene, figure = build_figure(view_index=2, show_reference=False)
    try:
        reference = scene.reference_view()
        points = np.concatenate([reference.detector_outline,
                                 reference.source_draw.reshape(1, 3),
                                 reference.detector_origin.reshape(1, 3)])
        assert geometry_viewer._within(points[:, TOP_COLUMNS],
                                       figure._limits['top'])
        assert geometry_viewer._within(points, figure._limits['scan'])
    finally:
        close(figure)


def test_the_comparison_is_named_in_the_top_view():
    """A comparison's detector carries the word "comparison", once."""
    scene, figure = build_figure(view_index=2)
    try:
        figure.set_compare(compare_overrides(scene))
        label = figure._compare['label_top']
        assert label.get_text() == 'comparison'
        assert label.axes is figure.ax_top
        corners = figure.compare_scene.view(2).detector_corners[
            :, TOP_COLUMNS]
        gap = np.linalg.norm(corners - np.asarray(label.xy)[None, :], axis=1)
        assert float(np.min(gap)) < 1e-9
        for axes in (figure.ax_side, figure.ax_3d):
            assert 'comparison' not in [text.get_text()
                                        for text in axes.texts]
        figure.set_compare(None)
        assert 'comparison' not in [text.get_text()
                                    for text in figure.ax_top.texts]
    finally:
        close(figure)


def test_the_volume_zoom_hides_the_3d_labels_outside_its_cube():
    """A 3D label whose point leaves the cube is hidden, not drawn outside it.

    Matplotlib does not clip a 3D text artist or a ``quiver`` arrowhead to the
    axes limits, so in the volume zoom the source's label was drawn where its
    point projects, which is outside the panel and on top of the rest of the
    figure.
    """
    _, figure = build_figure(view_index=2)
    try:
        assert figure._source_text_3d.get_visible() is True
        assert figure._detector_text_3d.get_visible() is True
        assert figure._reference_text_3d.get_visible() is True

        figure.set_zoom('volume')
        cube = np.array([figure.ax_3d.get_xlim(), figure.ax_3d.get_ylim(),
                         figure.ax_3d.get_zlim()])
        source = figure.scene.view(2).source_draw
        # The source is outside the cube, so its label is not drawn.
        assert (source < cube[:, 0]).any() or (source > cube[:, 1]).any()
        assert figure._source_text_3d.get_visible() is False
        assert figure._detector_text_3d.get_visible() is False
        assert figure._reference_text_3d.get_visible() is False
        assert figure._reference_arrow_3d.get_visible() is False

        figure.set_zoom('scan')
        assert figure._source_text_3d.get_visible() is True
        assert figure._reference_text_3d.get_visible() is True
        assert figure._reference_arrow_3d.get_visible() is True
    finally:
        close(figure)


def test_geometry_viewer_mirrors_slice_viewer_nonblocking_registry():
    """geometry_viewer keeps a nonblocking figure alive in a module registry,
    and the next blocking call closes it, as mbirtorch.slice_viewer does."""
    import matplotlib.pyplot as plt
    import geometry_viewer as module
    cfg = CONFIGS_BY_NAME['parallel']
    model = probe.build_model(cfg)
    module._NONBLOCKING_FIGURES.clear()
    first = module.geometry_viewer(model, block=False)
    assert first in module._NONBLOCKING_FIGURES
    assert plt.fignum_exists(first.figure.number)
    # Under Agg, show prints a line and returns, so the blocking call returns
    # at once and runs the registry's closing step.
    second = module.geometry_viewer(model, view_index=2, block=True)
    assert module._NONBLOCKING_FIGURES == []
    assert not plt.fignum_exists(first.figure.number)
    assert second.view_index == 2
    plt.close(second.figure)


def test_geometry_viewer_takes_the_viewer_options_explicitly():
    """The entry point exposes the figure's options as named arguments."""
    import matplotlib.pyplot as plt
    import geometry_viewer as module
    cfg = CONFIGS_BY_NAME['cone flat']
    model = probe.build_model(cfg)
    figure = module.geometry_viewer(model, view_index=1, show_trajectory=True,
                                    compare=dict(det_channel_offset=5.0),
                                    show_reference=False, zoom='volume',
                                    title='named', block=False)
    assert figure.view_index == 1
    assert figure.show_trajectory is True
    assert figure.compare_scene is not None
    module._NONBLOCKING_FIGURES.clear()
    plt.close(figure.figure)
