"""A web page that draws an mbirtorch scan geometry, built with Gradio.

What the app shows.  A user chooses one of six scan geometries, sets the scan
and the detector with a few numbers, and gets the geometry viewer's five-panel
figure: a 3D view, a top view of the xy plane, a side view of the yz plane, the
detector face in row and channel index, and a panel of derived numbers.  A
slider steps through the views.  Three checkboxes turn the source's path, the
3D zoom to the volume, and the angle-0 reference on and off.  A comparison
section draws a second geometry over the first from a few parameter overrides,
which is the calibration use: a vendor geometry against the same geometry with
an estimated offset.

Where the numbers come from.  ``geometry_defaults.default_parameters`` builds
the complete parameter dictionary, including the reconstruction shape and the
voxel pitch that an mbirtorch model constructor would choose.  Nothing here
imports mbirtorch or torch.  ``geometry_scene.GeometryScene`` turns the
dictionary into drawable primitives and ``geometry_viewer.GeometryFigure``
draws them.

Why the figure carries no widgets.  On the web the figure is a static image, so
the app builds it with ``widgets=False`` and puts every control in the page.
``blit=False`` is required as well: with blitting allowed, the viewer marks the
artists that a view change moves as animated, a full draw skips an animated
artist, and the source, the detector, and everything else that moves would be
missing from the image.

How to run it locally:

    cd plans/experiments/geometry_viewer/web
    pip install -r requirements.txt
    python app.py

The page then opens at the address Gradio prints.  Importing this module builds
the Blocks object and launches nothing, so a test can import it and launch it
itself.

Two packagings deploy this file, both assembled by ``gv5_build_web.py``.  The
first is ``web/lite/index.html``, a static Hugging Face Space that runs this
same Python in the browser under Gradio-Lite.  The second is ``web/space/``, a
Hugging Face Space with the Gradio SDK, which runs it on a server.  Every
component and argument used here exists in both Gradio 5.45, which is the
Gradio-Lite release the static page is pinned to, and in Gradio 6.
"""

import ast
import os
import sys
import time

import matplotlib

# Agg before any pyplot import: this process has no display, and Pyodide has
# no GUI toolkit at all.  The viewer defers its pyplot import to the first
# figure, so this line runs first whatever imports follow.
matplotlib.use('Agg')

import gradio as gr  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# The three modules this app draws with sit beside it in a Space and one
# directory up in the plans repository, so both places are on the path.  In
# Gradio-Lite the files are written into the working directory and __file__ may
# not be set, so the path setup is guarded.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                                   # pragma: no cover
    _HERE = os.getcwd()
for _directory in (_HERE, os.path.dirname(_HERE)):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

import geometry_defaults  # noqa: E402
from geometry_scene import GeometryScene, required_parameter_names  # noqa: E402
from geometry_viewer import GeometryFigure  # noqa: E402

# ── the page's fixed choices ─────────────────────────────────────────────────

#: The geometries the page offers, as the label a user picks and the geometry
#: kind and variant it builds.  The three cone entries are one kind.
GEOMETRY_CHOICES = ('cone', 'cone curved', 'cone helical', 'parallel',
                    'multiaxis', 'translation')

#: The geometry kind of each choice.
KIND_OF_CHOICE = dict((
    ('cone', 'cone'), ('cone curved', 'cone'), ('cone helical', 'cone'),
    ('parallel', 'parallel'), ('multiaxis', 'multiaxis'),
    ('translation', 'translation'),
))

#: The figure's size in inches and its resolution in dots per inch.  The
#: product is the image the page shows, 1080 by 675 pixels.  This is smaller
#: than the 15 by 9 inches the desktop viewer uses, because the browser
#: receives the figure as an image, and a smaller image is drawn and sent
#: faster.  The resolution reaches the figure through a matplotlib setting
#: that :func:`render` holds only while the figure is built, so the figure
#: carries the resolution from the moment it is created, the viewer's text
#: fitting measures the text at the size the image will have, and no other
#: figure in the process is affected.
WEB_FIGSIZE = (12.0, 7.5)
WEB_DPI = 90

#: The image format the plot component sends.  The default is webp, which the
#: matplotlib release inside Pyodide cannot write, so the format is named here.
PLOT_FORMAT = 'png'

#: The default scan: the view count, the detector shape, and the two
#: distances in ALU.  These are the numbers ``gv_show_example.py`` uses.
DEFAULT_NUM_VIEWS = 180
DEFAULT_NUM_DET_ROWS = 96
DEFAULT_NUM_DET_CHANNELS = 128
DEFAULT_SOURCE_DETECTOR_DIST = 4.0 * DEFAULT_NUM_DET_CHANNELS
DEFAULT_SOURCE_ISO_DIST = DEFAULT_SOURCE_DETECTOR_DIST / 2.0

#: The defaults of the three geometries that take one number of their own: the
#: helical travel in ALU, the multiaxis elevation in degrees, and the
#: translation grid.
DEFAULT_HELICAL_TRAVEL = 50.0
DEFAULT_ELEVATION_DEG = 30.0
DEFAULT_NUM_X_TRANSLATIONS = 5
DEFAULT_NUM_Z_TRANSLATIONS = 3
DEFAULT_X_SPACING = 20.0
DEFAULT_Z_SPACING = 12.0

#: The largest scan the page will draw.  A view change redraws the whole
#: figure here, and the number of views sets what the source's path and the
#: fit test cost, so a page that accepted a million views would hang.
MAX_NUM_VIEWS = 3600

#: The largest detector the page will draw.  The detector's size costs the
#: drawing almost nothing, because the drawing draws its outline and not its
#: pixels, so this bound is generous.
MAX_DETECTOR_SIDE = 100000

#: The example the comparison box starts with.
EXAMPLE_OVERRIDE_TEXT = 'det_channel_offset=13.0'

#: The smallest width in pixels of a number box that shares a row with
#: another one.  Gradio's own minimum of 160 pixels is wider than half of the
#: control column, so a pair of boxes wrapped onto two lines and the column
#: grew twice as tall as it needs to be.
PAIR_MIN_WIDTH = 110


# ── building one figure ──────────────────────────────────────────────────────

def scene_parameters(geometry, num_views, num_det_rows, num_det_channels,
                     delta_det_channel, delta_det_row, det_channel_offset,
                     det_row_offset, source_detector_dist, source_iso_dist,
                     angle_start_deg, angle_end_deg, elevation_deg,
                     helical_travel, num_x_translations, num_z_translations,
                     x_spacing, z_spacing, recon_shape=None):
    """The parameter dictionary and geometry kind for one set of page values.

    Args:
        geometry (str): one of :data:`GEOMETRY_CHOICES`.
        num_views, num_det_rows, num_det_channels (int): the sinogram shape.
            The translation geometry takes its view count from its grid
            instead, so ``num_views`` is ignored there.
        delta_det_channel, delta_det_row (float): the detector pitches in ALU.
        det_channel_offset, det_row_offset (float): the detector offsets in
            ALU.
        source_detector_dist, source_iso_dist (float): the two distances in
            ALU, used by the cone and translation geometries.
        angle_start_deg, angle_end_deg (float): the angular range in degrees.
        elevation_deg (float): the multiaxis elevation in degrees.
        helical_travel (float): the helical travel in ALU.
        num_x_translations, num_z_translations (int): the translation grid.
        x_spacing, z_spacing (float): the translation spacings in ALU.
        recon_shape (tuple of int, optional): a reconstruction shape that
            replaces the automatic one.

    Returns:
        (dict, str): the parameters and the geometry kind.
    """
    kind = KIND_OF_CHOICE[geometry]
    detector = dict(delta_det_channel=float(delta_det_channel),
                    delta_det_row=float(delta_det_row),
                    det_channel_offset=float(det_channel_offset),
                    det_row_offset=float(det_row_offset))
    num_det_rows = _counted(num_det_rows, 'the number of detector rows',
                            MAX_DETECTOR_SIDE)
    num_det_channels = _counted(num_det_channels,
                                'the number of detector channels',
                                MAX_DETECTOR_SIDE)

    if kind == 'translation':
        num_x = _counted(num_x_translations, 'the number of x translations',
                         MAX_NUM_VIEWS)
        num_z = _counted(num_z_translations, 'the number of z translations',
                         MAX_NUM_VIEWS)
        views = _counted(num_x * num_z, 'the number of translations',
                         MAX_NUM_VIEWS)
        vectors = geometry_defaults.translation_vectors(
            num_x, num_z, float(x_spacing), float(z_spacing))
        geometry_arguments = dict(
            translation_vectors=vectors,
            source_detector_dist=float(source_detector_dist),
            source_iso_dist=float(source_iso_dist))
    else:
        views = _counted(num_views, 'the number of views', MAX_NUM_VIEWS)
        if kind == 'multiaxis':
            geometry_arguments = dict(
                angles=geometry_defaults.azimuth_elevation_pairs(
                    views, angle_start_deg, angle_end_deg, elevation_deg))
        else:
            angles = geometry_defaults.view_angles(views, angle_start_deg,
                                                   angle_end_deg)
            geometry_arguments = dict(angles=angles)
        if kind == 'cone':
            geometry_arguments['source_detector_dist'] = float(
                source_detector_dist)
            geometry_arguments['source_iso_dist'] = float(source_iso_dist)
            geometry_arguments['use_curved_detector'] = (
                geometry == 'cone curved')
            if geometry == 'cone helical':
                geometry_arguments['helical_z_shifts'] = (
                    geometry_defaults.helical_z_shifts(views, helical_travel))

    shape = (views, num_det_rows, num_det_channels)
    params = geometry_defaults.default_parameters(kind, shape, **detector,
                                                  **geometry_arguments)
    if recon_shape is not None:
        params['recon_shape'] = recon_shape
    return params, kind


def render(geometry, num_views, num_det_rows, num_det_channels,
           delta_det_channel, delta_det_row, det_channel_offset,
           det_row_offset, source_detector_dist, source_iso_dist,
           angle_start_deg, angle_end_deg, elevation_deg, helical_travel,
           num_x_translations, num_z_translations, x_spacing, z_spacing,
           view_index, show_trajectory, zoom_to_volume, show_reference,
           compare_enabled, compare_text, recon_rows, recon_cols,
           recon_slices):
    """Draw one figure and return it with a short status.

    Every argument is a page control's value, in the order the page lists them.
    An invalid set of values raises ``gr.Error``, which the page shows as a
    message instead of a figure.  Under Pyodide the invalid set is reported in
    the status block instead, and the figure keeps its last value, because
    Gradio-Lite shows every exception a handler raises in a window with the
    Python traceback in front of the page.

    Returns:
        (Figure or dict, str): the matplotlib figure the plot component shows,
        or ``gr.update()`` to keep the figure it has, and the markdown of the
        status block.
    """
    started = time.perf_counter()
    try:
        recon_shape = _recon_shape_override(recon_rows, recon_cols,
                                            recon_slices)
        params, kind = scene_parameters(
            geometry, num_views, num_det_rows, num_det_channels,
            delta_det_channel, delta_det_row, det_channel_offset,
            det_row_offset, source_detector_dist, source_iso_dist,
            angle_start_deg, angle_end_deg, elevation_deg, helical_travel,
            num_x_translations, num_z_translations, x_spacing, z_spacing,
            recon_shape=recon_shape)
        overrides = (_parse_overrides(compare_text, kind)
                     if compare_enabled else None)
        scene = GeometryScene(params, kind)
        index = _checked_view_index(view_index, scene.num_views)
        # Close the figures of earlier calls: each render builds a new one, and
        # pyplot keeps every figure it makes until something closes it.
        plt.close('all')
        with matplotlib.rc_context({'figure.dpi': WEB_DPI}):
            figure = GeometryFigure(
                scene, view_index=index,
                show_trajectory=bool(show_trajectory),
                zoom='volume' if zoom_to_volume else 'scan',
                show_reference=bool(show_reference), compare=overrides,
                figsize=WEB_FIGSIZE, widgets=False, blit=False)
    except gr.Error as problem:
        if running_in_pyodide():
            return gr.update(), _refusal_markdown(problem.message)
        raise
    except (ValueError, TypeError, KeyError) as problem:
        if running_in_pyodide():
            return gr.update(), _refusal_markdown(str(problem))
        raise gr.Error(str(problem))
    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    return figure.figure, _status_markdown(scene, elapsed_ms)


def _refusal_markdown(message):
    """The status block when the values cannot be drawn: the reason, and a
    note that the figure shown is the last one drawn."""
    return (f'**Nothing drawn.** {message}  The figure above is the last one '
            'drawn.')


def _status_markdown(scene, elapsed_ms):
    """The status block: the render time, where the figure was drawn, and the
    two quantities the figure's text panel does not print.

    The figure's own text panel lists every derived quantity but two: the
    drawing distance, which is the distance used where a position is a drawing
    choice, and the sentence about the fan and cone angles.
    """
    quantities = scene.derived_quantities()
    where = 'in this browser' if running_in_pyodide() else 'on the server'
    lines = [
        f'**Rendered in {elapsed_ms:.0f} ms** {where}.  '
        f'Every control change draws the figure again.',
        '',
        f'- drawing distance: {quantities["drawing_distance"]:.3g} ALU',
        f'- about the angles: {quantities["angle_note"]}',
    ]
    return '\n'.join(lines)


# ── reading the page's values ────────────────────────────────────────────────

def _counted(value, name, largest):
    """One count from the page, as an integer between one and ``largest``."""
    if value is None:
        raise gr.Error(f'{name.capitalize()} is empty; give a whole number of '
                       'at least one.')
    try:
        count = int(round(float(value)))
    except (TypeError, ValueError):
        raise gr.Error(f'{name.capitalize()} must be a whole number; got '
                       f'{value!r}.')
    if count < 1:
        raise gr.Error(f'{name.capitalize()} must be at least one; got '
                       f'{count}.')
    if count > largest:
        raise gr.Error(f'{name.capitalize()} must be at most {largest}; got '
                       f'{count}.')
    return count


def _checked_view_index(view_index, num_views):
    """The view to draw, held inside the scan.

    The slider's maximum follows the view count, and a change of the count
    reaches the slider a moment after it reaches this function, so an index
    past the end is clamped rather than refused.
    """
    try:
        index = int(round(float(view_index)))
    except (TypeError, ValueError):
        index = 0
    return max(0, min(index, num_views - 1))


def _recon_shape_override(recon_rows, recon_cols, recon_slices):
    """The reconstruction shape from the advanced section, or None.

    All three numbers must be given together, because two numbers do not name
    a shape.  A blank box counts as not given.

    A zero counts as blank as well.  Gradio 5.45, which is the release the
    static packaging runs, sends zero for an empty integer number box, where
    Gradio 6 sends nothing at all.  Reading zero as blank makes the two
    releases behave the same, and it costs nothing: a reconstruction of zero
    rows is not a reconstruction, so no user means it.
    """
    entries = (recon_rows, recon_cols, recon_slices)
    given = [entry for entry in entries if not _is_blank_count(entry)]
    if not given:
        return None
    if len(given) != 3:
        raise gr.Error('Give all three reconstruction counts, or leave all '
                       'three blank for the automatic shape.')
    shape = []
    for entry, name in zip(entries, ('rows', 'columns', 'slices')):
        shape.append(_counted(entry, f'the number of recon {name}',
                              MAX_DETECTOR_SIDE))
    return tuple(shape)


def _is_blank_count(entry):
    """Whether a number box holds no count; see
    :func:`_recon_shape_override`."""
    if entry is None or entry == '':
        return True
    try:
        return int(round(float(entry))) == 0
    except (TypeError, ValueError):
        return False


def _parse_overrides(text, kind):
    """The comparison's parameter overrides, read from ``name=value`` lines.

    A blank line and a line starting with ``#`` are skipped.  A value is read
    as a Python literal, so ``13.0``, ``True``, and ``(64, 64, 32)`` all work.

    Args:
        text (str): the box's contents.
        kind (str): the geometry kind, whose parameter names are the names
            allowed here.

    Returns:
        dict or None: the overrides, or None when the box holds nothing.
    """
    allowed = required_parameter_names(kind)
    overrides = {}
    for number, line in enumerate(str(text or '').splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            raise gr.Error(f'Line {number} of the comparison overrides is not '
                           f'a name=value line: {line!r}.')
        name, _, value_text = line.partition('=')
        name, value_text = name.strip(), value_text.strip()
        if name not in allowed:
            raise gr.Error(f'{name!r} is not a parameter of a {kind} '
                           f'geometry.  The names this geometry uses are '
                           f'{list(allowed)}.')
        try:
            value = ast.literal_eval(value_text)
        except (ValueError, SyntaxError):
            raise gr.Error(f'Cannot read the value on line {number} of the '
                           f'comparison overrides: {value_text!r}.  Write a '
                           'number, True or False, or a tuple such as '
                           '(64, 64, 32).')
        overrides[name] = value
    if not overrides:
        raise gr.Error('The comparison is on but no override was given.  '
                       'Write a line such as '
                       f'{EXAMPLE_OVERRIDE_TEXT} or turn the comparison off.')
    return overrides


# ── the controls that other controls change ──────────────────────────────────

def view_maximum(geometry, num_views, num_x_translations,
                 num_z_translations, view_index):
    """An update that puts the slider's maximum at the last view index and
    keeps the slider's value inside the new range.

    The value is clamped here and not only in :func:`render`, because Gradio
    checks a slider's value against its maximum before any function runs, and
    a value past the new maximum is refused there with an error.  A count that
    is not a positive number leaves the slider as it is, and the render then
    reports the count.
    """
    kind = KIND_OF_CHOICE.get(geometry, 'cone')
    try:
        if kind == 'translation':
            count = int(num_x_translations) * int(num_z_translations)
        else:
            count = int(num_views)
    except (TypeError, ValueError):
        return gr.update()
    if count < 1:
        return gr.update()
    count = min(count, MAX_NUM_VIEWS)
    try:
        index = int(round(float(view_index)))
    except (TypeError, ValueError):
        index = 0
    return gr.update(maximum=count - 1, value=max(0, min(index, count - 1)))


def group_visibility(geometry):
    """Updates that show each geometry's own controls and hide the others.

    The order is the one :data:`GEOMETRY_GROUPS` gives: the two distances, the
    multiaxis elevation, the helical travel, and the translation grid.
    """
    kind = KIND_OF_CHOICE.get(geometry, 'cone')
    return (gr.update(visible=kind in ('cone', 'translation')),
            gr.update(visible=kind == 'multiaxis'),
            gr.update(visible=geometry == 'cone helical'),
            gr.update(visible=kind == 'translation'),
            gr.update(visible=kind != 'translation'))


# ── the page ─────────────────────────────────────────────────────────────────

HEADER = """\
# mbirtorch geometry viewer

Choose a geometry and set the scan.  The figure shows a 3D view, a top view of
the xy plane, a side view of the yz plane, the detector face in row and channel
index, and the derived numbers.  Every panel draws negative z at the top, and
the top and side views draw y to the left, so the source of the view at angle 0
is on the left and its detector is on the right.
"""


def build_page():
    """Build the Blocks page and return it.

    Returns:
        gr.Blocks: the page, not launched.
    """
    with gr.Blocks(title='mbirtorch geometry viewer') as page:
        gr.Markdown(HEADER)
        with gr.Row():
            with gr.Column(scale=1):
                geometry = gr.Dropdown(list(GEOMETRY_CHOICES), value='cone',
                                       label='geometry')
                with gr.Accordion('the scan', open=True):
                    # The view count and the angular range belong to the
                    # five geometries whose views are angles.  A translation
                    # scan's views are its positions, so both are hidden
                    # there and the grid below takes their place.
                    with gr.Column(visible=True) as views_group:
                        num_views = gr.Number(
                            value=DEFAULT_NUM_VIEWS, precision=0,
                            label='number of views')
                        with gr.Row():
                            angle_start = gr.Number(
                                value=0.0, label='first angle (deg)',
                                min_width=PAIR_MIN_WIDTH)
                            angle_end = gr.Number(
                                value=360.0, label='last angle (deg)',
                                min_width=PAIR_MIN_WIDTH)
                    with gr.Column(visible=False) as elevation_group:
                        elevation_deg = gr.Number(
                            value=DEFAULT_ELEVATION_DEG,
                            label='elevation (deg)',
                            info='Held at this value in every view.')
                    with gr.Column(visible=False) as helical_group:
                        helical_travel = gr.Number(
                            value=DEFAULT_HELICAL_TRAVEL,
                            label='helical travel (ALU)',
                            info='Spread evenly over the views.')
                    with gr.Column(visible=False) as translation_group:
                        with gr.Row():
                            num_x_translations = gr.Number(
                                value=DEFAULT_NUM_X_TRANSLATIONS, precision=0,
                                label='x translations',
                                min_width=PAIR_MIN_WIDTH)
                            num_z_translations = gr.Number(
                                value=DEFAULT_NUM_Z_TRANSLATIONS, precision=0,
                                label='z translations',
                                min_width=PAIR_MIN_WIDTH)
                        with gr.Row():
                            x_spacing = gr.Number(
                                value=DEFAULT_X_SPACING,
                                label='x spacing (ALU)',
                                min_width=PAIR_MIN_WIDTH)
                            z_spacing = gr.Number(
                                value=DEFAULT_Z_SPACING,
                                label='z spacing (ALU)',
                                min_width=PAIR_MIN_WIDTH)
                    with gr.Column(visible=True) as distance_group:
                        with gr.Row():
                            source_detector_dist = gr.Number(
                                value=DEFAULT_SOURCE_DETECTOR_DIST,
                                label='source-detector distance (ALU)',
                                min_width=PAIR_MIN_WIDTH)
                            source_iso_dist = gr.Number(
                                value=DEFAULT_SOURCE_ISO_DIST,
                                label='source-iso distance (ALU)',
                                min_width=PAIR_MIN_WIDTH)
                with gr.Accordion('the detector', open=True):
                    with gr.Row():
                        num_det_rows = gr.Number(
                            value=DEFAULT_NUM_DET_ROWS, precision=0,
                            label='detector rows', min_width=PAIR_MIN_WIDTH)
                        num_det_channels = gr.Number(
                            value=DEFAULT_NUM_DET_CHANNELS, precision=0,
                            label='detector channels',
                            min_width=PAIR_MIN_WIDTH)
                    with gr.Row():
                        delta_det_channel = gr.Number(
                            value=1.0, label='channel pitch (ALU)',
                            min_width=PAIR_MIN_WIDTH)
                        delta_det_row = gr.Number(
                            value=1.0, label='row pitch (ALU)',
                            min_width=PAIR_MIN_WIDTH)
                    with gr.Row():
                        det_channel_offset = gr.Number(
                            value=0.0, label='channel offset (ALU)',
                            min_width=PAIR_MIN_WIDTH)
                        det_row_offset = gr.Number(
                            value=0.0, label='row offset (ALU)',
                            min_width=PAIR_MIN_WIDTH)
                with gr.Accordion('compare with a second geometry',
                                  open=False):
                    compare_enabled = gr.Checkbox(
                        value=False, label='draw the comparison')
                    compare_text = gr.Textbox(
                        value=EXAMPLE_OVERRIDE_TEXT, lines=3,
                        label='parameter overrides, one name=value per line',
                        info='The second geometry is drawn dashed, and the '
                             'text panel lists what differs.')
                with gr.Accordion('advanced', open=False):
                    gr.Markdown('Leave these blank for the reconstruction '
                                'shape the model would choose.')
                    with gr.Row():
                        recon_rows = gr.Number(value=None, precision=0,
                                               label='rows', min_width=80)
                        recon_cols = gr.Number(value=None, precision=0,
                                               label='columns', min_width=80)
                        recon_slices = gr.Number(value=None, precision=0,
                                                 label='slices', min_width=80)
            with gr.Column(scale=3):
                plot = gr.Plot(label='the geometry', format=PLOT_FORMAT)
                view_index = gr.Slider(
                    minimum=0, maximum=DEFAULT_NUM_VIEWS - 1, step=1,
                    value=0, label='view')
                with gr.Row():
                    show_trajectory = gr.Checkbox(value=False,
                                                  label='source path')
                    zoom_to_volume = gr.Checkbox(value=False,
                                                 label='3D zoom to volume')
                    show_reference = gr.Checkbox(value=True,
                                                 label='angle-0 reference')
                status = gr.Markdown('')

        # The render inputs, in the order render() takes them.
        inputs = [geometry, num_views, num_det_rows, num_det_channels,
                  delta_det_channel, delta_det_row, det_channel_offset,
                  det_row_offset, source_detector_dist, source_iso_dist,
                  angle_start, angle_end, elevation_deg, helical_travel,
                  num_x_translations, num_z_translations, x_spacing, z_spacing,
                  view_index, show_trajectory, zoom_to_volume, show_reference,
                  compare_enabled, compare_text, recon_rows, recon_cols,
                  recon_slices]
        outputs = [plot, status]

        # The geometry choice shows and hides the controls that belong to one
        # geometry.  A control that sets the view count first updates the
        # slider's range and value, and the figure is drawn after that update,
        # so that the slider value the drawing reads is inside the new range.
        # Every other control draws the figure directly.
        geometry.change(group_visibility, geometry,
                        [distance_group, elevation_group, helical_group,
                         translation_group, views_group])
        view_count_inputs = [geometry, num_views, num_x_translations,
                             num_z_translations]
        for control in view_count_inputs:
            control.change(view_maximum, view_count_inputs + [view_index],
                           view_index).then(render, inputs, outputs)
        for control in inputs:
            if not any(control is counted for counted in view_count_inputs):
                control.change(render, inputs, outputs)

        # Draw the default geometry when the page opens.
        page.load(render, inputs, outputs)
    return page


#: The page, built at import time so that a test can launch it.
demo = build_page()


def running_in_pyodide():
    """Whether this module is running in a browser, under Pyodide.

    Gradio-Lite shows the app that ``launch`` registers, and it runs this file
    to get one.  Whether it runs the file as the program depends on the
    Gradio-Lite release, so the launch below does not depend on that alone.
    """
    return sys.platform == 'emscripten' or 'pyodide' in sys.modules


if __name__ == '__main__' or running_in_pyodide():
    demo.launch()
