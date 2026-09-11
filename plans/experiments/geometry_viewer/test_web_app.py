"""Tests for the web packaging of the geometry viewer.

The tests cover three things.  The first is the render function of ``web/app.py``:
it must return a matplotlib figure for each of the six geometries the page
offers, for a comparison, and for a reconstruction shape given by hand, and it
must raise ``gr.Error`` with a plain sentence for a bad number and for an
override line it cannot read.  Its third return value is the comparison table,
which is shown only while a comparison is drawn.  The second is the build
script: every Python file inside ``web/lite/index.html`` must decode back to
its source byte for byte, and both Space directories must hold the files a
Hugging Face Space needs.  The third is the whole page: the test launches the
Gradio app, opens it in a headless Chromium through Playwright, waits for the
plot, moves the view slider, and checks that the plot's image changed.

Nothing here imports mbirtorch or torch, because the web app does not.  The
test that pins ``geometry_defaults.py`` against the real models is
``test_geometry_defaults.py``.

Run:
    cd plans/experiments/geometry_viewer
    MPLBACKEND=Agg python -m pytest -q test_web_app.py

The end-to-end test needs Playwright and a Chromium.  It skips with a message
when either is missing, and it writes a full-page screenshot to
``figures/gv5_space_screenshot.png``.
"""

import html.parser
import os
import re
import socket
import sys
import textwrap
import time
import types

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')   # the tests open no window
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'web'))

import gradio as gr  # noqa: E402

import app  # noqa: E402
import gv5_build_web as build_web  # noqa: E402

#: Where the end-to-end screenshot is written.  The figures directory is
#: ignored by git, as every image in this plan directory is.
SCREENSHOT = os.path.join(HERE, 'figures', 'gv5_space_screenshot.png')

#: How long the end-to-end test waits for the page and for a redraw, in
#: milliseconds.  The first render happens while the page loads, and a render
#: takes about half a second on this machine, so both bounds are generous.
PAGE_TIMEOUT_MS = 120_000
REDRAW_TIMEOUT_MS = 60_000

#: Where a preinstalled Chromium may sit when Playwright's own default path
#: does not find one.  The first entry that exists is used.
CHROMIUM_CANDIDATES = (
    '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    '/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell',
)

#: The scan the timing test measures: the page's default cone geometry, and an
#: 1800-view helical scan, which is the size ``gv4_timing.py`` uses.
TIMED_HELICAL_VIEWS = 1800

#: The largest render time the timing test accepts, in seconds.  The point of
#: the test is the number it prints; this bound only catches a render that has
#: stopped being interactive at all.
RENDER_TIME_CEILING_S = 60.0


def render_arguments(**changes):
    """The page's default control values, as ``render`` takes them.

    Args:
        **changes: values to replace, by the name ``render`` uses.

    Returns:
        list: the positional arguments of :func:`app.render`.
    """
    values = dict(
        geometry='cone',
        num_views=app.DEFAULT_NUM_VIEWS,
        num_det_rows=app.DEFAULT_NUM_DET_ROWS,
        num_det_channels=app.DEFAULT_NUM_DET_CHANNELS,
        delta_det_channel=1.0,
        delta_det_row=1.0,
        det_channel_offset=0.0,
        det_row_offset=0.0,
        source_detector_dist=app.DEFAULT_SOURCE_DETECTOR_DIST,
        source_iso_dist=app.DEFAULT_SOURCE_ISO_DIST,
        angle_start_deg=0.0,
        angle_end_deg=360.0,
        elevation_deg=app.DEFAULT_ELEVATION_DEG,
        helical_travel=app.DEFAULT_HELICAL_TRAVEL,
        num_x_translations=app.DEFAULT_NUM_X_TRANSLATIONS,
        num_z_translations=app.DEFAULT_NUM_Z_TRANSLATIONS,
        x_spacing=app.DEFAULT_X_SPACING,
        z_spacing=app.DEFAULT_Z_SPACING,
        view_index=0,
        show_trajectory=False,
        zoom_to_volume=False,
        show_reference=True,
        camera_elevation_deg=app.CAMERA_ELEVATION_DEG,
        camera_azimuth_deg=app.CAMERA_AZIMUTH_DEG,
        compare_enabled=False,
        compare_text='',
        recon_rows=None,
        recon_cols=None,
        recon_slices=None,
    )
    unknown = set(changes) - set(values)
    assert not unknown, f'render takes no argument named {sorted(unknown)}.'
    values.update(changes)
    return list(values.values())


# ── the render function ──────────────────────────────────────────────────────

@pytest.mark.parametrize('geometry', app.GEOMETRY_CHOICES)
def test_render_returns_a_figure(geometry):
    """Each geometry renders to a matplotlib figure at the web figure size.

    No comparison is asked for here, so the third output hides the comparison
    plot and carries no figure.
    """
    figure, status, comparison = app.render(
        *render_arguments(geometry=geometry, view_index=2))
    assert isinstance(figure, plt.Figure)
    assert tuple(figure.get_size_inches()) == app.WEB_FIGSIZE
    assert figure.dpi == app.WEB_DPI
    assert 'Rendered in' in status
    assert comparison['visible'] is False and 'value' not in comparison
    plt.close(figure)


def test_render_draws_a_comparison():
    """A comparison override draws a second geometry.

    The figure then holds the comparison's own artists, which the viewer keeps
    in a list of its own, so the check is that the list is not empty.  The
    third output shows the comparison plot and carries the viewer's second
    figure, which is the table of every difference.
    """
    arguments = render_arguments(compare_enabled=True,
                                 compare_text='det_channel_offset=13.0')
    figure, _, comparison = app.render(*arguments)
    assert isinstance(figure, plt.Figure)
    # Two more axes lines exist in the comparison case than without it; the
    # text panel's comparison block is the visible difference.
    texts = [artist.get_text() for artist in figure.findobj(plt.Text)]
    assert any('det_channel_offset' in text for text in texts), (
        'the text panel should list the parameter the comparison changed')

    assert comparison['visible'] is True
    table = comparison['value']
    assert isinstance(table, plt.Figure) and table is not figure
    table_texts = [artist.get_text() for artist in table.findobj(plt.Text)]
    assert any('det_channel_offset' in text for text in table_texts), (
        'the comparison table should hold the parameter that changed')
    plt.close(figure)
    plt.close(table)


def test_render_takes_a_recon_shape():
    """The advanced section's three counts replace the automatic shape."""
    figure, _, _ = app.render(*render_arguments(recon_rows=40, recon_cols=44,
                                                recon_slices=12))
    titles = [artist.get_text() for artist in figure.findobj(plt.Text)]
    assert any('(40, 44, 12)' in text for text in titles), (
        'the figure should report the reconstruction shape it was given')
    plt.close(figure)


def test_under_pyodide_a_bad_value_is_reported_in_the_status(monkeypatch):
    """Under Pyodide a bad value keeps the figure and explains itself in the
    status block, because Gradio-Lite shows a raised exception as a traceback
    window in front of the page.  A ``pyodide`` module in ``sys.modules`` is
    how the app tells that it runs there."""
    monkeypatch.setitem(sys.modules, 'pyodide', types.ModuleType('pyodide'))
    figure, status, comparison = app.render(*render_arguments(num_views=0))
    assert isinstance(figure, dict) and figure.get('__type__') == 'update'
    assert 'Nothing drawn' in status and 'at least one' in status
    # The comparison plot keeps whatever it holds, as the figure does.
    assert comparison == gr.update()


def test_the_camera_fields_turn_the_3d_panel():
    """The two camera numbers reach the 3D panel, and a blank box keeps the
    viewer's default.  The 3D axes come first in the figure's axes list.

    The default is checked against the viewer's own constant and not against
    a name of this module, because the module has a ``DEFAULT_ELEVATION_DEG``
    of its own, the multiaxis scan's elevation, and a first version of the
    camera fields took that angle for the camera's default without any test
    noticing.
    """
    import geometry_viewer
    figure, _, _ = app.render(*render_arguments(camera_elevation_deg=10.0,
                                               camera_azimuth_deg=30.0))
    axes_3d = figure.axes[0]
    assert axes_3d.elev == pytest.approx(10.0)
    assert axes_3d.azim == pytest.approx(30.0)
    plt.close(figure)
    figure, _, _ = app.render(*render_arguments(camera_elevation_deg=None,
                                                camera_azimuth_deg=None))
    assert figure.axes[0].elev == pytest.approx(
        geometry_viewer.DEFAULT_ELEVATION_DEG)
    assert figure.axes[0].azim == pytest.approx(
        geometry_viewer.DEFAULT_AZIMUTH_DEG)
    assert app.CAMERA_ELEVATION_DEG == geometry_viewer.DEFAULT_ELEVATION_DEG
    assert app.CAMERA_ELEVATION_DEG != app.DEFAULT_ELEVATION_DEG
    plt.close(figure)


def test_render_refuses_zero_views():
    """A view count of zero raises gr.Error with a plain sentence."""
    with pytest.raises(gr.Error) as raised:
        app.render(*render_arguments(num_views=0))
    assert 'at least one' in str(raised.value)


def test_render_refuses_an_unreadable_override():
    """An override line that is not name=value raises gr.Error."""
    with pytest.raises(gr.Error) as raised:
        app.render(*render_arguments(compare_enabled=True,
                                     compare_text='det_channel_offset 13.0'))
    assert 'name=value' in str(raised.value)

    with pytest.raises(gr.Error) as raised:
        app.render(*render_arguments(compare_enabled=True,
                                     compare_text='det_channel_offset=wide'))
    assert 'Cannot read the value' in str(raised.value)

    with pytest.raises(gr.Error) as raised:
        app.render(*render_arguments(compare_enabled=True,
                                     compare_text='pitch=1.0'))
    assert 'not a parameter' in str(raised.value)


def test_a_blank_recon_shape_is_the_automatic_one():
    """Blank reconstruction counts keep the automatic shape, and so do zeros.

    Gradio 5.45 sends zero for an empty integer number box, and Gradio 6 sends
    nothing, so both have to mean the automatic shape.  The static packaging
    runs the older release, where the page would otherwise refuse to draw
    anything until a user filled the advanced section in.
    """
    automatic, _, _ = app.render(*render_arguments())
    blank_title = automatic.get_suptitle()
    plt.close(automatic)
    for entries in (dict(), dict(recon_rows=0, recon_cols=0, recon_slices=0)):
        figure, _, _ = app.render(*render_arguments(**entries))
        assert figure.get_suptitle() == blank_title
        plt.close(figure)


def test_render_refuses_a_partial_recon_shape():
    """Two of the three reconstruction counts do not name a shape."""
    with pytest.raises(gr.Error) as raised:
        app.render(*render_arguments(recon_rows=40, recon_cols=44))
    assert 'all three' in str(raised.value)


def test_the_slider_maximum_follows_the_view_count():
    """The view slider's maximum is the last view index of the scan, and its
    value is kept inside the new range.

    The translation geometry counts its views from its grid, so its maximum
    comes from the two counts and not from the view number.  A count below one
    leaves the slider alone, and the render reports the count.
    """
    update = app.view_maximum('cone', 240, 5, 3, 7)
    assert update['maximum'] == 239 and update['value'] == 7
    update = app.view_maximum('translation', 240, 5, 3, 40)
    assert update['maximum'] == 14 and update['value'] == 14
    assert 'maximum' not in app.view_maximum('cone', 0, 5, 3, 7)


def test_group_visibility_matches_the_geometry():
    """Each geometry shows its own controls and hides the others."""
    distances, elevation, helical, translation, views = app.group_visibility(
        'cone helical')
    assert distances['visible'] and helical['visible'] and views['visible']
    assert not elevation['visible'] and not translation['visible']

    distances, elevation, helical, translation, views = app.group_visibility(
        'translation')
    assert distances['visible'] and translation['visible']
    assert not views['visible'] and not helical['visible']

    distances, elevation, helical, translation, views = app.group_visibility(
        'multiaxis')
    assert elevation['visible'] and views['visible']
    assert not distances['visible'] and not translation['visible']


# ── the build script ─────────────────────────────────────────────────────────

class GradioFileReader(html.parser.HTMLParser):
    """Collect the name and text content of every ``<gradio-file>`` element."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.files = {}
        self.requirements = []
        self._name = None
        self._in_requirements = False
        self._parts = []

    def handle_starttag(self, tag, attributes):
        if tag == 'gradio-file':
            self._name = dict(attributes).get('name')
            self._parts = []
        elif tag == 'gradio-requirements':
            self._in_requirements = True
            self._parts = []

    def handle_endtag(self, tag):
        if tag == 'gradio-file' and self._name is not None:
            self.files[self._name] = ''.join(self._parts)
            self._name = None
        elif tag == 'gradio-requirements':
            self.requirements = ''.join(self._parts).split()
            self._in_requirements = False
        self._parts = []

    def handle_data(self, data):
        if self._name is not None or self._in_requirements:
            self._parts.append(data)


@pytest.fixture(scope='module')
def built():
    """Run the build script once and return the parsed static page.

    Returns:
        (str, GradioFileReader): the text of ``index.html`` and its parser.
    """
    build_web.build(verbose=False)
    with open(os.path.join(build_web.LITE, 'index.html'), 'r',
              encoding='utf-8') as handle:
        text = handle.read()
    reader = GradioFileReader()
    reader.feed(text)
    return text, reader


def test_every_module_round_trips_through_the_static_page(built):
    """Each Python file inside index.html decodes back to its source.

    The element's text content is the source with one newline in front of it,
    which is the newline that separates the source from the opening tag.
    """
    _, reader = built
    assert set(reader.files) == set(build_web.MODULE_FILES)
    for name in build_web.MODULE_FILES:
        source = build_web.read_source(name)
        assert reader.files[name] == '\n' + source, (
            f'{name} does not decode back to its source')
        # A file that decodes to its source is also a file Python can read,
        # which is what the browser will do with it.
        compile(reader.files[name], name, 'exec')


def test_the_static_page_is_pinned_and_asks_for_matplotlib(built):
    """The page loads one Gradio-Lite release and one extra package."""
    text, reader = built
    assert build_web.LITE_SCRIPT in text
    assert build_web.LITE_STYLESHEET in text
    assert '@gradio/lite@5.45.0' in build_web.LITE_SCRIPT
    assert reader.requirements == ['matplotlib']
    assert 'entrypoint' in text.split('<gradio-file name="app.py"')[1][:20]


def test_both_packagings_hold_the_files_a_space_needs(built):
    """The two directories hold their files, and both READMEs carry the front
    matter the site reads."""
    for name in build_web.MODULE_FILES + build_web.WEB_FILES + ('icon.png',):
        path = os.path.join(build_web.SPACE, name)
        assert os.path.getsize(path) > 0, f'{path} is missing or empty'
    for name in ('index.html', 'README.md', 'icon.png'):
        path = os.path.join(build_web.LITE, name)
        assert os.path.getsize(path) > 0, f'{path} is missing or empty'

    for directory, sdk, app_file in ((build_web.SPACE, 'gradio', 'app.py'),
                                     (build_web.LITE, 'static',
                                      'index.html')):
        with open(os.path.join(directory, 'README.md'), 'r',
                  encoding='utf-8') as handle:
            readme = handle.read()
        front_matter = readme.split('---')[1]
        for entry in (f'sdk: {sdk}', f'app_file: {app_file}',
                      'license: bsd-3-clause', 'title:', 'emoji:',
                      'colorFrom:', 'colorTo:', 'pinned: false',
                      'short_description:'):
            assert entry in front_matter, f'{directory} README lacks {entry}'
        description = [line for line in front_matter.splitlines()
                       if line.startswith('short_description:')][0]
        assert len(description.split(':', 1)[1].strip(' "')) < 80


def test_the_static_page_holds_the_runtime_dependencies_at_fixed_versions(
        built):
    """The pin file names exact versions, the page's shim carries every one of
    them, and the Python the shim runs in the worker compiles.

    The runtime resolves gradio's dependencies from PyPI when the page loads,
    and a moved dependency stopped the page three times before the pins were
    added; ``gv5_web_findings.md`` records each one.
    """
    text, _ = built
    pins = build_web.read_lite_pins()
    for name in ('huggingface-hub', 'filelock', 'anyio'):
        assert name in pins, f'{name} is not pinned'
    for name, specifier in pins.items():
        assert re.fullmatch(r'==\d+(\.\d+)*', specifier), (
            f'{name} is not held at one version: {specifier}')
    compile(build_web.lite_pin_patch(pins), 'lite_pin_patch', 'exec')
    # The page carries the patches as JSON.  The names survive that
    # unchanged, and the quotes of the two anchors are escaped once.
    for name in pins:
        assert name in text
    for anchor in (build_web.LITE_WHEEL_INSTALL, build_web.LITE_RUN_SYNC_MOCK):
        assert anchor.replace('"', '\\"') in text
    assert 'Lite runtime shim' in text


#: The guide for other Thingy builders, which carries a copy of the shim.
GUIDE = os.path.join(HERE, '..', '..', 'features', 'geometry_viewer',
                     'gradio_lite_space_guide.md')


def test_the_guide_carries_the_same_shim_as_the_page():
    """The shim pasted into the guide is the one the page is built with.

    The guide is what other builders copy, so its copy must not drift from the
    build.  The guide holds the shim without the four spaces of indentation
    the page gives it.
    """
    with open(GUIDE, 'r', encoding='utf-8') as handle:
        guide = handle.read()
    start = guide.index('```html\n<script>')
    end = guide.index('</script>\n```', start) + len('</script>')
    in_guide = guide[start + len('```html\n'):end]
    built = textwrap.dedent(build_web.lite_shim(build_web.read_lite_pins()))
    assert in_guide == built.rstrip('\n')


def test_the_icon_is_a_square_tile(built):
    """The icon is 400 by 400 pixels and is not blank."""
    for directory in (build_web.LITE, build_web.SPACE):
        image = plt.imread(os.path.join(directory, 'icon.png'))
        assert image.shape[0] == image.shape[1] == build_web.ICON_PIXELS
        # The tile is drawn on a light background with dark lines on it, so
        # its darkest pixel is much darker than its background.
        assert float(np.min(image[:, :, :3])) < 0.4


# ── the whole page, in a browser ─────────────────────────────────────────────

def free_port():
    """A port no one is listening on."""
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return int(probe.getsockname()[1])


def chromium_executable():
    """The path of a preinstalled Chromium, or None to use Playwright's own.

    Playwright looks for a browser under a version-numbered directory of its
    own choosing.  A machine that has a Chromium installed under another
    version needs to be told where it is.
    """
    for candidate in CHROMIUM_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def test_the_page_redraws_in_a_browser():
    """Launch the app, open it in Chromium, move the slider, see a new image.

    What this test establishes.  The render function returning a figure is not
    the same as the page working: the plot component has to send the figure to
    the browser, and a control change has to reach the server and come back as
    a new image.  This test checks that whole path.  It also writes a full-page
    screenshot, which is what a reader looks at to check the layout.
    """
    playwright_module = pytest.importorskip(
        'playwright.sync_api',
        reason='Playwright is not installed, so the page cannot be opened')
    executable = chromium_executable()

    port = free_port()
    app.demo.launch(prevent_thread_lock=True, server_port=port, share=False,
                    quiet=True)
    try:
        with playwright_module.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(
                    headless=True, executable_path=executable,
                    args=['--no-sandbox'])
            except Exception as problem:            # pragma: no cover
                pytest.skip(f'No Chromium to open the page with: {problem}')
            try:
                page = browser.new_page(viewport={'width': 1500,
                                                  'height': 1150})
                page.goto(f'http://127.0.0.1:{port}', wait_until='networkidle',
                          timeout=PAGE_TIMEOUT_MS)
                plot = page.locator('img').first
                plot.wait_for(timeout=PAGE_TIMEOUT_MS)
                first = page.evaluate(
                    'document.querySelector("img").src.length')
                assert first > 10_000, ('the plot should be a rendered image, '
                                        f'not {first} characters')
                first_source = page.evaluate(
                    'document.querySelector("img").src')

                # Move the view slider one view on, through the page.
                slider = page.locator('input[type=range]').first
                slider.focus()
                page.keyboard.press('ArrowRight')
                page.wait_for_function(
                    'source => document.querySelector("img").src !== source',
                    arg=first_source, timeout=REDRAW_TIMEOUT_MS)
                second_source = page.evaluate(
                    'document.querySelector("img").src')
                assert second_source != first_source, (
                    'the plot did not change when the view changed')

                os.makedirs(os.path.dirname(SCREENSHOT), exist_ok=True)
                page.screenshot(path=SCREENSHOT, full_page=True)
                assert os.path.getsize(SCREENSHOT) > 10_000
            finally:
                browser.close()
    finally:
        app.demo.close()


# ── how long a render takes ──────────────────────────────────────────────────

def test_render_times_are_measured_and_printed():
    """Measure and print the render time of two scans.

    The two scans are the page's default cone geometry, at 180 views, and an
    1800-view helical scan, which is the size ``gv4_timing.py`` measures a
    slider step on.  A render here builds the whole figure, because the web
    packaging has no partial-redraw path: the browser gets a new image for
    every control change.  ``gv5_web_findings.md`` records the numbers of a
    run.
    """
    times = {}
    for name, arguments in (
            ('default cone, 180 views', render_arguments()),
            (f'helical, {TIMED_HELICAL_VIEWS} views',
             render_arguments(geometry='cone helical',
                              num_views=TIMED_HELICAL_VIEWS,
                              helical_travel=30.0))):
        started = time.perf_counter()
        figure, _, _ = app.render(*arguments)
        times[name] = time.perf_counter() - started
        plt.close(figure)

    print()
    for name, seconds in times.items():
        print(f'render, {name}: {1000.0 * seconds:.0f} ms')
    for name, seconds in times.items():
        assert seconds < RENDER_TIME_CEILING_S, (
            f'{name} took {seconds:.1f} s, which is not an interactive page')
