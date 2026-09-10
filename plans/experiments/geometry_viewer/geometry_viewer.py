"""Matplotlib drawing of an mbirtorch scan geometry, in five panels.

What this module is for.  ``geometry_scene.GeometryScene`` turns a tomography
model's numbers into points and polylines.  This module draws those primitives
so that a user can see the geometry: where the source and the detector sit,
where the reconstruction volume sits, which way the gantry turns, and whether
the volume projects inside the detector.  It owns no geometry of its own.
Every position it draws comes from the scene, and every number it prints comes
from ``GeometryScene.derived_quantities``.

The five panels.  ``GeometryFigure`` builds one figure with a 2 by 3 grid.  The
left column holds a 3D view for orientation.  The middle and right columns hold
a top view in the xy plane, a side view in the yz plane, a detector face in
index coordinates, and a text panel of derived numbers.  The four drawing
panels show one view of the scan at a time, and ``set_view`` moves to another.
The point where the central ray meets the detector is called the detector iso,
which is the name the group's reference slide uses, and the center of the
detector grid is called the detector center.

The picture drawn.  The object stays fixed and the source and the detector
carry the view's motion, which is how a scanner user thinks of a scan.  The
projector code does the opposite, and the scene handles the conversion.

The display convention: negative z is the top of every panel.  Array indices
increase from top to bottom when an array is printed or shown, and the slice
index k runs along +z, so a drawing with -z at the top shows the volume the way
its array is indexed and the way ``imshow`` shows one slice (Greg,
2026-09-10).  The object frame itself does not change.  It stays right-handed
with +z along the increasing slice index, and every primitive the scene reports
is unchanged.  Only the presentation changes: the three 2D panels invert the
axes that the convention turns around, and the 3D panel rolls its camera.  The
module constant :data:`Z_UP_SIGN` is the sign of the object-frame z that points
up the screen, and each panel reads it in one place, so setting it to +1 draws
+z up again.

The screen orientation of each panel.  The reference picture is the group's
slide "Parallel Beam Geometry - top view", from Balke et al., Separable Models
for cone-beam MBIR Reconstruction, 2018.  In it the beam runs from left to
right, with the source on the left and the detector on the right.  Every panel
here follows that picture.  The top view is the xy plane seen from -z, with y
increasing to the left and x increasing downward.  The side view is the yz
plane seen from +x, with y increasing to the left and z increasing downward.
The detector face has the channel index increasing to the right and row 0 at
the top, which is the view from the source toward the detector.  In all three
the source of a view at angle 0 is on the left.

The 3D camera.  The default camera sits 25 degrees above the drawing's top,
which is the -z side, and 40 degrees off the +x axis toward -y, and it is
rolled by 180 degrees so that -z is up.  From there the 3D picture agrees with
the top view.  The y axis runs to the left, the x axis runs down the screen,
-z is up, and the source of a view at angle 0 sits at the back on the left.
The 40 degrees off the x axis matter.  Near an azimuth of zero the eye looks
along x, so the x axis and the z axis both run up and down the screen and the
detector stands edge on.  A detector that is upright is then hard to tell from
one lying flat.  The camera can be dragged with the mouse, and ``set_view``
keeps whatever camera the user has set.

Why that camera is rolled.  The roll is what inverts this panel's z, in place
of a reversed pair of z limits.  Reversed limits invert the z axis too, and
matplotlib draws them by flipping the z axis of the world it projects.  That
flip mirrors the picture.  The depth matplotlib sorts its artists by, and the
faces of the axes box it draws, then belong to an eye on the other side of the
object, and the x and y tick labels move over the panel's title.  The roll
turns the same view upside down instead.  The panel then stays a picture taken
from one viewpoint, and its z ticks read negative at the top.

The widgets.  A slider under the panels steps through the views, and three
toggles sit beside it.  The first turns the source's path over all views on and
off.  The second switches the 3D panel between the whole scan and a close view
of the volume, because a 12 ALU volume drawn to scale in a 200 ALU scan is
twenty pixels wide.  The third turns the angle-0 reference on and off.  The
widgets follow the slice viewer of ``mbirtorch/viewer.py``: an integer-stepped
``Slider`` with ``drawon`` off, and ``CheckButtons`` for the toggles.

The labels.  The source and the detector carry a short text label in the 3D
view, the top view, and the side view.  The detector iso carries one in the 3D
view and the top view, and the detector's pixel-0 marker carries one in the top
view.  Each label names the element it sits beside, which one panel's legend
cannot do for the other panels (Greg, 2026-09-09).  A label moves with the
element it names, so it is a moving artist like that element.  The labels of
one panel are placed so that no two of them overlap and none reaches outside
the panel, which a test checks by measuring them.

The angle-0 reference.  The source and the detector are drawn a second time,
where they sit when the view action is the identity.  That position comes from
``GeometryScene.reference_view``.  These artists do not move with the slider,
so they show the current view against a fixed one, and the dotted arrow through
them is the direction the geometry projects at angle 0.  They are drawn dotted
and partly transparent in the 3D view and the top view, and the third toggle
turns them off.  The top view names them with a caption in its upper left
corner, because the ray they lie on crosses the middle of that panel, where the
labels of the source and the detector are.

How a redraw stays fast.  Every artist is created once and its data is replaced
in place, so a view change calls ``set_data`` and never ``clear``.  The panel
limits are computed from a sample of views and then held, so the ticks, the
grids, the legends, and the text block belong to a background that a view
change does not touch.  A view change restores that background, redraws only
the artists that moved, and blits, which is the same partial-redraw idea the
slice viewer uses.  Anything that changes the background instead, such as a
toggle or a new comparison, repaints the whole figure.  ``gv4_timing.md``
records what the two paths cost on an 1800-view helical scan.

The comparison overlay.  A second geometry can be drawn over the first in one
color with dashed lines, from a second scene, a second model, or a dictionary
of parameter overrides.  The text panel then lists every parameter and derived
quantity that differs.  This is the calibration use: a vendor geometry against
the same geometry with an estimated offset.

Import discipline.  This module imports numpy and the matplotlib base package
at import time, and nothing else.  ``pyplot``, the widgets, and the 3D toolkit
are imported inside :func:`_load_pyplot`, on the first figure construction.
Importing this module therefore never resolves a matplotlib backend and never
touches a GUI toolkit, which is the rule ``mbirtorch/viewer.py`` follows.

Colors.  One color per element, listed in :data:`COLORS`, is used in every
panel where the element appears.  The two index markers are the ones to look at
first: the marker at detector pixel (row 0, channel 0) and the marker at voxel
(0, 0, 0) show the index orientation of the data, and a mirrored channel order
or a wrong offset sign is visible from those two alone.
"""

import textwrap

import numpy as np
import matplotlib  # base package only; no GUI toolkit is touched at import

from geometry_scene import GeometryScene

__all__ = ['GeometryFigure', 'show_geometry', 'COLORS',
           'DEFAULT_ELEVATION_DEG', 'DEFAULT_AZIMUTH_DEG',
           'VOLUME_BOX_EDGES', 'ZOOM_MODES', 'BLIT_BACKENDS',
           'Z_UP_SIGN', 'TOP_PANEL_COLUMNS', 'SIDE_PANEL_COLUMNS',
           'ISO_NAME', 'CENTER_NAME']


# --- names used in every panel ---

#: What the point where the central ray meets the detector is called.  The name
#: is the one the group's reference slide uses, and every panel uses it.
ISO_NAME = 'detector iso'

#: What the center of the detector grid is called.  The two detector offsets
#: move this point away from the detector iso.
CENTER_NAME = 'detector center'


# --- the display convention ---

#: The sign of the object-frame z that points up the screen.  The value -1
#: draws -z at the top of every panel, which is the convention this module
#: follows; see the module docstring.  The value +1 inverts no axis and draws
#: +z up.  Each panel reads this constant in one place: the three 2D panels
#: through :func:`_screen_pair`, which orders a pair of axes limits, and the 3D
#: panel through :func:`_camera_roll_deg`, which rolls its camera.  The
#: geometry itself carries no sign flip.
Z_UP_SIGN = -1


def _z_up_is_negative():
    """Whether -z points up the screen; see :data:`Z_UP_SIGN`."""
    return Z_UP_SIGN < 0


def _top_view_title():
    """The top panel's title, which names the side it is seen from."""
    side = 'left' if _z_up_is_negative() else 'right'
    return ('Top view, the xy plane, seen from -z (the top)\n'
            f'y increases to the {side}')


def _side_view_title():
    """The side panel's title, which names the side it is seen from."""
    return 'Side view, the yz plane, seen from +x'


def _detector_view_title():
    """The first line of the detector panel's title, naming the row order.

    Row 0 at the top is the view from the source toward the detector.  Row 0
    at the bottom, which +1 gives, is the same grid seen from behind the
    detector, so that title names the row order alone.
    """
    if _z_up_is_negative():
        return 'Detector face, seen from the source, row 0 at the top'
    return 'Detector face, row 0 at the bottom'


def _camera_roll_deg():
    """How far the 3D camera is rolled, so that -z is at the top of the panel.

    The 3D panel turns its z axis around with the camera and not with its axes
    limits.  Reversed limits would do it too, but matplotlib draws reversed
    limits by flipping the z axis of the world it projects, and that mirrors
    the picture: the depth it sorts artists by, and the faces of the axes box
    it draws, then belong to an eye on the other side.  A roll of 180 degrees
    turns the same physical view upside down instead, which leaves the picture
    a picture of the geometry and puts the negative z ticks at the top.
    """
    return CAMERA_ROLL_DEG if _z_up_is_negative() else 0.0


def _convention_note():
    """The text panel's sentence naming the display convention."""
    return ('Drawn with -z up.' if _z_up_is_negative()
            else 'Drawn with +z up.')


def _screen_step(delta):
    """Which way a step along a projected panel's axis points on the screen.

    Both projected panels turn both of their axes around under the display
    convention, so one rule serves the horizontal and the vertical axis of
    each.

    Args:
        delta (float): a step in the object coordinate the panel puts on that
            axis.

    Returns:
        int: 1 when the step points to the right or up the screen, and -1 when
        it points to the left or down.  The two answers swap with
        :data:`Z_UP_SIGN`.
    """
    forward = 1 if float(delta) >= 0.0 else -1
    return -forward if _z_up_is_negative() else forward


def _screen_bottom(values):
    """The end of a run of values that a panel draws at the bottom of the
    screen.

    A panel's vertical axis increases downward under the display convention, so
    the largest value is at the bottom.  A label that is to hang below a drawn
    object is placed at this value.
    """
    values = np.asarray(values, dtype=np.float64)
    return float(np.max(values) if _z_up_is_negative() else np.min(values))


def _screen_pair(low, high):
    """One axis's limits, ordered for the display convention.

    Under the convention a panel's axes increase downward or to the left, and
    matplotlib draws such an axis from a pair of limits in decreasing order.
    Which axes those are is stated per panel where the limits are applied.

    Args:
        low, high (float): the axis's limits, in increasing order.

    Returns:
        tuple: the pair, reversed when -z points up the screen.
    """
    if _z_up_is_negative():
        return (float(high), float(low))
    return (float(low), float(high))


#: Which object coordinates the top view puts on its horizontal and its
#: vertical axis, as indices into (x, y, z).  The top view is the xy plane seen
#: from -z, with y across the screen and x down it.
TOP_PANEL_COLUMNS = (1, 0)

#: The same for the side view, which is the yz plane seen from +x, with y
#: across the screen and z down it.
SIDE_PANEL_COLUMNS = (1, 2)

#: Which two of a view's four corner rays bound the plane the top view draws.
#: ``ViewScene.corner_rays`` runs over the detector's corners, and these two
#: sit at the extremes of the channel direction, which is the direction the top
#: view spreads the detector along.
TOP_EDGE_RAYS = (0, 1)

#: The same for the side view, whose plane the two corners at the extremes of
#: the row direction bound.
SIDE_EDGE_RAYS = (0, 3)


# --- appearance ---

#: One color per drawn element.  Every panel uses these, so an element keeps
#: its color across the figure.
COLORS = {
    'source': '#ff7f0e',        # orange
    'detector': '#1f77b4',      # blue
    'rays': '#c0c0c0',          # light gray
    'central_ray': '#111111',   # near black
    'volume': '#9467bd',        # purple
    'ror': '#17becf',           # cyan
    'axis': '#555555',          # gray
    'pixel0': '#e7298a',        # magenta
    'voxel0': '#2ca02c',        # green
    'trajectory': '#8c564b',    # brown
    'overshoot': '#d62728',     # red
    'compare': '#6b8e00',       # dark yellow-green
}

TITLE_FONT_SIZE = 9
LABEL_FONT_SIZE = 8
TICK_FONT_SIZE = 7
ANNOTATION_FONT_SIZE = 7
LEGEND_FONT_SIZE = 6.5
TEXT_PANEL_FONT_SIZE = 7.5
WIDGET_FONT_SIZE = 8

#: Characters per line in the text panel's wrapped sentences.  The panel's
#: aligned "name : value" lines are shorter than this, so this width sets the
#: panel's overall text width.
TEXT_PANEL_WRAP_WIDTH = 52

#: How far to the side a label sits from the point it names, in points.  The
#: vertical gap is chosen per label instead, so that two labels near one point
#: sit on different lines.
LABEL_GAP_POINTS = 6

#: The vertical gap of the labels at the detector iso from that point, in
#: points, and the extra gap the second of them takes so that the two sit on
#: different lines.  One line of the annotation font is about eleven points
#: tall at the sizes this module uses.
ISO_LABEL_GAP_POINTS = 6.0
LABEL_LINE_POINTS = 11.0

SOURCE_MARKER_SIZE = 13
INDEX_MARKER_SIZE = 8
INDEX_MARKER_WIDTH = 1.8

#: Color of the two corner rays that bound a 2D panel's plane.  They are drawn
#: darker than the other two so that the fan and the cone read as outlines.
EMPHASIZED_RAY_COLOR = '#8a8a8a'

#: The dash pattern and line width of every comparison line.  The comparison
#: is drawn dashed so that it can be told from the primary geometry in a black
#: and white print as well as by color.
COMPARE_DASHES = (0, (5.0, 2.0))
COMPARE_LINEWIDTH = 1.4

#: The alpha, the dash pattern, and the line width of the angle-0 reference.
#: The reference sits behind the view drawn and must not be read as part of it,
#: so it is dotted and partly transparent.
REFERENCE_ALPHA = 0.45
REFERENCE_DOTS = (0, (1.0, 2.0))
REFERENCE_LINEWIDTH = 1.2

#: The fraction of the reference central ray that its arrowhead spans.  The
#: head is drawn from a short segment at the detector end, which is how the
#: rotation arc's head is drawn.
REFERENCE_ARROW_FRACTION = 0.06

#: Where the angle-0 reference's label sits in the top view, in that panel's
#: axes coordinates.  It is a caption in the upper left corner rather than a
#: label on the ray; see :meth:`GeometryFigure._create_reference_artists`.
REFERENCE_LABEL_CORNER = (0.015, 0.985)

#: The 3D camera, in degrees.  The eye sits 25 degrees above the drawing's top,
#: which is the -z side, and 40 degrees off the +x axis toward -y.  From there
#: y runs to the left and x runs down the screen, as in the top view, and the
#: source of a view at angle 0 sits at the back on the left.  The 40 degrees
#: off the x axis matter: near an azimuth of 0 the x axis and the z axis both
#: run up and down the screen, and the detector stands edge on, so a panel
#: that is upright is hard to tell from one lying flat.  See the module
#: docstring.
DEFAULT_ELEVATION_DEG = -25.0
DEFAULT_AZIMUTH_DEG = -40.0

#: How far the 3D camera is rolled about its own axis, in degrees, to put -z at
#: the top of the panel.  This is the 3D panel's one reading of Z_UP_SIGN; see
#: :func:`_camera_roll_deg`.
CAMERA_ROLL_DEG = 180.0

#: Points sampled around an ellipse when drawing the region of reconstruction.
ELLIPSE_SAMPLES = 65

#: Points sampled along each ray the 3D panel draws.  A ray is a straight
#: segment and two points would draw it, but the 3D panel clips a line point by
#: point at its axes limits, so a segment whose two ends are both outside the
#: limits disappears.  In the volume zoom every ray has both ends outside, and
#: sampling the segment keeps the part that crosses the panel.
RAY_SAMPLES = 33

#: Points sampled along one projected volume edge on the detector face, used
#: only when the edge crosses the detector's boundary and has to be split into
#: an inside part and an overshooting part.
EDGE_CLIP_SAMPLES = 64

#: Fractional margin added around the data of a 2D panel.
PANEL_MARGIN = 0.10

#: The two states of the 3D panel's zoom control.  ``'scan'`` puts the source,
#: the detector, and the volume in one cube.  ``'volume'`` puts a cube around
#: the volume box alone and lets the axes limits clip the rays and the axis.
ZOOM_MODES = ('scan', 'volume')
DEFAULT_ZOOM = 'scan'

#: Width of the ``'volume'`` zoom cube, as a multiple of the volume's largest
#: extent.  Three extents leave the volume filling the middle third of the
#: panel with room for the rotation axis and the nearest rays around it.
ZOOM_VOLUME_WIDTH_FACTOR = 3.0

#: How many views are sampled when the panel limits are computed.  The limits
#: are then held, so that a view change does not move the ticks and the
#: background can be reused.  A view whose content falls outside the limits
#: makes them grow and forces one full redraw.
LIMIT_SAMPLE_VIEWS = 16

#: At most this many differing entries are listed in the text panel's
#: comparison section, and the font that section uses.  The panel holds about
#: thirty-six lines of its own font size, and the derived quantities and the
#: footer take twenty-seven of them, so a comparison between two unrelated
#: geometries has to be capped or it would run off the panel.  The cap falls
#: further when the panel turns out to be too short for it; see
#: ``_place_text_blocks``.
MAX_COMPARISON_ENTRIES = 6

#: The text panel's font size while a comparison is drawn.  The derived
#: quantities alone nearly fill the panel at ``TEXT_PANEL_FONT_SIZE``, so the
#: whole panel is set smaller to make room for the comparison section.
COMPARING_FONT_SIZE = 6.0

#: The fraction of the text panel's height its three blocks may fill.  The
#: last line's descenders sit below the line the measurement counts, so a
#: block set to the full height reaches a few pixels past the panel.
TEXT_PANEL_FILL = 0.97

#: The smallest font the text panel will shrink to when its blocks are taller
#: than the panel; see ``GeometryFigure._fit_text_font``.  A saved figure's
#: numbers stop being readable below this.  Only the two parallel-type
#: geometries with a comparison drawn reach it: their drawing note is six lines
#: where a cone scan's is two.
TEXT_PANEL_FONT_MINIMUM = 5.0

#: Widget rectangles in figure coordinates, as (left, bottom, width, height).
#: The slider is shorter than it was, because the row now carries a third
#: toggle beside it.
SLIDER_RECT = (0.07, 0.045, 0.38, 0.025)
TRAJECTORY_CHECK_RECT = (0.50, 0.015, 0.12, 0.075)
ZOOM_CHECK_RECT = (0.645, 0.015, 0.15, 0.075)
REFERENCE_CHECK_RECT = (0.815, 0.015, 0.15, 0.075)

#: The top of the panel grid is unchanged; its bottom leaves room for the
#: widget row.
GRID_BOTTOM = 0.115


def _corner_edges():
    """The twelve edges of the volume box, as index pairs into its corners.

    ``GeometryScene.volume_corners`` returns the eight corners in the order of
    three nested sign loops over x, then y, then z, so corner
    ``4 * ix + 2 * iy + iz`` carries sign ``ix`` in x, ``iy`` in y, and
    ``iz`` in z.  Two corners are joined by an edge when exactly one of those
    three signs differs, which is when their indices differ in one bit.
    """
    edges = []
    for first in range(8):
        for second in range(first + 1, 8):
            if bin(first ^ second).count('1') == 1:
                edges.append((first, second))
    return tuple(edges)


#: The twelve edges of the volume box; see :func:`_corner_edges`.
VOLUME_BOX_EDGES = _corner_edges()

#: A walk around the volume box's footprint in the xy plane, as corner
#: indices.  These four corners share the low z sign, so their x and y values
#: are the four combinations of the box's x and y extents.
_XY_FOOTPRINT_WALK = (0, 2, 6, 4, 0)

#: A walk around the volume box's face in the yz plane, as corner indices.
#: These four corners share the low x sign.
_YZ_FACE_WALK = (0, 1, 3, 2, 0)

#: Backends where the partial-redraw (blit) fast path is used.  These are the
#: two ``mbirtorch/viewer.py`` verifies: Agg for headless runs and TkAgg for
#: the interactive sessions the fast path exists for.  Everywhere else a view
#: change repaints the whole figure.
BLIT_BACKENDS = {'agg', 'tkagg'}

# Backends that have no interactive window; show_geometry warns under these.
NONINTERACTIVE_BACKENDS = {'agg', 'pdf', 'ps', 'svg', 'template', 'cairo'}


# Filled in by _load_pyplot on the first figure construction, so that importing
# this module never resolves a matplotlib backend.
plt = None
Poly3DCollection = None
Rectangle = None
FancyArrowPatch = None
Slider = None
CheckButtons = None
IdentityTransform = None
Bbox = None


def _load_pyplot():
    """Import pyplot, the widgets, and the drawing classes on first use."""
    global plt, Poly3DCollection, Rectangle, FancyArrowPatch
    global Slider, CheckButtons, IdentityTransform, Bbox
    if plt is not None:
        return
    import matplotlib.pyplot as _plt
    from mpl_toolkits.mplot3d.art3d import (
        Poly3DCollection as _Poly3DCollection)
    from matplotlib.patches import (Rectangle as _Rectangle,
                                    FancyArrowPatch as _FancyArrowPatch)
    from matplotlib.widgets import (Slider as _Slider,
                                    CheckButtons as _CheckButtons)
    from matplotlib.transforms import (IdentityTransform as _IdentityTransform,
                                       Bbox as _Bbox)
    plt = _plt
    Poly3DCollection = _Poly3DCollection
    Rectangle = _Rectangle
    FancyArrowPatch = _FancyArrowPatch
    Slider = _Slider
    CheckButtons = _CheckButtons
    IdentityTransform = _IdentityTransform
    Bbox = _Bbox


# --- small formatting and geometry-free drawing helpers ---

def _three_figures(value):
    """A number formatted to three significant figures.

    Zero is added to the value so that a negative zero, which several offsets
    produce, prints as ``0`` rather than as ``-0``.
    """
    return f'{float(value) + 0.0:.3g}'


def _as_scene(model_or_scene, **scene_kwargs):
    """A ``GeometryScene``, built from a model when one is given."""
    if isinstance(model_or_scene, GeometryScene):
        if scene_kwargs:
            raise TypeError('Scene options apply only when a model is given; '
                            'a GeometryScene is already built.')
        return model_or_scene
    return GeometryScene.from_model(model_or_scene, **scene_kwargs)


def _ellipse_points(center, semi_axis_x, semi_axis_y, height):
    """Points around an axis-aligned ellipse at one height, (N, 3).

    The region of reconstruction is an elliptic cylinder about the rotation
    axis, and the scene reports it as a center, two semi-axes, and a pair of
    heights.  This turns that description into a polyline.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, ELLIPSE_SAMPLES)
    x = center[0] + semi_axis_x * np.cos(angle)
    y = center[1] + semi_axis_y * np.sin(angle)
    z = np.full_like(angle, float(height))
    return np.stack([x, y, z], axis=1)


def _point_3d(point):
    """One point as three one-element arrays, ready for ``set_data_3d``.

    The 3D line artist reads the shape of its own data when any coordinate is
    not finite, so its data has to be arrays and not lists.
    """
    point = np.asarray(point, dtype=np.float64).reshape(3)
    return point[0:1], point[1:2], point[2:3]


def _sampled_segment(start, end, count=RAY_SAMPLES):
    """A straight segment as ``count`` points from ``start`` to ``end``.

    See :data:`RAY_SAMPLES` for why a segment is drawn with more than its two
    ends.
    """
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    fraction = np.linspace(0.0, 1.0, count)[:, None]
    return start[None, :] + fraction * (end - start)[None, :]


def _joined(parts, width=3):
    """Several polylines joined into one array, separated by rows of NaN.

    One line artist per polyline costs one artist to update and one artist to
    draw for every segment of a drawing.  A single artist whose data carries a
    NaN row between polylines draws the same picture, because matplotlib breaks
    a line at a non-finite point.  The volume box, the four corner rays, and
    the twelve projected edges on the detector face are each drawn this way.

    Args:
        parts (sequence): the polylines, each (N, width).
        width (int): the number of columns, 3 for object-frame points and 2
            for detector index pairs.

    Returns:
        ndarray: the joined polyline, (M, width), empty when ``parts`` is.
    """
    pieces = []
    separator = np.full((1, width), np.nan)
    for index, part in enumerate(parts):
        if index:
            pieces.append(separator)
        pieces.append(np.asarray(part, dtype=np.float64).reshape(-1, width))
    if not pieces:
        return np.zeros((0, width))
    return np.concatenate(pieces, axis=0)


def _runs_of_equal_flags(flags):
    """Index slices of consecutive equal entries, overlapping by one entry.

    The overlap makes neighboring runs share a point, so polylines drawn from
    the runs join up instead of leaving a gap.

    Args:
        flags (ndarray): a boolean array.

    Returns:
        list of (slice, bool): each run's slice and the value it holds.
    """
    flags = np.asarray(flags, dtype=bool)
    if flags.size == 0:
        return []
    breaks = np.flatnonzero(flags[1:] != flags[:-1]) + 1
    starts = np.concatenate([[0], breaks])
    stops = np.concatenate([breaks, [flags.size]])
    runs = []
    for start, stop in zip(starts, stops):
        # Extend by one so that this run reaches the next run's first point.
        runs.append((slice(start, min(stop + 1, flags.size)),
                     bool(flags[start])))
    return runs


def _bounds(points, margin=PANEL_MARGIN):
    """The (low, high) pair per column of ``points``, with a fractional margin.

    Args:
        points (ndarray): (N, 2) or (N, 3).
        margin (float): fraction of the largest extent added on each side.

    Returns:
        tuple: one (low, high) pair per column.
    """
    points = np.asarray(points, dtype=np.float64)
    low = np.nanmin(points, axis=0)
    high = np.nanmax(points, axis=0)
    pad = margin * float(np.max(high - low))
    if pad <= 0.0:
        pad = 1.0
    return tuple((float(low[index] - pad), float(high[index] + pad))
                 for index in range(points.shape[1]))


def _cube_bounds(points, margin=PANEL_MARGIN):
    """One cubic (low, high) triple that holds ``points``.

    The three axes get the same extent, which is the largest of the three data
    extents.  One ALU is then the same length along x, y, and z, so an angle in
    the drawing is the angle in the geometry.  The cost is empty space along
    the short axes.  The alternative, an axis box shaped like the data, gives a
    long thin tunnel for a cone geometry whose source-detector distance is many
    times the volume's size, and that tunnel is unreadable when the camera
    looks along it.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    low = np.nanmin(points, axis=0)
    high = np.nanmax(points, axis=0)
    center = 0.5 * (low + high)
    extent = float(np.max(high - low)) * (1.0 + 2.0 * margin)
    if extent <= 0.0:
        extent = 1.0
    return tuple((float(center[index] - 0.5 * extent),
                  float(center[index] + 0.5 * extent))
                 for index in range(3))


def _far_corner(view, columns):
    """The detector corner a panel puts farthest from the pixel-0 marker.

    A label on the detector goes here.  The pixel-0 marker sits at one end of
    the detector, so the far end holds a label that neither runs into the
    pixel-0 label nor lands in the middle of the panel, where the rays are.

    Args:
        view (ViewScene): the view's primitives.
        columns (sequence of int): which object coordinates the panel draws,
            as indices into (x, y, z).  The 3D panel draws all three.

    Returns:
        ndarray: the corner, (3,).
    """
    corners = np.asarray(view.detector_corners, dtype=np.float64)
    columns = list(columns)
    pixel0 = np.asarray(view.detector_pixel0, dtype=np.float64)[columns]
    distance = np.linalg.norm(corners[:, columns] - pixel0[None, :], axis=1)
    return corners[int(np.argmax(distance))]


def _place_beside(label, point, side=None, vertical=None):
    """Move a label beside a point in a 2D panel.

    By default the horizontal offset takes the sign that points toward the
    middle of the panel, so that a label on a point at the panel's edge reads
    into the panel and not off it.

    Args:
        label: an annotation whose ``textcoords`` is ``'offset points'``.
        point (sequence): the panel-plane position of the thing it names.
        side (int, optional): 1 to put the label to the right of the point and
            -1 to put it to the left.  The default chooses the side that reads
            into the panel.
        vertical (float, optional): the vertical offset from the point, in
            points.  The default keeps the offset the label was created with.
    """
    if side is None:
        low, high = label.axes.get_xlim()
        past_middle = float(point[0]) > 0.5 * (low + high)
        # A panel whose horizontal axis increases to the left draws a point
        # past the middle of that axis on the left half of the screen, where
        # its label has to read to the right to stay in the panel.
        if label.axes.xaxis_inverted():
            past_middle = not past_middle
        side = -1 if past_middle else 1
    label.set_ha('right' if side < 0 else 'left')
    if vertical is None:
        vertical = float(label.xyann[1])
    else:
        vertical = float(vertical)
        label.set_va('bottom' if vertical >= 0.0 else 'top')
    label.xyann = (side * LABEL_GAP_POINTS, vertical)
    label.xy = (float(point[0]), float(point[1]))


def _within(points, limits):
    """Whether every point lies inside the given per-column limits.

    A pair of limits may come in either order, because a panel drawn with the
    display convention holds the limits of an inverted axis in decreasing
    order.  Each pair is therefore read as its smaller and its larger value
    and not as a low and a high.
    """
    points = np.asarray(points, dtype=np.float64)
    for index, pair in enumerate(limits):
        low, high = min(pair), max(pair)
        column = points[:, index]
        column = column[np.isfinite(column)]
        if column.size == 0:
            continue
        if float(np.min(column)) < low or float(np.max(column)) > high:
            return False
    return True


def _fit_text(quantities):
    """The fit statement of the text panel: whether the volume fits, and by how
    much the worst corner misses when it does not."""
    if quantities['volume_fits_detector']:
        return 'yes'
    overshoot = _three_figures(quantities['worst_overshoot_pixels'])
    return f'no (worst overshoot {overshoot} px)'


#: Numbers in a comparison line are printed to this many significant figures,
#: and to ``COMPARISON_LONG_FIGURES`` when three figures make the two values
#: read as equal.  A difference the panel lists has to be a difference the
#: reader can see.
COMPARISON_FIGURES = 3
COMPARISON_LONG_FIGURES = 7


def _value_text(value, figures=COMPARISON_FIGURES):
    """One parameter or derived value, in a form a text line can carry.

    A per-view array is named by its shape rather than printed, because a
    comparison line has room for a number and not for 1800 of them.  A short
    array, such as a shape or one translation vector, is printed in full.
    """
    if value is None:
        return 'not set'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, str):
        return value if len(value) <= 28 else value[:25] + '...'
    array = np.asarray(value)
    if array.ndim == 0 or array.size == 1:
        return f'{float(array.reshape(-1)[0]) + 0.0:.{figures}g}'
    if array.ndim == 1 and array.size <= 4:
        return ('(' + ', '.join(f'{float(entry) + 0.0:.{figures}g}'
                                for entry in array) + ')')
    return f'array{tuple(int(n) for n in array.shape)}'


def _difference_text(name, mine, theirs):
    """One comparison line, as ``name: primary -> comparison``.

    The two values are printed to more figures when three make them read as
    the same number, which happens for a quantity that a changed parameter
    moves only a little.
    """
    left, right = _value_text(mine), _value_text(theirs)
    if left == right:
        left = _value_text(mine, COMPARISON_LONG_FIGURES)
        right = _value_text(theirs, COMPARISON_LONG_FIGURES)
    return f'{name}: {left} -> {right}'


class GeometryFigure:
    """A five-panel matplotlib figure of one scan geometry.

    The figure shows one view of the scan.  :meth:`set_view` moves to another
    view, :meth:`set_show_trajectory` turns the path of the source over all
    views on and off, :meth:`set_zoom` switches the 3D panel between the whole
    scan and the volume, :meth:`set_show_reference` turns the angle-0 reference
    on and off, and :meth:`set_compare` overlays a second geometry.  The slider
    and the three toggles under the panels call the same methods.

    This class calls no matplotlib window function.  Use :meth:`save` to write
    a file, :meth:`show` to open a window, or :func:`show_geometry` to do both
    steps at once.

    Args:
        model_or_scene: a ``GeometryScene``, or a ``TomographyModel`` from
            which one is built with ``GeometryScene.from_model``.
        view_index (int, optional): the view to draw.  Defaults to 0.
        show_trajectory (bool, optional): whether to draw the source's path
            over all views.  Defaults to False.
        figsize (tuple, optional): the figure size in inches.
        title (str, optional): the figure's suptitle.  Defaults to a line
            naming the geometry kind and the model's shapes.
        elevation_deg, azimuth_deg (float, optional): the initial 3D camera.
        compare (optional): a second geometry to draw over the first.  A
            ``GeometryScene``, a ``TomographyModel``, or a dictionary of
            parameter overrides applied to a copy of this scene's parameters,
            for example ``dict(det_channel_offset=12.5)``.  None draws no
            comparison.
        zoom (str, optional): the 3D panel's zoom, one of :data:`ZOOM_MODES`.
        show_reference (bool, optional): whether to draw the source and the
            detector at their angle-0 position as a fixed reference.  Defaults
            to True.
        widgets (bool, optional): whether to build the slider and the three
            toggles.  Defaults to True.
        blit (bool, optional): whether a view change may use the partial-redraw
            fast path.  Defaults to True.  False forces a full repaint per
            change, which is slower and is the reference the timing script
            compares against.

    Attributes:
        scene (GeometryScene): the geometry drawn.
        compare_scene (GeometryScene or None): the second geometry drawn.
        figure: the matplotlib ``Figure``.
        panel_axes (tuple): the five axes, in the order 3D view, top view, side
            view, detector face, text panel.
        view_slider: the view ``Slider``, or None when the scan has one view or
            ``widgets`` is False.
        trajectory_check, zoom_check, reference_check: the three
            ``CheckButtons``, or None when ``widgets`` is False.
        detector_volume_edges (ndarray): the twelve projected volume edges the
            detector-face panel drew, (12, 2, 2), as (row, channel) pairs taken
            from ``ViewScene.volume_outline_on_detector``.
    """

    def __init__(self, model_or_scene, view_index=0, show_trajectory=False,
                 figsize=(15.0, 9.0), title=None,
                 elevation_deg=DEFAULT_ELEVATION_DEG,
                 azimuth_deg=DEFAULT_AZIMUTH_DEG,
                 compare=None, zoom=DEFAULT_ZOOM, show_reference=True,
                 widgets=True, blit=True):
        _load_pyplot()
        self.scene = _as_scene(model_or_scene)
        self.compare_scene = None
        self._view_index = self._checked_view_index(view_index)
        self._show_trajectory = bool(show_trajectory)
        self._show_reference = bool(show_reference)
        self._zoom = self._checked_zoom(zoom)
        self._elevation_deg = float(elevation_deg)
        self._azimuth_deg = float(azimuth_deg)

        self.enable_blit = bool(blit)
        self._background = None
        self._suspend_blit = False
        self._syncing_widgets = False
        self._slider_region = None

        self._trajectory = None
        self._compare_trajectory = None
        self._reference = None
        self._limits = {}
        self._limit_views = set()
        self.detector_volume_edges = None

        self._quantities = self.scene.derived_quantities()
        self._compare_quantities = None
        self._compare_line_limit = None
        self._text_font_size = TEXT_PANEL_FONT_SIZE

        # Every artist, in four groups.  _moving holds the artists a view
        # change updates, as (axes, artist) pairs, because a partial redraw
        # draws an artist through its axes.  _compare_moving and
        # _compare_static hold the comparison overlay, which set_compare
        # creates and removes.  _reference_artists holds the angle-0
        # reference, which never moves and which its toggle hides.
        self._moving = []
        self._compare_moving = []
        self._compare_static = []
        self._reference_artists = []
        self._arrow_3d = None

        self._build_panels(figsize, title)
        self._create_widgets(widgets)
        self._create_artists()
        if compare is not None:
            self._install_compare(compare)
        self._create_legends()
        self.figure.canvas.mpl_connect('draw_event', self._on_draw_event)
        self._refresh(rebuild_limits=True)

    @classmethod
    def from_model(cls, model, scene_kwargs=None, **kwargs):
        """Build a figure from an mbirtorch model.

        Args:
            model: a ``TomographyModel`` subclass instance.
            scene_kwargs (dict, optional): passed to ``GeometryScene``.
            **kwargs: passed to the constructor.

        Returns:
            GeometryFigure
        """
        scene = GeometryScene.from_model(model, **(scene_kwargs or {}))
        return cls(scene, **kwargs)

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def view_index(self):
        """The view currently drawn."""
        return self._view_index

    @property
    def show_trajectory(self):
        """Whether the source's path over all views is drawn."""
        return self._show_trajectory

    @property
    def zoom(self):
        """The 3D panel's zoom, one of :data:`ZOOM_MODES`."""
        return self._zoom

    @property
    def show_reference(self):
        """Whether the angle-0 reference is drawn."""
        return self._show_reference

    def set_view(self, view_index):
        """Draw another view.

        Only the artists that move are redrawn, and the 3D camera and the panel
        limits are kept, so a camera the user has dragged survives the change
        and the ticks do not move under the drawing.

        Args:
            view_index (int): the view to draw.
        """
        self._view_index = self._checked_view_index(view_index)
        self._sync_slider()
        self._refresh()

    def set_show_trajectory(self, flag):
        """Turn the source's path over all views on or off.

        The path is one polyline per panel, whatever the number of views, and
        it is computed once and kept.

        Args:
            flag (bool): whether to draw the path.
        """
        flag = bool(flag)
        if flag == self._show_trajectory:
            return
        self._show_trajectory = flag
        self._sync_trajectory_check()
        for _, artist in self._trajectory_artists():
            artist.set_visible(flag)
        # The path reaches beyond one view's source position, so the panel
        # limits change with it and the whole figure repaints.
        self._refresh(rebuild_limits=True)

    def set_zoom(self, mode):
        """Switch the 3D panel between the whole scan and the volume.

        In ``'scan'`` the source, the detector, and the volume share one cube.
        In ``'volume'`` the cube is about :data:`ZOOM_VOLUME_WIDTH_FACTOR`
        volume extents wide and centered on the volume, and the axes limits
        clip the rays and the rotation axis where they leave it.

        Args:
            mode (str): one of :data:`ZOOM_MODES`.
        """
        mode = self._checked_zoom(mode)
        if mode == self._zoom:
            return
        self._zoom = mode
        self._sync_zoom_check()
        self._refresh(rebuild_limits=True)

    def set_show_reference(self, flag):
        """Turn the angle-0 reference on or off.

        The reference does not move with the slider, so it belongs to the
        background, and turning it off repaints the whole figure.  The
        source-path toggle works the same way.

        Args:
            flag (bool): whether to draw the reference.
        """
        flag = bool(flag)
        if flag == self._show_reference:
            return
        self._show_reference = flag
        self._sync_reference_check()
        for _, artist in self._reference_artists:
            artist.set_visible(flag)
        self._refresh(rebuild_limits=True)

    def set_compare(self, compare):
        """Draw a second geometry over the first, or stop drawing one.

        Args:
            compare: a ``GeometryScene``, a ``TomographyModel``, a dictionary
                of parameter overrides applied to a copy of this scene's
                parameters, or None to remove the comparison.
        """
        self._remove_compare_artists()
        self.compare_scene = None
        self._compare_quantities = None
        self._compare_trajectory = None
        # A new comparison gets the whole line budget back; the old one may
        # have been cut down to fit.
        self._compare_line_limit = None
        if compare is not None:
            self._install_compare(compare)
        self._create_legends()
        self._update_static_text()
        self._refresh(rebuild_limits=True)

    def save(self, path, dpi=110):
        """Write the figure to an image file.

        The artists a view change updates are marked animated, which keeps them
        out of an ordinary full draw so that the partial redraw can put them
        back.  A saved file must hold them, so they are unmarked for the
        duration of the write.

        Args:
            path (str): the file to write.  The extension chooses the format.
            dpi (int, optional): dots per inch.
        """
        self._suspend_blit = True
        try:
            self._set_animated(False)
            self.figure.savefig(path, dpi=dpi, facecolor='white')
        finally:
            self._set_animated(self._animate_moving())
            self._suspend_blit = False
            self._background = None
        return path

    def show(self, block=True):
        """Open a window on the figure.

        On a backend with no window this prints a line and returns, leaving the
        figure available for :meth:`save`.

        Args:
            block (bool, optional): whether ``show`` waits for the window to
                close.
        """
        backend = matplotlib.get_backend().lower()
        if backend in NONINTERACTIVE_BACKENDS:
            print(f'The {backend} backend has no window; nothing was shown.  '
                  'Use GeometryFigure.save to write a file.')
            return
        plt.show(block=block)

    # ------------------------------------------------------------------
    # Checked arguments and small lookups
    # ------------------------------------------------------------------

    def _checked_view_index(self, view_index):
        view_index = int(view_index)
        if not 0 <= view_index < self.scene.num_views:
            raise IndexError(f'view_index {view_index} is outside '
                             f'[0, {self.scene.num_views}).')
        return view_index

    @staticmethod
    def _checked_zoom(mode):
        mode = str(mode)
        if mode not in ZOOM_MODES:
            raise ValueError(f'Unknown zoom {mode!r}; expected one of '
                             f'{ZOOM_MODES}.')
        return mode

    def _compare_index(self, view_index):
        """The view of the comparison that goes with a view of the primary.

        The comparison follows the slider, so the two share a view index.  A
        comparison with fewer views holds its last view instead of raising.
        """
        return min(int(view_index), self.compare_scene.num_views - 1)

    def _compare_view(self):
        """The comparison's ``ViewScene`` for the current view, or None."""
        if self.compare_scene is None:
            return None
        return self.compare_scene.view(self._compare_index(self._view_index))

    def _source_trajectory(self):
        """The source's path over all views, computed once and kept."""
        if self._trajectory is None:
            sources, _ = self.scene.trajectory()
            self._trajectory = sources
        return self._trajectory

    def _reference_view(self):
        """The angle-0 reference view, computed once and kept."""
        if self._reference is None:
            self._reference = self.scene.reference_view()
        return self._reference

    def _compare_source_trajectory(self):
        """The comparison's source path over all views, computed once."""
        if self._compare_trajectory is None:
            sources, _ = self.compare_scene.trajectory()
            self._compare_trajectory = sources
        return self._compare_trajectory

    def _trajectory_artists(self):
        """The (axes, artist) pairs that draw a source path."""
        pairs = [(axes, artist) for axes, artist in self._trajectory_lines]
        pairs.extend(self._compare_trajectory_lines)
        return pairs

    def _view_label(self, scene=None, view_index=None):
        """A short phrase naming what makes a view different.

        A rotating geometry is named by its angle, and the translation geometry
        by its translation vector.
        """
        scene = self.scene if scene is None else scene
        index = self._view_index if view_index is None else view_index
        if scene.kind == 'translation':
            vector = scene.translation_vectors[index]
            return ('translation ('
                    + ', '.join(_three_figures(value) for value in vector)
                    + ') ALU')
        angle = float(scene.angles[index])
        label = (f'angle {angle:.3f} rad '
                 f'({np.degrees(angle):.1f} deg)')
        if scene.kind == 'multiaxis':
            elevation = float(scene.elevations[index])
            label += f', elevation {elevation:.3f} rad'
        elif scene.kind == 'cone':
            shift = float(scene.z_shifts[index])
            if shift != 0.0:
                label += f', z shift {_three_figures(shift)} ALU'
        return label

    # ------------------------------------------------------------------
    # Layout and widgets
    # ------------------------------------------------------------------

    def _build_panels(self, figsize, title):
        """Create the figure and its five axes.

        The 3D view takes the whole left column, because it needs the room.
        The top view and the side view share the upper right, and the detector
        face and the text panel share the lower right.  The widget row goes
        under all of them.
        """
        self.figure = plt.figure(figsize=figsize)
        grid = self.figure.add_gridspec(
            2, 3, width_ratios=(1.35, 1.0, 1.0), left=0.04, right=0.985,
            bottom=GRID_BOTTOM, top=0.90, wspace=0.30, hspace=0.28)
        self.ax_3d = self.figure.add_subplot(grid[:, 0], projection='3d')
        self.ax_top = self.figure.add_subplot(grid[0, 1])
        self.ax_side = self.figure.add_subplot(grid[0, 2])
        self.ax_detector = self.figure.add_subplot(grid[1, 1])
        self.ax_text = self.figure.add_subplot(grid[1, 2])
        self.panel_axes = (self.ax_3d, self.ax_top, self.ax_side,
                           self.ax_detector, self.ax_text)

        if title is None:
            title = (f'{self._quantities["geometry_kind"]} geometry, '
                     f'sinogram {self._quantities["sinogram_shape_text"]}, '
                     f'recon {self._quantities["recon_shape_text"]}')
        self.figure.suptitle(title, fontsize=11)

        # The opaque rectangle a partial redraw paints over the slider row
        # before drawing it again; animated=True keeps it out of full draws.
        self._clear_rect = Rectangle((0, 0), 1, 1,
                                     facecolor=self.figure.get_facecolor(),
                                     edgecolor='none', animated=True,
                                     transform=IdentityTransform())
        self.figure.add_artist(self._clear_rect)

    def _create_widgets(self, wanted):
        """Create the view slider and the three toggles.

        The slider steps by one view and never by a fraction, and its own draw
        is turned off so that a step goes through this class's redraw instead.
        A scan with one view has nothing to slide, so the slider axes is
        hidden.  These are the conventions ``mbirtorch/viewer.py`` uses for its
        slice slider.
        """
        self.view_slider = None
        self.trajectory_check = None
        self.zoom_check = None
        self.reference_check = None
        self._slider_axes = None
        if not wanted:
            return

        self._slider_axes = self.figure.add_axes(SLIDER_RECT)
        if self.scene.num_views > 1:
            self.view_slider = Slider(
                self._slider_axes, label='View', valmin=0,
                valmax=self.scene.num_views - 1, valinit=self._view_index,
                valstep=1, valfmt='%0.0f')
            self.view_slider.drawon = False
            self.view_slider.label.set_fontsize(WIDGET_FONT_SIZE)
            self.view_slider.valtext.set_fontsize(WIDGET_FONT_SIZE)
            self.view_slider.on_changed(self._on_slider)
        else:
            self._slider_axes.set_visible(False)

        trajectory_axes = self.figure.add_axes(TRAJECTORY_CHECK_RECT)
        trajectory_axes.set_frame_on(False)
        self.trajectory_check = CheckButtons(
            trajectory_axes, ['source path'], [self._show_trajectory])
        for label in self.trajectory_check.labels:
            label.set_fontsize(WIDGET_FONT_SIZE)
        self.trajectory_check.on_clicked(self._on_trajectory_check)

        zoom_axes = self.figure.add_axes(ZOOM_CHECK_RECT)
        zoom_axes.set_frame_on(False)
        self.zoom_check = CheckButtons(
            zoom_axes, ['3D zoom to volume'], [self._zoom == 'volume'])
        for label in self.zoom_check.labels:
            label.set_fontsize(WIDGET_FONT_SIZE)
        self.zoom_check.on_clicked(self._on_zoom_check)

        reference_axes = self.figure.add_axes(REFERENCE_CHECK_RECT)
        reference_axes.set_frame_on(False)
        self.reference_check = CheckButtons(
            reference_axes, ['angle-0 reference'], [self._show_reference])
        for label in self.reference_check.labels:
            label.set_fontsize(WIDGET_FONT_SIZE)
        self.reference_check.on_clicked(self._on_reference_check)

    def _on_slider(self, value):
        """The view slider moved."""
        if self._syncing_widgets:
            return
        index = int(round(float(value)))
        if index == self._view_index:
            return
        self.set_view(index)

    def _on_trajectory_check(self, _label):
        """The source-path toggle was clicked."""
        if self._syncing_widgets:
            return
        self.set_show_trajectory(self.trajectory_check.get_status()[0])

    def _on_zoom_check(self, _label):
        """The 3D zoom toggle was clicked."""
        if self._syncing_widgets:
            return
        self.set_zoom('volume' if self.zoom_check.get_status()[0] else 'scan')

    def _on_reference_check(self, _label):
        """The angle-0 reference toggle was clicked."""
        if self._syncing_widgets:
            return
        self.set_show_reference(self.reference_check.get_status()[0])

    def _sync_slider(self):
        """Move the slider to the current view without calling back."""
        if self.view_slider is None:
            return
        if int(round(float(self.view_slider.val))) == self._view_index:
            return
        self._syncing_widgets = True
        try:
            self.view_slider.set_val(self._view_index)
        finally:
            self._syncing_widgets = False

    def _sync_trajectory_check(self):
        """Match the toggle to the state, without calling back."""
        if self.trajectory_check is None:
            return
        if self.trajectory_check.get_status()[0] == self._show_trajectory:
            return
        self._syncing_widgets = True
        try:
            self.trajectory_check.set_active(0)
        finally:
            self._syncing_widgets = False

    def _sync_zoom_check(self):
        """Match the zoom toggle to the state, without calling back."""
        if self.zoom_check is None:
            return
        if self.zoom_check.get_status()[0] == (self._zoom == 'volume'):
            return
        self._syncing_widgets = True
        try:
            self.zoom_check.set_active(0)
        finally:
            self._syncing_widgets = False

    def _sync_reference_check(self):
        """Match the reference toggle to the state, without calling back."""
        if self.reference_check is None:
            return
        if self.reference_check.get_status()[0] == self._show_reference:
            return
        self._syncing_widgets = True
        try:
            self.reference_check.set_active(0)
        finally:
            self._syncing_widgets = False

    # ------------------------------------------------------------------
    # Creating the artists
    # ------------------------------------------------------------------

    def _create_artists(self):
        """Create every artist once, for the primary geometry.

        The artists split in two.  A static artist is drawn by a full repaint
        and belongs to the background: the volume box, the region of
        reconstruction, the rotation axis or translation path, the voxel
        marker, the detector grid, the two detector-face reference markers, the
        angle-0 reference, and the text block.  A moving artist carries the
        current view and is redrawn on every view change: the source, the
        detector outline, the rays, the projected volume outline, the offset
        segments, the arc, the labels of the source and the detector, the panel
        titles, and the text panel's footer.  A moving artist is marked
        animated only where the partial redraw runs; see
        :meth:`_animate_moving`.
        """
        view = self.scene.view(self._view_index)
        self._create_3d_artists(view)
        self._create_top_artists(view)
        self._create_side_artists(view)
        self._create_detector_artists(view)
        self._create_reference_artists()
        self._create_text_artists()
        self._trajectory_lines = [(self.ax_3d, self._path_3d),
                                  (self.ax_top, self._path_top),
                                  (self.ax_side, self._path_side)]
        self._compare_trajectory_lines = []
        self._set_animated(self._animate_moving())

    def _set_animated(self, flag):
        """Mark or unmark every moving artist as animated."""
        for _, artist in self._moving + self._compare_moving:
            artist.set_animated(bool(flag))
        if self._arrow_3d is not None:
            self._arrow_3d.set_animated(bool(flag))

    def _moving_line(self, axes, color, **kwargs):
        """One empty line artist that a view change will fill in."""
        line, = axes.plot([], [], color=color, **kwargs)
        self._moving.append((axes, line))
        return line

    def _moving_line_3d(self, axes, color, **kwargs):
        """One empty 3D line artist that a view change will fill in."""
        line, = axes.plot(np.zeros(0), np.zeros(0), np.zeros(0),
                          color=color, axlim_clip=True, **kwargs)
        self._moving.append((axes, line))
        return line

    def _moving_text(self, axes, text, color, vertical):
        """One small label that a view change moves to what it names.

        The label is placed at a data point and offset from it in points, so
        that the gap between the label and its marker is the same gap at every
        panel scale.  :func:`_place_beside` sets the horizontal part of that
        offset per view and keeps the vertical part given here.

        Args:
            axes: the panel.
            text (str): the label.
            color: the color of the element the label names.
            vertical (float): the vertical offset from the data point, in
                points.  A positive offset puts the label above the point and
                a negative one puts it below.

        Returns:
            The annotation, already registered as a moving artist.
        """
        label = axes.annotate(text, xy=(0.0, 0.0),
                              textcoords='offset points',
                              xytext=(LABEL_GAP_POINTS, vertical),
                              fontsize=ANNOTATION_FONT_SIZE, color=color,
                              ha='left',
                              va='bottom' if vertical >= 0 else 'top')
        self._moving.append((axes, label))
        return label

    def _moving_text_3d(self, axes, text, color, vertical='baseline'):
        """One small 3D label that a view change moves to what it names.

        A 3D text artist has no offset in points, so the gap from the point it
        names is a leading space in the text, and the only vertical choice is
        the alignment.

        Args:
            axes: the 3D panel.
            text (str): the label.
            color: the color of the element the label names.
            vertical (str, optional): the vertical alignment, which puts the
                label on the line of its point, above it, or below it.
        """
        label = axes.text(0.0, 0.0, 0.0, text, color=color,
                          fontsize=ANNOTATION_FONT_SIZE,
                          verticalalignment=vertical)
        self._moving.append((axes, label))
        return label

    # --- the 3D panel ---

    def _create_3d_artists(self, view):
        """Create the 3D panel's artists and label its axes."""
        axes = self.ax_3d

        # Static: the volume box, the region of reconstruction, the rotation
        # axis or the translation path, and the voxel marker.  None of these
        # moves with the view, because the object is the thing held fixed.
        box = _joined([view.volume_corners[[first, second]]
                       for first, second in VOLUME_BOX_EDGES])
        axes.plot(box[:, 0], box[:, 1], box[:, 2], color=COLORS['volume'],
                  linewidth=1.0, label='volume box', axlim_clip=True)
        self._create_3d_ror(view)
        if view.rotation_axis is not None:
            segment = _sampled_segment(view.rotation_axis[0],
                                       view.rotation_axis[1])
            axes.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                      color=COLORS['axis'], linewidth=1.2, linestyle='--',
                      label='rotation axis', axlim_clip=True)
        if view.translation_path is not None:
            path = view.translation_path
            axes.plot(path[:, 0], path[:, 1], path[:, 2], color=COLORS['axis'],
                      linewidth=1.0, marker='o', markersize=2.5,
                      label='translation path', axlim_clip=True)
        axes.plot([view.voxel0_center[0]], [view.voxel0_center[1]],
                  [view.voxel0_center[2]], marker='x', linestyle='none',
                  markersize=INDEX_MARKER_SIZE,
                  markeredgewidth=INDEX_MARKER_WIDTH, color=COLORS['voxel0'],
                  label='voxel (0, 0, 0)', axlim_clip=True, zorder=6)
        self._path_3d, = axes.plot(np.zeros(0), np.zeros(0), np.zeros(0),
                                   color=COLORS['trajectory'],
                                   linewidth=1.0, linestyle='-.',
                                   label='source path', axlim_clip=True)
        self._path_3d.set_visible(self._show_trajectory)

        # Moving: the source, the detector, the rays, and the arc.
        source_label = ('source' if view.source is not None
                        else 'source (drawn)')
        self._source_3d = self._moving_line_3d(
            axes, COLORS['source'], marker='*', markersize=SOURCE_MARKER_SIZE,
            linestyle='none', label=source_label, zorder=6)
        self._detector_3d = self._moving_line_3d(
            axes, COLORS['detector'], linewidth=1.6, label='detector')
        self._face_3d = None
        if not self.scene.use_curved_detector:
            # A curved detector's face is not a polygon, so only its outline is
            # drawn.  A flat panel gets a lightly filled face, which tells the
            # near side of the detector from the far side.
            self._face_3d = Poly3DCollection(
                [view.detector_corners], facecolor=COLORS['detector'],
                alpha=0.12, edgecolor='none', axlim_clip=True)
            axes.add_collection3d(self._face_3d)
            self._moving.append((axes, self._face_3d))
        self._rays_3d = self._moving_line_3d(axes, COLORS['rays'],
                                             linewidth=0.7)
        self._central_3d = self._moving_line_3d(
            axes, COLORS['central_ray'], linewidth=1.1, label='central ray')
        self._pixel0_3d = self._moving_line_3d(
            axes, COLORS['pixel0'], marker='x', linestyle='none',
            markersize=INDEX_MARKER_SIZE,
            markeredgewidth=INDEX_MARKER_WIDTH,
            label='detector pixel (0, 0)', zorder=6)
        self._arc_3d = self._moving_line_3d(axes, COLORS['axis'],
                                            linewidth=1.4)
        # The arc's label hangs below the end of the arc, which is a few
        # degrees of travel from the source, so that it does not run into the
        # source's own label.
        self._arc_text_3d = self._moving_text_3d(axes, ' source travel',
                                                 COLORS['axis'],
                                                 vertical='top')
        # Two spaces, not one: the source's star marker is wide enough to
        # reach under a label that starts one space away.
        self._source_text_3d = self._moving_text_3d(axes, '  source',
                                                    COLORS['source'])
        self._detector_text_3d = self._moving_text_3d(axes, ' detector',
                                                      COLORS['detector'])
        # The iso label hangs below its point, because the angle-0 reference's
        # label sits at the reference detector, which is a small distance from
        # the detector iso when the view angle is small.
        self._iso_text_3d = self._moving_text_3d(axes, ' ' + ISO_NAME,
                                                 COLORS['central_ray'],
                                                 vertical='top')

        axes.set_xlabel('x (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_ylabel('y (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_zlabel('z (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.tick_params(labelsize=TICK_FONT_SIZE)
        axes.view_init(elev=self._elevation_deg, azim=self._azimuth_deg,
                       roll=_camera_roll_deg())
        axes.set_box_aspect((1.0, 1.0, 1.0))
        self._title_3d = axes.set_title('', fontsize=TITLE_FONT_SIZE)
        self._moving.append((axes, self._title_3d))

    def _create_3d_ror(self, view):
        """Draw the region of reconstruction, when there is one to draw.

        The region is an elliptic cylinder about the rotation axis, so it does
        not move with the view.  It is drawn as its two rings and four
        uprights, joined into one polyline; the uprights make the shape read as
        a cylinder rather than as two unrelated ellipses.
        """
        cylinder = view.ror_cylinder
        if cylinder is None:
            return
        lower = _ellipse_points(cylinder['center'], cylinder['semi_axis_x'],
                                cylinder['semi_axis_y'], cylinder['z_min'])
        upper = _ellipse_points(cylinder['center'], cylinder['semi_axis_x'],
                                cylinder['semi_axis_y'], cylinder['z_max'])
        parts = [lower, upper]
        for step in range(0, ELLIPSE_SAMPLES - 1, (ELLIPSE_SAMPLES - 1) // 4):
            parts.append(np.stack([lower[step], upper[step]]))
        rings = _joined(parts)
        self.ax_3d.plot(rings[:, 0], rings[:, 1], rings[:, 2],
                        color=COLORS['ror'], linewidth=0.9,
                        label='region of reconstruction', axlim_clip=True)

    # --- the two projected panels ---

    def _create_top_artists(self, view):
        """Create the top panel's artists: the xy plane seen from -z.

        The panel puts y on its horizontal axis, increasing to the left, and x
        on its vertical axis, increasing downward.  That is the orientation of
        the group's reference slide, in which the beam runs from left to right.
        """
        axes = self.ax_top
        first, second = TOP_PANEL_COLUMNS

        walk = view.volume_corners[list(_XY_FOOTPRINT_WALK)]
        axes.plot(walk[:, first], walk[:, second], color=COLORS['volume'],
                  linewidth=1.2)
        cylinder = view.ror_cylinder
        if cylinder is not None:
            ring = _ellipse_points(cylinder['center'], cylinder['semi_axis_x'],
                                   cylinder['semi_axis_y'], cylinder['z_min'])
            axes.plot(ring[:, first], ring[:, second], color=COLORS['ror'],
                      linewidth=0.9)
        if view.rotation_axis is not None:
            # The rotation axis is a point in this plane.
            axes.plot([0.0], [0.0], marker='+', markersize=9,
                      markeredgewidth=1.4, color=COLORS['axis'],
                      linestyle='none')
        if view.translation_path is not None:
            path = view.translation_path
            axes.plot(path[:, first], path[:, second], color=COLORS['axis'],
                      linewidth=1.2, marker='o', markersize=3.0)
            # The path is a few ALU across while the panel spans the
            # source-detector distance, so it needs a label to be recognized.
            # The label hangs below the path on the screen, because the
            # angle-0 reference's label runs along the ray above it.
            axes.annotate('translation path',
                          xy=(float(np.mean(path[:, first])),
                              _screen_bottom(path[:, second])),
                          textcoords='offset points', xytext=(0, -5),
                          ha='center', va='top',
                          fontsize=ANNOTATION_FONT_SIZE, color=COLORS['axis'])
        axes.plot([view.voxel0_center[first]], [view.voxel0_center[second]],
                  marker='x', linestyle='none', markersize=INDEX_MARKER_SIZE,
                  markeredgewidth=INDEX_MARKER_WIDTH, color=COLORS['voxel0'],
                  zorder=6)
        self._path_top, = axes.plot([], [], color=COLORS['trajectory'],
                                    linewidth=1.0, linestyle='-.')
        self._path_top.set_visible(self._show_trajectory)

        self._top = self._create_projected_moving_artists(axes)
        # Only the top view names the pixel-0 marker.  In the side view that
        # marker sits at the end of the detector, where the detector's own
        # label already is.  The label sits well above the marker, because a
        # scan whose detector is short against the panel puts the marker close
        # to the detector iso, whose own two labels take the lines nearer the
        # detector.
        self._top['pixel0_label'] = self._moving_text(
            axes, 'pixel (0,0)', COLORS['pixel0'], 14)
        self._arc_top = self._moving_line(axes, COLORS['axis'], linewidth=1.4)
        self._arrow_top = FancyArrowPatch(
            (0.0, 0.0), (0.0, 0.0), arrowstyle='-|>', mutation_scale=9,
            linewidth=1.4, color=COLORS['axis'], shrinkA=0.0, shrinkB=0.0)
        axes.add_patch(self._arrow_top)
        self._moving.append((axes, self._arrow_top))
        self._arc_text_top = axes.annotate(
            'source travel', xy=(0.0, 0.0), textcoords='offset points',
            xytext=(3, -8), fontsize=ANNOTATION_FONT_SIZE,
            color=COLORS['axis'])
        self._moving.append((axes, self._arc_text_top))
        axes.set_xlabel('y (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_ylabel('x (ALU)', fontsize=LABEL_FONT_SIZE)
        self._title_top = axes.set_title(_top_view_title(),
                                         fontsize=TITLE_FONT_SIZE)
        _finish_2d_panel(axes)

    def _create_side_artists(self, view):
        """Create the side panel's artists: the yz plane seen from +x.

        The panel puts y on its horizontal axis, increasing to the left, and z
        on its vertical axis, increasing downward, so that -z is at the top and
        the source of a view at angle 0 is on the left.
        """
        axes = self.ax_side
        first, second = SIDE_PANEL_COLUMNS

        walk = view.volume_corners[list(_YZ_FACE_WALK)]
        axes.plot(walk[:, first], walk[:, second], color=COLORS['volume'],
                  linewidth=1.2)
        axes.axhline(0.0, color=COLORS['axis'], linewidth=0.7, linestyle=':')

        # The volume's z center, which is recon_slice_offset, drawn as a
        # segment from z = 0 at the volume's high y edge and labeled above the
        # volume box.  A cone geometry's side view is many times wider than it
        # is tall, so the three labels of this panel go at three places along
        # it: the detector's at the near end, the volume's in the middle, and
        # the trajectory's in the far corner.
        z_min, z_max = self.scene.volume_z_range()
        z_center = 0.5 * (z_min + z_max)
        if z_center != 0.0:
            y_at = float(np.max(view.volume_corners[:, 1]))
            axes.plot([y_at, y_at], [0.0, z_center], color=COLORS['volume'],
                      linewidth=2.6, solid_capstyle='butt')
            # The label reads outward from the segment, away from the volume
            # box, and it sits above the segment's end on the screen.  Over
            # the middle of the box it ran into the row-offset label, and
            # below the segment it ran into the source's label, which hangs
            # below the source in this panel.  This panel is only a few labels
            # tall, so each of its labels needs its own place.
            outward = _screen_step(1.0)
            axes.annotate(f'recon_slice_offset {_three_figures(z_center)}',
                          xy=(y_at, z_center),
                          textcoords='offset points',
                          xytext=(outward * LABEL_GAP_POINTS,
                                  LABEL_GAP_POINTS),
                          fontsize=ANNOTATION_FONT_SIZE,
                          color=COLORS['volume'],
                          ha='left' if outward > 0 else 'right',
                          va='bottom')
        if view.rotation_axis is not None:
            segment = view.rotation_axis
            axes.plot(segment[:, first], segment[:, second],
                      color=COLORS['axis'], linewidth=1.2, linestyle='--')
        if view.translation_path is not None:
            path = view.translation_path
            axes.plot(path[:, first], path[:, second], color=COLORS['axis'],
                      linewidth=1.0, marker='o', markersize=2.5)
        axes.plot([view.voxel0_center[first]], [view.voxel0_center[second]],
                  marker='x', linestyle='none', markersize=INDEX_MARKER_SIZE,
                  markeredgewidth=INDEX_MARKER_WIDTH, color=COLORS['voxel0'],
                  zorder=6)
        self._path_side, = axes.plot([], [], color=COLORS['trajectory'],
                                     linewidth=1.0, linestyle='-.')
        self._path_side.set_visible(self._show_trajectory)
        self._path_note_side = axes.annotate(
            '', xy=(0.98, 0.04), xycoords='axes fraction', va='bottom',
            ha='right', fontsize=ANNOTATION_FONT_SIZE,
            color=COLORS['trajectory'])
        self._path_note_side.set_visible(self._show_trajectory)

        # The source's label goes below the source in this panel.  Above it
        # is where the recon_slice_offset label sits, and the two ran into
        # each other in the curved cone figure.
        self._side = self._create_projected_moving_artists(
            axes, source_vertical=-7, name_iso=False)
        axes.set_xlabel('y (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_ylabel('z (ALU)', fontsize=LABEL_FONT_SIZE)
        self._title_side = axes.set_title(_side_view_title(),
                                          fontsize=TITLE_FONT_SIZE)
        _finish_2d_panel(axes)

    def _create_projected_moving_artists(self, axes, source_vertical=7,
                                         name_iso=True):
        """The moving artists the top and side panels share.

        The two panels are the same scene projected onto two different planes,
        so one routine creates the source, the detector outline, the rays, the
        central ray, the pixel marker, and the offset segment for both.

        Args:
            axes: the panel.
            source_vertical (float, optional): the vertical offset of the
                source's label, in points.
            name_iso (bool, optional): whether to label the detector iso in
                this panel.

        Returns:
            dict: the artists, keyed by name.
        """
        artists = dict(
            rays=self._moving_line(axes, COLORS['rays'], linewidth=0.7),
            edge_rays=self._moving_line(axes, EMPHASIZED_RAY_COLOR,
                                        linewidth=1.1),
            detector=self._moving_line(axes, COLORS['detector'],
                                       linewidth=1.6),
            central=self._moving_line(axes, COLORS['central_ray'],
                                      linewidth=1.1),
            offset=self._moving_line(axes, COLORS['detector'], linewidth=2.6,
                                     solid_capstyle='butt'),
            source=self._moving_line(axes, COLORS['source'], marker='*',
                                     markersize=SOURCE_MARKER_SIZE,
                                     linestyle='none', zorder=6),
            pixel0=self._moving_line(axes, COLORS['pixel0'], marker='x',
                                     linestyle='none',
                                     markersize=INDEX_MARKER_SIZE,
                                     markeredgewidth=INDEX_MARKER_WIDTH,
                                     zorder=6),
        )
        # The source's label is offset a little farther than the others,
        # because the central ray ends at the source and a smaller offset put
        # the label on that line.
        artists['source_label'] = self._moving_text(
            axes, 'source', COLORS['source'], source_vertical)
        artists['detector_label'] = self._moving_text(
            axes, 'detector', COLORS['detector'], 4)
        # The detector iso is where the central ray meets the detector, and it
        # is the point the two detector offsets are measured from.  Only the
        # top view names it, for the reason the pixel-0 marker is named there
        # alone: the side view is short and wide, and its detector already
        # carries its own label and the row-offset label at the same end.
        # _update_projected_panel puts this label and the offset label below on
        # the side of the detector away from its far corner.
        if name_iso:
            artists['iso_label'] = self._moving_text(
                axes, ISO_NAME, COLORS['central_ray'],
                ISO_LABEL_GAP_POINTS)
        # The label of the detector offset this panel shows.  It names the
        # offset segment, which runs from the detector iso to the detector
        # center, so it hangs at the iso like the label above, one line
        # farther out.
        artists['offset_label'] = self._moving_text(
            axes, '', COLORS['detector'],
            ISO_LABEL_GAP_POINTS + LABEL_LINE_POINTS)
        return artists

    # --- the detector-face panel ---

    def _create_detector_artists(self, view):
        """Create the detector-face panel's artists.

        The horizontal axis is the channel index, increasing to the right, and
        the vertical axis is the row index, increasing downward with row 0 at
        the top.  That is the view from the source toward the detector with -z
        up, and it is how ``imshow`` shows one view of a sinogram.  The grid,
        the detector iso, and the detector center depend only on the detector
        parameters, so they are static.  Only the projected volume outline
        moves with the view.
        """
        axes = self.ax_detector
        num_rows = self.scene.num_det_rows
        num_channels = self.scene.num_det_channels

        axes.add_patch(Rectangle((-0.5, -0.5), num_channels, num_rows,
                                 facecolor=COLORS['detector'], alpha=0.10,
                                 edgecolor=COLORS['detector'], linewidth=1.4))
        landing_row, landing_channel = self.scene.uv_to_indices(0.0, 0.0)
        axes.plot([float(landing_channel)], [float(landing_row)], marker='o',
                  markersize=6, linestyle='none', markerfacecolor='none',
                  markeredgewidth=1.4, color=COLORS['central_ray'],
                  label=ISO_NAME)
        center_row, center_channel = self.scene.uv_to_indices(
            -self.scene.det_channel_offset, -self.scene.row_offset)
        axes.plot([float(center_channel)], [float(center_row)], marker='s',
                  markersize=5, linestyle='none', markerfacecolor='none',
                  markeredgewidth=1.4, color=COLORS['detector'],
                  label=CENTER_NAME)
        axes.plot([0.0], [0.0], marker='x', linestyle='none',
                  markersize=INDEX_MARKER_SIZE,
                  markeredgewidth=INDEX_MARKER_WIDTH, color=COLORS['pixel0'],
                  label='pixel (0,0) = sino[v, 0, 0]')

        self._edges_inside = self._moving_line(axes, COLORS['volume'],
                                               linewidth=1.2,
                                               label='volume box')
        # No legend label: this line exists in every view and carries data
        # only where the volume projects past the detector's edge.
        self._edges_outside = self._moving_line(axes, COLORS['overshoot'],
                                                linewidth=1.6)
        axes.set_aspect('equal', adjustable='box')
        axes.set_xlabel('channel index', fontsize=LABEL_FONT_SIZE)
        axes.set_ylabel('row index', fontsize=LABEL_FONT_SIZE)
        axes.tick_params(labelsize=TICK_FONT_SIZE)
        self._title_detector = axes.set_title('', fontsize=TITLE_FONT_SIZE)
        self._moving.append((axes, self._title_detector))

    # --- the angle-0 reference ---

    def _reference_label(self):
        """What the angle-0 reference's arrow is called in the top view.

        The name is two lines, because one line of it is wider than the top
        panel.  The translation geometry has no view angle, so its reference is
        the view action's identity said the other way: no translation.
        """
        if self.scene.kind == 'translation':
            return 'projection direction,\nno translation'
        return 'projection direction,\nangle 0'

    def _reference_short_label(self):
        """The same name, short enough for the 3D panel."""
        return ('no translation' if self.scene.kind == 'translation'
                else 'angle 0')

    def _create_reference_artists(self):
        """Draw the source and the detector at their angle-0 position.

        The reference is the geometry with the view action set to the identity,
        which ``GeometryScene.reference_view`` returns.  It answers what one
        view cannot: where the source and the detector stand at angle 0, and
        which way the geometry projects through the object from there (Greg,
        2026-09-09).

        These artists do not depend on the view, so they are ordinary static
        artists and they belong to the background.  Their toggle therefore
        repaints the whole figure.  The 3D view and the top view get them.  The
        side view is the yz plane, in which a rotation about z moves nothing,
        so a reference drawn there would lie on top of the view drawn.
        """
        reference = self._reference_view()
        self._reference_artists = []
        dotted = dict(linestyle=REFERENCE_DOTS, linewidth=REFERENCE_LINEWIDTH,
                      alpha=REFERENCE_ALPHA)
        hollow_star = dict(marker='*', linestyle='none',
                           markersize=SOURCE_MARKER_SIZE,
                           markerfacecolor='none',
                           markeredgecolor=COLORS['source'],
                           markeredgewidth=1.3, alpha=REFERENCE_ALPHA)
        source = reference.source_draw
        origin = reference.detector_origin
        outline = reference.detector_outline
        # The arrowhead is drawn from a short segment at the detector end of
        # the central ray, so that it points the way the geometry projects.
        head_start = origin + REFERENCE_ARROW_FRACTION * (source - origin)

        def keep(axes, artist):
            """Register one reference artist and set its visibility."""
            artist.set_visible(self._show_reference)
            self._reference_artists.append((axes, artist))
            return artist

        axes = self.ax_3d
        keep(axes, axes.plot([source[0]], [source[1]], [source[2]],
                             label='angle-0 reference', axlim_clip=True,
                             **hollow_star)[0])
        keep(axes, axes.plot(outline[:, 0], outline[:, 1], outline[:, 2],
                             color=COLORS['detector'], axlim_clip=True,
                             **dotted)[0])
        central = _sampled_segment(source, origin)
        keep(axes, axes.plot(central[:, 0], central[:, 1], central[:, 2],
                             color=COLORS['central_ray'], axlim_clip=True,
                             **dotted)[0])
        direction = origin - head_start
        self._reference_arrow_3d = keep(axes, axes.quiver(
            head_start[0], head_start[1], head_start[2],
            direction[0], direction[1], direction[2],
            color=COLORS['central_ray'], alpha=REFERENCE_ALPHA,
            arrow_length_ratio=3.0, linewidth=REFERENCE_LINEWIDTH))
        self._reference_arrow_point = head_start
        # The label goes at the middle of the reference detector's low-row
        # edge.  On the arrowhead, which lands at the middle of that detector,
        # it fell on the current detector's own label instead.
        corners = reference.detector_corners
        label_at = 0.5 * (corners[0] + corners[1])
        self._reference_text_3d = keep(axes, axes.text(
            label_at[0], label_at[1], label_at[2],
            ' ' + self._reference_short_label(),
            color=COLORS['central_ray'], alpha=REFERENCE_ALPHA,
            fontsize=ANNOTATION_FONT_SIZE))

        axes = self.ax_top
        across, down = TOP_PANEL_COLUMNS
        keep(axes, axes.plot([source[across]], [source[down]],
                             **hollow_star)[0])
        keep(axes, axes.plot(outline[:, across], outline[:, down],
                             color=COLORS['detector'], **dotted)[0])
        keep(axes, axes.plot([source[across], origin[across]],
                             [source[down], origin[down]],
                             color=COLORS['central_ray'], **dotted)[0])
        head = FancyArrowPatch(
            (head_start[across], head_start[down]),
            (origin[across], origin[down]),
            arrowstyle='-|>', mutation_scale=10,
            linewidth=REFERENCE_LINEWIDTH, color=COLORS['central_ray'],
            alpha=REFERENCE_ALPHA, shrinkA=0.0, shrinkB=0.0)
        axes.add_patch(head)
        keep(axes, head)
        # The top view's label reads along the ray, which is the line it
        # names.
        # The label sits in the panel's upper left corner and not on the ray
        # it names.  The ray runs across the middle of this panel, between the
        # source's labels at one end and the detector's three at the other,
        # and a two-line label anywhere along it ran into one of them in at
        # least one of the six geometries.  The corner is the one part of the
        # panel that no geometry draws in, and the dotted line, its hollow
        # star, and its arrowhead are the only dotted artists in the panel, so
        # the label is not ambiguous there.
        keep(axes, axes.annotate(
            self._reference_label(), xy=REFERENCE_LABEL_CORNER,
            xycoords='axes fraction', fontsize=ANNOTATION_FONT_SIZE,
            color=COLORS['central_ray'], alpha=REFERENCE_ALPHA, ha='left',
            va='top'))

    # --- the text panel ---

    def _create_text_artists(self):
        """Create the text panel's three text artists.

        The panel holds a static block at the top, a comparison block above the
        footer, and a footer that names the view drawn.  The split exists for
        speed: the static block is about twenty lines of monospace text, and
        rendering it costs more than every line and marker of the four drawing
        panels together, so a view change must not redraw it.
        """
        axes = self.ax_text
        axes.set_axis_off()
        self._static_text = axes.text(
            0.0, 1.0, '', transform=axes.transAxes,
            fontsize=TEXT_PANEL_FONT_SIZE, family='monospace', va='top',
            ha='left')
        # The comparison block sits under the static block, and
        # _place_compare_text moves it to where that block ends.
        self._compare_text = axes.text(
            0.0, 0.30, '', transform=axes.transAxes,
            fontsize=TEXT_PANEL_FONT_SIZE, family='monospace', va='top',
            ha='left', color=COLORS['compare'])
        self._footer_text = axes.text(
            0.0, 0.10, '', transform=axes.transAxes,
            fontsize=TEXT_PANEL_FONT_SIZE, family='monospace', va='top',
            ha='left')
        self._moving.append((axes, self._footer_text))
        axes.set_title('Derived quantities', fontsize=TITLE_FONT_SIZE)
        self._update_static_text()

    def _update_static_text(self, keep_font=False):
        """Fill in the static text block and the comparison block.

        Args:
            keep_font (bool, optional): whether to keep the font size the panel
                has now.  New content starts from the panel's own size, but a
                re-flow of the comparison section during the fitting passes
                must not undo a font size that fitting has already chosen.
        """
        quantities = self._quantities

        def triple(*keys):
            return ', '.join(_three_figures(quantities[key]) for key in keys)

        rows = [
            ('geometry', quantities['geometry_kind'], ''),
            ('views', str(quantities['num_views']), ''),
            ('sinogram (v, r, c)', quantities['sinogram_shape_text'], ''),
            ('recon (r, c, s)', quantities['recon_shape_text'], ''),
            ('magnification', _three_figures(quantities['magnification']), ''),
            ('fan angle', _three_figures(quantities['fan_angle_deg']), 'deg'),
            ('cone angle', _three_figures(quantities['cone_angle_deg']),
             'deg'),
            ('lateral FoV', _three_figures(quantities['lateral_fov_alu']),
             'ALU'),
            ('axial FoV', _three_figures(quantities['axial_fov_alu']), 'ALU'),
            ('voxel pitch x, y, z',
             triple('delta_voxel_x', 'delta_voxel_y', 'delta_voxel_z'), 'ALU'),
            ('volume extent x,y,z',
             triple('volume_extent_x', 'volume_extent_y', 'volume_extent_z'),
             'ALU'),
            ('volume z center', _three_figures(quantities['volume_z_center']),
             'ALU'),
            ('detector w by h',
             _three_figures(quantities['detector_width']) + ' by '
             + _three_figures(quantities['detector_height']), 'ALU'),
            ('det pitch chan, row',
             triple('detector_channel_pitch', 'detector_row_pitch'), 'ALU'),
            ('det center du, dv',
             triple('detector_center_u', 'detector_center_v'), 'ALU'),
            ('helical travel',
             _three_figures(quantities['helical_travel_alu']), 'ALU'),
            ('volume fits det', _fit_text(quantities), ''),
        ]
        width = max(len(name) for name, _, _ in rows)
        lines = [f'{name:<{width}} : {value}{" " + unit if unit else ""}'
                 for name, value, unit in rows]
        lines.append('')
        # Two sentences about the drawing itself: what the offsets of the
        # detector center are measured from, and which way z is drawn.  The
        # second names the display convention, which every panel follows.
        for note in (f'(du, dv) from {ISO_NAME} to {CENTER_NAME}.',
                     _convention_note(), quantities['drawing_note']):
            lines.extend(textwrap.wrap(note, width=TEXT_PANEL_WRAP_WIDTH))
        self._static_text.set_text('\n'.join(lines))

        compare_lines = self._comparison_lines()
        self._compare_text.set_text('\n'.join(compare_lines))
        self._compare_text.set_visible(bool(compare_lines))
        # New content starts from the panel's own font size; _fit_text_font
        # takes it down from there if the blocks are taller than the panel.
        if not keep_font:
            self._text_font_size = (COMPARING_FONT_SIZE
                                    if self.compare_scene is not None
                                    else TEXT_PANEL_FONT_SIZE)
        self._set_text_font(self._text_font_size)

    def _set_text_font(self, size):
        """Put one font size on the text panel's three blocks."""
        for artist in (self._static_text, self._compare_text,
                       self._footer_text):
            artist.set_fontsize(float(size))

    def _fit_text_font(self, renderer, panel):
        """Shrink the text panel's font until its blocks fit the panel.

        How tall the three blocks are depends on the geometry.  A parallel or
        multiaxis scan's drawing note is five or six lines where a cone scan's
        is two, and at the panel's own font size those blocks ran past the
        bottom of the panel and into the widget row.  The font size is
        therefore scaled by the room the panel has, which is measured with the
        renderer for the reason :meth:`_place_text_blocks` measures.  The scale
        is proportional, so one step lands close and the caller's next pass
        settles it.

        Returns:
            bool: whether the font changed, which means the figure has to be
            drawn again before the blocks can be placed.
        """
        blocks = [self._static_text]
        if self._compare_text.get_visible():
            blocks.append(self._compare_text)
        blocks.append(self._footer_text)
        counts = [artist.get_text().count('\n') + 1 for artist in blocks]
        # One blank line separates one block from the next.
        lines = sum(counts) + len(blocks) - 1
        extent = self._static_text.get_window_extent(renderer)
        line_height = extent.height / max(counts[0], 1)
        needed = lines * line_height
        room = TEXT_PANEL_FILL * panel.height
        # Half a line of tolerance, because a font's line height does not
        # scale exactly with its size: without the tolerance each pass found
        # the blocks a little too tall and shrank them again, and the passes
        # never settled.
        if needed <= room + 0.5 * line_height or line_height <= 0.0:
            return False
        size = max(TEXT_PANEL_FONT_MINIMUM,
                   self._text_font_size * room / needed)
        if size >= self._text_font_size - 0.05:
            return False
        self._text_font_size = size
        self._set_text_font(size)
        return True

    def _comparison_lines(self):
        """The text panel's comparison section, as a list of lines.

        Each entry reads ``name: primary -> comparison``.  The section is
        capped twice: at :data:`MAX_COMPARISON_ENTRIES` entries, and at the
        number of lines the panel has room for, which
        :meth:`_place_text_blocks` measures.  Two unrelated geometries differ
        in every entry, and a section that listed them all would run off the
        panel.  What is left out is counted in a last line.
        """
        if self.compare_scene is None:
            return []
        rows = self.scene.differences(self.compare_scene)
        lines = ['Comparison (dashed):']
        if not rows:
            lines.append('  nothing differs')
            return lines

        blocks = [textwrap.wrap(_difference_text(name, mine, theirs),
                                width=TEXT_PANEL_WRAP_WIDTH,
                                initial_indent='  ',
                                subsequent_indent='      ')
                  for name, mine, theirs in rows[:MAX_COMPARISON_ENTRIES]]
        limit = self._compare_line_limit
        shown = 0
        for block in blocks:
            if limit is not None and len(lines) + len(block) + 1 > limit:
                break
            lines.extend(block)
            shown += 1
        if shown < len(rows):
            lines.append(f'  and {len(rows) - shown} more')
        return lines

    def _footer_lines(self):
        """The text panel's footer, as a list of lines."""
        lines = [f'view drawn : {self._view_index}']
        lines.extend(textwrap.wrap('view: ' + self._view_label(),
                                   width=TEXT_PANEL_WRAP_WIDTH))
        if self.compare_scene is not None:
            index = self._compare_index(self._view_index)
            if index != self._view_index:
                lines.extend(textwrap.wrap(
                    f'comparison holds view {index}: '
                    + self._view_label(self.compare_scene, index),
                    width=TEXT_PANEL_WRAP_WIDTH))
        return lines

    # --- the legends ---

    def _create_legends(self):
        """Build the legend of the 3D panel and of the detector face.

        The legends are rebuilt when a comparison is added or removed, because
        the comparison adds an entry to each of them.
        """
        self.ax_3d.legend(loc='upper left', fontsize=LEGEND_FONT_SIZE,
                          framealpha=0.85, borderpad=0.3, labelspacing=0.25)
        self.ax_detector.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE,
                                framealpha=0.85, borderpad=0.3,
                                labelspacing=0.25)

    def _place_text_blocks(self):
        """Stack the text panel's three blocks from the top of the panel.

        The static block, the comparison block, and the footer follow one
        another with a blank line between them.  Where one block ends is
        measured with the renderer rather than estimated from the font size,
        because the line spacing of a text block depends on the font
        matplotlib resolves.  An estimate put the blocks on top of each other.

        Returns:
            bool: whether a block moved, which means the figure has to be
            drawn again.
        """
        canvas = self.figure.canvas
        if not hasattr(canvas, 'get_renderer'):
            return False
        renderer = canvas.get_renderer()
        panel = self.ax_text.get_window_extent(renderer)
        if panel.height <= 0.0:
            return False

        def line_height(artist):
            extent = artist.get_window_extent(renderer)
            count = artist.get_text().count('\n') + 1
            return extent, extent.height / max(count, 1)

        def place(artist, top):
            """Put an artist's top at a display height.

            Returns:
                bool: whether the artist moved.
            """
            target = float((top - panel.y0) / panel.height)
            if abs(target - float(artist.get_position()[1])) < 0.004:
                return False
            artist.set_position((0.0, target))
            return True

        # The font size is chosen first, because the places of the blocks
        # depend on how tall a line is.  Measuring an artist reports its
        # current font and not the font of the last draw, so the size and the
        # places settle in one pass.
        moved = self._fit_text_font(renderer, panel)
        extent, height = line_height(self._static_text)
        next_top = extent.y0 - height
        if self._compare_text.get_visible():
            moved |= place(self._compare_text, next_top)
            extent, compare_height = line_height(self._compare_text)
            next_top = extent.y0 - compare_height
        moved |= place(self._footer_text, next_top)
        if self._compare_text.get_visible():
            moved |= self._limit_comparison_lines(renderer, panel,
                                                  compare_height)
        return moved

    def _limit_comparison_lines(self, renderer, panel, line_height):
        """Cut the comparison section down to the lines the panel has room for.

        The three text blocks together can be taller than the panel, which a
        comparison between two unrelated geometries makes likely.  The
        comparison section is the one that gives way, because the derived
        quantities and the view drawn are what the panel is for.  The limit
        only ever falls, so this settles instead of adding and dropping the
        same entry.

        Returns:
            bool: whether the section was cut, which means the figure has to be
            drawn again.
        """
        compare = self._compare_text.get_window_extent(renderer)
        footer = self._footer_text.get_window_extent(renderer)
        room = compare.y1 - panel.y0 - footer.height - line_height
        fits = int(max(room, 0.0) // max(line_height, 1.0))
        current = self._compare_text.get_text().count('\n') + 1
        if fits >= current:
            return False
        limit = fits if self._compare_line_limit is None else min(
            fits, self._compare_line_limit)
        if limit == self._compare_line_limit:
            return False
        self._compare_line_limit = limit
        self._update_static_text(keep_font=True)
        return True

    # ------------------------------------------------------------------
    # The comparison overlay
    # ------------------------------------------------------------------

    def _install_compare(self, compare):
        """Build the comparison scene and its artists."""
        self.compare_scene = self._as_compare_scene(compare)
        self._compare_quantities = self.compare_scene.derived_quantities()
        self._compare_trajectory = None
        self._create_compare_artists()
        self._update_static_text()

    def _as_compare_scene(self, compare):
        """A ``GeometryScene`` for the comparison.

        A dictionary of parameter overrides is applied to a copy of this
        scene's parameters, so that a comparison can be asked for as
        ``compare=dict(det_channel_offset=12.5)``.  A model is read the same
        way the primary scene was, with the same drawing options.
        """
        if isinstance(compare, GeometryScene):
            return compare
        if isinstance(compare, dict):
            return self.scene.with_parameters(compare)
        return GeometryScene.from_model(compare,
                                        **self.scene.drawing_options())

    def _create_compare_artists(self):
        """Create the comparison's artists in all four drawing panels.

        The comparison is drawn in one color with dashed lines: its detector
        outline, its source, its central ray, its projected volume outline on
        the detector face, and its pixel-0 marker.  The volume box is shared
        unless the two volumes differ, in which case the comparison's box is
        drawn too.
        """
        color = COLORS['compare']
        dashed = dict(linestyle=COMPARE_DASHES, linewidth=COMPARE_LINEWIDTH)

        def moving(axes, three_d=False, label=None, **kwargs):
            if three_d:
                line, = axes.plot(np.zeros(0), np.zeros(0), np.zeros(0),
                                  color=color, axlim_clip=True, label=label,
                                  **kwargs)
            else:
                line, = axes.plot([], [], color=color, label=label, **kwargs)
            self._compare_moving.append((axes, line))
            return line

        self._compare = {}
        self._compare['detector_3d'] = moving(
            self.ax_3d, three_d=True, label='comparison', **dashed)
        self._compare['central_3d'] = moving(self.ax_3d, three_d=True,
                                             **dashed)
        self._compare['source_3d'] = moving(
            self.ax_3d, three_d=True, marker='*',
            markersize=SOURCE_MARKER_SIZE - 3, linestyle='none',
            markerfacecolor='none', markeredgewidth=1.4)
        self._compare['pixel0_3d'] = moving(
            self.ax_3d, three_d=True, marker='+', linestyle='none',
            markersize=INDEX_MARKER_SIZE, markeredgewidth=INDEX_MARKER_WIDTH)
        for name, axes in (('top', self.ax_top), ('side', self.ax_side)):
            self._compare[f'detector_{name}'] = moving(axes, **dashed)
            self._compare[f'central_{name}'] = moving(axes, **dashed)
            self._compare[f'source_{name}'] = moving(
                axes, marker='*', markersize=SOURCE_MARKER_SIZE - 3,
                linestyle='none', markerfacecolor='none', markeredgewidth=1.4)
            self._compare[f'pixel0_{name}'] = moving(
                axes, marker='+', linestyle='none',
                markersize=INDEX_MARKER_SIZE,
                markeredgewidth=INDEX_MARKER_WIDTH)
        # The comparison is named once, beside its detector in the top view,
        # because a label in every panel would say the same thing four times.
        # It hangs farther below the detector than the primary's labels do, so
        # that it does not run into the channel-offset label.
        label = self.ax_top.annotate(
            'comparison', xy=(0.0, 0.0), textcoords='offset points',
            xytext=(LABEL_GAP_POINTS, -14), fontsize=ANNOTATION_FONT_SIZE,
            color=color, ha='left', va='top')
        self._compare_moving.append((self.ax_top, label))
        self._compare['label_top'] = label
        self._compare['edges_detector'] = moving(
            self.ax_detector, label='comparison', **dashed)
        self._compare['landing_detector'] = moving(
            self.ax_detector, marker='o', markersize=6, linestyle='none',
            markerfacecolor='none', markeredgewidth=1.4)
        self._compare['center_detector'] = moving(
            self.ax_detector, marker='s', markersize=5, linestyle='none',
            markerfacecolor='none', markeredgewidth=1.4)

        # The volume box is the object, and the object is what a comparison
        # usually shares.  Only a comparison whose recon parameters differ gets
        # its own box, drawn as a static artist because the object does not
        # move with the view.
        compare_corners = self.compare_scene.volume_corners()
        if not np.allclose(compare_corners, self.scene.volume_corners()):
            box = _joined([compare_corners[[first, second]]
                           for first, second in VOLUME_BOX_EDGES])
            line, = self.ax_3d.plot(box[:, 0], box[:, 1], box[:, 2],
                                    color=color, axlim_clip=True, **dashed)
            self._compare_static.append((self.ax_3d, line))
            for axes, walk, columns in (
                    (self.ax_top, _XY_FOOTPRINT_WALK, TOP_PANEL_COLUMNS),
                    (self.ax_side, _YZ_FACE_WALK, SIDE_PANEL_COLUMNS)):
                points = compare_corners[list(walk)]
                line, = axes.plot(points[:, columns[0]],
                                  points[:, columns[1]], color=color,
                                  **dashed)
                self._compare_static.append((axes, line))

        # The comparison's own detector grid, when its detector has a different
        # shape.  An identical grid would only draw a dashed line on top of the
        # primary's rectangle.
        if (self.compare_scene.num_det_rows != self.scene.num_det_rows
                or self.compare_scene.num_det_channels
                != self.scene.num_det_channels):
            grid = self.ax_detector.add_patch(Rectangle(
                (-0.5, -0.5), self.compare_scene.num_det_channels,
                self.compare_scene.num_det_rows, facecolor='none',
                edgecolor=color, linewidth=COMPARE_LINEWIDTH,
                linestyle='--'))
            self._compare_static.append((self.ax_detector, grid))

        # The comparison's source path follows the same toggle as the
        # primary's.
        self._compare_trajectory_lines = []
        for axes in (self.ax_3d, self.ax_top, self.ax_side):
            if axes is self.ax_3d:
                line, = axes.plot(np.zeros(0), np.zeros(0), np.zeros(0),
                                  color=color, linewidth=1.0, linestyle='-.',
                                  axlim_clip=True)
            else:
                line, = axes.plot([], [], color=color, linewidth=1.0,
                                  linestyle='-.')
            line.set_visible(self._show_trajectory)
            self._compare_static.append((axes, line))
            self._compare_trajectory_lines.append((axes, line))
        self._set_animated(self._animate_moving())

    def _remove_compare_artists(self):
        """Remove every comparison artist from its axes."""
        for _, artist in self._compare_moving + self._compare_static:
            artist.remove()
        self._compare_moving = []
        self._compare_static = []
        self._compare_trajectory_lines = []
        self._compare = {}

    # ------------------------------------------------------------------
    # Updating the artists for one view
    # ------------------------------------------------------------------

    def _refresh(self, rebuild_limits=False):
        """Update every moving artist for the current view and repaint.

        The panel limits are rebuilt when they are asked for, when they are
        not set yet, or when the current view's content would fall outside
        them.  A rebuild moves the ticks, so it forces a full repaint; the
        common case reuses the background and repaints only what moved.

        Args:
            rebuild_limits (bool): whether to rebuild the limits in any case.
        """
        view = self.scene.view(self._view_index)
        compare_view = self._compare_view()
        if rebuild_limits or not self._limits_hold(view, compare_view):
            self._compute_limits(view, compare_view)
            self._apply_limits()
            rebuild_limits = True

        self._update_moving_artists(view, compare_view)
        if rebuild_limits:
            self._full_redraw()
        else:
            self._fast_redraw()

    def _update_moving_artists(self, view, compare_view):
        """Replace the data of every moving artist, in place."""
        self._update_3d_panel(view)
        self._update_projected_panel(self._top, view, TOP_PANEL_COLUMNS,
                                     TOP_EDGE_RAYS,
                                     self.scene.det_channel_offset,
                                     'det_channel_offset')
        self._update_projected_panel(self._side, view, SIDE_PANEL_COLUMNS,
                                     SIDE_EDGE_RAYS, self.scene.row_offset,
                                     'det_row_offset')
        self._update_arc_and_trajectory(view)
        self._update_detector_panel(view)
        self._clip_3d_artists(view)
        self._footer_text.set_text('\n'.join(self._footer_lines()))
        self._update_titles()
        if compare_view is not None:
            self._update_compare_artists(compare_view)

    def _clip_3d_artists(self, view):
        """Hide the 3D artists that lie outside the 3D panel's cube.

        Two kinds of 3D artist are not clipped by the axes limits, the way a
        3D line drawn with ``axlim_clip`` is: a text artist and an arrowhead
        built by ``quiver``.  In the volume zoom the source sits far outside
        the cube, and its label would be drawn where that point projects,
        which is outside the panel and on top of the rest of the figure.  Each
        such artist is hidden when the point it is attached to is outside the
        cube.

        Args:
            view (ViewScene): the current view's primitives.
        """
        cube = (self.ax_3d.get_xlim(), self.ax_3d.get_ylim(),
                self.ax_3d.get_zlim())
        arc = view.rotation_direction_arc
        artists = [(self._source_text_3d, view.source_draw, True),
                   (self._detector_text_3d, _far_corner(view, (0, 1, 2)),
                    True),
                   (self._iso_text_3d, view.detector_origin, True),
                   (self._arc_text_3d, None if arc is None else arc[-1],
                    arc is not None),
                   (self._arrow_3d, None if arc is None else arc[-2],
                    arc is not None and self._arrow_3d is not None),
                   (self._reference_text_3d,
                    self._reference_text_3d.get_position_3d(),
                    self._show_reference),
                   (self._reference_arrow_3d, self._reference_arrow_point,
                    self._show_reference)]
        for artist, point, drawn in artists:
            if artist is None:
                continue
            if not drawn:
                artist.set_visible(False)
                continue
            point = np.asarray(point, dtype=np.float64).reshape(1, 3)
            artist.set_visible(_within(point, cube))

    def _update_titles(self):
        """Put the view drawn into the titles of the drawing panels."""
        label = self._view_label()
        self._title_3d.set_text(f'3D view, view {self._view_index}, {label}')
        # The row order is a display choice, so the detector panel's title
        # names it on its own line above the view drawn.
        self._title_detector.set_text(
            f'{_detector_view_title()}\nview {self._view_index}, {label}')

    def _update_3d_panel(self, view):
        """Update the 3D panel's moving artists."""
        self._source_3d.set_data_3d(*_point_3d(view.source_draw))
        outline = view.detector_outline
        self._detector_3d.set_data_3d(outline[:, 0], outline[:, 1],
                                      outline[:, 2])
        if self._face_3d is not None:
            self._face_3d.set_verts([view.detector_corners])
            # A 3D collection projects its vertices when the whole axes is
            # drawn.  A partial redraw draws the artist alone, so the
            # projection is asked for here.  Before the first draw the axes
            # has no projection matrix, and that first draw makes one.
            if self.ax_3d.M is not None:
                self._face_3d.do_3d_projection()
        rays = _joined([_sampled_segment(ray[0], ray[1])
                        for ray in view.corner_rays])
        self._rays_3d.set_data_3d(rays[:, 0], rays[:, 1], rays[:, 2])
        central = _sampled_segment(view.source_draw, view.detector_origin)
        self._central_3d.set_data_3d(central[:, 0], central[:, 1],
                                     central[:, 2])
        self._pixel0_3d.set_data_3d(*_point_3d(view.detector_pixel0))
        self._iso_text_3d.set_position_3d(tuple(view.detector_origin))
        self._source_text_3d.set_position_3d(tuple(view.source_draw))
        self._detector_text_3d.set_position_3d(
            tuple(_far_corner(view, (0, 1, 2))))
        if self._show_trajectory:
            path = self._source_trajectory()
            self._path_3d.set_data_3d(path[:, 0], path[:, 1], path[:, 2])

    def _update_projected_panel(self, artists, view, columns, edge_pair,
                                offset, offset_name):
        """Update one projected panel's moving artists.

        Args:
            artists (dict): the panel's artists, from
                :meth:`_create_projected_moving_artists`.
            view (ViewScene): the current view's primitives.
            columns (tuple): which object coordinates go on the horizontal and
                the vertical axis, as indices into (x, y, z);
                :data:`TOP_PANEL_COLUMNS` or :data:`SIDE_PANEL_COLUMNS`.
            edge_pair (tuple): which two of the four corner rays bound this
                panel's plane, as indices into ``view.corner_rays``.
            offset (float): the detector offset this panel shows, which is
                ``det_channel_offset`` for the top view and the row offset for
                the side view.
            offset_name (str): the name of that offset, for its label.
        """
        first, second = columns

        def flat(points):
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            return points[:, first], points[:, second]

        rays = view.corner_rays
        # The two corner rays that bound this plane are drawn darker, so that
        # the fan in the top view and the cone in the side view read as
        # outlines.  The top view's pair is the two corners at the extremes of
        # the channel direction and the side view's is the two at the extremes
        # of the row direction.
        other_pair = tuple(index for index in range(4)
                           if index not in edge_pair)
        edge = _joined([rays[index] for index in edge_pair])
        other = _joined([rays[index] for index in other_pair])
        artists['edge_rays'].set_data(*flat(edge))
        artists['rays'].set_data(*flat(other))

        artists['detector'].set_data(*flat(view.detector_outline))
        artists['central'].set_data(*flat(np.stack([view.source_draw,
                                                    view.detector_origin])))
        artists['source'].set_data(*flat(view.source_draw))
        artists['pixel0'].set_data(*flat(view.detector_pixel0))

        # Each label goes beside the thing it names.  The detector's goes at
        # the end of the detector farthest from the pixel-0 marker, which
        # keeps it off the middle of the panel, where the rays and the offset
        # segment are, and away from the pixel-0 label.  Every label takes the
        # panel's own side rule, which reads it into the panel and so keeps it
        # inside the panel at either end of the detector.
        anchor = _far_corner(view, columns)
        pixel0 = view.detector_pixel0
        _place_beside(artists['source_label'],
                      (view.source_draw[first], view.source_draw[second]))
        _place_beside(artists['detector_label'],
                      (anchor[first], anchor[second]))
        pixel0_label = artists.get('pixel0_label')
        if pixel0_label is not None:
            _place_beside(pixel0_label, (pixel0[first], pixel0[second]))

        # The labels at the detector iso stack on the side of the detector
        # away from its far corner, where the detector's own label is.  A
        # detector short against the panel puts the iso and that corner close
        # together, and stacking the other way ran the labels into each other.
        iso_point = (view.detector_origin[first], view.detector_origin[second])
        away = -_screen_step(anchor[second] - view.detector_origin[second])
        iso_label = artists.get('iso_label')
        gap = ISO_LABEL_GAP_POINTS
        if iso_label is not None:
            _place_beside(iso_label, iso_point, vertical=away * gap)
            gap += LABEL_LINE_POINTS
        offset_label = artists['offset_label']
        offset_label.set_visible(float(offset) != 0.0)
        if float(offset) != 0.0:
            offset_label.set_text(f'{offset_name} {_three_figures(offset)}')
            _place_beside(offset_label, iso_point, vertical=away * gap)

        # The offset segment runs from the detector iso to the center of the
        # detector grid.  Projected onto this panel it shows exactly the offset
        # this panel is about, because the other offset is perpendicular to the
        # plane.
        if float(offset) == 0.0:
            artists['offset'].set_data([], [])
        else:
            artists['offset'].set_data(
                *flat(np.stack([view.detector_origin, view.detector_center])))

    def _update_arc_and_trajectory(self, view):
        """Update the rotation arc, its arrowhead, and the source path.

        The arc, its direction, and its arrowhead direction all come from
        ``ViewScene.rotation_direction_arc``, whose last two points give the
        direction the head points.  The 3D arrowhead is drawn by matplotlib's
        own ``quiver``, which builds it from a start and a direction, so it is
        made again for each view rather than updated.
        """
        arc = view.rotation_direction_arc
        if self._arrow_3d is not None:
            self._arrow_3d.remove()
            self._arrow_3d = None
        for artist in (self._arc_3d, self._arc_text_3d, self._arc_top,
                       self._arrow_top, self._arc_text_top):
            artist.set_visible(arc is not None)
        if arc is not None:
            self._arc_3d.set_data_3d(arc[:, 0], arc[:, 1], arc[:, 2])
            self._arc_text_3d.set_position_3d(tuple(arc[-1]))
            start, end = arc[-2], arc[-1]
            direction = end - start
            self._arrow_3d = self.ax_3d.quiver(
                start[0], start[1], start[2],
                direction[0], direction[1], direction[2],
                color=COLORS['axis'], arrow_length_ratio=3.0, linewidth=1.4)
            self._arrow_3d.set_animated(self._animate_moving())
            # quiver widens the axes limits to hold its arrow, so the panel's
            # own limits go back on afterwards.
            self._apply_3d_limits()

            top_first, top_second = TOP_PANEL_COLUMNS
            self._arc_top.set_data(arc[:, top_first], arc[:, top_second])
            self._arrow_top.set_positions(
                (start[top_first], start[top_second]),
                (end[top_first], end[top_second]))
            # The arc's label reads into the panel, because the arc's end
            # sits at the source's radius, which is near a panel edge.
            _place_beside(self._arc_text_top,
                          (end[top_first], end[top_second]))

        if self._show_trajectory:
            path = self._source_trajectory()
            self._path_top.set_data(path[:, TOP_PANEL_COLUMNS[0]],
                                    path[:, TOP_PANEL_COLUMNS[1]])
            self._path_side.set_data(path[:, SIDE_PANEL_COLUMNS[0]],
                                     path[:, SIDE_PANEL_COLUMNS[1]])
            travel = self._quantities['helical_travel_alu']
            self._path_note_side.set_text(
                f'source z range {_three_figures(travel)} ALU')
        for artist in (self._path_top, self._path_side, self._path_3d,
                       self._path_note_side):
            artist.set_visible(self._show_trajectory)

    def _update_detector_panel(self, view):
        """Update the projected volume outline on the detector face.

        The twelve edges are drawn as two polylines rather than as twelve
        artists: one for the parts on the detector and one for the parts past
        its edge, joined by rows of NaN.  An edge that crosses the boundary is
        sampled and split, so the red part starts exactly where the outline
        leaves the grid.
        """
        outline = np.asarray(view.volume_outline_on_detector, dtype=np.float64)
        edges = np.stack([outline[[first, second]]
                          for first, second in VOLUME_BOX_EDGES])
        self.detector_volume_edges = edges
        inside, outside = self._split_edges(edges)
        # The panel's axes are (channel, row) and the scene reports
        # (row, channel).
        self._edges_inside.set_data(inside[:, 1], inside[:, 0])
        self._edges_outside.set_data(outside[:, 1], outside[:, 0])

    def _split_edges(self, edges):
        """The projected edges, split into the parts on and off the detector.

        Args:
            edges (ndarray): the twelve edges, (12, 2, 2), as (row, channel).

        Returns:
            (ndarray, ndarray): the parts inside the grid and the parts outside
            it, each a NaN-separated polyline of (row, channel) pairs.
        """
        inside_parts, outside_parts = [], []
        for edge in edges:
            if self._inside_detector(edge).all():
                inside_parts.append(edge)
                continue
            fraction = np.linspace(0.0, 1.0, EDGE_CLIP_SAMPLES)[:, None]
            points = edge[0][None, :] + fraction * (edge[1] - edge[0])[None, :]
            inside = self._inside_detector(points)
            for span, is_inside in _runs_of_equal_flags(inside):
                piece = points[span]
                if piece.shape[0] < 2:
                    continue
                (inside_parts if is_inside else outside_parts).append(piece)
        return (_joined(inside_parts, 2), _joined(outside_parts, 2))

    def _inside_detector(self, row_channel):
        """Whether each (row, channel) pair lies on the detector grid."""
        pairs = np.asarray(row_channel, dtype=np.float64).reshape(-1, 2)
        return ((pairs[:, 0] >= -0.5)
                & (pairs[:, 0] <= self.scene.num_det_rows - 0.5)
                & (pairs[:, 1] >= -0.5)
                & (pairs[:, 1] <= self.scene.num_det_channels - 0.5))

    def _update_compare_artists(self, view):
        """Update the comparison overlay for its view."""
        compare = self._compare
        outline = view.detector_outline
        compare['detector_3d'].set_data_3d(outline[:, 0], outline[:, 1],
                                           outline[:, 2])
        central = np.stack([view.source_draw, view.detector_origin])
        compare['central_3d'].set_data_3d(central[:, 0], central[:, 1],
                                          central[:, 2])
        source = view.source_draw
        pixel0 = view.detector_pixel0
        compare['source_3d'].set_data_3d(*_point_3d(source))
        compare['pixel0_3d'].set_data_3d(*_point_3d(pixel0))
        for name, (first, second) in (('top', TOP_PANEL_COLUMNS),
                                      ('side', SIDE_PANEL_COLUMNS)):
            compare[f'detector_{name}'].set_data(outline[:, first],
                                                 outline[:, second])
            compare[f'central_{name}'].set_data(central[:, first],
                                                central[:, second])
            compare[f'source_{name}'].set_data([source[first]],
                                               [source[second]])
            compare[f'pixel0_{name}'].set_data([pixel0[first]],
                                               [pixel0[second]])

        # The comparison's label sits beside the end of its detector that is
        # farthest from its pixel-0 marker, as the primary's does.
        anchor = _far_corner(view, TOP_PANEL_COLUMNS)
        _place_beside(compare['label_top'],
                      (anchor[TOP_PANEL_COLUMNS[0]],
                       anchor[TOP_PANEL_COLUMNS[1]]))

        indices = np.asarray(view.volume_outline_on_detector,
                             dtype=np.float64)
        edges = _joined([indices[[first, second]]
                         for first, second in VOLUME_BOX_EDGES], 2)
        compare['edges_detector'].set_data(edges[:, 1], edges[:, 0])
        landing_row, landing_channel = self.compare_scene.uv_to_indices(
            0.0, 0.0)
        compare['landing_detector'].set_data([float(landing_channel)],
                                             [float(landing_row)])
        center_row, center_channel = self.compare_scene.uv_to_indices(
            -self.compare_scene.det_channel_offset,
            -self.compare_scene.row_offset)
        compare['center_detector'].set_data([float(center_channel)],
                                            [float(center_row)])
        if self._show_trajectory:
            path = self._compare_source_trajectory()
            for axes, line in self._compare_trajectory_lines:
                if axes is self.ax_3d:
                    line.set_data_3d(path[:, 0], path[:, 1], path[:, 2])
                elif axes is self.ax_top:
                    line.set_data(path[:, TOP_PANEL_COLUMNS[0]],
                                  path[:, TOP_PANEL_COLUMNS[1]])
                else:
                    line.set_data(path[:, SIDE_PANEL_COLUMNS[0]],
                                  path[:, SIDE_PANEL_COLUMNS[1]])

    # ------------------------------------------------------------------
    # Panel limits
    # ------------------------------------------------------------------

    def _object_point_groups(self, view):
        """Every object-frame point one view's drawing occupies.

        The panel limits are worked out from these, so the list must hold
        everything a panel draws in object coordinates.  The detector-face
        panel is in index coordinates and is handled separately.
        """
        groups = [view.volume_corners, view.detector_outline,
                  view.source_draw.reshape(1, 3),
                  view.detector_origin.reshape(1, 3),
                  view.detector_center.reshape(1, 3),
                  view.corner_rays.reshape(-1, 3)]
        if view.rotation_axis is not None:
            groups.append(view.rotation_axis)
        if view.translation_path is not None:
            groups.append(view.translation_path)
        if view.rotation_direction_arc is not None:
            groups.append(view.rotation_direction_arc)
        return groups

    def _limit_sample_indices(self):
        """The views the panel limits are computed from.

        A scan of 1800 views is not walked through: the limits come from
        :data:`LIMIT_SAMPLE_VIEWS` views spread over the scan, plus the view
        drawn, plus any view that has been found to fall outside the limits
        before.  A rotating geometry's views are rotations of one another about
        z, so a sample bounds them all in radius; the check in
        :meth:`_limits_hold` covers what a sample can miss.
        """
        count = self.scene.num_views
        if count <= LIMIT_SAMPLE_VIEWS:
            sample = set(range(count))
        else:
            sample = set(int(index) for index in np.round(
                np.linspace(0, count - 1, LIMIT_SAMPLE_VIEWS)))
        sample |= self._limit_views
        sample.add(self._view_index)
        self._limit_views = sample
        return sorted(sample)

    def _compute_limits(self, view, compare_view):
        """Work out and store the limits of the four drawing panels."""
        groups = []
        outlines = []
        for index in self._limit_sample_indices():
            sampled = self.scene.view(index)
            groups.extend(self._object_point_groups(sampled))
            outlines.append(sampled.volume_outline_on_detector)
            if self.compare_scene is not None:
                other = self.compare_scene.view(self._compare_index(index))
                groups.extend(self._object_point_groups(other))
                outlines.append(other.volume_outline_on_detector)
        groups.extend(self._object_point_groups(view))
        outlines.append(view.volume_outline_on_detector)
        if compare_view is not None:
            groups.extend(self._object_point_groups(compare_view))
            outlines.append(compare_view.volume_outline_on_detector)
        if self._show_trajectory:
            groups.append(self._source_trajectory())
            if self.compare_scene is not None:
                groups.append(self._compare_source_trajectory())
        # The angle-0 reference is measured whether or not it is drawn, so
        # that turning it on cannot put it outside limits already set.  The
        # limits are computed once, and the reference never moves.
        reference = self._reference_view()
        groups.extend([reference.source_draw.reshape(1, 3),
                       reference.detector_outline,
                       reference.detector_origin.reshape(1, 3)])

        points = np.concatenate([np.asarray(group, dtype=np.float64)
                                 .reshape(-1, 3) for group in groups])
        self._limits['scan'] = _cube_bounds(points)
        self._limits['volume'] = self._volume_cube()
        self._limits['top'] = _bounds(points[:, list(TOP_PANEL_COLUMNS)])
        self._limits['side'] = _bounds(points[:, list(SIDE_PANEL_COLUMNS)])

        index_points = np.concatenate(
            [np.asarray(outline, dtype=np.float64).reshape(-1, 2)
             for outline in outlines]
            + [np.array([[-0.5, -0.5],
                         [self.scene.num_det_rows - 0.5,
                          self.scene.num_det_channels - 0.5]])])
        margin = PANEL_MARGIN * max(self.scene.num_det_rows,
                                    self.scene.num_det_channels)
        row_low, row_high = (float(np.min(index_points[:, 0])) - margin,
                             float(np.max(index_points[:, 0])) + margin)
        channel_low, channel_high = (
            float(np.min(index_points[:, 1])) - margin,
            float(np.max(index_points[:, 1])) + margin)
        self._limits['detector'] = ((row_low, row_high),
                                    (channel_low, channel_high))

    def _volume_cube(self):
        """The cube the ``'volume'`` zoom uses, as three (low, high) pairs.

        The cube is centered on the volume box and is
        :data:`ZOOM_VOLUME_WIDTH_FACTOR` times the volume's largest extent
        wide, so the volume fills the middle of the panel and the rotation axis
        and the nearest rays are still in the picture.
        """
        corners = self.scene.volume_corners()
        if self.compare_scene is not None:
            corners = np.concatenate([corners,
                                      self.compare_scene.volume_corners()])
        low = corners.min(axis=0)
        high = corners.max(axis=0)
        center = 0.5 * (low + high)
        extent = ZOOM_VOLUME_WIDTH_FACTOR * float(np.max(high - low))
        if extent <= 0.0:
            extent = 1.0
        return tuple((float(center[index] - 0.5 * extent),
                      float(center[index] + 0.5 * extent))
                     for index in range(3))

    def _limits_hold(self, view, compare_view):
        """Whether the current view's content fits the limits already set."""
        if not self._limits:
            return False
        groups = self._object_point_groups(view)
        outlines = [view.volume_outline_on_detector]
        if compare_view is not None:
            groups.extend(self._object_point_groups(compare_view))
            outlines.append(compare_view.volume_outline_on_detector)
        points = np.concatenate([np.asarray(group, dtype=np.float64)
                                 .reshape(-1, 3) for group in groups])
        if not _within(points[:, list(TOP_PANEL_COLUMNS)],
                       self._limits['top']):
            return False
        if not _within(points[:, list(SIDE_PANEL_COLUMNS)],
                       self._limits['side']):
            return False
        if self._zoom == 'scan' and not _within(points, self._limits['scan']):
            return False
        index_points = np.concatenate(
            [np.asarray(outline, dtype=np.float64).reshape(-1, 2)
             for outline in outlines])
        return _within(index_points, self._limits['detector'])

    def _apply_limits(self):
        """Put the stored limits on the four drawing panels.

        Autoscaling is turned off afterwards, so that no artist's data can move
        a panel's limits and invalidate the background behind it.
        """
        self._apply_3d_limits()
        # Both projected panels put y on their horizontal axis and turn it
        # around, so that y increases to the left and the source of a view at
        # angle 0 is on the left.  Their vertical axes are x for the top view
        # and z for the side view, and both increase downward.  These two
        # lines are the top and side panels' one reading of Z_UP_SIGN.
        for axes, key in ((self.ax_top, 'top'), (self.ax_side, 'side')):
            horizontal, vertical = self._limits[key]
            axes.set_xlim(*_screen_pair(*horizontal))
            axes.set_ylim(*_screen_pair(*vertical))
            axes.set_autoscalex_on(False)
            axes.set_autoscaley_on(False)
        # The detector face keeps the channel index increasing to the right and
        # turns the row axis around, which puts row 0 at the top.
        rows, channels = self._limits['detector']
        self.ax_detector.set_xlim(*channels)
        self.ax_detector.set_ylim(*_screen_pair(*rows))
        self.ax_detector.set_autoscalex_on(False)
        self.ax_detector.set_autoscaley_on(False)

    def _apply_3d_limits(self):
        """Put the current zoom's cube on the 3D panel.

        The limits are the cube in increasing order on all three axes.  What
        puts -z at the top of this panel is the camera's roll and not a
        reversed pair of limits; see :func:`_camera_roll_deg`.
        """
        cube = self._limits['volume' if self._zoom == 'volume' else 'scan']
        axes = self.ax_3d
        axes.set_xlim(*cube[0])
        axes.set_ylim(*cube[1])
        axes.set_zlim(*cube[2])
        axes.set_box_aspect((1.0, 1.0, 1.0))
        axes.set_autoscalex_on(False)
        axes.set_autoscaley_on(False)
        axes.set_autoscalez_on(False)

    # ------------------------------------------------------------------
    # Painting: full repaints and partial redraws
    # ------------------------------------------------------------------

    def _animate_moving(self):
        """Whether the moving artists are marked animated.

        A full draw skips an animated artist, and only the partial-redraw path
        paints one.  So the moving artists may be animated only where that path
        runs: on a backend in ``BLIT_BACKENDS`` whose canvas reports blit
        support, with blitting enabled.  Everywhere else they stay ordinary
        artists that a full draw paints.  Marking them animated on such a
        backend would leave the source, the detector, and everything the slider
        moves invisible, which is what happened on the macosx backend before
        this rule existed (Greg, 2026-09-09).
        """
        canvas = self.figure.canvas
        return (self.enable_blit
                and matplotlib.get_backend().lower() in BLIT_BACKENDS
                and bool(getattr(canvas, 'supports_blit', False)))

    def _blit_usable(self):
        """Whether the partial-redraw fast path can be used right now.

        The rule follows ``mbirtorch/viewer.py``: only on a backend where the
        fast path is verified, and only when the canvas reports blit support.
        Everywhere else a view change repaints the whole figure, which is
        correct and slower.  A save suspends the path for its duration.
        """
        return self._animate_moving() and not self._suspend_blit

    def _full_redraw(self):
        """Repaint the whole figure and cache the new background."""
        self._background = None
        canvas = self.figure.canvas
        if not self._blit_usable():
            canvas.draw_idle()
            return
        canvas.draw()
        for _ in range(3):
            # Measuring the text blocks needs a renderer, so their font size
            # and their places are settled after the first draw and the figure
            # is drawn again.  Two passes settle it; the third is a guard.
            if not self._place_text_blocks():
                break
            self._background = None
            canvas.draw()
        if self._background is None:
            # The backend did not report the draw, so the background is taken
            # here instead of in the draw handler.
            self._capture_background()

    def _fast_redraw(self):
        """Restore the background, redraw what moved, and blit."""
        canvas = self.figure.canvas
        if not self._blit_usable():
            canvas.draw_idle()
            return
        if self._background is None:
            self._full_redraw()
            return
        canvas.restore_region(self._background)
        self._redraw_slider_row()
        self._draw_moving_and_blit()

    def _on_draw_event(self, event):
        """Take a new background whenever the whole figure is repainted.

        A full repaint happens when the user drags the 3D camera, resizes the
        window, or when this class asks for one.  The moving artists are
        animated and so are absent from that repaint, which makes it exactly
        the background a partial redraw needs.
        """
        if self._suspend_blit or not self._blit_usable():
            return
        if event is not None and event.canvas is not self.figure.canvas:
            return
        self._capture_background()

    def _capture_background(self):
        """Store the current canvas as the background and draw what moves."""
        canvas = self.figure.canvas
        try:
            self._background = canvas.copy_from_bbox(self.figure.bbox)
        except Exception:
            self._background = None
            return
        self._draw_moving_and_blit()

    def _draw_moving_and_blit(self):
        """Draw every moving artist onto the canvas and blit the figure."""
        canvas = self.figure.canvas
        for axes, artist in self._moving + self._compare_moving:
            if artist.get_visible():
                axes.draw_artist(artist)
        if self._arrow_3d is not None and self._arrow_3d.get_visible():
            self.ax_3d.draw_artist(self._arrow_3d)
        canvas.blit(self.figure.bbox)
        try:
            canvas.flush_events()
        except NotImplementedError:
            pass

    def _redraw_slider_row(self):
        """Repaint the slider, which the restored background holds stale.

        The slider's bar and value text are not animated, so the background
        carries them as they were when it was taken.  An opaque rectangle is
        painted over the row and the slider axes is drawn again on top, which
        is what ``mbirtorch/viewer.py`` does for its own slider rows.
        """
        if self.view_slider is None or not self._slider_axes.get_visible():
            return
        canvas = self.figure.canvas
        renderer = canvas.get_renderer()
        try:
            region = self._slider_axes.get_tightbbox(renderer).padded(6)
        except Exception:
            region = self._slider_axes.bbox.padded(6)
        # Union with the last region so that a value text that grew shorter
        # leaves no part of the old one behind.
        if self._slider_region is not None:
            region = Bbox.union([region, self._slider_region])
        self._slider_region = region
        self._clear_rect.set_bounds(region.x0, region.y0, region.width,
                                    region.height)
        self._clear_rect.draw(renderer)
        self._slider_axes.draw(renderer)


def _finish_2d_panel(axes):
    """Give a 2D panel equal aspect, small tick labels, and a light grid.

    The aspect is kept equal by reshaping the axes box rather than by padding
    the data, so a panel whose content is wide and short draws as a wide short
    strip.  Padding the data instead would squeeze the content into a band
    across the middle of the panel.
    """
    axes.set_aspect('equal', adjustable='box')
    axes.tick_params(labelsize=TICK_FONT_SIZE)
    axes.grid(True, linewidth=0.3, alpha=0.4)


def show_geometry(model_or_scene, view_index=0, show_trajectory=False,
                  block=True, **kwargs):
    """Build a geometry figure and open a window on it.

    Args:
        model_or_scene: a ``GeometryScene`` or a ``TomographyModel``.
        view_index (int, optional): the view to draw.
        show_trajectory (bool, optional): whether to draw the source's path.
        block (bool, optional): whether ``show`` waits for the window to close.
        **kwargs: passed to :class:`GeometryFigure`.

    Returns:
        GeometryFigure: the figure, so that a caller can save it or change the
        view after the window closes.
    """
    figure = GeometryFigure(model_or_scene, view_index=view_index,
                            show_trajectory=show_trajectory, **kwargs)
    figure.show(block=block)
    return figure
