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

The picture drawn.  The object stays fixed and the source and the detector
carry the view's motion, which is how a scanner user thinks of a scan.  The
projector code does the opposite, and the scene handles the conversion.

The 3D camera.  The default camera sits 22 degrees above the xy plane at an
azimuth of -70 degrees, so the eye looks down from the -y side and 20 degrees
around toward +x.  The +y axis then runs away from the eye into the picture, so
for a view angle of zero the source sits at the back, and the +z axis runs up
the screen.  The 20 degrees off the -y axis matter: at an azimuth of exactly
-90 degrees the y axis and the z axis both run up the screen, and a detector
that stands upright is then hard to tell from one lying flat.  The camera can
be dragged with the mouse, and ``set_view`` keeps whatever camera the user has
set.

Import discipline.  This module imports numpy and the matplotlib base package
at import time, and nothing else.  ``pyplot`` and the 3D toolkit are imported
inside :func:`_load_pyplot`, on the first figure construction.  Importing this
module therefore never resolves a matplotlib backend and never touches a GUI
toolkit, which is the rule ``mbirtorch/viewer.py`` follows.

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
           'VOLUME_BOX_EDGES']


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
}

TITLE_FONT_SIZE = 9
LABEL_FONT_SIZE = 8
TICK_FONT_SIZE = 7
ANNOTATION_FONT_SIZE = 7
LEGEND_FONT_SIZE = 6.5
TEXT_PANEL_FONT_SIZE = 7.5

#: Characters per line in the text panel's wrapped sentences.  The panel's
#: aligned "name : value" lines are shorter than this, so this width sets the
#: panel's overall text width.
TEXT_PANEL_WRAP_WIDTH = 52

SOURCE_MARKER_SIZE = 13
INDEX_MARKER_SIZE = 8
INDEX_MARKER_WIDTH = 1.8

#: Color of the two corner rays that bound a 2D panel's plane.  They are drawn
#: darker than the other two so that the fan and the cone read as outlines.
EMPHASIZED_RAY_COLOR = '#8a8a8a'

#: The 3D camera, in degrees.  See the module docstring.
DEFAULT_ELEVATION_DEG = 22.0
DEFAULT_AZIMUTH_DEG = -70.0

#: Points sampled around an ellipse when drawing the region of reconstruction.
ELLIPSE_SAMPLES = 65

#: The angle the rotation-direction arc sweeps, in radians, and how many points
#: it is drawn with.  The arc starts at the source and follows the circle the
#: source travels on, so it needs no radius of its own.
ROTATION_ARC_SWEEP = 0.5
ROTATION_ARC_SAMPLES = 17

#: Points sampled along one projected volume edge on the detector face, used
#: only when the edge crosses the detector's boundary and has to be split into
#: an inside part and an overshooting part.
EDGE_CLIP_SAMPLES = 64

#: Fractional margin added around the data of a 2D panel.
PANEL_MARGIN = 0.10


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


# Filled in by _load_pyplot on the first figure construction, so that importing
# this module never resolves a matplotlib backend.
plt = None
Poly3DCollection = None
Rectangle = None

# Backends that have no interactive window; show_geometry warns under these.
NONINTERACTIVE_BACKENDS = {'agg', 'pdf', 'ps', 'svg', 'template', 'cairo'}


def _load_pyplot():
    """Import pyplot and the drawing classes on first use."""
    global plt, Poly3DCollection, Rectangle
    if plt is not None:
        return
    import matplotlib.pyplot as _plt
    from mpl_toolkits.mplot3d.art3d import (
        Poly3DCollection as _Poly3DCollection)
    from matplotlib.patches import Rectangle as _Rectangle
    plt = _plt
    Poly3DCollection = _Poly3DCollection
    Rectangle = _Rectangle


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


class GeometryFigure:
    """A five-panel matplotlib figure of one scan geometry.

    The figure shows one view of the scan.  :meth:`set_view` moves to another
    view and :meth:`set_show_trajectory` turns the path of the source over all
    views on and off.  Both redraw the drawing panels.

    This class calls no matplotlib window function.  Use :meth:`save` to write
    a file, or :func:`show_geometry` to open a window.

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

    Attributes:
        scene (GeometryScene): the scene drawn.
        figure: the matplotlib ``Figure``.
        panel_axes (tuple): the five axes, in the order 3D view, top view, side
            view, detector face, text panel.
        detector_volume_edges (ndarray): the twelve projected volume edges the
            detector-face panel drew, (12, 2, 2), as (row, channel) pairs taken
            from ``ViewScene.volume_outline_on_detector``.
    """

    def __init__(self, model_or_scene, view_index=0, show_trajectory=False,
                 figsize=(15.0, 9.0), title=None,
                 elevation_deg=DEFAULT_ELEVATION_DEG,
                 azimuth_deg=DEFAULT_AZIMUTH_DEG):
        _load_pyplot()
        self.scene = _as_scene(model_or_scene)
        self._view_index = self._checked_view_index(view_index)
        self._show_trajectory = bool(show_trajectory)
        self._trajectory = None
        self._elevation_deg = float(elevation_deg)
        self._azimuth_deg = float(azimuth_deg)
        self.detector_volume_edges = None

        self._quantities = self.scene.derived_quantities()
        self._build_panels(figsize, title)
        self._draw_all()

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

    def set_view(self, view_index):
        """Draw another view.

        The four drawing panels are cleared and redrawn.  The 3D camera is
        preserved, so a camera the user has dragged survives the change.

        Args:
            view_index (int): the view to draw.
        """
        self._view_index = self._checked_view_index(view_index)
        self._draw_all()

    def set_show_trajectory(self, flag):
        """Turn the source's path over all views on or off.

        Args:
            flag (bool): whether to draw the path.
        """
        self._show_trajectory = bool(flag)
        self._draw_all()

    def save(self, path, dpi=110):
        """Write the figure to an image file.

        Args:
            path (str): the file to write.  The extension chooses the format.
            dpi (int, optional): dots per inch.
        """
        self.figure.savefig(path, dpi=dpi, facecolor='white')
        return path

    def _checked_view_index(self, view_index):
        view_index = int(view_index)
        if not 0 <= view_index < self.scene.num_views:
            raise IndexError(f'view_index {view_index} is outside '
                             f'[0, {self.scene.num_views}).')
        return view_index

    def _drawn_corner_rays(self, view):
        """The four rays to the detector corners, as this figure draws them.

        A geometry with a finite source gets the scene's own corner rays, which
        run from the source to the four corners.  A parallel-type geometry has
        no source, and the scene's drawn stand-in would then give four rays
        converging on a point, which is the picture of a cone beam and not of a
        parallel one.  For those geometries each ray is drawn instead as a
        segment ending at its corner and running back along the scene's ray
        direction, as far back as the drawn source is.  Both the direction and
        the length come from the ``ViewScene``, so this adds no geometry of its
        own.

        Returns:
            ndarray: four segments, (4, 2, 3), each from a starting point to a
            detector corner.
        """
        if not self.scene.is_parallel_type:
            return view.corner_rays
        length = float(np.linalg.norm(view.source_draw - view.detector_origin))
        starts = view.detector_corners - length * view.ray_direction[None, :]
        return np.stack([starts, view.detector_corners], axis=1)

    def _source_trajectory(self):
        """The source's path over all views, computed once and kept."""
        if self._trajectory is None:
            sources, _ = self.scene.trajectory()
            self._trajectory = sources
        return self._trajectory

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_panels(self, figsize, title):
        """Create the figure and its five axes.

        The 3D view takes the whole left column, because it needs the room.
        The top view and the side view share the upper right, and the detector
        face and the text panel share the lower right.
        """
        self.figure = plt.figure(figsize=figsize)
        grid = self.figure.add_gridspec(
            2, 3, width_ratios=(1.35, 1.0, 1.0), left=0.04, right=0.985,
            bottom=0.06, top=0.90, wspace=0.30, hspace=0.28)
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

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_all(self):
        """Clear the five panels and draw the current view into them."""
        # Keep whatever camera the 3D panel has, so that a redraw does not
        # undo a rotation the user made with the mouse.
        if self.ax_3d.has_data():
            self._elevation_deg = float(self.ax_3d.elev)
            self._azimuth_deg = float(self.ax_3d.azim)
        for axes in self.panel_axes:
            axes.clear()

        view = self.scene.view(self._view_index)
        self._draw_3d_panel(view)
        self._draw_top_panel(view)
        self._draw_side_panel(view)
        self._draw_detector_panel(view)
        self._draw_text_panel(view)

    def _view_label(self):
        """A short phrase naming what makes the current view different.

        A rotating geometry is named by its angle, and the translation geometry
        by its translation vector.
        """
        index = self._view_index
        if self.scene.kind == 'translation':
            vector = self.scene.translation_vectors[index]
            return ('translation ('
                    + ', '.join(_three_figures(value) for value in vector)
                    + ') ALU')
        angle = float(self.scene.angles[index])
        label = (f'angle {angle:.3f} rad '
                 f'({np.degrees(angle):.1f} deg)')
        if self.scene.kind == 'multiaxis':
            elevation = float(self.scene.elevations[index])
            label += f', elevation {elevation:.3f} rad'
        elif self.scene.kind == 'cone':
            shift = float(self.scene.z_shifts[index])
            if shift != 0.0:
                label += f', z shift {_three_figures(shift)} ALU'
        return label

    # --- the 3D panel ---

    def _draw_3d_panel(self, view):
        """Draw the source, detector, volume, axis, and rays in three axes."""
        axes = self.ax_3d
        extent_points = [view.volume_corners, view.detector_outline,
                         view.source_draw.reshape(1, 3),
                         view.detector_origin.reshape(1, 3)]

        # The source, drawn as a star.  A parallel-type geometry has no source
        # position, and the scene then supplies a drawn stand-in.
        source_label = ('source' if view.source is not None
                        else 'source (drawn)')
        axes.plot([view.source_draw[0]], [view.source_draw[1]],
                  [view.source_draw[2]], marker='*',
                  markersize=SOURCE_MARKER_SIZE, linestyle='none',
                  color=COLORS['source'], label=source_label, zorder=6)

        # The detector: its outline always, and its face filled when it is a
        # flat panel.  A curved detector's face is not a polygon, so only the
        # outline is drawn for it.
        axes.plot(view.detector_outline[:, 0], view.detector_outline[:, 1],
                  view.detector_outline[:, 2], color=COLORS['detector'],
                  linewidth=1.6, label='detector')
        if not self.scene.use_curved_detector:
            face = Poly3DCollection([view.detector_corners],
                                    facecolor=COLORS['detector'], alpha=0.12,
                                    edgecolor='none')
            axes.add_collection3d(face)

        # The four rays to the detector corners, and the central ray from the
        # source to where it meets the detector.
        corner_rays = self._drawn_corner_rays(view)
        extent_points.append(corner_rays.reshape(-1, 3))
        for ray in corner_rays:
            axes.plot(ray[:, 0], ray[:, 1], ray[:, 2], color=COLORS['rays'],
                      linewidth=0.7)
        axes.plot([view.source_draw[0], view.detector_origin[0]],
                  [view.source_draw[1], view.detector_origin[1]],
                  [view.source_draw[2], view.detector_origin[2]],
                  color=COLORS['central_ray'], linewidth=1.1,
                  label='central ray')

        # The volume box, as its twelve edges.
        for first, second in VOLUME_BOX_EDGES:
            edge = view.volume_corners[[first, second]]
            axes.plot(edge[:, 0], edge[:, 1], edge[:, 2],
                      color=COLORS['volume'], linewidth=1.0)
        axes.plot([], [], [], color=COLORS['volume'], linewidth=1.0,
                  label='volume box')

        self._draw_3d_ror(view)
        self._draw_3d_axis_or_path(view)

        # The two index markers.
        axes.plot([view.detector_pixel0[0]], [view.detector_pixel0[1]],
                  [view.detector_pixel0[2]], marker='x', linestyle='none',
                  markersize=INDEX_MARKER_SIZE,
                  markeredgewidth=INDEX_MARKER_WIDTH, color=COLORS['pixel0'],
                  label='detector pixel (0, 0)', zorder=6)
        axes.plot([view.voxel0_center[0]], [view.voxel0_center[1]],
                  [view.voxel0_center[2]], marker='x', linestyle='none',
                  markersize=INDEX_MARKER_SIZE,
                  markeredgewidth=INDEX_MARKER_WIDTH, color=COLORS['voxel0'],
                  label='voxel (0, 0, 0)', zorder=6)

        if self._show_trajectory:
            path = self._source_trajectory()
            axes.plot(path[:, 0], path[:, 1], path[:, 2],
                      color=COLORS['trajectory'], linewidth=1.0,
                      linestyle='-.', label='source path')
            extent_points.append(path)

        axes.set_xlabel('x (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_ylabel('y (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_zlabel('z (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.tick_params(labelsize=TICK_FONT_SIZE)
        axes.set_title(f'3D view, view {self._view_index}, '
                       f'{self._view_label()}', fontsize=TITLE_FONT_SIZE)
        axes.view_init(elev=self._elevation_deg, azim=self._azimuth_deg)
        _set_common_scale_3d(axes, extent_points)
        axes.legend(loc='upper left', fontsize=LEGEND_FONT_SIZE,
                    framealpha=0.85, borderpad=0.3, labelspacing=0.25)

    def _draw_3d_ror(self, view):
        """Draw the region of reconstruction, when there is one to draw."""
        cylinder = view.ror_cylinder
        if cylinder is None:
            return
        axes = self.ax_3d
        lower = _ellipse_points(cylinder['center'], cylinder['semi_axis_x'],
                                cylinder['semi_axis_y'], cylinder['z_min'])
        upper = _ellipse_points(cylinder['center'], cylinder['semi_axis_x'],
                                cylinder['semi_axis_y'], cylinder['z_max'])
        for ring in (lower, upper):
            axes.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=COLORS['ror'],
                      linewidth=0.9)
        # A few uprights make the shape read as a cylinder rather than as two
        # unrelated ellipses.
        for step in range(0, ELLIPSE_SAMPLES - 1, (ELLIPSE_SAMPLES - 1) // 4):
            axes.plot([lower[step, 0], upper[step, 0]],
                      [lower[step, 1], upper[step, 1]],
                      [lower[step, 2], upper[step, 2]], color=COLORS['ror'],
                      linewidth=0.6)
        axes.plot([], [], [], color=COLORS['ror'], linewidth=0.9,
                  label='region of reconstruction')

    def _draw_3d_axis_or_path(self, view):
        """Draw the rotation axis with its direction arc, or the object path.

        A rotating geometry gets the rotation axis as a dashed line and a short
        arc with an arrowhead showing which way the source travels as the view
        index grows.  The translation geometry does not rotate, so the path of
        its translation vectors takes the axis's place.
        """
        axes = self.ax_3d
        if view.rotation_axis is not None:
            segment = view.rotation_axis
            axes.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                      color=COLORS['axis'], linewidth=1.2, linestyle='--',
                      label='rotation axis')
            arc = self._rotation_direction_arc(view)
            if arc is not None:
                axes.plot(arc[:, 0], arc[:, 1], arc[:, 2],
                          color=COLORS['axis'], linewidth=1.4)
                start, end = arc[-2], arc[-1]
                axes.quiver(start[0], start[1], start[2],
                            end[0] - start[0], end[1] - start[1],
                            end[2] - start[2], color=COLORS['axis'],
                            arrow_length_ratio=3.0, linewidth=1.4)
                axes.text(arc[-1, 0], arc[-1, 1], arc[-1, 2],
                          ' source travel', color=COLORS['axis'],
                          fontsize=ANNOTATION_FONT_SIZE)
        if view.translation_path is not None:
            path = view.translation_path
            axes.plot(path[:, 0], path[:, 1], path[:, 2], color=COLORS['axis'],
                      linewidth=1.0, marker='o', markersize=2.5,
                      label='translation path')

    def _rotation_direction_arc(self, view):
        """An arc showing which way the source travels, as an annotation.

        The arc starts at the source and follows the circle about the rotation
        axis that the source travels on, so its radius and height are the
        source's own.  It sweeps a fixed angle in the direction the source
        moves as the view index grows.  The conventions record states that
        direction: the object turns counterclockwise seen from the +z axis for
        a growing view angle, so in a drawing that holds the object fixed the
        source turns clockwise.  The sign of the step between this view's angle
        and the next one is read from the scene, so a model whose angles
        decrease gets an arc the other way.

        Returns:
            ndarray or None: the arc, (N, 3), or None when the geometry does
            not rotate, has only one view, or holds two views at one angle.
        """
        if view.rotation_axis is None or self.scene.num_views < 2:
            return None
        angles = np.asarray(self.scene.angles, dtype=np.float64)
        index = self._view_index
        neighbor = index + 1 if index + 1 < angles.size else index - 1
        step = angles[neighbor] - angles[index]
        if neighbor < index:
            step = -step
        if step == 0.0:
            return None

        source = view.source_draw
        radius = float(np.hypot(source[0], source[1]))
        height = float(source[2])
        if radius <= 0.0:
            return None
        start = float(np.arctan2(source[1], source[0]))
        # A growing view angle carries the source clockwise seen from +z, which
        # is a falling azimuth, so the sweep takes the sign of -step.
        sweep = -np.sign(step) * ROTATION_ARC_SWEEP
        azimuth = start + np.linspace(0.0, sweep, ROTATION_ARC_SAMPLES)
        return np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth),
                         np.full(ROTATION_ARC_SAMPLES, height)], axis=1)

    # --- the two projected panels ---

    def _draw_top_panel(self, view):
        """Draw the xy plane: the fan, the footprint, and the offset in u."""
        axes = self.ax_top
        self._draw_projected_scene(axes, view, first_axis=0, second_axis=1,
                                   corner_pair=(0, 1),
                                   volume_walk=_XY_FOOTPRINT_WALK)

        # The region of reconstruction is an ellipse in this plane.
        cylinder = view.ror_cylinder
        if cylinder is not None:
            ring = _ellipse_points(cylinder['center'], cylinder['semi_axis_x'],
                                   cylinder['semi_axis_y'], cylinder['z_min'])
            axes.plot(ring[:, 0], ring[:, 1], color=COLORS['ror'],
                      linewidth=0.9)

        # The rotation axis is a point in this plane; the arc shows its sense.
        if view.rotation_axis is not None:
            axes.plot([0.0], [0.0], marker='+', markersize=9,
                      markeredgewidth=1.4, color=COLORS['axis'],
                      linestyle='none')
            arc = self._rotation_direction_arc(view)
            if arc is not None:
                axes.plot(arc[:, 0], arc[:, 1], color=COLORS['axis'],
                          linewidth=1.4)
                axes.annotate('', xy=(arc[-1, 0], arc[-1, 1]),
                              xytext=(arc[-2, 0], arc[-2, 1]),
                              arrowprops=dict(arrowstyle='-|>', linewidth=1.4,
                                              color=COLORS['axis']))
                axes.annotate('source travel', xy=(arc[-1, 0], arc[-1, 1]),
                              textcoords='offset points', xytext=(3, -8),
                              fontsize=ANNOTATION_FONT_SIZE,
                              color=COLORS['axis'])
        if view.translation_path is not None:
            path = view.translation_path
            axes.plot(path[:, 0], path[:, 1], color=COLORS['axis'],
                      linewidth=1.2, marker='o', markersize=3.0)
            # The path is a few ALU across while the panel spans the
            # source-detector distance, so it needs a label to be recognized.
            axes.annotate('translation path',
                          xy=(float(np.max(path[:, 0])),
                              float(np.max(path[:, 1]))),
                          textcoords='offset points', xytext=(4, 4),
                          fontsize=ANNOTATION_FONT_SIZE, color=COLORS['axis'])

        # The offset segment is short, so its label hangs below the detector.
        if self._draw_detector_offset(axes, view, 0, 1,
                                      self.scene.det_channel_offset):
            _offset_label(axes, view, 0, 1,
                          'det_channel_offset '
                          + _three_figures(self.scene.det_channel_offset),
                          xytext=(0, -5), va='top', ha='center')

        if self._show_trajectory:
            path = self._source_trajectory()
            axes.plot(path[:, 0], path[:, 1], color=COLORS['trajectory'],
                      linewidth=1.0, linestyle='-.')

        axes.set_xlabel('x (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_ylabel('y (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_title('Top view, the xy plane, seen from +z',
                       fontsize=TITLE_FONT_SIZE)
        _finish_2d_panel(axes)

    def _draw_side_panel(self, view):
        """Draw the yz plane: the cone, the z extent, and the offset in v."""
        axes = self.ax_side
        self._draw_projected_scene(axes, view, first_axis=1, second_axis=2,
                                   corner_pair=(0, 3),
                                   volume_walk=_YZ_FACE_WALK)

        # A cone geometry's side view is many times wider than it is tall,
        # because its width is the source-detector distance and its height is
        # the detector height.  The three labels below therefore go at three
        # different places along the panel: the detector's at the far end, the
        # volume's in the middle, and the trajectory's in the top corner.
        # Labels placed beside one another would overlap in so short a panel.
        axes.axhline(0.0, color=COLORS['axis'], linewidth=0.7, linestyle=':')

        # The volume's z center, which is recon_slice_offset, drawn as a
        # segment from z = 0 at the volume's high y edge and labeled above the
        # volume box.
        z_min, z_max = self.scene.volume_z_range()
        z_center = 0.5 * (z_min + z_max)
        if z_center != 0.0:
            y_at = float(np.max(view.volume_corners[:, 1]))
            axes.plot([y_at, y_at], [0.0, z_center], color=COLORS['volume'],
                      linewidth=2.6, solid_capstyle='butt')
            axes.annotate(f'recon_slice_offset {_three_figures(z_center)}',
                          xy=(y_at, max(z_max, z_center)),
                          textcoords='offset points', xytext=(0, 4),
                          fontsize=ANNOTATION_FONT_SIZE,
                          color=COLORS['volume'], ha='center', va='bottom')

        if view.rotation_axis is not None:
            segment = view.rotation_axis
            axes.plot(segment[:, 1], segment[:, 2], color=COLORS['axis'],
                      linewidth=1.2, linestyle='--')
        if view.translation_path is not None:
            path = view.translation_path
            axes.plot(path[:, 1], path[:, 2], color=COLORS['axis'],
                      linewidth=1.0, marker='o', markersize=2.5)

        if self._draw_detector_offset(axes, view, 1, 2, self.scene.row_offset):
            _offset_label(axes, view, 1, 2,
                          'det_row_offset '
                          + _three_figures(self.scene.row_offset),
                          xytext=(0, -5), va='top', ha='left')

        if self._show_trajectory:
            path = self._source_trajectory()
            axes.plot(path[:, 1], path[:, 2], color=COLORS['trajectory'],
                      linewidth=1.0, linestyle='-.')
            travel = self._quantities['helical_travel_alu']
            axes.annotate(f'source z range {_three_figures(travel)} ALU',
                          xy=(0.98, 0.04), xycoords='axes fraction',
                          va='bottom', ha='right',
                          fontsize=ANNOTATION_FONT_SIZE,
                          color=COLORS['trajectory'])

        axes.set_xlabel('y (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_ylabel('z (ALU)', fontsize=LABEL_FONT_SIZE)
        axes.set_title('Side view, the yz plane, seen from +x',
                       fontsize=TITLE_FONT_SIZE)
        _finish_2d_panel(axes)

    def _draw_projected_scene(self, axes, view, first_axis, second_axis,
                              corner_pair, volume_walk):
        """Draw the elements the top and side views share.

        The two panels are the same scene projected onto two different planes,
        so the source, the detector, the rays, the volume, and the two index
        markers are drawn by one routine.

        Args:
            axes: the panel's axes.
            view (ViewScene): the primitives of the current view.
            first_axis, second_axis (int): which object coordinates go on the
                horizontal and vertical axes, as indices into (x, y, z).
            corner_pair (tuple of int): the two detector corners whose rays
                bound the drawing in this plane.  The top view uses the two
                corners at the extremes of the channel direction, and the side
                view the two at the extremes of the row direction.
            volume_walk (tuple of int): corner indices of a closed walk around
                the volume's outline in this plane.
        """
        def flat(points):
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            return points[:, first_axis], points[:, second_axis]

        source_h, source_v = flat(view.source_draw)
        axes.plot(source_h, source_v, marker='*',
                  markersize=SOURCE_MARKER_SIZE, linestyle='none',
                  color=COLORS['source'], zorder=6)

        outline_h, outline_v = flat(view.detector_outline)
        axes.plot(outline_h, outline_v, color=COLORS['detector'],
                  linewidth=1.6)

        # Every corner ray, with the two that bound this plane drawn darker.
        for index, ray in enumerate(self._drawn_corner_rays(view)):
            ray_h, ray_v = flat(ray)
            emphasize = index in corner_pair
            axes.plot(ray_h, ray_v,
                      color=(EMPHASIZED_RAY_COLOR if emphasize
                             else COLORS['rays']),
                      linewidth=0.7 if not emphasize else 1.1)

        central_h, central_v = flat(np.stack([view.source_draw,
                                              view.detector_origin]))
        axes.plot(central_h, central_v, color=COLORS['central_ray'],
                  linewidth=1.1)

        walk = view.volume_corners[list(volume_walk)]
        walk_h, walk_v = flat(walk)
        axes.plot(walk_h, walk_v, color=COLORS['volume'], linewidth=1.2)

        pixel_h, pixel_v = flat(view.detector_pixel0)
        axes.plot(pixel_h, pixel_v, marker='x', linestyle='none',
                  markersize=INDEX_MARKER_SIZE,
                  markeredgewidth=INDEX_MARKER_WIDTH, color=COLORS['pixel0'],
                  zorder=6)
        voxel_h, voxel_v = flat(view.voxel0_center)
        axes.plot(voxel_h, voxel_v, marker='x', linestyle='none',
                  markersize=INDEX_MARKER_SIZE,
                  markeredgewidth=INDEX_MARKER_WIDTH, color=COLORS['voxel0'],
                  zorder=6)

    def _draw_detector_offset(self, axes, view, first_axis, second_axis,
                              offset):
        """Draw the detector offset as a segment, and say whether it was drawn.

        The segment runs from the point where the central ray meets the
        detector to the center of the detector grid.  Projected onto this
        panel's plane it shows exactly the offset this panel is about, because
        the other offset is perpendicular to the plane.  The segment is a few
        pixels long in a panel that spans the whole source-detector distance,
        so its name and value go in the panel's corner note instead of beside
        it.

        Returns:
            bool: whether a segment was drawn, which is whether the offset is
            nonzero.
        """
        if float(offset) == 0.0:
            return False
        start = view.detector_origin
        end = view.detector_center
        axes.plot([start[first_axis], end[first_axis]],
                  [start[second_axis], end[second_axis]],
                  color=COLORS['detector'], linewidth=2.6,
                  solid_capstyle='butt')
        return True

    # --- the detector-face panel ---

    def _draw_detector_panel(self, view):
        """Draw the detector grid and the volume's projected outline on it.

        The horizontal axis is the channel index and the vertical axis is the
        row index, with row 0 at the bottom so that the row index increases
        upward, matching the row coordinate v along +z.
        """
        axes = self.ax_detector
        num_rows = self.scene.num_det_rows
        num_channels = self.scene.num_det_channels

        axes.add_patch(Rectangle((-0.5, -0.5), num_channels, num_rows,
                                 facecolor=COLORS['detector'], alpha=0.10,
                                 edgecolor=COLORS['detector'], linewidth=1.4))

        outline = np.asarray(view.volume_outline_on_detector, dtype=np.float64)
        edges = np.stack([outline[[first, second]]
                          for first, second in VOLUME_BOX_EDGES])
        self.detector_volume_edges = edges
        for edge in edges:
            self._draw_detector_edge(axes, edge)

        # Where the central ray lands, and where the grid's center sits.  The
        # two coincide only when both detector offsets are zero.
        landing_row, landing_channel = self.scene.uv_to_indices(0.0, 0.0)
        axes.plot([float(landing_channel)], [float(landing_row)], marker='o',
                  markersize=6, linestyle='none',
                  markerfacecolor='none', markeredgewidth=1.4,
                  color=COLORS['central_ray'], label='central ray lands')
        center_row, center_channel = self.scene.uv_to_indices(
            -self.scene.det_channel_offset, -self.scene.row_offset)
        axes.plot([float(center_channel)], [float(center_row)], marker='s',
                  markersize=5, linestyle='none', markerfacecolor='none',
                  markeredgewidth=1.4, color=COLORS['detector'],
                  label='grid center')
        axes.plot([0.0], [0.0], marker='x', linestyle='none',
                  markersize=INDEX_MARKER_SIZE,
                  markeredgewidth=INDEX_MARKER_WIDTH, color=COLORS['pixel0'],
                  label='pixel (0, 0)')
        axes.plot([], [], color=COLORS['volume'], linewidth=1.2,
                  label='volume box')

        margin = PANEL_MARGIN * max(num_channels, num_rows)
        low_channel = min(-0.5, float(np.min(outline[:, 1]))) - margin
        high_channel = max(num_channels - 0.5,
                           float(np.max(outline[:, 1]))) + margin
        low_row = min(-0.5, float(np.min(outline[:, 0]))) - margin
        high_row = max(num_rows - 0.5, float(np.max(outline[:, 0]))) + margin
        axes.set_xlim(low_channel, high_channel)
        axes.set_ylim(low_row, high_row)
        axes.set_aspect('equal', adjustable='box')
        axes.set_xlabel('channel index', fontsize=LABEL_FONT_SIZE)
        axes.set_ylabel('row index', fontsize=LABEL_FONT_SIZE)
        axes.tick_params(labelsize=TICK_FONT_SIZE)
        axes.set_title(f'Detector face, view {self._view_index}, '
                       f'{self._view_label()}', fontsize=TITLE_FONT_SIZE)
        axes.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE,
                    framealpha=0.85, borderpad=0.3, labelspacing=0.25)

    def _draw_detector_edge(self, axes, edge):
        """Draw one projected volume edge, red where it leaves the detector.

        Args:
            axes: the detector-face axes.
            edge (ndarray): the edge's two endpoints, (2, 2), as
                (row, channel).
        """
        inside_start = self._inside_detector(edge[0])
        inside_end = self._inside_detector(edge[1])
        if inside_start and inside_end:
            axes.plot(edge[:, 1], edge[:, 0], color=COLORS['volume'],
                      linewidth=1.2)
            return
        # The edge crosses the detector's boundary, so it is sampled and drawn
        # as runs of points that are inside and runs that are outside.  The
        # sampled points lie on the straight line between the two projected
        # corners, which is where the edge is drawn in any case.
        fraction = np.linspace(0.0, 1.0, EDGE_CLIP_SAMPLES)[:, None]
        points = edge[0][None, :] + fraction * (edge[1] - edge[0])[None, :]
        inside = np.array([self._inside_detector(point) for point in points])
        for span, is_inside in _runs_of_equal_flags(inside):
            piece = points[span]
            if piece.shape[0] < 2:
                continue
            axes.plot(piece[:, 1], piece[:, 0],
                      color=COLORS['volume'] if is_inside
                      else COLORS['overshoot'],
                      linewidth=1.2 if is_inside else 1.6)

    def _inside_detector(self, row_channel):
        """Whether a (row, channel) index pair lies on the detector grid."""
        row, channel = float(row_channel[0]), float(row_channel[1])
        return (-0.5 <= row <= self.scene.num_det_rows - 0.5
                and -0.5 <= channel <= self.scene.num_det_channels - 0.5)

    # --- the text panel ---

    def _draw_text_panel(self, view):
        """Print the derived quantities and the drawing note."""
        axes = self.ax_text
        axes.set_axis_off()
        quantities = self._quantities

        def triple(*keys):
            return ', '.join(_three_figures(quantities[key]) for key in keys)

        rows = [
            ('geometry', quantities['geometry_kind'], ''),
            ('views', str(quantities['num_views']), ''),
            ('view drawn', str(self._view_index), ''),
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
            ('det center u, v',
             triple('detector_center_u', 'detector_center_v'), 'ALU'),
            ('helical travel',
             _three_figures(quantities['helical_travel_alu']), 'ALU'),
            ('volume fits det', _fit_text(quantities), ''),
        ]
        width = max(len(name) for name, _, _ in rows)
        lines = [f'{name:<{width}} : {value}{" " + unit if unit else ""}'
                 for name, value, unit in rows]

        lines.append('')
        lines.extend(textwrap.wrap('view drawn: ' + self._view_label(),
                                   width=TEXT_PANEL_WRAP_WIDTH))
        lines.append('')
        lines.extend(textwrap.wrap(quantities['drawing_note'],
                                   width=TEXT_PANEL_WRAP_WIDTH))

        axes.text(0.0, 1.0, '\n'.join(lines), transform=axes.transAxes,
                  fontsize=TEXT_PANEL_FONT_SIZE, family='monospace',
                  va='top', ha='left')
        axes.set_title('Derived quantities', fontsize=TITLE_FONT_SIZE)


def _fit_text(quantities):
    """The fit statement of the text panel: whether the volume fits, and by how
    much the worst corner misses when it does not."""
    if quantities['volume_fits_detector']:
        return 'yes'
    overshoot = _three_figures(quantities['worst_overshoot_pixels'])
    return f'no (worst overshoot {overshoot} px)'


def _set_common_scale_3d(axes, point_groups):
    """Give the three 3D axes one common scale, in a cubic box.

    The three axes get the same extent, which is the largest of the three data
    extents, and the box is a cube.  One ALU is then the same length along x,
    y, and z, so an angle in the drawing is the angle in the geometry.  The
    cost is empty space along the short axes.  The alternative, an axis box
    shaped like the data, gives a long thin tunnel for a cone geometry whose
    source-detector distance is many times the volume's size, and that tunnel
    is unreadable when the camera looks along it.
    """
    points = np.concatenate([np.asarray(group, dtype=np.float64).reshape(-1, 3)
                             for group in point_groups])
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = 0.5 * (low + high)
    extent = float(np.max(high - low)) * (1.0 + 2.0 * PANEL_MARGIN)
    if extent <= 0.0:
        extent = 1.0
    axes.set_xlim(center[0] - 0.5 * extent, center[0] + 0.5 * extent)
    axes.set_ylim(center[1] - 0.5 * extent, center[1] + 0.5 * extent)
    axes.set_zlim(center[2] - 0.5 * extent, center[2] + 0.5 * extent)
    axes.set_box_aspect((1.0, 1.0, 1.0))


def _offset_label(axes, view, first_axis, second_axis, text, xytext, va, ha):
    """Label a detector offset segment, placed away from the detector outline.

    Args:
        axes: the panel's axes.
        view (ViewScene): the current view's primitives.
        first_axis, second_axis (int): the panel's two object coordinates.
        text (str): the label.
        xytext (tuple): the label's offset from the segment, in points.
        va, ha (str): the label's vertical and horizontal alignment.
    """
    start = view.detector_origin
    end = view.detector_center
    axes.annotate(text,
                  xy=(0.5 * (start[first_axis] + end[first_axis]),
                      0.5 * (start[second_axis] + end[second_axis])),
                  textcoords='offset points', xytext=xytext,
                  fontsize=ANNOTATION_FONT_SIZE, color=COLORS['detector'],
                  va=va, ha=ha)


def _finish_2d_panel(axes):
    """Give a 2D panel equal aspect, a margin, and small tick labels.

    The aspect is kept equal by reshaping the axes box rather than by padding
    the data, so a panel whose content is wide and short draws as a wide short
    strip.  Padding the data instead would squeeze the content into a band
    across the middle of the panel.
    """
    axes.set_aspect('equal', adjustable='box')
    axes.margins(PANEL_MARGIN)
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
    backend = matplotlib.get_backend().lower()
    if backend in NONINTERACTIVE_BACKENDS:
        print(f'The {backend} backend has no window; nothing was shown.  '
              'Use GeometryFigure.save to write a file.')
        return figure
    plt.show(block=block)
    return figure
