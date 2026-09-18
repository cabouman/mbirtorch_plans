# Geometry sketches in other packages

Date: 2026-09-09.  Purpose: record what TIGRE, LEAP, and CIL draw, so that the
figure design of Increment 3 in `geometry_viewer_plan.md` can borrow from them
deliberately.  Sources were read at the commits the comparison studies cite.

## What each package draws

**TIGRE** (`Python/tigre/utilities/visualization/plot_geometry.py`, 247 lines).  One
3D axis for one view angle.  It draws the source as a point, the detector as a
filled rectangle placed by the detector offsets, the volume as its six faces with
low opacity, the x, y, and z axes as arrows, and two dash-dot circles in the plane
of rotation for the source and detector trajectories.  The axis limits are set from
the source and detector distances so the drawing is to scale.  A second function
animates the trajectory over the view angles.

**LEAP** (`src/leapctype.py`, `sketch_system` and `drawCT`).  One 3D axis, viewed from
above by default (`view_init(90, -90)`), for one view or for a list of views.  It
draws the detector outline, the central ray in green, red rays from the source to
the four detector corners, and the volume as a translucent box.  It draws a tilted
detector by rotating the corner points about the central ray, and a helical scan by
drawing the source and detector at their per-view height.  The modular geometry
loops over views and draws each source and detector module.

**CIL** (`Wrappers/Python/cil/utilities/display.py`, class `_ShowGeometry`).  One 3D
axis with a legend.  It draws the world axes as labeled arrows, the detector outline
with arrows for its row and column directions, the source as a star, dashed rays
from the source to the detector corners, the rotation axis as an arrow, the volume
as a dotted box, and a dashed arc with an arrowhead for the rotation direction.  Two
markers are unusual and useful: an x at detector pixel 0 and an x at voxel 0, so the
index orientation of the data is visible in the drawing.  A 2D geometry is drawn
as a 3D one seen from above.

## What the mbirtorch viewer takes from them

- From all three: the source point, the detector outline, the corner rays, the
  volume box, and the trajectory drawn to scale in one 3D axis.
- From CIL: the marker at detector pixel 0 and at voxel 0, and the rotation
  direction arc.  A wrong offset sign or a mirrored channel order is visible from
  these two markers alone.
- From LEAP: the top-down default camera, and drawing several views at once.
- From TIGRE: the axis limits set from the source and detector distances.

None of the three has a 2D top view, a 2D side view, a detector-face panel, or a
text panel of derived quantities.  Those four panels are the part of the mbirtorch
design that goes beyond the references.
