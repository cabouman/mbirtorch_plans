"""Open the interactive geometry viewer on an example model.

Run this from a checkout of mbirtorch_plans in an environment where mbirtorch
is installed and a display is available:

    cd plans/experiments/geometry_viewer
    python gv_show_example.py

Which geometry is shown.  ``GEOMETRY`` chooses one of the six geometries the
viewer supports, and ``GEOMETRIES`` lists their names.  Each one is built from
parameters alone, by the function named in ``BUILDERS``, so the script needs no
data and no scanner files.  The parameters every geometry shares, and the ones
each geometry adds, are the constants under "run parameters" below.

What the window shows.  The figure has a 3D view, a top view of the xy plane,
a side view of the yz plane, a detector face in row and channel index, and a
panel of derived numbers.  Every panel is drawn with negative z at the top, and
the top and side views put y to the left, so the source of a view at angle 0 is
on the left and the detector is on the right.

The controls.  A slider steps through the views, and three toggles sit beside
it.  "source path" draws the source's trajectory over all views.  "3D zoom to
volume" switches the 3D panel between the whole scan and a cube around the
volume.  "angle-0 reference", on by default, draws the source and the detector
at their zero-angle position as faint dotted outlines, with a dotted central
ray whose arrowhead shows the projection direction at angle 0.  The 3D panel
can be rotated with the mouse.  Closing the window ends the script.

To view your own model instead, pass it to ``show_geometry`` in place of the
model this script builds, for example the one that a scanner reader returns:

    scan = mbirtorch.preprocess.nsi.get_sino_and_model(dataset_dir)
    show_geometry(scan['model'])

To compare two geometries, pass a second model or a dictionary of parameter
overrides as ``compare``; the second geometry is drawn dashed and the text
panel lists every parameter and derived quantity that differs.
"""

import numpy as np

import mbirtorch
from geometry_viewer import show_geometry

# ── run parameters ───────────────────────────────────────────────────────────

#: The geometries this script can show, and the one it shows.
GEOMETRIES = ('cone', 'cone_curved', 'cone_helical', 'parallel', 'multiaxis',
              'translation')
GEOMETRY = 'cone'

# The detector and the views, shared by every geometry.  The translation
# geometry sets its own view count, because its views are its translations.
NUM_VIEWS = 180
NUM_DET_ROWS = 96
NUM_DET_CHANNELS = 128

# The source and detector distances of the three cone geometries and of the
# translation geometry, in ALU.
SOURCE_DETECTOR_DIST = 4.0 * NUM_DET_CHANNELS
SOURCE_ISO_DIST = SOURCE_DETECTOR_DIST / 2.0

# Detector offsets in ALU, applied to every geometry.  Change one and watch
# the detector-iso marker move on the detector face and the offset label move
# in the top and side views.
DET_CHANNEL_OFFSET = 3.0
DET_ROW_OFFSET = -2.0

# The helical geometry's axial travel in ALU, spread evenly over the views.
# Fifty ALU is about half the detector's height at the rotation axis, so the
# source path covers half a detector over the scan.
HELICAL_TRAVEL_ALU = 50.0

# The multiaxis geometry's elevation in degrees, held at one value for every
# view.  The elevation is the angle at which the source looks up at the
# object, and the model warns above 45 degrees.
ELEVATION_DEG = 30.0

# The translation geometry's grid of object positions.  The view count is the
# product of the two counts, and the spacings are in ALU.
NUM_X_TRANSLATIONS = 5
NUM_Z_TRANSLATIONS = 3
X_TRANSLATION_SPACING = 20.0
Z_TRANSLATION_SPACING = 12.0

# Set to None for no comparison, or to a dictionary of parameter overrides.
# The example shifts the detector by ten more channels.
COMPARE = dict(det_channel_offset=DET_CHANNEL_OFFSET + 10.0)

VIEW_INDEX = 0
SHOW_TRAJECTORY = True


# ── one builder per geometry ─────────────────────────────────────────────────

def view_angles(num_views=NUM_VIEWS):
    """One full turn of view angles, in radians."""
    return np.linspace(0.0, 2.0 * np.pi, num_views, endpoint=False)


def build_cone():
    """A circular cone beam scan with a flat detector."""
    return mbirtorch.ConeBeamModel(
        (NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS), view_angles(),
        source_detector_dist=SOURCE_DETECTOR_DIST,
        source_iso_dist=SOURCE_ISO_DIST, compile_mode='off')


def build_cone_curved():
    """The same scan with a curved detector.

    The detector is then a cylinder of radius ``source_detector_dist`` whose
    axis passes through the source, and the 3D and top views draw its outline
    as an arc.
    """
    return mbirtorch.ConeBeamModel(
        (NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS), view_angles(),
        source_detector_dist=SOURCE_DETECTOR_DIST,
        source_iso_dist=SOURCE_ISO_DIST, use_curved_detector=True,
        compile_mode='off')


def build_cone_helical():
    """A helical cone beam scan.

    The per-view z shifts run over ``HELICAL_TRAVEL_ALU``, centered on zero.  A
    positive shift moves the object toward negative z, so in the viewer's
    drawing, which holds the object fixed, the source and the detector move
    toward positive z as the scan runs.  Positive z is drawn downward, so the
    source path descends the screen.
    """
    travel = float(HELICAL_TRAVEL_ALU)
    z_shifts = np.linspace(-0.5 * travel, 0.5 * travel, NUM_VIEWS)
    return mbirtorch.ConeBeamModel(
        (NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS), view_angles(),
        source_detector_dist=SOURCE_DETECTOR_DIST,
        source_iso_dist=SOURCE_ISO_DIST, helical_z_shifts=z_shifts,
        compile_mode='off')


def build_parallel():
    """A parallel beam scan.

    The parallel model has no source position and no detector position, so the
    viewer draws both at a drawing distance from the volume and says so in its
    text panel.
    """
    return mbirtorch.ParallelBeamModel(
        (NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS), view_angles(),
        compile_mode='off')


def build_multiaxis():
    """A multiaxis parallel scan at one elevation.

    Each view carries an azimuth and an elevation.  The azimuths run over one
    full turn and the elevation is held at ``ELEVATION_DEG``, which tilts every
    ray out of the xy plane by that angle.
    """
    azimuths = view_angles()
    elevations = np.full(NUM_VIEWS, np.radians(ELEVATION_DEG))
    angle_pairs = np.stack([azimuths, elevations], axis=1)
    return mbirtorch.MultiAxisParallelModel(
        (NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS), angle_pairs,
        compile_mode='off')


def build_translation():
    """A translation scan over a grid of object positions.

    ``mbirtorch.gen_translation_vectors`` lays the positions out as a grid of
    ``NUM_X_TRANSLATIONS`` by ``NUM_Z_TRANSLATIONS`` steps, so the scan has
    that many views.  The source and the detector sit as in the flat cone
    geometry, and the object translates instead of rotating.
    """
    vectors = mbirtorch.gen_translation_vectors(
        NUM_X_TRANSLATIONS, NUM_Z_TRANSLATIONS, X_TRANSLATION_SPACING,
        Z_TRANSLATION_SPACING)
    num_views = NUM_X_TRANSLATIONS * NUM_Z_TRANSLATIONS
    return mbirtorch.TranslationModel(
        (num_views, NUM_DET_ROWS, NUM_DET_CHANNELS), vectors,
        source_detector_dist=SOURCE_DETECTOR_DIST,
        source_iso_dist=SOURCE_ISO_DIST, compile_mode='off')


#: The builder of each geometry, by name.
BUILDERS = dict(cone=build_cone, cone_curved=build_cone_curved,
                cone_helical=build_cone_helical, parallel=build_parallel,
                multiaxis=build_multiaxis, translation=build_translation)


def build_model(geometry=GEOMETRY):
    """Build one geometry's model and set the two detector offsets.

    Args:
        geometry (str, optional): one of :data:`GEOMETRIES`.

    Returns:
        A ``TomographyModel`` subclass instance.
    """
    if geometry not in BUILDERS:
        raise ValueError(f'Unknown geometry {geometry!r}; expected one of '
                         f'{GEOMETRIES}.')
    model = BUILDERS[geometry]()
    # no_warning suppresses the notice that a reconstruction parameter was
    # changed by hand; the offsets are the point of the example.
    model.set_params(det_channel_offset=DET_CHANNEL_OFFSET,
                     det_row_offset=DET_ROW_OFFSET, no_warning=True)
    return model


# ── Show it ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ct_model = build_model(GEOMETRY)
    show_geometry(ct_model, view_index=VIEW_INDEX,
                  show_trajectory=SHOW_TRAJECTORY, compare=COMPARE,
                  block=True)
