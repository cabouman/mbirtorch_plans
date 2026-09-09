"""Open the interactive geometry viewer on an example model.

Run this from a checkout of mbirtorch_plans in an environment where mbirtorch
is installed and a display is available:

    cd plans/experiments/geometry_viewer
    python gv_show_example.py

The window has a view slider, a "source path" toggle that draws the source's
trajectory over all views, and a "3D zoom to volume" toggle.  The 3D panel can
be rotated with the mouse.  Closing the window ends the script.

To view your own model instead, replace the block under "Build the model" with
the model you have, for example the one that a scanner reader returns:

    ct_model = mbirtorch.preprocess.nsi.get_sino_and_model(dataset_dir)['model']

To compare two geometries, pass a second model or a dictionary of parameter
overrides as ``compare``; the second geometry is drawn dashed and the text
panel lists every parameter and derived quantity that differs.
"""

import numpy as np

import mbirtorch
from geometry_viewer import show_geometry

# ── run parameters ───────────────────────────────────────────────────────────

NUM_VIEWS = 180
NUM_DET_ROWS = 96
NUM_DET_CHANNELS = 128
SOURCE_DETECTOR_DIST = 4.0 * NUM_DET_CHANNELS
SOURCE_ISO_DIST = SOURCE_DETECTOR_DIST / 2.0

# Detector offsets in ALU.  Change one and watch the central-ray marker move
# on the detector face and the offset label move in the top and side views.
DET_CHANNEL_OFFSET = 3.0
DET_ROW_OFFSET = -2.0

# Set to None for no comparison, or to a dictionary of parameter overrides.
# The example shifts the detector by ten more channels.
COMPARE = dict(det_channel_offset=DET_CHANNEL_OFFSET + 10.0)

VIEW_INDEX = 0
SHOW_TRAJECTORY = True

# ── Build the model ──────────────────────────────────────────────────────────

angles = np.linspace(0.0, 2.0 * np.pi, NUM_VIEWS, endpoint=False)
ct_model = mbirtorch.ConeBeamModel((NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS),
                                   angles,
                                   source_detector_dist=SOURCE_DETECTOR_DIST,
                                   source_iso_dist=SOURCE_ISO_DIST,
                                   compile_mode='off')
ct_model.set_params(det_channel_offset=DET_CHANNEL_OFFSET,
                    det_row_offset=DET_ROW_OFFSET, no_warning=True)

# ── Show it ──────────────────────────────────────────────────────────────────

show_geometry(ct_model, view_index=VIEW_INDEX, show_trajectory=SHOW_TRAJECTORY,
              compare=COMPARE, block=True)
