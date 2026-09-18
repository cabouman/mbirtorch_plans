"""Does the rotation estimate on the no-metal NSI scan recover when the band reaches the object's
structure?

The band sweep of ``real_scan_band_height.py`` took the comparison band up to 201 rows, which is
100 rows on each side of the central plane, and the estimate moved only from 0.047 to 0.071
degrees against the vendor's 0.167.  This object's cross-row structure sits 470 to 752 rows from
the central plane, so no band of that sweep reached it.  This job runs the same sweep on the same
scan at two heights that do reach it: 501 rows, whose edge at 250 rows is still short of the
structure, and 1001 rows, whose edge at 500 rows is past the nearest of it.  If the estimate
moves to the vendor's value at 1001 rows and not at 501, the estimate needs the structure itself;
if it moves at neither, something else anchors it.

Everything else, the loading, the fixed channel offset, the recorded fields, and the band
statistic, is the band-height job's, imported and rerun with these settings.  The view comparison
and the geometry vectors of that job are not repeated.  The default band runs again first, as the
tie to the earlier sweep.

Run through ``real_scan_band_reach.sbatch``, which sets this job's results directory.
"""
import os

import real_scan_band_height as sweep

# The settings that differ from the band-height job.  The 1001-row band holds about 11 GB per
# band array on this scan, and the comparison holds a few of them, so the batch file asks for two
# GPUs to get the host memory.
sweep.DATASETS = (
    dict(name='nsi_no_metal', reader='nsi',
         path='/depot/bouman/data/Lilly/demo_nsi_vert_no_metal_all_views.tgz'),
)
sweep.BAND_ROWS = (None, 501, 1001)
sweep.VIEW_COMPARISON_DATASET = 'none'
sweep.VECTOR_DATASET = 'none'

# Results go to this job's own file, in the directory the batch file exported before the
# interpreter started.
sweep.JSONL = os.path.join(sweep.RESULTS, 'real_scan_band_reach.jsonl')
sweep.first_job.JSONL = sweep.JSONL

if __name__ == '__main__':
    sweep.main()
