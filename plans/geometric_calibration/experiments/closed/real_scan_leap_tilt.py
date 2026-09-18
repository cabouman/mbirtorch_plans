"""LEAP's tilt estimate on the real NSI and Zeiss scans with the full detector height.

The earlier jobs gave LEAP's ``estimate_tilt`` a band of 128 detector rows, and it returned the bound
of its search on two NSI scans, 0.006 degrees on the third, and -0.402 degrees on the Zeiss scan
whose rotation is near zero.  LEAP's method compares conjugate projections, those 180 degrees apart,
after rebinning fan-beam or cone-beam data to parallel coordinates, over every row it is given.  A
detector rotation shifts each row's content along the channels in proportion to the row's height
above the central plane, so a band of 128 rows removes most of that signal.  This job gives LEAP the
whole detector and asks the same question.  It also records LEAP's own cost, the mean square of its
``conjugate_difference``, at a grid of tilt angles, so the record shows what LEAP's metric prefers.

Three scans run: the NSI phantom without metal, whose detector rotation direct reconstructions put
at the vendor's 0.167 degrees; the same phantom with a metal insert; and the Zeiss ball grid array,
whose rotation the module and a row-band fit both put below 0.02 degrees.  Each scan runs LEAP twice,
once on the whole detector and once on the 128-row band the earlier jobs used, so the height is the
only thing that changes.  LEAP's centerCol is set from the vendor's offset, as LEAP's documentation
asks.  The results are transcribed to ``real_scan_leap_tilt.md`` beside this file.
"""
import math
import os
import resource
import sys
import time
import traceback
from gc import collect as collect_garbage

os.environ.setdefault('MBIRTORCH_NUM_DEVICES', '1')

import numpy as np
import torch

import mbirtorch
from mbirtorch.preprocess import geometry_calibration as gc

import real_scan_validation as first_job
import real_scan_followup as second_job
from real_scan_validation import central_band, git_commit, record
from real_scan_followup import load_scan

torch.set_num_threads(14)

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
DATASETS = (
    dict(name='nsi_no_metal', reader='nsi',
         path='/depot/bouman/data/Lilly/demo_nsi_vert_no_metal_all_views.tgz'),
    dict(name='nsi_metal', reader='nsi',
         path='/depot/bouman/data/Lilly/demo_nsi_vert_metal_all_views.tgz'),
    dict(name='bga', reader='zeiss',
         path='/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm'),
)
BAND_ROWS = 128                                   # the band the earlier jobs gave LEAP
TILT_GRID_DEGREES = np.arange(-0.4, 0.401, 0.05)  # LEAP's cost is evaluated at these angles

RESULTS = first_job.RESULTS
DATA = first_job.DATA
JSONL = os.path.join(RESULTS, 'real_scan_leap_tilt.jsonl')
first_job.JSONL = JSONL


def leap_model(scan, num_rows, center_row):
    """A LEAP cone-beam model of the scan with ``num_rows`` detector rows and the given center row.

    LEAP's view angle is 180 degrees minus mbirtorch's, and the channel axis runs the other way, as
    the LEAP comparison established.  centerCol is set from the vendor's offset, negated for the
    channel reversal.
    """
    from leapctype import tomographicModels
    leapct = tomographicModels()
    leapct.set_gpu(0)
    angles = gc._view_angles(scan.ct_model)
    phis = np.ascontiguousarray(180.0 - np.degrees(angles), dtype=np.float32)
    center_col = (scan.num_channels - 1) / 2.0 - scan.vendor_offset / scan.delta_det_channel
    leapct.set_conebeam(scan.num_views, num_rows, scan.num_channels, scan.delta_det_row,
                        scan.delta_det_channel, center_row, center_col, phis,
                        float(scan.ct_model.get_params('source_iso_dist')),
                        float(scan.ct_model.get_params('source_detector_dist')))
    leapct.set_flatDetector()
    return leapct, center_col


def leap_on_rows(scan, lo, hi, label):
    """LEAP's tilt estimate and cost curve on detector rows ``[lo, hi)``."""
    start = time.perf_counter()
    data = np.ascontiguousarray(scan.sino[:, lo:hi][:, :, ::-1], dtype=np.float32)
    copy_seconds = time.perf_counter() - start
    center_row = scan.central_row - lo
    leapct, center_col = leap_model(scan, hi - lo, center_row)
    start = time.perf_counter()
    tilt = float(leapct.estimate_tilt(data))
    tilt_seconds = time.perf_counter() - start
    costs = []
    start = time.perf_counter()
    for alpha in TILT_GRID_DEGREES:
        difference = leapct.conjugate_difference(data, alpha=float(alpha), centerCol=center_col)
        costs.append(float(np.mean(np.asarray(difference, dtype=np.float64) ** 2)))
    curve_seconds = time.perf_counter() - start
    del data
    collect_garbage()
    record(scan.name, 'leap_tilt', tilt_seconds, rows=label, row_window=[int(lo), int(hi)],
           num_rows=int(hi - lo), center_row=center_row, center_col=center_col,
           tilt_degrees=tilt, at_search_bound=bool(abs(abs(tilt) - 5.0) < 1e-3),
           cost_grid_degrees=[float(a) for a in TILT_GRID_DEGREES], costs=costs,
           cost_argmin_degrees=float(TILT_GRID_DEGREES[int(np.argmin(costs))]),
           cost_ratio=max(costs) / max(min(costs), 1e-30), copy_seconds=copy_seconds,
           curve_seconds=curve_seconds)


def run_dataset(spec):
    name = spec['name']
    dataset_start = time.perf_counter()
    scan = None
    try:
        scan = load_scan(spec)
        if scan is None:
            return
        for label, (lo, hi) in (('full', (0, scan.num_rows)), ('band', central_band(scan.ct_model, BAND_ROWS))):
            try:
                leap_on_rows(scan, lo, hi, label)
            except Exception:
                record(name, 'leap_tilt', 0.0, rows=label, traceback=traceback.format_exc())
    except Exception:
        record(name, 'error', time.perf_counter() - dataset_start, traceback=traceback.format_exc())
    finally:
        record(name, 'resources', time.perf_counter() - dataset_start,
               max_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2)
        del scan
        collect_garbage()


def main():
    os.makedirs(RESULTS, exist_ok=True)
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(mbirtorch.__file__)))
    record('job', 'environment', 0.0, torch=torch.__version__,
           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
           mbirtorch=mbirtorch.__version__, mbirtorch_commit=git_commit(package_root),
           results=RESULTS, jsonl=JSONL, band_rows=BAND_ROWS,
           tilt_grid_degrees=[float(a) for a in TILT_GRID_DEGREES], argv=sys.argv)
    for spec in DATASETS:
        run_dataset(spec)
    print('REAL_SCAN_LEAP_TILT DONE', flush=True)


if __name__ == '__main__':
    main()
