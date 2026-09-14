"""Load the ORNL Inconel scan the way the collaborators' comparison scripts do.

Their parameter reader (utils.Get_projection_parameters_standalone_fast) mines
the scanner log, reads the raw counts, forms the statistical weights from the
dark-subtracted counts, and applies the per-view shifts.  Their scripts then
discard the reader's projections and read the beam-hardening and scatter
corrected projections from a tiff.  This module does the same and returns
everything in (view, row, column) order.

The reader needs about 90 GB of host memory and several minutes, so the first
call caches the weights and the parameter dictionaries under CACHE_DIR and
later calls read the cache.

An optional view step and detector step shrink the problem for a smoke test:
views are taken every view_step, detector pixels are block averaged over
det_step x det_step, and the volume is shrunk by the same factor.
"""
import os
import pickle
import time

import numpy as np
import tifffile

SCAN_DIR = '/depot/bouman/data/ORNL/mbirtorch_vs_leap/483-103_2132x8/'
TIFF_PATH = ('/depot/bouman/data/ORNL/mbirtorch_vs_leap/'
             '483-103_2132x8_proj_BHC_SC_2132vws.tiff')
CACHE_DIR = '/scratch/gautschi/buzzard/leap_ornl/data/'
WEIGHTS_CACHE = os.path.join(CACHE_DIR, 'weights_vrc_float32.npy')
PARAMS_CACHE = os.path.join(CACHE_DIR, 'params.pkl')


def log(msg):
    print(f'[ornl_data {time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _read_raw():
    from utils.Get_projection_parameters_standalone_fast import proj_and_params
    proj, weights, proj_params, miscalib, vol_params, rec_params = proj_and_params(SCAN_DIR)
    del proj
    weights = np.ascontiguousarray(weights.transpose([1, 0, 2]), dtype=np.float32)
    params = dict(proj_params=proj_params, miscalib=miscalib,
                  vol_params=vol_params, rec_params=rec_params)
    return weights, params


def _block_mean(array, step):
    """Average step x step blocks of the last two axes, one view at a time."""
    views, rows, cols = array.shape
    rows_out, cols_out = rows // step, cols // step
    out = np.empty((views, rows_out, cols_out), dtype=np.float32)
    for v in range(views):
        block = array[v, :rows_out * step, :cols_out * step].astype(np.float32)
        out[v] = block.reshape(rows_out, step, cols_out, step).mean(axis=(1, 3))
    return out


def load(view_step=1, det_step=1):
    """Return (projections, weights, params) as float32 (view, row, column)."""
    t0 = time.time()
    if os.path.exists(WEIGHTS_CACHE) and os.path.exists(PARAMS_CACHE):
        log('reading cached weights and parameters')
        weights = np.load(WEIGHTS_CACHE)
        with open(PARAMS_CACHE, 'rb') as f:
            params = pickle.load(f)
    else:
        log("reading the raw scan with the collaborators' reader")
        weights, params = _read_raw()
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(WEIGHTS_CACHE, weights)
        with open(PARAMS_CACHE, 'wb') as f:
            pickle.dump(params, f)
        log('cached the weights and parameters')
    log(f'reading the corrected projections from {TIFF_PATH}')
    proj = tifffile.imread(TIFF_PATH)
    log(f'tiff read: shape {proj.shape} dtype {proj.dtype}, {time.time() - t0:.0f} s so far')

    if view_step > 1:
        proj = proj[::view_step]
        weights = weights[::view_step]
        params['proj_params']['angles'] = params['proj_params']['angles'][::view_step]
    if det_step > 1:
        proj = _block_mean(proj, det_step)
        weights = _block_mean(weights, det_step)
        cone = params['proj_params']['cone_params']
        cone['pix_x'] *= det_step
        cone['pix_y'] *= det_step
        vol = params['vol_params']
        for key in ('n_vox_x', 'n_vox_y', 'n_vox_z'):
            vol[key] = int(vol[key]) // det_step
        vol['vox_xy'] = np.float32(vol['vox_xy'] * det_step)
        vol['vox_z'] = np.float32(vol['vox_z'] * det_step)
    else:
        proj = np.ascontiguousarray(proj, dtype=np.float32)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    params['proj_params']['dims'] = np.array([proj.shape[1], proj.shape[0], proj.shape[2]])
    log(f'projections {proj.shape} weights {weights.shape} '
        f'views {len(params["proj_params"]["angles"])}, loaded in {time.time() - t0:.0f} s')
    return proj, weights, params


def describe(params):
    """Print the geometry the collaborators' scripts derive from the reader."""
    cone = params['proj_params']['cone_params']
    vol = params['vol_params']
    mis = params['miscalib']
    print('  detector pitch (mm):', cone['pix_x'], cone['pix_y'])
    print('  source to axis (mm):', cone['src_orig'],
          ' source to detector (mm):', cone['src_orig'] + cone['orig_det'])
    print('  volume:', vol['n_vox_x'], vol['n_vox_y'], vol['n_vox_z'],
          ' voxel (mm):', vol['vox_xy'], vol['vox_z'])
    print('  miscalibration delta_u, delta_v (mm):', mis['delta_u'], mis['delta_v'])
    angles = params['proj_params']['angles']
    print(f'  angles: {len(angles)} from {np.rad2deg(angles[0]):.4f} to '
          f'{np.rad2deg(angles[-1]):.4f} degrees')
