"""Split the FDK of the ORNL scan into its parts for both codes on the same
four cards: filtering, back projection, and the movement of data between host
and devices.  Each part is timed twice so the second figure is warm.

mbirtorch is pinned to four devices, which is the count its policy chose in
the full run.  The sinogram is placed on the devices once, then the filter
and the back projection run in their device forms, then the volume is
gathered to the host, which is the sequence recon_direct runs.  LEAP's FBP
is timed whole, then its filter and its plain back projection separately;
LEAP reads the host array and streams it itself.
"""
import sys
import time

import numpy as np

import ornl_data

arm = sys.argv[1]
proj, weights, params = ornl_data.load()
del weights
cone = params['proj_params']['cone_params']
vol = params['vol_params']
mis = params['miscalib']
angles = params['proj_params']['angles']
sod = cone['src_orig']
sdd = cone['src_orig'] + cone['orig_det']


def report(label, seconds):
    print(f'{label}: {seconds:.2f} s', flush=True)


if arm == 'mbirtorch':
    import torch
    import mbirtorch

    def sync():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)

    model = mbirtorch.ConeBeamModel(proj.shape, angles, source_detector_dist=sdd, source_iso_dist=sod)
    model.set_params(recon_shape=(int(vol['n_vox_x']), int(vol['n_vox_y']), int(vol['n_vox_z'])),
                     delta_det_channel=cone['pix_y'], delta_det_row=cone['pix_x'],
                     delta_voxel=vol['vox_xy'], det_channel_offset=-mis['delta_u'],
                     det_row_offset=-mis['delta_v'], use_ror_mask=False, verbose=0)
    model.configure_devices(num_devices=torch.cuda.device_count())
    for rep in range(2):
        sync(); t = time.time()
        recon = model.recon_direct(proj)
        sync(); report(f'recon_direct whole, run {rep}', time.time() - t)
    for rep in range(2):
        sync(); t = time.time()
        sino_dev = model._shard_sinogram(proj)
        sync(); report(f'host to devices (shard sinogram), run {rep}', time.time() - t)
        t = time.time()
        filtered = model.fdk_filter(sino_dev, output_sharded=True)
        sync(); report(f'fdk_filter on devices, run {rep}', time.time() - t)
        t = time.time()
        recon_dev = model.back_project(filtered, output_sharded=True)
        sync(); report(f'back_project on devices, run {rep}', time.time() - t)
        t = time.time()
        recon = model._gather_recon(recon_dev)
        sync(); report(f'devices to host (gather recon), run {rep}', time.time() - t)
        del sino_dev, filtered, recon_dev
    print('placement:', model.placement if hasattr(model, 'placement') else 'n/a', flush=True)
elif arm == 'leap':
    from leapctype import tomographicModels
    leapct = tomographicModels()
    dims = params['proj_params']['dims']
    numRows, numCols = int(dims[0]), int(dims[2])
    scan_angles = (angles * 180 / np.pi).astype('float32')
    pixelWidth, pixelHeight = cone['pix_x'], cone['pix_y']
    centerRow = numRows / 2 - mis['delta_v'] / pixelHeight - 0.5
    centerCol = numCols / 2 - mis['delta_u'] / pixelWidth - 0.5
    magnif = sdd / sod
    leapct.set_conebeam(len(scan_angles), numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol,
                        scan_angles, sod, sdd)
    leapct.set_volume(numX=int(vol['n_vox_x']), numY=int(vol['n_vox_y']), numZ=int(vol['n_vox_z']),
                      voxelWidth=pixelWidth / magnif, voxelHeight=pixelHeight / magnif)
    leapct.set_diameterFOV(1.0e16)
    print('LEAP GPUs in use:', list(leapct.get_gpus()), 'projector:', leapct.get_projector() if hasattr(leapct, 'get_projector') else 'n/a', flush=True)
    for rep in range(2):
        t = time.time()
        f = leapct.FBP(proj)
        report(f'FBP whole, run {rep}', time.time() - t)
    f = leapct.allocate_volume()
    for rep in range(2):
        t = time.time()
        leapct.backproject(proj, f)
        report(f'backproject (unfiltered data, plain adjoint), run {rep}', time.time() - t)
    q = proj.copy()
    t = time.time()
    leapct.filterProjections(q)
    report('filterProjections in place on the host copy', time.time() - t)
    t = time.time()
    leapct.weightedBackproject(q, f)
    report('weightedBackproject of the filtered data', time.time() - t)
print('PIECES DONE', flush=True)
