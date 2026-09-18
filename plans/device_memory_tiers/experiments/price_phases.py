"""Print every ledger phase for the ORNL geometry on four devices with the
torch-body batch terms removed, to see which phase holds the measured peak."""
import numpy as np
import mbirtorch
GB = 1024 ** 3
BODY_TERMS = ('back batch', 'forward batch')
views, rows, cols = 2132, 1456, 1840
sod, sdd, pitch = 110.080, 808.814, 0.127
angles = -np.deg2rad(0.2284520 + np.arange(views) * 360.0 / views)
for k in (4,):
    model = mbirtorch.ConeBeamModel((views, rows, cols), angles, source_detector_dist=sdd, source_iso_dist=sod)
    model.set_params(recon_shape=(1360, 1360, 1296), delta_det_channel=pitch, delta_det_row=pitch,
                     delta_voxel=pitch * sod / sdd, det_channel_offset=0.880, det_row_offset=0.181,
                     use_ror_mask=False, no_warning=True, verbose=0)
    model.configure_devices(devices=['cpu'] * k)
    ledger = model._build_memory_ledger(workload='recon', weights='placeholder', init_recon='placeholder')
    print(f'== {k} devices: phase totals on the lead device, body batch terms removed')
    rows_out = []
    for ph in ledger.phases:
        body = sum(v[0] for n, v in ph.terms if n in BODY_TERMS)
        rows_out.append((ph.per_device[0] - body, ph.name, ph))
    for total, name, ph in sorted(rows_out, reverse=True)[:7]:
        print(f'  {total / GB:7.2f} GiB  {name}')
    for total, name, ph in sorted(rows_out, reverse=True)[:3]:
        print(f'  terms of {name}:')
        for n, v in ph.terms:
            if v[0] >= 0.05 * GB:
                print(f'     {n:<36s} {v[0] / GB:6.2f} GiB')
