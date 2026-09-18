"""Price the device arrays of a VCD reconstruction of the ORNL Inconel scan
with mbirtorch's memory ledger, on one, two, and four devices.

The geometry is the one the collaborators' scripts build: 2132 views, a
1456 x 1840 detector at 0.127 mm, source to axis 110.080 mm, source to
detector 808.814 mm, and a 1360 x 1360 x 1296 volume at the detector pitch
divided by the magnification.  Weights are passed, as their scripts do.
Pricing runs on CPU devices, so the per-subset transients are the torch
bodies' and overstate an H100 peak; the persistent terms depend on the
geometry and the device count only.
"""
import numpy as np
import mbirtorch

GB = 1024 ** 3
PERSISTENT = ('error sinogram', 'weights', 'flat recon', 'hessian diagonal',
              'prox input', 'partitions (lead device)', 'library workspace')


def price(num_devices):
    views, rows, cols = 2132, 1456, 1840
    sod, sdd, pitch = 110.080, 808.814, 0.127
    angles = -np.deg2rad(0.2284520 + np.arange(views) * 360.0 / views)
    model = mbirtorch.ConeBeamModel((views, rows, cols), angles,
                                    source_detector_dist=sdd, source_iso_dist=sod)
    model.set_params(recon_shape=(1360, 1360, 1296), delta_det_channel=pitch,
                     delta_det_row=pitch, delta_voxel=pitch * sod / sdd,
                     det_channel_offset=0.880, det_row_offset=0.181,
                     use_ror_mask=False, no_warning=True, verbose=0)
    model.configure_devices(devices=['cpu'] * num_devices)
    ledger = model._build_memory_ledger(workload='recon', weights='placeholder')
    phases = ledger.phases

    def total(phase):
        return max(phase.per_device)

    dominant = max(phases, key=total)
    subset_phases = [p for p in phases if p.name.startswith('subset ')]
    widest = max(subset_phases, key=total)
    lead = int(np.argmax(widest.per_device))
    persistent = sum(v[lead] for n, v in widest.terms if n in PERSISTENT)
    print(f'== {num_devices} device(s): recon shape {tuple(model.get_params("recon_shape"))}')
    print(f'   dominant phase: {dominant.name}: {total(dominant) / GB:.2f} GB on the lead device')
    print(f'   widest subset step: {widest.name}: {total(widest) / GB:.2f} GB on the lead device')
    for n, v in widest.terms:
        if v[lead] >= 0.05 * GB:
            print(f'     {n:<40s} {v[lead] / GB:8.2f} GB')
    print(f'   persistent set on the lead device: {persistent / GB:.2f} GB')
    print(f'   per-device totals of the widest step: '
          + ', '.join(f'{x / GB:.2f}' for x in widest.per_device) + ' GB')


for k in (1, 2, 4):
    price(k)
