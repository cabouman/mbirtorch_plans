"""Price the device-resident arrays of a VCD reconstruction with the memory
ledger, for three cone-beam geometries, with and without a proximal input.

The ledger is the library's own model of what a reconstruction allocates.
For each case the script reports three quantities: the persistent set, which
is every array the loop holds for its whole run; the widest subset step of
the loop, which is the per-device peak the device policy fits to memory; and
the recon-shaped residents, which are the arrays a host-resident layout
would remove from the device.  It also reports the pixel count the
partitions cover, since the default region-of-reconstruction mask leaves
part of the grid out of every subset, and the dominant phase of the whole
run.

The ledger prices the projection bodies the model would run on the pricing
device.  On this Mac those are the torch bodies, whose per-subset transients
are far larger than the Triton kernels' on an H100, so the widest-step
totals printed here overstate a GPU run's peak.  The persistent terms depend
on the geometry only.

Run from the mbirtorch repository root; results print to stdout.
"""

import numpy as np

import mbirtorch

# Run parameters.
CASES = [
    ('dense cone beam, 512-class, 1.5 N views', 512, 768),
    ('dense cone beam, 2048-class, 1.5 N views', 2048, 3072),
    ('sparse cone beam, 2048-class, 0.25 N views', 2048, 512),
]
GB = 1024 ** 3
PERSISTENT = ('error sinogram', 'weights', 'flat recon', 'hessian diagonal',
              'prox input', 'partitions (lead device)', 'library workspace')
RECON_SHAPED = ('flat recon', 'hessian diagonal', 'prox input')


def price(name, n, views, prox):
    angles = np.linspace(0, 2 * np.pi, views, endpoint=False)
    model = mbirtorch.ConeBeamModel((views, n, n), angles,
                                    source_detector_dist=4 * n, source_iso_dist=2 * n)
    model.set_params(no_warning=True, verbose=0)
    model.configure_devices(devices=['cpu'])
    call_arrays = {'weights': 'placeholder'}
    if prox:
        call_arrays['prox_input'] = 'placeholder'
    ledger = model._build_memory_ledger(workload='recon', **call_arrays)
    phases = ledger.phases

    def total(phase):
        return phase.per_device[0]

    def term_sum(phase, names):
        return sum(values[0] for term_name, values in phase.terms if term_name in names)

    dominant = max(phases, key=total)
    subset_phases = [p for p in phases if p.name.startswith('subset ')]
    widest = max(subset_phases, key=total)
    persistent = term_sum(widest, PERSISTENT)
    recon_shaped = term_sum(widest, RECON_SHAPED)
    rows, cols = model.get_params('recon_shape')[:2]
    grid = int(rows) * int(cols)
    covered = int(ledger.num_pixels_full)

    print(f'== {name}, prox={prox}')
    print(f'   recon shape {tuple(model.get_params("recon_shape"))}, '
          f'sinogram shape {(views, n, n)}')
    print(f'   pixels covered by the partitions: {covered} of {grid} '
          f'({covered / grid:.4f} of the grid)')
    print(f'   dominant phase of the run: {dominant.name}: {total(dominant) / GB:.2f} GB')
    print(f'   widest subset step: {widest.name}: {total(widest) / GB:.2f} GB')
    for term_name, values in widest.terms:
        value = values[0]
        if value >= 0.01 * GB:
            print(f'     {term_name:<40s} {value / GB:8.2f} GB')
    print(f'   persistent set: {persistent / GB:.2f} GB')
    print(f'   recon-shaped residents: {recon_shaped / GB:.2f} GB')
    print(f'   recon-shaped share of the persistent set: {100 * recon_shaped / persistent:.1f} percent')
    print(f'   recon-shaped share of the widest subset step: '
          f'{100 * recon_shaped / total(widest):.1f} percent')
    for phase in subset_phases:
        if phase.name.endswith('(granularity 4)'):
            print(f'     {phase.name:<50s} {total(phase) / GB:8.2f} GB')


def main():
    for name, n, views in CASES:
        for prox in (False, True):
            price(name, n, views, prox)


if __name__ == '__main__':
    main()
