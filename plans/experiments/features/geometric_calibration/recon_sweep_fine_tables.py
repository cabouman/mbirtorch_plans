"""Print the record of a ``recon_sweep_fine`` job as tables.

The job writes one JSON line per measurement, and the lines that hold the score curves are long.
This script reads the file and prints what a reader needs: the environment, each scan's loading
facts, the slices with their timing and identity checks, each slice's score at the named candidates,
the trimmed curves with their minima, half widths, depths, and floors, the resources, and the closing
comparison between the two scans.  It runs with numpy-free standard Python.  Run parameters are at
the top; the file is named by the environment variable the job's batch file also sets, so the same
directory serves both.
"""
import json
import math
import os

# ── run parameters ────────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.environ.get('RECON_SWEEP_FINE_RESULTS', 'results_recon_sweep_fine')
JSONL_NAME = 'recon_sweep_fine.jsonl'


def load(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fmt(value, digits=4):
    """A short text form of a number, a bool, or a missing value."""
    if value is None:
        return '-'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, int):
        return str(value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return 'nan'
    if abs(value) >= 1e-2 or value == 0.0:
        return f'{value:.{digits}f}'
    return f'{value:.3e}'


def print_environment(entries):
    for e in entries:
        if e['kind'] == 'kernel':
            print('KERNEL')
            print(f"  {e.get('kernel')}, extra row margin {e.get('extra_row_margin')}, "
                  f"reflect rows {e.get('reflect_rows')}")
    for e in entries:
        if e['kind'] == 'environment':
            print('ENVIRONMENT')
            for key in ('torch', 'gpu', 'mbirtorch', 'mbirtorch_commit', 'mbirtorch_file', 'results',
                        'candidate_ranges_degrees', 'candidate_step_degrees', 'named_candidates_degrees',
                        'slice_fractions', 'blur_voxels', 'parabola_points', 'half_width_rise',
                        'downsample_factor'):
                print(f'  {key}: {e.get(key)}')
            print()


def print_scan(entries, name):
    print(f'===================== {name} =====================')
    for e in entries:
        if e['dataset'] != name:
            continue
        if e['kind'] == 'scan':
            print(f"  scan: shape {e.get('shape')} load {fmt(e['seconds'], 1)} s, "
                  f"vendor tilt {fmt(e.get('vendor_det_rotation_degrees'), 5)} deg, "
                  f"vendor offset {fmt(e.get('vendor_offset_channels'), 3)} ch, "
                  f"central row {fmt(e.get('central_row'), 2)}, coverage "
                  f"{fmt(e.get('angular_coverage_degrees'), 2)} deg")
        if e['kind'] == 'half_models':
            print(f"  half models: views {e.get('num_views')} -> even {e['even']['num_views']}, "
                  f"odd {e['odd']['num_views']}; geometry reset even {e['even']['half_geometry_reset']}, "
                  f"odd {e['odd']['half_geometry_reset']}; largest slice z difference "
                  f"{fmt(e['even']['largest_slice_z_difference'])}, "
                  f"{fmt(e['odd']['largest_slice_z_difference'])}; recon_shape {e.get('recon_shape')}, "
                  f"delta_voxel {fmt(e.get('delta_voxel'), 5)}, aspect {e.get('voxel_row_aspect')}")
            print(f"  candidates ({len(e['candidates_degrees'])}): "
                  + ', '.join(f'{c:.4f}' for c in e['candidates_degrees']))
            print(f"  named: {e.get('named_degrees')} at indices {e.get('named_indices')}")
        if e['kind'] in ('skip', 'error'):
            print(f"  {e['kind'].upper()}: {e.get('reason')} {str(e.get('traceback', ''))[-600:]}")

    slices = [e for e in entries if e['dataset'] == name and e['kind'] == 'slice']
    print('\n  SLICES')
    print('  slice | rows from plane | z (ALU) | even s | odd s | identity s | identity rel max | '
          'identity rms rel | mean square at vendor | shape')
    for e in slices:
        if 'traceback' in e:
            print(f"  slice {e.get('slice_index')}: TRACEBACK {e['traceback'][-500:]}")
            continue
        print(f"  {e['slice_index']} | {fmt(e['rows_from_central_plane'], 1)} | {fmt(e['z_alu'], 2)} | "
              f"{fmt(e['seconds_even'], 1)} | {fmt(e['seconds_odd'], 1)} | {fmt(e['seconds_identity'], 1)} | "
              f"{fmt(e['identity_rel_max'])} | {fmt(e['identity_rms_rel'])} | "
              f"{fmt(e['mean_square_at_identity'])} | {e['slice_shape']}")

    scored = [e for e in slices if 'scores' in e]
    if scored:
        named = scored[0]['named_indices']
        candidates = scored[0]['candidates_degrees']
        print('\n  PER-SLICE SCORES (mean parity) at the named candidates, and each slice\'s lowest candidate')
        for blur in scored[0]['blur_voxels']:
            key = str(float(blur))
            print(f'   blur {key}:')
            for e in scored:
                row = e['scores']['mean'][key]
                finite = [(v, i) for i, v in enumerate(row) if v is not None and not math.isnan(v)]
                arg = min(finite)[1] if finite else -1
                at = ', '.join(f'{label} {fmt(row[idx])}' for label, idx in named.items())
                lowest = fmt(candidates[arg], 4) if arg >= 0 else '-'
                print(f"     slice {e['slice_index']} ({fmt(e['rows_from_central_plane'], 0)} rows): "
                      f"{at}; lowest {lowest} deg")

    curves = [e for e in entries if e['dataset'] == name and e['kind'] == 'curve']
    for kind in ('raw', 'normalized'):
        print(f'\n  TRIMMED CURVES, {kind}')
        print('  blur | parity | lowest deg | at end | fit deg | half width deg | opens up | depth | '
              'floor | depth/floor | interior minima | score at named | per-slice fit minima deg | '
              'dropped at the first candidate')
        for e in curves:
            s = e[kind]
            p = s['parabola']
            ratio = s.get('depth_over_floor')
            named = ', '.join(f'{k} {fmt(v)}' for k, v in s.get('score_at_named', {}).items())
            slice_fit = ', '.join(fmt(v, 4) for v in s['slice_locations']['parabola_degrees'])
            dropped = s['dropped_slices'][0] if s['dropped_slices'] else []
            print(f"  {e['blur']} | {e['parity']} | {fmt(s['argmin_degrees'], 4)} | {fmt(s['argmin_at_end'])} | "
                  f"{fmt(p['location_degrees'], 4)} | {fmt(p['half_width_degrees'], 4)} | "
                  f"{fmt(p['opens_upward'])} | {fmt(s['depth'])} | {fmt(s.get('floor'))} | "
                  f"{fmt(ratio, 2) if ratio is not None else '-'} | {s['interior_minima']} | {named} | "
                  f"{slice_fit} | {dropped}")
    for e in entries:
        if e['dataset'] == name and e['kind'] == 'resources':
            print(f"\n  resources: {fmt(e['seconds'], 1)} s, max rss {fmt(e.get('max_rss_gb'), 1)} GB, "
                  f"gpu peak {fmt(e.get('gpu_peak_gb'), 2)} GB")
    print()


def print_gates(entries):
    gates = [e for e in entries if e['kind'] == 'gates']
    if not gates:
        return
    print('===================== CLOSING COMPARISON =====================')
    for kind in ('raw', 'normalized'):
        print(f'  {kind}:')
        print('  blur | first min | half width | second min | half width | difference | agree | '
              'metal in bracket | depth>floor first | second')
        for e in gates:
            g = e[kind]
            a, b = g['first'], g['second']
            print(f"  {e['blur']} | {fmt(a['minimum_degrees'], 4)} | {fmt(a['half_width_degrees'], 4)} | "
                  f"{fmt(b['minimum_degrees'], 4)} | {fmt(b['half_width_degrees'], 4)} | "
                  f"{fmt(g['difference_degrees'], 4)} | {fmt(g['agree_within_half_width'])} | "
                  f"{fmt(g['metal_min_inside_bracket'])} | {fmt(a.get('depth_exceeds_floor'))} | "
                  f"{fmt(b.get('depth_exceeds_floor'))}")


def main():
    entries = load(os.path.join(RESULTS_DIR, JSONL_NAME))
    print_environment(entries)
    for name in sorted({e['dataset'] for e in entries if e['dataset'] != 'job'}):
        print_scan(entries, name)
    print_gates(entries)


if __name__ == '__main__':
    main()
