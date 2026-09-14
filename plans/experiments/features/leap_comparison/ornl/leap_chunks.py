"""Price LEAP's chunks for the full-size ORNL geometry from its slab formulas.

Forward projection with several GPUs takes 128 detector rows per job and
needs the volume slices those rows can see; back projection takes 128 slices
per job and needs the detector rows that see them.  The formulas follow
parameters.cpp (sliceRangeNeededForProjection, rowRangeNeededForBackprojection)
for a cone beam with the field-of-view radius at the volume's furthest voxel.
"""
import math

views, rows, cols = 2132, 1456, 1840
pitch, sod, sdd = 0.127, 110.080, 808.814
nx, ny, nz = 1360, 1360, 1296
vox = pitch * sod / sdd
center_row = rows / 2 - (-0.180975) / pitch - 0.5
half_width = 0.5 * (nx - 1) * vox
radius = math.hypot(half_width, half_width)
z0 = -0.5 * (nz - 1) * vox
GB = 1024 ** 3
proj_bytes = views * rows * cols * 4
vol_bytes = nx * ny * nz * 4


def slices_for_rows(first, last):
    v_lo = ((first - center_row) * pitch - 0.5 * pitch) / sdd
    v_hi = ((last - center_row) * pitch + 0.5 * pitch) / sdd
    dmin, dmax = sod - radius - vox, sod + radius + vox
    z_lo = min(v_lo * dmin, v_lo * dmax) - 0.5 * vox
    z_hi = max(v_hi * dmin, v_hi * dmax) + 0.5 * vox
    lo = max(0, min(nz - 1, math.floor((z_lo - z0) / vox)))
    hi = max(0, min(nz - 1, math.ceil((z_hi - z0) / vox)))
    return lo, hi


def rows_for_slices(first, last):
    z_lo = first * vox + z0 - 0.5 * vox
    z_hi = last * vox + z0 + 0.5 * vox
    dmin, dmax = sod - radius - vox, sod + radius + vox
    vs = [z_lo / dmin, z_lo / dmax, z_hi / dmin, z_hi / dmax]
    r_lo = min(vs) * sdd / pitch + center_row - 1
    r_hi = max(vs) * sdd / pitch + center_row + 1
    lo = max(0, min(rows - 1, math.floor(r_lo)))
    hi = max(0, min(rows - 1, math.ceil(r_hi)))
    return lo, hi


print(f'sinogram {proj_bytes / GB:.2f} GiB, volume {vol_bytes / GB:.2f} GiB, '
      f'field-of-view radius {radius:.2f} mm')
print('forward projection, 128-row jobs:')
worst = 0
for first in range(0, rows, 128):
    last = min(first + 127, rows - 1)
    lo, hi = slices_for_rows(first, last)
    need = (last - first + 1) / rows * proj_bytes + (hi - lo + 1) / nz * vol_bytes
    worst = max(worst, need)
    print(f'  rows {first:4d}-{last:4d}: slices {lo:4d}-{hi:4d} ({hi - lo + 1:3d}), '
          f'projection chunk {(last - first + 1) / rows * proj_bytes / GB:.2f} GiB, '
          f'slab {(hi - lo + 1) / nz * vol_bytes / GB:.2f} GiB')
print(f'  widest job {worst / GB:.2f} GiB plus LEAP\'s 0.25 GB reserve')
print('back projection, 128-slice jobs:')
worst = 0
for first in range(0, nz, 128):
    last = min(first + 127, nz - 1)
    lo, hi = rows_for_slices(first, last)
    need = (last - first + 1) / nz * vol_bytes + (hi - lo + 1) / rows * proj_bytes
    worst = max(worst, need)
    print(f'  slices {first:4d}-{last:4d}: rows {lo:4d}-{hi:4d} ({hi - lo + 1:3d}), '
          f'slab {(last - first + 1) / nz * vol_bytes / GB:.2f} GiB, '
          f'projection rows {(hi - lo + 1) / rows * proj_bytes / GB:.2f} GiB')
print(f'  widest job {worst / GB:.2f} GiB plus LEAP\'s 0.25 GB reserve')
