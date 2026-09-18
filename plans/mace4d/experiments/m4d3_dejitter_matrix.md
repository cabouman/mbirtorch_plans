# m4d3: the temporal dejitter filter as one matrix

## What the script does

The mbirjax temporal filter `_dejitter_4d_dct` zeroes a few DCT-I modes along
the frame axis and leaves the spatial axes untouched, so it is a linear map on
the frame axis alone.  That map is one N by N matrix, where N is the number of
frames.  This script copies the filter, builds its matrix by filtering each
unit frame impulse, checks that the matrix reproduces the filter, reports the
properties of the matrix, and times the filter form against the matrix form.

No file in the mbirtorch package was changed.  The copy of the filter differs
from the mbirjax original in one way: it takes a `workers` argument that it
hands to scipy's `dct` and `idct`.  With `workers=None` it calls them exactly
as the original does.

## Command

```
cd "/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch" && \
/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python \
  "/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/experiments/features/mace4d/m4d3_dejitter_matrix.py" \
  2>&1 | tee "/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/experiments/features/mace4d/results/m4d3_dejitter_matrix.txt"
```

## Environment

Python 3.11.15, torch 2.13.0, numpy 2.4.6, scipy 1.17.1.  Machine: Apple M3
Max, 103 GB memory, macOS.  Devices used: cpu and mps.
`torch.get_num_threads()` reported 10.

Filter settings for every part: period 6, harmonics True, band width 1,
working dtype float32.

## Matrix against filter, and matrix properties

The agreement check ran on random float32 arrays of shape (N, 16, 16, 8).

| N | max abs difference, matrix against filter | max abs value of the input |
| --- | --- | --- |
| 3 | 0.000000e+00 | 4.138789 |
| 12 | 7.152557e-07 | 3.988657 |
| 30 | 1.192093e-06 | 4.663319 |

| N | max abs(M - M^T) | max abs(M M - M) | rank, numpy default tolerance | rank, float32 tolerance |
| --- | --- | --- | --- | --- |
| 3 | 0.000000e+00 | 0.000000e+00 | 0 | 0 |
| 12 | 4.470348e-08 | 2.233025e-08 | 12 | 4 |
| 30 | 5.960464e-08 | 1.487817e-07 | 30 | 22 |

M for N = 3 is the 3 by 3 zero matrix.

The singular values fall into two groups.  At N = 12 there are 4 values equal
to 1 and 8 values between 5.7e-10 and 5.2e-08.  At N = 30 there are 22 values
equal to 1 and 8 values between 1.7e-09 and 4.0e-08.

## Timing

Input x of shape (30, 256, 256, 64) float32, which is 125,829,120 elements
and 0.503 GB.  Each number is the mean of 2 timed runs after one warm-up run.

| form | seconds |
| --- | --- |
| scipy filter, chunk_size=None | 1.526 |
| scipy filter, workers=-1 | 0.210 |
| torch CPU matrix form | 0.044 |
| torch mps matrix form, matmul only | 0.025 |

The scipy result and the torch CPU matrix result differ by at most
2.145767e-06.

## What the numbers say

The matrix reproduces the filter to at most 1.2e-06 on inputs whose largest
value is about 4.7.  The matrix is symmetric to 6.0e-08 and idempotent to
1.5e-07.  These are the sizes of difference that float32 arithmetic produces
at these array sizes.

The rank must be read with a float32 tolerance.  The matrix is built in
float32, so its zero singular values come out near 1e-08, and the numpy
default tolerance is a float64 one that counts them as nonzero.  At a float32
tolerance the rank is 0 at N = 3, 4 at N = 12, and 22 at N = 30.  In each case
the rank equals N minus the number of DCT-I modes the filter zeroes.

At N = 3 with period 6 the filter zeroes every mode, so the matrix is zero and
the filter returns an array of zeros.  The frame count and the period together
decide whether any temporal content survives.

The scipy filter takes 1.526 s and the same filter with `workers=-1` takes
0.210 s, a factor of 7.3.  The torch CPU matrix form takes 0.044 s, which is
35 times faster than the single-threaded scipy filter and 4.8 times faster
than the threaded one.  The mps matmul takes 0.025 s.

The mps time covers the matrix multiply only.  The data was already on the
device and the 0.5 GB transfer in each direction is not included.

Nothing failed and nothing was skipped.
