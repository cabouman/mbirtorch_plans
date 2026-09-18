# m4d2: the host-side consensus update

## What the script does

MACE4D runs one consensus update per iteration on four pairs of full-size 4D
arrays.  The update reads W and X, then writes new W, a new consensus average
xbar, and the percent change of xbar.  This script times three implementations
of that same arithmetic: the numpy in-place form copied from the mbirjax MACE4D
loop, a torch CPU form with the same sequence of in-place operations and one
scratch tensor, and a torch CPU form that uses no scratch array.

No file in the mbirtorch package was changed.

## Command

```
cd "/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch" && \
/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python \
  "/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/experiments/features/mace4d/m4d2_host_consensus_update.py" \
  2>&1 | tee "/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/experiments/features/mace4d/results/m4d2_host_consensus_update.txt"
```

## Environment

Python 3.11.15, torch 2.13.0, numpy 2.4.6.  Machine: Apple M3 Max, 103 GB
memory, macOS.  The whole script runs on the host; no device is used.
`torch.get_num_threads()` reported 10, which is the default in this
environment.

Array shape (8, 192, 192, 192) float32, which is 56,623,104 elements and
0.226 GB per array.  Three updates were timed per implementation, with
beta = [0.5, 1/6, 1/6, 1/6] and rho = 0.5.

## Timing and agreement

| implementation | seconds per update | max abs W difference from (a) | max abs xbar difference from (a) |
| --- | --- | --- | --- |
| (a) numpy in-place | 0.272 | reference | reference |
| (b) torch, one scratch | 0.157 | 0.000000e+00 | 0.000000e+00 |
| (c) torch, no scratch | 0.108 | 2.861023e-06 | 4.768372e-07 |

Both differences are below the 1e-5 tolerance.

Change statistic, in percent:

| update | (a) numpy | (b) torch scratch | (c) torch no scratch | relative difference from (a) |
| --- | --- | --- | --- | --- |
| 1 | 57.736198 | 57.735575 | 57.735575 | 1.080e-05 for both |
| 2 | 0 | 0 | 0 | not defined at zero |
| 3 | 0 | 0 | 0 | not defined at zero |

Peak resident memory for the whole process was 5.871 GB
(`ru_maxrss` = 5871091712 bytes, which macOS reports in bytes).

## What the numbers say

The torch form with one scratch tensor gives W and xbar that are bitwise
identical to the numpy form.  It takes 0.157 s per update against 0.272 s for
numpy, a factor of 1.7.

The torch form without a scratch array differs from numpy by at most 2.9e-06
on W and 4.8e-07 on xbar.  That form groups the scale factors differently, so
its float32 rounding differs.  It takes 0.108 s per update, a factor of 2.5
faster than numpy and 1.5 faster than the torch form with a scratch tensor.

The change statistic from torch differs from the numpy one by 1.08e-05
relative, which is within the 1e-4 tolerance.  The two libraries accumulate a
float32 2-norm over 56.6 million elements in different orders, and that is the
size of difference such an accumulation produces.

X does not change between the three updates in this measurement, so xbar is
the same after every update.  The change statistic is therefore exactly zero
on the second and third updates.  The relative agreement test only applies to
the first update.

Implementations (a) and (b) allocate two full-size arrays per update, for z
and xbar, because that is what the mbirjax loop does.  Implementation (c)
allocates none: it reuses two buffers, zeroes them in place, and writes the
difference xbar - xbar_prev into z, which is no longer needed by that point.

Nothing failed and nothing was skipped.
