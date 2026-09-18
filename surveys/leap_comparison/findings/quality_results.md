# Noisy reconstruction quality: LEAP against mbirtorch

Measured 2026-09-03 on the Purdue gautschi cluster, one NVIDIA H100 80GB HBM3
per job except the device-policy check, which used four.  Every number below is
read from the JSON files in
`../../experiments/features/leap_comparison/results/`:

| file | slurm job | what it holds |
|---|---|---|
| `quality_results.jsonl` | 15855240 | the sweep, the curves at N = 256, 512 and 1024, the steady-state times, the default-stop runs |
| `quality_probe_results.jsonl` | 15855290 | the check that the sweep winners are not sitting on a grid boundary |
| `autopin_results.jsonl` | 15855229 | mbirtorch's automatic device policy against a pinned four-device layout |
| `quality_warm_results.jsonl` | 15856252 | the warm-process curves, the measured warm direct reconstructions, and the N = 512 slice sets |

Software and hardware are as in `leap_benchmark_results.md`: LEAP 1.26 built
from source, mbirtorch 0.0.2 at commit 26bd0ea, torch 2.13.0+cu130, Python
3.11.16, NVIDIA H100 80GB HBM3, driver 595.71.05.

## What was measured

Geometry, unchanged from the timing study: circular cone beam, flat detector,
source to axis 1000 mm, source to detector 2000 mm, detector pixel 2 * 256/N mm,
voxel 1 * 256/N mm, N views uniform over 360 degrees, N x N x N volume.

**Phantom.**  The timing study's three spheres at the same centres and radii,
rescaled to attenuation per mm: the large sphere 0.02, the two inclusions 0.01
and 0.04.  The longest line through it measured 4.167 nepers at N = 256 and
4.152 at N = 512 and 4.149 at N = 1024.

**Noiseless sinogram.**  Forward projected with **mbirtorch**, not LEAP.  That
is an inverse crime, but a symmetric one: the timing study measured the two
forward projectors to agree to 0.052 percent at N = 256, and this study
re-checked the same comparison at each size it ran (0.000522 at N = 256,
0.000281 at N = 512, 0.000148 at N = 1024).  Neither library is reconstructing data that only its own
projector could have made, and the same argument would hold had LEAP's
projector been used instead.

**Noise.**  counts = Poisson(I0 exp(-p)) with I0 = 10000 photons per detector
pixel at every N and a fixed seed (1234); sinogram = -log(max(counts, 1) / I0);
weights = counts / I0.  The lowest count drawn was 120 at N = 256 and 110 at both
N = 512 and N = 1024, so the log was never actually clipped.  One noisy sinogram and one
weight array are written per size and both libraries read the same two arrays,
LEAP through the view and detector mapping below.  mbirtorch receives them as
`weights=`; LEAP receives the same array as RWLS's `W` argument, documented
there as "weights, should be the same size as g".

**Alignment.**  LEAP's view j matches mbirtorch's view i at j = (N/2 - i) mod N,
that is phi_LEAP = 180 degrees - phi_mbirtorch, together with a reversal of the
detector column index and no change to the detector rows; volumes map between
mbirtorch's (y, x, z) and LEAP's (z, y, x) by transposition.  This was
established in the timing study at N = 256 and is re-checked here at every size
rather than assumed, by projecting the phantom with LEAP and comparing against
the mapped mbirtorch projection.

**Metric.**  NRMSE = ||x - phantom|| / ||phantom|| over the voxels inside the
inscribed cylinder of radius N/2 - 2 voxels, over all slices.  Accumulated in
float64 in chunks along the first axis.

**Procedure.**  Each library first reconstructs directly from the noisy
sinogram (mbirtorch `recon_fdk`, LEAP `FBP`), and that reconstruction is the
starting point for every iterative run.  For each iteration count k the library
runs from that start for exactly k iterations, independently: mbirtorch
`recon(..., init_recon=direct, max_iterations=k, stop_threshold_change_pct=0.0)`,
LEAP `RWLS(g, f, k, ...)` on a fresh copy of the direct reconstruction.  All k
for one library at one size share a process, so compilation is paid once in that
process rather than once per k.

## Direct reconstructions from the noisy data

| N | LEAP `FBP` NRMSE | mbirtorch `recon_fdk` NRMSE |
|---|---|---|
| 256 | 0.09987 | 0.11237 |
| 512 | 0.11810 | 0.14153 |
| 1024 | 0.15638 | 0.19245 |

## Parameter sweep at N = 256

Best NRMSE reached at any k <= 100, per setting.  Each library's grid is the one
it was asked for; the extra rows marked "probe" come from job 15855290, which
was added because mbirtorch's winner sat at the edge of its grid and a boundary
minimum is not shown to be a minimum.

**mbirtorch**, sharpness at the default snr_db of 30, plus two snr_db values at
sharpness 0.  Its shipped defaults are sharpness 1.0 and snr_db 30.0.

| setting | best NRMSE | at k |
|---|---|---|
| sharpness -3 (probe) | 0.05231 | 15 |
| sharpness -2 (probe) | 0.04442 | 20 |
| **sharpness -1** | **0.04226** | 100 |
| sharpness 0, snr_db 25 | 0.04704 | 100 |
| sharpness 0 (snr_db 30) | 0.05651 | 100 |
| sharpness 0, snr_db 35 | 0.09446 | 10 |
| sharpness 1 (the default) | 0.09311 | 10 |
| sharpness 2 | 0.11471 | 2 |

**LEAP**, RWLS with an anisotropic TV filter, `filterSequence(1.0)` holding
`TV(leapct, delta, p=1.2, weight)`, SQS preconditioner, non-negativity on.

| TV weight | delta 0.001 | delta 0.01 |
|---|---|---|
| 0.01 | 0.08132 | 0.08128 |
| 0.1 | 0.08094 | 0.08044 |
| 1 | 0.07158 | 0.06524 |
| 3 (probe) | 0.05423 | not run |
| **10** | **0.04542** | 0.06504 |
| 30 (probe) | 0.04780 | not run |
| 100 | 0.05929 | 0.10210 |

**Winners: mbirtorch sharpness -1 at the default snr_db of 30; LEAP TV weight 10
at delta 0.001.**  Both are interior minima of the extended grid, so neither was
chosen at a boundary.  Both winners are three to four times more smoothing than
each library's own starting point (mbirtorch's default sharpness is 1, and the
TV weight used in the timing study was 1), which is what this noise level asks
for.  Note also that mbirtorch's sharpness 2 and LEAP's delta 0.01 with weight
100 both get *worse* with more iterations: under-regularized runs converge to a
noisier answer than they started from.

## Curves at N = 512, both libraries at their best setting

![NRMSE against wall time at N = 512](../../experiments/features/leap_comparison/results/quality_nrmse_vs_time_512.png)

The cold-process figures are kept under a `_coldprocess` suffix
(`quality_nrmse_vs_time_256_coldprocess.png`, `..._512_coldprocess.png`,
`..._1024_coldprocess.png`); the unsuffixed names now hold the warm-process
curves of the section below.

Wall time is the direct reconstruction plus k iterations, as measured in a
process whose runs ascend in k, so the k = 1 and k = 2 entries carry
mbirtorch's compilation and the curve runs backwards at its left end.  The
warm-process section below re-measures all of this with that cost removed; the
NRMSE values are identical and only the times change.

| k | LEAP NRMSE | LEAP time (s) | mbirtorch NRMSE | mbirtorch time (s) |
|---|---|---|---|---|
| 1 | 0.09909 | 2.654 | 0.13549 | 13.010 |
| 2 | 0.09359 | 3.585 | 0.10826 | 15.289 |
| 3 | 0.09091 | 4.697 | 0.06788 | 8.138 |
| 5 | 0.08623 | 6.922 | 0.04632 | 11.583 |
| 7 | 0.08247 | 9.085 | 0.04113 | 15.014 |
| 10 | 0.07760 | 12.397 | 0.03887 | 20.145 |
| 15 | 0.07060 | 17.918 | 0.03798 | 28.712 |
| 20 | 0.06464 | 23.437 | 0.03778 | 37.294 |
| 30 | 0.05548 | 34.475 | 0.03769 | 54.434 |
| 50 | 0.04594 | 56.555 | 0.03765 | 88.621 |
| 75 | 0.04127 | 84.193 | 0.03763 | 131.588 |
| 100 | 0.03902 | 111.777 | 0.03762 | 174.229 |

The mbirtorch times at k = 1 and k = 2 are larger than at k = 3 because those
two runs came first in their process and paid the compilation; the same effect
is visible at N = 256.

## Curves at N = 1024, the confirmation

Four iteration counts rather than twelve, at the same best settings.

| k | LEAP NRMSE | LEAP time (s) | mbirtorch NRMSE | mbirtorch time (s) |
|---|---|---|---|---|
| 5 | 0.11085 | 106.347 | 0.04898 | 127.916 |
| 10 | 0.09988 | 191.501 | 0.03498 | 188.568 |
| 20 | 0.08258 | 362.103 | 0.03274 | 358.621 |
| 40 | 0.05675 | 703.057 | 0.03246 | 698.231 |

At this size the two iterations cost almost exactly the same, 18.72 s for LEAP
against 18.08 s for mbirtorch, so the columns are close to a direct comparison
of convergence: at every k mbirtorch's error is between two and three times
smaller.

## Curves at N = 256

| k | LEAP NRMSE | LEAP time (s) | mbirtorch NRMSE | mbirtorch time (s) |
|---|---|---|---|---|
| 1 | 0.08804 | 0.287 | 0.10754 | 17.980 |
| 2 | 0.08400 | 0.259 | 0.08456 | 13.425 |
| 3 | 0.08205 | 0.337 | 0.06131 | 11.251 |
| 5 | 0.07852 | 0.492 | 0.04833 | 12.111 |
| 7 | 0.07564 | 0.648 | 0.04489 | 12.970 |
| 10 | 0.07195 | 0.883 | 0.04321 | 14.279 |
| 15 | 0.06688 | 1.271 | 0.04249 | 16.457 |
| 20 | 0.06288 | 1.659 | 0.04234 | 18.675 |
| 30 | 0.05737 | 2.441 | 0.04228 | 22.933 |
| 50 | 0.05158 | 3.995 | 0.04227 | 32.244 |
| 75 | 0.04783 | 5.949 | 0.04226 | 43.344 |
| 100 | 0.04542 | 7.895 | 0.04226 | 54.407 |

The mbirtorch column here carries a large fixed offset: the direct
reconstruction in that process took 10.694 s because it was the first GPU work
the process did and paid the FDK compilation.  The same reconstruction in the
sweep process took 3.087 s.  This is why the compile-free column below exists.

## Steady-state cost per iteration

Three k = 10 runs at the end of each process, so only the first could carry
compilation.  The value quoted is the last run divided by 10.

| N | library | the three runs (s) | steady state (s/iteration) |
|---|---|---|---|
| 256 | LEAP | 0.848, 0.848, 0.848 | 0.08480 |
| 256 | mbirtorch | 3.682, 3.683, 3.687 | 0.36869 |
| 512 | LEAP | 12.093, 12.090, 12.091 | 1.20909 |
| 512 | mbirtorch | 16.621, 16.993, 16.645 | 1.66453 |
| 1024 | LEAP | 187.198 (one run) | 18.71982 |
| 1024 | mbirtorch | 180.792 (one run) | 18.07920 |

Only one k = 10 run was made at N = 1024, to keep the job inside its walltime.
It is still compile-free: it came after the four iteration-count runs in the
same process, so anything that had to be compiled already had been.

## Iterations and time to a common target

The target at each N is 1.02 times the larger of the two libraries' best NRMSE
at that N, so it is a quality both libraries demonstrably reach.

Three clocks are given, because they answer different questions.  **Measured**
is the direct reconstruction plus the k iterations, exactly as timed in that
process; it is what a user waits for, including whatever compilation that
process had not yet paid.  **Iterations only** is k times the steady-state cost
of an iteration, which carries no compilation at all.  **Warm total** adds a
warm direct reconstruction to that: the `fdk` timings from the timing study's
`raw_results.jsonl` at the same geometry and size (mbirtorch 0.02819 s and LEAP
0.02985 s at N = 256, 0.33642 s and 0.29598 s at N = 512, 4.98008 s and
4.31366 s at N = 1024).  Borrowing those is
sound because a direct reconstruction's cost does not depend on the data, and it
is necessary because in this study each process ran its direct reconstruction
once, as its first GPU work, so every one of them carries compilation: 10.694 s
for mbirtorch at N = 256 in the process that produced the curve above, against
3.087 s for the same call in the sweep process and 0.028 s warm.

| N | target NRMSE | library | first k at or below target | NRMSE there | measured (s) | iterations only (s) | warm total (s) |
|---|---|---|---|---|---|---|---|
| 256 | 0.04633 | LEAP | 100 | 0.04542 | 7.895 | 8.480 | 8.510 |
| 256 | 0.04633 | mbirtorch | 7 | 0.04489 | 12.970 | 2.581 | 2.609 |
| 512 | 0.03980 | LEAP | 100 | 0.03902 | 111.777 | 120.909 | 121.205 |
| 512 | 0.03980 | mbirtorch | 10 | 0.03887 | 20.145 | 16.645 | 16.982 |
| 1024 | 0.05788 | LEAP | 40 | 0.05675 | 703.057 | 748.793 | 753.106 |
| 1024 | 0.05788 | mbirtorch | 5 | 0.04898 | 127.916 | 90.396 | 95.376 |

NRMSE at fixed iteration counts:

| N | library | k = 10 | k = 20 | k = 50 | k = 100 |
|---|---|---|---|---|---|
| 256 | LEAP | 0.07195 | 0.06288 | 0.05158 | 0.04542 |
| 256 | mbirtorch | 0.04321 | 0.04234 | 0.04227 | 0.04226 |
| 512 | LEAP | 0.07760 | 0.06464 | 0.04594 | 0.03902 |
| 512 | mbirtorch | 0.03887 | 0.03778 | 0.03765 | 0.03762 |
| 1024 | LEAP | 0.09988 | 0.08258 | not run | not run |
| 1024 | mbirtorch | 0.03498 | 0.03274 | not run | not run |

The shape of the result is the same at all three sizes, and the gap widens with
size.  LEAP's iteration is the cheaper one at the small sizes — 4.3 times
cheaper at N = 256, 1.4 times at N = 512 — but the advantage is gone by
N = 1024, where mbirtorch's iteration is 3.4 percent cheaper.  What decides the
comparison is that mbirtorch's VCD reaches the common target in an order of
magnitude fewer iterations: 7 against 100 at N = 256, 10 against 100 at N = 512,
5 against 40 at N = 1024.

Once compilation is out of the way, mbirtorch reaches the target 3.3 times
faster at N = 256, 7.1 times at N = 512 and 7.9 times at N = 1024.  On the
measured clock, which includes compilation, it is 5.5 times faster at N = 512,
5.5 times at N = 1024, and 1.6 times *slower* at N = 256, where the fixed cost
is a large fraction of a short run.

At N = 256 and N = 512, LEAP first reaches the target at k = 100, the last point
measured, so its curve had not flattened and a longer run might do better than
the target it set.  At N = 1024 it reaches the target at k = 40, also the last
point measured.  mbirtorch's curve is flat from k = 20 onward at every size.

## Warm-process re-measurement (job 15856252)

Greg spotted that the measured mbirtorch curve above runs backwards at its left
end: the k = 1 run is slower than the k = 5 run.  That is a property of the run
order, not of the timer, and the cause is worth recording because it will recur
in any study that sweeps iteration counts.

**Why two compilations, and why at iterations 1 and 2.**  mbirtorch's default
partition sequence is `[2, 4, 6] + [7, 8, 9, 10] * 25`
(`mbirtorch/_utils.py`, line 104), indexing into the default granularity
`[1, 2, 4, 8, 16, 32, 64, 128, 128, 128, 128]`.  So iteration 1 sweeps 4 pixel
subsets, iteration 2 sweeps 16, iteration 3 sweeps 64, and iterations 4 onward
sweep 128.  The subset count sets the pixel-batch shape the projector kernels
see, and `maybe_compile` in `mbirtorch/projectors.py` wraps those kernels with
`torch.compile(fn)` at its default `dynamic=None`, which is automatic dynamic
shapes: the first shape is specialized, the *second* distinct shape triggers one
recompilation to a shape-general kernel, and every later shape reuses that.  Two
compilation events, at iterations 1 and 2 — exactly where the excess time
appears, and nowhere else.

The warm re-run measures that directly.  Each process now discards one direct
reconstruction and two reconstructions (k = 3 and k = 5) before any timing.
Every NRMSE is identical to the cold run to five decimals, so only the clock
changed:

| k | warm iterations (s) | cold iterations (s) | cold minus warm (s) |
|---|---|---|---|
| 1 | 2.556 | 9.522 | +6.966 |
| 2 | 3.457 | 11.801 | +8.344 |
| 3 | 4.688 | 4.650 | -0.038 |
| 5 | 8.118 | 8.094 | -0.023 |
| 7 | 11.536 | 11.526 | -0.010 |
| 10 | 16.683 | 16.657 | -0.026 |
| 15 | 25.263 | 25.224 | -0.039 |
| 20 | 33.826 | 33.805 | -0.020 |
| 30 | 50.956 | 50.946 | -0.010 |
| 50 | 85.172 | 85.132 | -0.040 |
| 75 | 127.882 | 128.100 | +0.218 |
| 100 | 170.808 | 170.740 | -0.067 |

The excess is 6.966 s at k = 1 and 8.344 s at k = 2 and then vanishes: every
k >= 3 agrees with the cold run to better than 0.07 s except k = 75, which
differs by 0.218 s.  Total one-time cost 15.31 s at N = 512.

The discarded warm-up runs, timed so the record says what they cost.  The
mbirtorch k = 3 run is the expensive one at every size and its k = 5 run is
already clean, which is the same statement as the table above.  LEAP has nothing
to compile and its warm-up runs cost what its timed runs do.

| N | library | discarded direct (s) | discarded k = 3 (s) | discarded k = 5 (s) |
|---|---|---|---|---|
| 256 | leap | 0.035 | 0.406 | 0.461 |
| 256 | mbirtorch | 6.721 | 12.212 | 1.455 |
| 512 | leap | 0.300 | 4.463 | 6.573 |
| 512 | mbirtorch | 3.409 | 9.258 | 8.052 |
| 1024 | leap | 4.305 | 67.850 | 102.172 |
| 1024 | mbirtorch | 8.049 | 80.615 | 95.714 |

### Warm direct reconstructions, finally measured

Best of three in each process.  These are the numbers the target table above had
to borrow from the timing study, and they confirm the borrowing: mbirtorch
0.02835 against 0.02819 s at N = 256, 0.33461 against 0.33642 at N = 512,
4.98876 against 4.98008 at N = 1024, all within one percent.

| N | library | best of three (s) | the three runs (s) | NRMSE |
|---|---|---|---|---|
| 256 | leap | 0.02988 | 0.0305, 0.0299, 0.0301 | 0.09987 |
| 256 | mbirtorch | 0.02835 | 0.0303, 0.0285, 0.0284 | 0.11237 |
| 512 | leap | 0.29452 | 0.2953, 0.2945, 0.2946 | 0.11810 |
| 512 | mbirtorch | 0.33461 | 0.3390, 0.3346, 0.3348 | 0.14153 |
| 1024 | leap | 4.28985 | 4.2916, 4.2899, 4.2903 | 0.15638 |
| 1024 | mbirtorch | 4.98876 | 5.0009, 4.9893, 4.9888 | 0.19245 |

### Warm-process curves

Wall time is the warm direct reconstruction plus k iterations.  Both curves are
now monotone in time.

![NRMSE against wall time at N = 512, warm process](../../experiments/features/leap_comparison/results/quality_nrmse_vs_time_512.png)

**N = 512**

| k | LEAP NRMSE | LEAP warm time (s) | mbirtorch NRMSE | mbirtorch warm time (s) |
|---|---|---|---|---|
| 1 | 0.09909 | 2.467 | 0.13549 | 2.893 |
| 2 | 0.09359 | 3.578 | 0.10826 | 3.794 |
| 3 | 0.09091 | 4.691 | 0.06788 | 5.025 |
| 5 | 0.08623 | 6.918 | 0.04632 | 8.455 |
| 7 | 0.08247 | 9.143 | 0.04113 | 11.874 |
| 10 | 0.07760 | 12.479 | 0.03887 | 17.020 |
| 15 | 0.07060 | 18.041 | 0.03798 | 25.600 |
| 20 | 0.06464 | 23.604 | 0.03778 | 34.163 |
| 30 | 0.05548 | 34.729 | 0.03769 | 51.293 |
| 50 | 0.04594 | 57.030 | 0.03765 | 85.509 |
| 75 | 0.04127 | 85.282 | 0.03763 | 128.219 |
| 100 | 0.03902 | 112.851 | 0.03762 | 171.145 |

**N = 256**

| k | LEAP NRMSE | LEAP warm time (s) | mbirtorch NRMSE | mbirtorch warm time (s) |
|---|---|---|---|---|
| 1 | 0.08804 | 0.178 | 0.10754 | 0.273 |
| 2 | 0.08400 | 0.256 | 0.08456 | 0.363 |
| 3 | 0.08205 | 0.334 | 0.06131 | 0.590 |
| 5 | 0.07852 | 0.490 | 0.04833 | 1.472 |
| 7 | 0.07564 | 0.647 | 0.04489 | 2.350 |
| 10 | 0.07195 | 0.882 | 0.04321 | 3.668 |
| 15 | 0.06688 | 1.273 | 0.04249 | 5.863 |
| 20 | 0.06288 | 1.664 | 0.04234 | 8.039 |
| 30 | 0.05737 | 2.447 | 0.04228 | 12.407 |
| 50 | 0.05158 | 4.258 | 0.04227 | 21.142 |
| 75 | 0.04783 | 7.277 | 0.04226 | 32.101 |
| 100 | 0.04542 | 7.921 | 0.04226 | 43.007 |

**N = 1024** (two iteration counts only)

| k | LEAP NRMSE | LEAP warm time (s) | mbirtorch NRMSE | mbirtorch warm time (s) |
|---|---|---|---|---|
| 5 | 0.11085 | 106.823 | 0.04898 | 100.684 |
| 10 | 0.09988 | 192.611 | 0.03498 | 185.862 |

### Iterations and time to the common target, on the warm clock

The same rule as before, applied to the warm curves: 1.02 times the larger of
the two libraries' best NRMSE among the k that were measured.  The table two
sections above uses the cold-process clock; this one uses the warm clock.

| N | target NRMSE | library | first k at or below target | NRMSE there | warm time (s) |
|---|---|---|---|---|---|
| 256 | 0.04633 | LEAP | 100 | 0.04542 | 7.921 |
| 256 | 0.04633 | mbirtorch | 7 | 0.04489 | 2.350 |
| 512 | 0.03980 | LEAP | 100 | 0.03902 | 112.851 |
| 512 | 0.03980 | mbirtorch | 10 | 0.03887 | 17.020 |
| 1024 | 0.10188 | LEAP | 10 | 0.09988 | 192.611 |
| 1024 | 0.10188 | mbirtorch | 5 | 0.04898 | 100.684 |

At N = 256 and N = 512 the target is identical to the cold-process one, because
the NRMSE values are identical and only the clock changed; mbirtorch reaches it
**3.4 times faster** at N = 256 and **6.6 times faster** at N = 512.

**The N = 1024 row of this table is not comparable to the cold-process one.**
The warm run measured only k = 5 and k = 10 there, so LEAP's best is 0.09988
rather than the 0.05675 it reached at k = 40 in the cold run, and the target is
correspondingly looser (0.10188 against 0.05788).  For an N = 1024 target read
the cold-process table.  What the warm N = 1024 points show without any target
argument is that mbirtorch at k = 10 (NRMSE 0.03498 in 185.862 s) is already
better than LEAP's best cold-run result at k = 40 (0.05675 in 703.057 s), at a
quarter of the time.

## Reconstruction images at N = 512

The slices are in `../../experiments/features/leap_comparison/results/slices_512/` as .npy, three planes per volume, for the
phantom and five reconstructions.  A note on the planes: **no axis-aligned plane
contains both inner sphere centres.**  Sphere A (value 0.5) is cut by coronal
planes with y between -45 and 5 mm and sphere B (value 2.0) by planes with y
between 17 and 53 mm, and those ranges are disjoint; the z ranges are disjoint
too.  So instead of one coronal slice through both, each figure has three rows:
the centre axial slice at z = N/2 (which cuts only the large sphere, because
sphere A is exactly tangent to that plane), a coronal plane through sphere A, and
a coronal plane through sphere B, with the zoom inset on the sphere in that row.

| panel | NRMSE | warm wall time (s) |
|---|---|---|
| phantom | 0.00000 | 0.000 |
| leap_direct | 0.11810 | 0.295 |
| mbirtorch_direct | 0.14153 | 0.335 |
| leap_iterative_k14 | 0.07191 | 16.806 |
| mbirtorch_iterative_k10 | 0.03887 | 16.928 |
| leap_iterative_k100 | 0.03902 | 112.793 |

![matched wall time at N = 512](../../experiments/features/leap_comparison/results/quality_slices_matched_time_512.png)

Figures, all reconstructions on a common gray window of 0 to 0.045 per mm:

* `quality_slices_matched_time_512.png` — phantom, LEAP k = 14, mbirtorch
  k = 10.  Matched wall time, 16.81 s against 16.93 s: at the same cost
  mbirtorch's error is 0.03887 against LEAP's 0.07191, and the difference shows
  plainly as noise in the LEAP panels.
* `quality_slices_matched_quality_512.png` — phantom, LEAP k = 100, mbirtorch
  k = 10.  Matched quality, 0.03902 against 0.03887: LEAP needs 112.79 s to
  mbirtorch's 16.93 s, a factor of 6.7.
* `quality_slices_direct_512.png` — phantom, LEAP `FBP`, mbirtorch `recon_fdk`.
* `quality_slices_difference_512.png` — absolute difference from the phantom for
  all five reconstructions on one common scale, chosen as the largest 99.9th
  percentile over the panels drawn and printed in the figure title.

## mbirtorch with its own stopping rule left in place

`recon(..., max_iterations=100)` with the default
`stop_threshold_change_pct=0.2`, at the same best setting.  The iteration count
is read from the four per-iteration traces in the returned dictionary
(`fm_rmse`, `stop_threshold_change_pct`, `alpha_values`, `delta_norm_per_slice`),
which agree with each other; `partition_sequence` has 100 entries because it is
the planned sequence, not a record of what ran.

| N | stopped after | NRMSE there | at or below the target? | iteration time (s) |
|---|---|---|---|---|
| 256 | 9 of 100 | 0.04357 | yes (target 0.04633) | 3.234 |
| 512 | 10 of 100 | 0.03887 | yes (target 0.03980) | 16.618 |
| 1024 | 12 of 100 | 0.03392 | yes (target 0.05788) | 214.006 |

The default rule stops within a few iterations of the k this study found by
hand, and the reconstruction it returns meets the common target at all three
sizes, comfortably so at N = 1024.

## mbirtorch's automatic device policy against a pinned layout

The timing study left a question open: at N = 1024 on four GPUs, ten iterations
took 109.03 s under the automatic policy and 75.81 s pinned to four devices, but
the automatic arm ran first in that job and so paid the cold compilation.  Job
15855229 separates the two by running three reconstructions per arm in one
process, and by running the **pinned** arm first, so any ordering advantage now
favours the automatic arm.

| arm | run 1 (s) | run 2 (s) | run 3 (s) | steady state (s/iteration) |
|---|---|---|---|---|
| pinned to `cuda:0..3` | 99.233 | 66.046 | 65.954 | 6.595 |
| automatic policy | 77.821 | 68.246 | 68.183 | 6.818 |

**The earlier gap was almost all one-time cost.**  In steady state the automatic
policy costs 3.4 percent more per iteration than the pinned layout, not the
44 percent the single-run comparison suggested.

The policy's message, printed once at the first reconstruction, is:

```
Using 4 CUDA device(s) for this reconstruction (was 1).  configure_devices(num_devices=n) pins it.
```

The recorded layout after the first bare `forward_project` is `['cuda']` in the
automatic arm and `['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']` in the pinned one,
confirming the timing study's finding that the automatic policy widens at the
first reconstruction and not before.  Per-GPU peak memory differs slightly
between the arms: 19163, 12499, 12499, 11951 MiB automatic against 18057, 13525,
13525, 12977 MiB pinned, so the automatic layout leaves a little more on the
lead device.

## What did not work

1. **LEAP's RWLS has no `numSubsets` argument**, so the ordered-subsets variant
   asked for could not be added at the best setting.  Its signature is
   `RWLS(self, g, f, numIter, filters=None, W=None, preconditioner=None,
   nonnegativityConstraint=True)`.  `SART` and `ASDPOCS` do take `numSubsets`,
   but they are different algorithms and swapping one in would not have been the
   measurement requested.

2. **No run in this study measured a warm direct reconstruction.**  Each
   process ran `recon_fdk` or `FBP` exactly once, as its first GPU work, so
   every direct reconstruction timed here carries compilation, and mbirtorch's
   varies wildly between processes for that reason: 10.694 s and 3.087 s for the
   same call at N = 256.  The "warm total" column therefore borrows the timing
   study's warm `fdk` timings rather than measuring its own.  A cleaner design
   would have run the direct reconstruction three times per process, as the
   iterative one was; that was not done.

3. **The wall times for small k in the cold-process tables are inflated by
   compilation** in whichever run comes first in a process, and the runs ascend
   in k, so the small-k runs carry it.  That is now measured rather than
   estimated, in the warm-process section: 6.966 s at k = 1 and 8.344 s at
   k = 2 at N = 512, and nothing at k >= 3.  The cold-process tables are kept
   as they were measured; read the warm-process tables for a clock without it.

4. **LEAP's curve never reached a converged optimum.**  It was still
   descending at the last measured point at all three sizes — k = 100 at
   N = 256 and N = 512, k = 40 at N = 1024 — so its "best NRMSE", and therefore
   the common target, is set by where the measurement stopped rather than by
   where the algorithm settles.  A longer run would lower LEAP's best and
   tighten the target, which would move mbirtorch's iteration count up.
   mbirtorch's own curve is flat to four digits from k = 20 onward at every
   size, so its best is a real limit.

5. **N = 1024 has four iteration counts, one steady-state run, and no k = 50 or
   k = 100 point.**  Twelve iteration counts there would have cost about five
   hours per library.  The N = 1024 rows are a confirmation of the shape found
   at the two smaller sizes, not an independent measurement of equal density.

6. **LEAP's warm k = 75 point at N = 256 is out of line with its neighbours**:
   7.247 s against 4.258 s at k = 50 and 7.891 s at k = 100, where a linear
   cost would give about 6.4 s.  The cold-process run at the same point read
   5.915 s.  It looks like a transient on the node rather than a property, and
   it does not touch any conclusion: the k = 100 point that sets the target
   reads 7.891 s warm against 7.861 s cold.

7. **The N = 1024 warm curve has only k = 5 and k = 10**, so its common target
   is much looser than the cold-process one and the two N = 1024 target rows
   must not be compared.  Twelve iteration counts there would have cost about
   five hours per library.

8. **This document was moved while this study was running.**  Another session
   relocated it from
   `plans/experiments/features/leap_comparison/results/quality_results.md` to
   `plans/features/leap_comparison/quality_results.md` and committed it there
   (commit 0da862f).  The warm-process and image sections were written into the
   relocated copy rather than the old path, so there is one document, not two.
   The JSON files, figures and slices all remain under
   `plans/experiments/features/leap_comparison/results/`.

9. **One phantom, one noise level, one seed.**  Nothing here separates the two
   libraries' image quality in general; it compares two particular algorithms at
   their best setting on one problem.
