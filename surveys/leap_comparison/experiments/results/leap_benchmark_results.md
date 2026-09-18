# LEAP against mbirtorch on one H100

Measured 2026-09-02 on the Purdue gautschi cluster: one NVIDIA H100 for
everything except the multi-GPU section, which used four.  Every number below is
read from the JSON files in this directory:

| file | slurm job | what it holds |
|---|---|---|
| `raw_results.jsonl` | 15847177 | the main single-GPU table at N = 256, 512, 1024, and the cross-checks |
| `smoke_results.jsonl` | 15847177 | the N = 64 smoke pass |
| `repeat_results.jsonl` | 15847391 | three repeated reconstructions at N = 256 |
| `repeat2_results.jsonl` | 15848539 | three repeated reconstructions at N = 512 and N = 1024 |
| `multigpu_results.jsonl` | 15848540 | the four-GPU pass at N = 1024 |

The slurm logs for all four jobs are here as well.

The two libraries do the same geometry on the same phantom, and their forward
projections agree to 0.05 percent, so the timings are of comparable work.  The
iterative reconstructions are **not** comparable in that sense: they are
different algorithms with different regularizers, and only their cost per
iteration is being reported, not their quality.

## Hardware, versions, and how each library was installed

| item | value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3, 81559 MiB, driver 595.71.05 (node h000, `ai` partition) |
| allocation | 1 GPU, 14 cores, `-A bouman -p ai -q normal` |
| Python | 3.11.16 |
| torch | 2.13.0+cu130 |
| numpy | 2.4.6 |
| LEAP | 1.26, built from source from github.com/LLNL/LEAP `main` |
| mbirtorch | 0.0.2, github.com/cabouman/mbirtorch branch `greg_dev`, commit 26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2 |
| LEAP build toolchain | cmake 3.31.8, nvcc 12.6.1 (cluster `cuda/12.6.1` module), gcc 11.5.0 |

LEAP's `libleapct.so` links `libcublas.so.12` and `libcufft.so.11` from the
CUDA 12.6.1 module, while torch carries its own CUDA 13.0 runtime.  The two
never share a process here: the harness starts one process per library, so this
mismatch was never put to the test.

Environment: a bare venv at `/scratch/gautschi/buzzard/leap_cmp/venv`, built
from the `mbirtorch` conda environment's interpreter, with mbirtorch and leapct
installed into it and that conda environment's site-packages appended through a
`.pth` path line so torch, triton and numpy resolve from there.

## Geometry, in each library's own terms

The same physical setup at every size: circular cone beam, flat detector,
magnification 2, N views over a full 360 degrees, N detector rows, N detector
columns, an N x N x N volume.  The detector pixel is 2 * 256/N mm and the voxel
is 1 * 256/N mm, so the volume is always a 256 mm cube and the detector always
covers it.

| quantity | value at N | mbirtorch parameter | LEAP parameter |
|---|---|---|---|
| views | N | `sinogram_shape[0]`, `angles` | `numAngles`, `phis` |
| detector rows x cols | N x N | `sinogram_shape[1:]` | `numRows`, `numCols` |
| detector pixel | 2 * 256/N mm | `delta_det_row`, `delta_det_channel` | `pixelHeight`, `pixelWidth` |
| voxel | 1 * 256/N mm | `delta_voxel` | `voxelWidth`, `voxelHeight` |
| volume | N x N x N | `recon_shape` | `numX`, `numY`, `numZ` |
| source to axis | 1000 mm | `source_iso_dist` | `sod` |
| source to detector | 2000 mm | `source_detector_dist` | `sdd` |
| detector centre | (N-1)/2 in both directions | default (`det_row_offset = det_channel_offset = 0`) | `centerRow = centerCol = 0.5*(N-1)` |
| angles | uniform, full turn | `np.linspace(0, 2*pi, N, endpoint=False)` radians | `np.linspace(0, 360, N, endpoint=False)` degrees |
| detector shape | flat | `use_curved_detector=False` (default) | `set_flatDetector()` |

LEAP's own parameter printout for N = 256 (from the job log) confirms this:
256 angles over 360 degrees, 256 x 256 detector, 2.000000 mm x 2.000000 mm
pixels, centre pixel 127.500000, 127.500000, sod 1000 mm, sdd 2000 mm,
256 x 256 x 256 voxels at 1.000000 mm, FOV [-128, 128] in each direction.

Phantom: a sphere of radius 89.6 mm and value 1 centred at the origin, with a
sphere of radius 25 mm and value 0.5 centred at (30, -20, 25) mm and a sphere of
radius 18 mm and value 2 centred at (-25, 35, -30) mm inside it.  It is defined
by physical positions, so the radii in voxels scale with N.  One array is built
per size in mbirtorch's index order (row = y, column = x, slice = z); LEAP reads
the same array transposed to its (z, y, x) order.

## Timings and peak memory

Best of 3 timed repeats after 1 warmup, with `torch.cuda.synchronize()` before
each timer stop, except the iterative reconstruction, which is a single run (see
the section after this one).  "device peak" is the maximum of
`nvidia-smi --query-gpu=memory.used` sampled every 0.1 s during the measurement,
filtered to this job's GPU by UUID; it includes the CUDA context and whatever
the torch caching allocator is holding.  "torch peak" is
`torch.cuda.max_memory_allocated`, reset before every measurement; it sees only
torch allocations, so for LEAP it counts the input and output tensors this
harness allocates but none of LEAP's own `cudaMalloc` working memory.

| N | operation | LEAP time (s) | mbirtorch time (s) | LEAP device peak (GiB) | mbirtorch device peak (GiB) | LEAP torch peak (GiB) | mbirtorch torch peak (GiB) |
|---|---|---|---|---|---|---|---|
| 256 | forward projection | 0.02587 | 0.03518 | 0.79 | 1.25 | 0.125 | 0.57 |
| 256 | back projection | 0.01609 | 0.02015 | 0.86 | 1.30 | 0.19 | 0.47 |
| 256 | FDK / FBP | 0.02985 | 0.02819 | 0.88 | 1.35 | 0.25 | 0.60 |
| 256 | iterative recon, 10 iterations | 0.9658 | 19.28 | 1.63 | 1.78 | 0.94 | 0.73 |
| 512 | forward projection | 0.3997 | 0.5479 | 2.11 | 3.97 | 1.00 | 3.12 |
| 512 | back projection | 0.2530 | 0.3070 | 2.61 | 4.23 | 1.50 | 2.62 |
| 512 | FDK / FBP | 0.2960 | 0.3364 | 3.11 | 5.28 | 2.00 | 3.62 |
| 512 | iterative recon, 10 iterations | 12.56 | 32.06 | 8.74 | 7.31 | 7.50 | 4.65 |
| 1024 | forward projection | 6.314 | 8.636 | 12.61 | 17.92 | 8.00 | 17.02 |
| 1024 | back projection | 4.070 | 4.837 | 16.61 | 20.86 | 12.00 | 15.55 |
| 1024 | FDK / FBP | 4.314 | 4.980 | 20.61 | 28.93 | 16.00 | 23.55 |
| 1024 | iterative recon, 10 iterations | 192.7 | 200.2 | 65.62 | 45.03 | 60.00 | 36.10 |

Both libraries fit at N = 1024 on one 80 GB H100.  LEAP's iterative
reconstruction is the closest call: a 65.62 GiB device peak on a card with
81559 MiB (79.65 GiB) total.

Per iteration on that single run (total divided by 10).  These are first-run
figures and so include mbirtorch's one-time compilation; the steady-state
figures are in the next section but one.

| N | LEAP RWLS+TV (s/iter) | mbirtorch VCD+qGGMRF (s/iter) |
|---|---|---|
| 256 | 0.09658 | 1.928 |
| 512 | 1.256 | 3.206 |
| 1024 | 19.27 | 20.02 |

### The iterative reconstructions are different algorithms

LEAP: `FBP` to start, then exactly 10 `RWLS` iterations (preconditioned
conjugate gradient on a regularized weighted least squares cost) with an
anisotropic TV regularizer (`filterSequence(1.0)` holding
`TV(leapct, delta=0.025, p=1.2, weight=1.0)`), `preconditioner='SQS'`,
non-negativity on, and weights set to all ones.  The log shows it printing
"RWLS iteration 1 of 10" through "10 of 10".

mbirtorch: `recon(..., max_iterations=10, stop_threshold_change_pct=0.0)`, which
is Multi-Granular Vector Coordinate Descent with a qGGMRF prior, with early
stopping switched off so exactly 10 iterations run.  With `init_recon=None` it
starts from its own FDK reconstruction, which is why LEAP is given an FBP start
here: both timings then cover "sinogram in, 10 iterations out".

These solve different optimization problems with different priors, so the times
are a cost-per-iteration comparison and nothing more.

### mbirtorch pays a one-time compilation cost on the first reconstruction

mbirtorch compiles parts of its solver with `torch.compile` on first use, so a
single timed run charges that one-time cost to the iteration time.  The
reconstruction was therefore run three times in one process at every size, on a
cold inductor cache directory: N = 256 in job 15847391, N = 512 and N = 1024 in
job 15848539.  Totals for 10 iterations:

| N | LEAP run 1 | LEAP run 2 | LEAP run 3 | mbirtorch run 1 | mbirtorch run 2 | mbirtorch run 3 |
|---|---|---|---|---|---|---|
| 256 | 0.9851 | 0.8847 | 0.8782 | 13.7416 | 3.4640 | 3.4643 |
| 512 | 12.5149 | 12.3965 | 12.6005 | 31.6319 | 15.7836 | 15.8913 |
| 1024 | 192.7014 | 192.6374 | 192.7966 | 196.8213 | 177.4380 | 177.3758 |

LEAP's three runs agree to well under one percent at every size: it has no
first-run cost to pay.  mbirtorch's first run costs 10.28 s more than its third
at N = 256, 15.74 s more at N = 512 and 19.45 s more at N = 1024 — an overhead
that grows only slowly with problem size, so its share of the total falls from
75 percent at N = 256 to 10 percent at N = 1024.

Steady-state cost per iteration, taken from run 3 divided by 10:

| N | LEAP RWLS+TV (s/iter) | mbirtorch VCD+qGGMRF (s/iter) | ratio LEAP : mbirtorch |
|---|---|---|---|
| 256 | 0.08782 | 0.34643 | 1 : 3.94 |
| 512 | 1.26005 | 1.58913 | 1 : 1.26 |
| 1024 | 19.27966 | 17.73758 | 1 : 0.92 |

The gap closes as the problem grows, and at N = 1024 mbirtorch's iteration is
the cheaper of the two.  This is still a cost-per-iteration comparison between
two different algorithms, not a statement about how much either iteration
achieves.

Peak memory in these repeated runs is stable: LEAP's device peak is identical
across all three runs at every size (1673, 8953 and 67193 MiB), and mbirtorch's
is identical across runs 2 and 3 (1759, 7483 and 46107 MiB).

Note that job 15847391 ran on the same node as job 15847177, on a different GPU,
partly at the same time, and job 15848539 ran alongside the multi-GPU job on a
different node.  Every forward, back and FDK timing they re-measured matches the
main job's to within 0.69 percent (N = 512 mbirtorch forward 0.5483 s against
0.5479 s), so the overlap did not measurably disturb the numbers.

## Multi-GPU at N = 1024

Job 15848540, four H100s on one node (`--gpus-per-node=4`), same geometry and
phantom.  Three arms, each its own process.  The single-GPU column repeats the
one-GPU numbers from the table above for comparison.

The memory sampler reads each GPU separately, so the per-device peaks say
directly which operations actually spread: a device holding only about 529 MiB
is holding a CUDA context and nothing else.

| arm | operation | time (s) | speedup over 1 GPU | per-GPU device peak (MiB) | spread? |
|---|---|---|---|---|---|
| mbirtorch, automatic policy | forward projection | 8.623 | 1.00x | 18349, 529, 529, 529 | no, 1 GPU |
| mbirtorch, automatic policy | back projection | 4.841 | 1.00x | 21365, 529, 529, 529 | no, 1 GPU |
| mbirtorch, automatic policy | FDK | 1.429 | 3.48x | 16189, 9899, 9899, 8839 | yes, 4 GPUs |
| mbirtorch, automatic policy | 10 iterations | 109.03 | 1.84x | 20189, 13525, 13525, 12977 | yes, 4 GPUs |
| mbirtorch, pinned to 4 | forward projection | 2.189 | 3.95x | 9439, 6269, 6269, 6269 | yes, 4 GPUs |
| mbirtorch, pinned to 4 | back projection | 1.331 | 3.63x | 12301, 7851, 7851, 6791 | yes, 4 GPUs |
| mbirtorch, pinned to 4 | FDK | 1.432 | 3.48x | 14415, 9967, 9967, 8907 | yes, 4 GPUs |
| mbirtorch, pinned to 4 | 10 iterations | 75.81 | 2.64x | 18055, 13523, 13523, 12975 | yes, 4 GPUs |
| LEAP, `set_gpus([0,1,2,3])` | forward projection | 2.214 | 2.85x | 1941, 1813, 1941, 1941 | yes, 4 GPUs |
| LEAP, `set_gpus([0,1,2,3])` | back projection | 1.645 | 2.47x | 1909, 1909, 1909, 1909 | yes, 4 GPUs |
| LEAP, `set_gpus([0,1,2,3])` | FBP | 1.849 | 2.33x | 2777, 2777, 2777, 2777 | yes, 4 GPUs |
| LEAP, `set_gpus([0,1,2,3])` | 10 iterations | 313.75 | 0.61x | 12827, 2261, 2777, 2521 | partly, see below |

Three things this shows.

**LEAP's multi-GPU path requires host arrays, and that is what makes its
iterative reconstruction slower on four GPUs than on one.**  `leapctype` passes
"is this array on the cpu" through to the library, and data already sitting on
one GPU is processed on that GPU alone; only host arrays get chunked across the
GPUs named by `set_gpus`.  So this arm uses numpy arrays and every call pays
host-to-device and device-to-host copies.  For a single projection that trade is
worth it — forward projection is 2.85x faster than the single-GPU on-device time
of 6.314 s — but RWLS makes many such calls per iteration, and the transfers
turn a 192.7 s reconstruction into a 313.7 s one.  The per-GPU peaks for that
row also show the work is not evenly spread: 12827 MiB on GPU 0 against about
2500 MiB on the others.  Its device peaks are low throughout because LEAP streams
chunks through the GPUs rather than holding the whole volume on each.

**mbirtorch's automatic policy did not spread the two bare projection calls.**
The layout it settled on at the first `forward_project` was `['cuda']`, one
device, and the per-GPU peaks confirm the other three sat idle for both forward
and back projection.  The documented behaviour is that the choice is made at the
first reconstruction, and indeed `recon_fdk` and `recon` both did spread across
all four.  A user who wants the projectors themselves spread has to say so with
`configure_devices(num_devices=4)`, which is the second arm.

**The pinned arm reconstructs faster than the automatic one**, 75.81 s against
109.03 s, although both used four GPUs for the reconstruction itself.  In the
automatic arm the sinogram was produced by a one-device `forward_project` and so
had to be redistributed across the four devices inside the timed region; in the
pinned arm it was already in four-device form.

**mbirtorch on four GPUs uses more total memory than LEAP but less per device
than it does on one**: its N = 1024 reconstruction peak falls from 46107 MiB on
one GPU to 18055 MiB on the busiest of four.

## Correctness cross-checks

All at N = 256, single GPU.

**a. Forward projections of the same phantom, LEAP against mbirtorch.**
Normalized RMSE (`||leap - mbirtorch|| / ||mbirtorch||`) after alignment:
**0.0005220**, that is **0.052 percent**.  Fitting a single global scale factor
first changes nothing: 0.0005220 with a best scale of 0.99999962, so the two
libraries also agree on the absolute scale of a line integral in mm.  Mean
sinogram values: mbirtorch 46.13872, LEAP 46.13875.

The alignment needed was a **detector column flip** together with a **view
index reversal and shift**, and **no detector row flip**.  Written out at
N = 256, LEAP's view j lines up with mbirtorch's view i as j = (128 - i) mod 256,
which for uniform angles over a full turn is

    phi_LEAP = 180 degrees - phi_mbirtorch

combined with reversing the detector column index.  No transpose of the
sinogram axes was needed: both libraries return (views, detector rows, detector
columns), and the detector row direction already matched.

The search that found this covered detector column flip (2) x view reversal (2)
x view shift (N), scored on per-view detector-column profiles, and then detector
row flip (2) scored on the full sinograms.  Because the angles are uniform over
a full turn, every candidate convention of the form phi -> +/- phi + constant is
one of the searched view permutations, so the search covers the plausible angle
conventions rather than sampling a few by hand.

**b. Each library's direct reconstruction against the phantom.**

| library | NRMSE against phantom | NRMSE after best global scale | best scale |
|---|---|---|---|
| mbirtorch `recon_fdk` | 0.06090 | 0.06077 | 1.00393 |
| LEAP `FBP` | 0.06468 | 0.06441 | 1.00590 |

Both reconstructions came out in the orientation the geometry predicted; the
z-flipped alternative scored 0.1728 and 0.1738 respectively, so the volume
orientation is not ambiguous.  A direct reconstruction is not expected to
reproduce the phantom exactly, so these say the geometry is right, not that
either algorithm is exact.

**c. Adjoint check, `<A x, y>` against `<x, A^T y>` with uniform random x and y.**

| library | `<A x, y>` | `<x, A^T y>` | relative difference |
|---|---|---|---|
| mbirtorch | 823575664.979739 | 823575663.9585674 | 1.240e-09 |
| LEAP | 810585892.0919025 | 810581095.4006543 | 5.918e-06 |

mbirtorch's forward and back projectors are exact adjoints of each other by
construction, and the reading is at float32 summation noise.  LEAP's pair is
adjoint to about 6e-6 relative, which is consistent with its forward and back
projectors being separately derived approximations of the same operator rather
than an exact transpose pair.

## Smoke pass at N = 64

An N = 64 pass ran first in the same job to exercise every code path.  Its
numbers are in `smoke_results.jsonl` and are not part of the table above; they
are recorded because the LEAP adjoint reading there, 3.881e-04 relative, is much
looser than at N = 256, which is worth knowing before drawing conclusions from
the N = 256 adjoint number alone.  Forward projection agreement at N = 64 was
0.001724 (0.17 percent).

## What did not work

1. **There is no `leapct` package on PyPI.**  `pip install leapct` cannot work.
   `https://pypi.org/pypi/leapct/json` returns HTTP 404, as do `leap-ct`,
   `leapctype` and `LEAP-CT`.  LEAP was instead built from a clone of
   github.com/LLNL/LEAP with
   `python -m pip install --no-cache-dir --no-deps -v .`, which runs
   `etc/build.sh` (cmake, then nvcc) and installs version 1.26.  The build took
   a few minutes on a login node and needed no GPU.  A conda-forge `leapct`
   1.26 binary package exists as an alternative and was not used here.

2. **LEAP's install requirements are incomplete.**  The first import failed:

   ```
     File "/scratch/gautschi/buzzard/leap_cmp/venv/lib/python3.11/site-packages/leapctype.py", line 16, in <module>
       import imageio
   ModuleNotFoundError: No module named 'imageio'
   ```

   `leapctype.py` imports `imageio` at module scope, but `setup.py` lists only
   `numpy` and `torch` in `install_requires`.  Fixed with `pip install imageio`.

3. **The torch environment named in the task setup no longer exists.**
   `/scratch/gautschi/buzzard/torch_p0/env/bin/python` is gone, and
   `/scratch/gautschi/buzzard/pcdrecon_regression/env` has no torch.  The conda
   environment `/home/buzzard/.conda/envs/mbirtorch` (Python 3.11.16, torch
   2.13.0+cu130, triton 3.7.1, numpy 2.4.6) was used as the base instead.

4. **The main table's iterative row is a single run and includes mbirtorch's
   one-time compilation.**  This is now quantified rather than left open: job
   15848539 repeated the reconstruction three times at N = 512 and N = 1024, so
   the steady-state cost per iteration is known at all three sizes (see the
   table above).  What is still not separated is compilation from execution
   *within* a run — the harness times whole calls, so the first-run overhead is
   measured as a difference between runs, not attributed to particular work.

5. **mbirtorch's automatic device policy left three of four GPUs idle for bare
   forward and back projection.**  This is documented behaviour, not a fault
   (the choice is made at the first reconstruction), but it means a script that
   calls `forward_project` alone on a multi-GPU node gets one GPU unless it
   calls `configure_devices`.  Measured, not inferred: the per-GPU peaks were
   18349, 529, 529, 529 MiB.

6. **LEAP's four-GPU iterative reconstruction was slower than its one-GPU one**,
   313.75 s against 192.69 s, because its multi-GPU path only engages for arrays
   on the host and RWLS then pays host-device copies on every projection.  No
   way to get LEAP's multi-GPU chunking with data resident on the GPUs was found
   in `leapctype`.

7. **Not attempted.**  Multi-GPU at N = 256 and N = 512, a 2-GPU point for a
   scaling curve, CPU projectors, curved detectors, helical scans, short scans,
   and reconstruction quality at matched regularization strength.  Nothing here
   says anything about which library produces the better image.

## Reproducing

On the cluster, in `/scratch/gautschi/buzzard/leap_cmp`:

```
sbatch leap_cmp_gautschi.sbatch           # the main table, all three sizes
sbatch leap_cmp_repeat_gautschi.sbatch    # repeated iterative runs at N = 256
sbatch leap_cmp_repeat2_gautschi.sbatch   # repeated iterative runs at N = 512 and 1024
sbatch leap_cmp_multigpu_gautschi.sbatch  # the four-GPU pass at N = 1024
```

`bench_leap_vs_mbirtorch.py` takes `--mode {phantom,mbirtorch,leap,compare}`,
`--N`, and `--results-dir`, and appends one JSON object per measurement to the
file named by `--jsonl` as soon as that measurement is taken.  `--devices`
selects one GPU (the default), every visible GPU (`all`), or, for mbirtorch,
its own automatic policy (`auto`).  `--iterative-repeats` sets how many times
the 10-iteration reconstruction runs in the same process.
