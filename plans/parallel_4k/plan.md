# Parallel-beam reconstruction at 4096 x 4096 x 900 views: plan

Status: ACTIVE
Updated: 2026-09-23
Code: mbirtorch branch parallel_4k_script from prerelease 498294d; the files under experiments/parallel_4k/ are staged and not committed
Next step: Decide whether the streaming harness becomes the collaborators' script, and which interfaces the library exposes for it; then measure the 4K case on gautschi (the first job, 16629880, was cancelled 2026-09-23 pending that discussion).

## Goal

Collaborators need to reconstruct a parallel-beam scan with a 4096 x 4096
detector and 900 views.  The volume is 4096 x 4096 x 4096 voxels, which is
275 GB in float32, and the sinogram is 60 GB.  The deliverable is a script
they can run, plus an estimate of the GPUs, the host memory, and the time
the run needs.

## What exists

`ParallelBeamModel.recon_split_sino` reconstructs the volume in bands of
detector rows and stitches the bands on the host.  Detector row r is slice r
in parallel beam, so the bands separate exactly in the forward model.  The
overlap of 5 rows on each side of a seam serves the prior.  When
`slices_per_part` is not given, the method chooses the fewest parts whose
largest part the memory ledger prices to fit the visible GPUs.

The script `experiments/parallel_4k/recon_parallel_split.py` in the
mbirtorch repository wraps this method.  It takes a sinogram file, the
angles, and an output path.  Its `--plan` mode prints the part count, the
peak memory per GPU, and the host memory without reconstructing, from a
sinogram file or from a shape alone.  A separate script,
`make_synthetic_sinogram.py`, writes a phantom sinogram by forward projecting
an ellipsoid phantom band by band, and checks a reconstruction against the
phantom, so the reconstruction scripts hold only the end-user task.
`README.md` beside the scripts holds the tables below, and
`gautschi_4k.sbatch` is the benchmark job.

## Device memory, from the ledger

The ledger prices the 4K case at about 0.27 GiB per slice held on one GPU,
with the compiled prior and the Triton bodies.  The peak phase is the
subset prior at granularity 4, which holds nine cylinder arrays over one
quarter of the pixels.  The table gives the modeled peak of the worst GPU
in GiB, with weights supplied.

| parts | slices per part | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 1078 | 540 | 270 | 136 |
| 2 | 2053 | 541 | 271 | 136 | 69 |
| 3 | 1376 | 363 | 182 | 92 | 46.5 |
| 4 | 1034 | 273 | 137 | 69 | 35 |
| 5 | 830 | 219 | 110 | 56 | 29 |
| 8 | 522 | 139 | 70 | 36 | 20 |
| 16 | 266 | 71 | 36 | 20 | 13 |
| 24 | 181 | 49 | 25 | 14 | 9.5 |

The fit rule is 1.15 times the peak at or below the card's free memory.
An 80 GB card therefore holds about 250 slices at a time, and parts times
GPUs must reach about 17.  The library chooses 3 parts on 8 GPUs, 5 on 4,
9 on 2, and 17 on 1.  On 40 GB cards the counts are 5, 9, 18, and 36.  The
ledger agreed with measured peaks within 7 percent at the 2048 class
(`archive/torch_port/active/multigpu_findings.md`, section 1.46), so these
counts should hold within one part.

Two details matter when pricing.  With compilation off the ledger charges
16 prior cylinders instead of 9, and the peak rises by 30 percent.  Without
CUDA the ledger prices the torch fallback bodies, whose batch transients
are far larger than the kernels', so the script's plan mode substitutes
the kernel cost functions when it runs on a machine without a GPU.

## Host memory

The stitch is the host peak.  `stitch_arrays` concatenates twice per part,
so at the last seam it holds the list of parts, the volume stitched so far,
and the new volume.  At 4K that is 2.7 to 2.9 volumes: 730 GB with 3 parts
and 810 GB with 17.  The sinogram adds 60 GB unless it is memory mapped
from a `.npy` file, and the weights add another 60 GB.

On gautschi the `ai` partition gives 9200 MB of host memory per CPU and
14 CPUs per GPU, and it refuses `--mem`.  The 4K stitch therefore needs the
whole node, which is 8 GPUs and 1030 GB.  A half node with 4 GPUs has
515 GB and cannot run the 4K case today, although its GPUs could.

The streaming harness `experiments/parallel_4k/stream_recon_parallel.py`
removes this limit without changing the library.  It reads one band of
sinogram rows from disk, reconstructs it with `recon`, blends the seam with
the previous band's last slices read back from the output file, writes the
band, and moves on.  The host holds one band of the sinogram, its weights,
and one reconstructed band, about 90 GB at 4K with 5 bands.  Its output
equals `recon_split_sino`'s to 3e-7 NRMSE on the same seed, and a progress
file makes a run resumable.  It uses one private call,
`_fits_available_devices`, to size the bands.

## Time, estimated

The estimate scales the dashboard's measured reconstruction of a
1024 x 1008 x 992 volume on H100 cards (prerelease, 2026-09-16).  A
3-iteration run took 18.96 s on one card.  About 4.2 s of that is the
setup, so one iteration costs about 4.9 s.  Four cards ran it 3.21 times
faster.  The 4K case has 61 times the projection work, counted as views
times pixels in the mask times slices, and 69 times the voxels.  One 4K
iteration should therefore cost about 5 minutes on one H100.  Fifteen
iterations then take about 1.5 hours on one GPU, 25 to 30 minutes on four,
and 15 to 20 minutes on eight if the scaling continues.  These numbers are
extrapolations by a factor of 60, and the measurement below replaces them.

## The measurement

Job 16629880 on gautschi was to run `gautschi_4k.sbatch` on one node with 8
H100 cards.  Greg cancelled it on 2026-09-23 pending discussion of the
streaming harness, so nothing has been measured yet.  When a job is
submitted again, the plan is this.  It times cells of 900 views at 1024, 1536, and 2048 rows and
channels on 1, 4, and 8 GPUs, each as one part.  Each cell runs with 3 and
with 6 iterations, so that the difference is the cost of three
steady-state iterations, and each run repeats once so that the second run
is warm.  The job then reconstructs the 4K case on 8 GPUs with the
library's own part count, and again with 3 parts if that run fails.  The
record is `experiments/recon4k_gautschi.md`.

## Decisions pending

1. Whether the streaming harness, rather than `recon_split_sino`, is what
   collaborators run, and whether the library adopts its loop.
2. Which interfaces the library exposes for it: the band-size check that
   `_fits_available_devices` performs, and a slab-wise gather of a
   reconstruction from the GPUs to disk.
3. Where the script should live for collaborators: `experiments/parallel_4k/`
   in mbirtorch today, or a documented demo.
4. Whether to add a parallel-beam test for `recon_split_sino`;
   `tests/test_split_sino.py` covers cone beam only.
