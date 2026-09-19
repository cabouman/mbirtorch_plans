# LEAP vs mbirtorch: features and performance, version 2

Status: ACTIVE
Updated: 2026-09-18
Code: mbirtorch efeca90 (greg_dev, 2026-09-14) and LEAP v1.26 at 0c8846f4 for the measurements
Next step: The follow-ups in the survey: a fixed-quality comparison against the ORNL OGM2 loop, the fusion prototype on the ORNL scan, an adjoint check on a texture-free LEAP build, a real scan with a known geometry error, and more than one noise level.

Date: 2026-09-14.  Versions compared: LEAP v1.26 at commit `0c8846f4`, mbirtorch 0.0.2 at commit
`efeca90` on branch `greg_dev`, the 4D reconstruction work at commit `8304b25` on branch
`mace_4d_dev`, and the ORNL `leapMBIR` scripts at commit `044f38a`.

This version supersedes `leap_comparison.md` of 2026-09-02, deleted on 2026-09-18 (`git show cd4aac3:plans/features/leap_comparison/leap_comparison.md`).  Four things changed since that version.
mbirtorch gained geometry calibration, a geometry viewer, a 4D reconstruction branch, and lower GPU
memory.  New measurements corrected what the first version said about LEAP's adjoint pair.  A
reproduction of an ORNL comparison measured both packages on a real scan on four GPUs.  The LEAP
side is the same code as before.

## Summary

LEAP and mbirtorch solve overlapping problems with different designs.  LEAP is a CUDA package of CT
algorithms, covering many geometries, many direct and iterative reconstruction methods, and many
correction and calibration utilities.  mbirtorch is a model-based reconstruction package built on
PyTorch, with one main reconstruction algorithm, four geometries, geometry calibration, a geometry
viewer, and automatic multi-GPU execution.  A consensus loop for plug-and-play priors and a 4D
reconstruction are on a development branch.  LEAP covers more capabilities.  mbirtorch covers fewer
capabilities, and supports each one with automatic parameter selection, multi-GPU execution, and
tests.

The two packages made the same choice on the most important design question.  Both use a
separable-footprint projector model, and both supply a matched forward and back projector pair.  An
adjoint of a linear operator is its exact transpose.  The two pairs are adjoint to different
accuracies.  mbirtorch's pair is adjoint to float32 rounding, at a relative difference of 1.240e-09
at N = 256 and 4.068e-09 at N = 64.  LEAP's pair is adjoint up to a quantization error, at 5.918e-06
at N = 256 and 3.881e-04 at N = 64.  Both LEAP kernels read their input through a CUDA texture,
which is a read path whose hardware interpolates between neighboring samples, and the interpolation
weights are rounded to eight fractional bits.

The LEAP features that mbirtorch lacks are these, ranked by value to mbirtorch users:

1. Offset-scan and truncated-object support in direct and iterative reconstruction.
2. A low-memory execution mode, which splits projections into chunks that fit GPU memory and runs a
   large scan on small GPUs from host-resident arrays.  A chunk is a piece of one projection or one
   volume that LEAP streams through a GPU and then discards.
3. Scatter correction, from a constant transmission offset up to a physics-based first-order model.
4. Analytic ray-traced phantoms.
5. Polychromatic and dual-energy physics through XrayPhysics, including spectrum-based
   beam-hardening correction.
6. A modular geometry with an arbitrary source position and detector pose per view.
7. The remaining calibration utilities, which are the source offset `tau`, the ball-phantom fit, the
   inconsistency reconstruction, and the MTF measurement.
8. Alternative iterative algorithms, which are SIRT, SART, MLEM, OSEM, MLTR, ASD-POCS, and RDLS, and
   a native total-variation prior.
9. A fan-beam geometry as its own model.
10. Detector deblur.

Three reasons account for every change from version 1's ranking.  Calibration moved down, because
mbirtorch now estimates the center of rotation and the detector rotation from the data.  The
low-memory mode moved up, because the ORNL measurement showed the size of the difference on a real
scan.  Physics corrections moved up, because the ORNL users apply LEAP's scatter and beam-hardening
corrections to every scan.

mbirtorch's advantages over LEAP are these, with the released capabilities first:

1. Automatic sharding of one reconstruction across GPUs.  Sharding means splitting the volume by
   slice and the sinogram by view and running the pieces together, and a memory ledger prices every
   candidate layout before the first large allocation.
2. An adjoint pair exact to float32 rounding, checked by tests.
3. VCD, which is Multi-Granular Vectorized Coordinate Descent, with a qGGMRF prior, automatic
   parameters, and a relative-change stop rule.  It reached a fixed image quality in far fewer
   iterations than LEAP's regularized weighted least squares, RWLS, with a total-variation prior.
4. A proximal map and a denoiser that accept data already split across GPUs, with a MACE consensus
   loop on a development branch and a multi-slice fusion prototype in research scripts.  MACE is
   multi-agent consensus equilibrium, a loop in which several agents each map an array to the array
   that agent prefers, and the consensus is the weighted average they agree on at a fixed point.
5. Geometry calibration from the sinogram.  Its offset estimator agrees with vendor values on real
   scans and was more accurate than LEAP's inside its search window.  Its rotation estimator has an
   object-dependent zero point, and the documentation tells users to prefer a vendor value.
6. A geometry viewer with five panels, two data overlays, and a geometry comparison.
7. Vendor scanner readers and one-call model construction.
8. Installation from PyPI with no compiler, on CUDA, CPU, and Apple MPS.
9. Tests, continuous integration, and documentation, at 988 test functions in 45 files, 31
   documentation pages, and 11 demo scripts.
10. Capabilities LEAP does not have at all, which are translation tomography, multi-axis parallel
    beam, automated view selection, and hyperspectral neutron support.
11. 4D reconstruction of a continuous scan of a moving object, on a development branch and nearly
    complete.

Five results carry the comparison.  mbirtorch reached a fixed NRMSE target in 7 iterations
against LEAP's 100 at N = 256, in 10 against 100 at N = 512, and in 5 against 40 at N = 1024.  NRMSE
is the root-mean-square difference between two arrays divided by the root-mean-square value of the
reference array.  On a warm clock mbirtorch reached that target 3.4 times faster than LEAP at
N = 256 and 6.6 times faster at N = 512.  On the ORNL scan on four H100 GPUs, fifteen iterations
took 619.0 s for mbirtorch and 829.8 s for the ORNL OGM2 loop, which is the collaborators' own
Nesterov OGM2 loop calling LEAP's projectors.  Peak GPU memory over the four GPUs was 156.4 GiB for
mbirtorch and 22.2 GiB for that loop, and peak host memory was 72.6 GiB against 175.4 GiB.  Changes
committed on 2026-09-14 lowered mbirtorch's combined GPU figure from 156.32 GiB to 124.65 GiB in
runs that pinned four GPUs.  The quality result is one phantom at one noise level, and the ORNL
result is one scan.  The records are `surveys/leap_comparison/findings/quality_results.md`, `surveys/leap_comparison/findings/ornl_reproduction.md`, and
`plans/device_memory_tiers/experiments/dm1_record.md`, whose paths are in Sources.

## Scope and versions

This document compares the features and the recorded performance of the two packages.  The evidence
has three parts: two sourced inventories prepared for version 1, the LEAP source at the pinned
commit, and the measurement records in this repository.

The four pinned versions are these.  LEAP is v1.26 at commit
`0c8846f42b2e59340d5559fc1271d590a292f9a0`, dated 2024-12-14.  mbirtorch is 0.0.2 at commit
`efeca90ff0aa2aaf5326617ab899ad794af9db67` on branch `greg_dev`, dated 2026-09-14, which is 14
commits after the commit version 1 compared.  The 4D reconstruction work is at commit
`8304b25ec4f1877a236946b809312779fae38917` on branch `mace_4d_dev`, dated 2026-09-14.  The ORNL code
is `leapMBIR` at commit `044f38ae2a6bc4509f565ba0384b426998bd43fc`, dated 2026-09-12.

No mbirtorch performance number in this document was measured at `efeca90`.  The head-to-head
benchmark ran at `26bd0ea`, the ORNL reproduction at `41fca86` of 2026-09-12, and the after-change
memory runs at `23c4a43`.

The LEAP side has not moved since version 1.  A GitHub API read on 2026-09-14 reported v1.26,
published 2024-12-14T23:24:36Z, as still the latest release, and the unreleased `version_two` branch
as still at commit `0c42bb2d71f8fee0120ab7a024ddbf1d1b21ba17` of 2026-07-25.  Every LEAP measurement
below used a cluster build of the pinned commit.

## What changed since the first comparison

### On the mbirtorch side

- A geometry calibration module, written 2026-09-04 to 2026-09-06.  Its public functions are
  `CalibrationResult`, `build_reduced_problem`, `reduce_sinogram`, `parameter_sweep`,
  `check_rotation_direction`, `apply_calibration`, `estimate_det_channel_offset`,
  `estimate_det_rotation`, and `conjugate_difference`.
- A geometry viewer subpackage, `mbirtorch/viewers/`, added 2026-09-12, with
  `TomographyModel.project_points` as the geometric map its panels share.
- Reconstruction geometry rules, also 2026-09-12.  `copy_ct_model` keeps the parent's reconstruction
  geometry, and the automatic passes center the volume on the illuminated detector band.
- GPU memory work on 2026-09-14, in commits `a225319`, `3101e93`, and `23c4a43`.  The error sinogram
  is formed in place, the sinogram reductions are chunked, the initial scale is computed without
  sinogram-sized temporaries, and `get_memory_stats` reports the allocator pool.
- The shard gather rebuilt, in commit `efeca90`, and a 4D reconstruction branch, `mace_4d_dev`.
- A multi-slice fusion prototype with a network denoiser.  It lives in mbirtorch's `experiments/drunet/` and in
  `mbirtorch_applications/nsi/msf_recon.py`, and it is not in the package on `greg_dev`.  On
  `mace_4d_dev` the loop and the agents are in the package as `mbirtorch/mace.py`.

### On the LEAP side

New measurements on the same LEAP code produced three findings:

- The adjoint finding.  LEAP's pair is matched up to the quantization of its texture interpolation
  weights.
- The multi-GPU memory finding.  On four GPUs with host-resident arrays the ORNL OGM2 loop held
  22.2 GiB across the four GPUs, against mbirtorch's 156.4 GiB.  On one GPU LEAP does not chunk.
- The calibration finding.  LEAP's `estimate_tilt` and `find_centerCol` were measured against
  mbirtorch's estimators on synthetic data and on three real scans.

### The ORNL comparison

Collaborators at ORNL reconstructed one Metrotom scan of an Inconel additive-manufacturing sample
with LEAP and with mbirtorch on four A100 40 GB GPUs, and reported that LEAP needed one sixth of the
GPU memory at about the same time per iteration.

## Feature comparison

Two inventory files are cited by abbreviation below:

- `LEAP-inv` = `surveys/leap_comparison/leap_inventory.md`.
- `MT-inv` = `surveys/leap_comparison/mbirtorch_inventory.md`, deleted on 2026-09-18 and
  readable with `git show cd4aac3:plans/features/leap_comparison/leap_comparison_sources/mbirtorch_inventory.md`, which
  describes mbirtorch at commit `26bd0ea`.

### Geometries

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Parallel and cone beam, helical, curved detector | yes | yes | [src/parameters.h#L620](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/parameters.h#L620); [mbirtorch/cone_beam.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/cone_beam.py) |
| Fan beam | yes, own type | no class | [src/leapctype.py#L545](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L545); `MT-inv` section 2 |
| Cone-parallel | yes | no | [src/leapctype.py#L431](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L431) |
| Modular, per-view source and detector pose | yes | no | [src/leapctype.py#L638](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L638) |
| Laminography | via modular beam | via multi-axis parallel | [demo_leapctype/d32_laminography.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d32_laminography.py); [mbirtorch/multiaxis_parallel.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/multiaxis_parallel.py) |
| Multi-axis parallel, per-view elevation | no | yes | [mbirtorch/multiaxis_parallel.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/multiaxis_parallel.py) |
| Translation tomography | no | yes, alpha | [mbirtorch/translation_model.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/translation_model.py) |
| Detector rotation as a model parameter | yes, `tiltAngle`, applied inside the cone-beam kernels, capped at 5 degrees | estimated from the data and applied by resampling the sinogram, capped at 5 degrees | [src/leapctype.py#L750](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L750) and [src/projectors_SF.cu#L1790](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors_SF.cu#L1790); `apply_calibration` and `_MAX_DET_ROTATION` in [mbirtorch/preprocess/geometry_calibration.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/preprocess/geometry_calibration.py) |
| Symmetric objects, attenuated Radon transform | yes | no | [src/leapctype.py#L5767](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5767) |

LEAP selects each geometry by a runtime parameter rather than by a separate class, and two mbirtorch
geometries have no LEAP counterpart.  Both packages cap a detector rotation at 5 degrees.

### Projector models and adjoints

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Model family | separable footprint | separable footprint | [documentation/LEAP.tex](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex); [mbirtorch/horizontal_fan.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/horizontal_fan.py) |
| Matched forward and back pair | yes, up to texture quantization | yes, to float32 rounding | [src/projectors_SF.cu#L2253-L2259](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors_SF.cu#L2253-L2259); [docs/source/usr_autograd.rst](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/docs/source/usr_autograd.rst) |
| Input read through CUDA textures with hardware interpolation | yes, both kernels | no, global memory with weights computed in the kernel | [src/projectors_SF.cu#L2367-L2372](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors_SF.cu#L2367-L2372); [mbirtorch/projectors.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/projectors.py) |
| Unmatched fast backprojector option | yes, `set_projector('VD')` | no | [src/leapctype.py#L5788](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5788) |
| Alternative model for hard cases | extended SF, Joseph for modular | one model per geometry | [src/projectors.cpp#L74-L80](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors.cpp#L74-L80) |
| Adjointness checked by an automated test | no | yes, at 1e-4 relative | `LEAP-inv` section 13.3; [tests/test_adjoint.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/tests/test_adjoint.py) |
| Implementation language | CUDA C plus OpenMP C++ | PyTorch plus Triton | `LEAP-inv` section 10.1; [docs/source/dev_projector_kernels.rst](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/docs/source/dev_projector_kernels.rst) |
| Precision | float32 only | float32 only | `LEAP-inv` section 4.5; `MT-inv` section 3 |

LEAP's forward kernel reads the volume through a 3D texture with linear interpolation, and its back
kernel reads the projections the same way, so every fetch interpolates in hardware.  A fetch
coordinate's fractional part is held in a 9-bit fixed point format with 8 bits of fractional value,
per the CUDA C Programming Guide, PG-02829-001_v10.0 of October 2018, Appendix G section G.2, page
243.  That rounding is why LEAP's forward and back projectors are not exact transposes.

### Direct reconstruction

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| FBP and FDK | yes, all geometries | yes, all four geometries | [src/leapctype.py#L2595](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2595); `recon_fdk` in [mbirtorch/cone_beam.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/cone_beam.py) |
| Ramp filter choices | 7 orders, 0 through 12 | 1, "ramp" only | [src/leapctype.py#L5831](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5831); [mbirtorch/tomography_utils.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/tomography_utils.py) |
| Low-pass on the ramp filter | yes | no | [src/leapctype.py#L5854](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5854) |
| Short-scan Parker weighting | yes, applied automatically | no | [src/ray_weighting_cpu.cpp#L66](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/ray_weighting_cpu.cpp#L66) |
| Non-equispaced view weighting | yes | no | [src/ray_weighting_cpu.cpp#L343](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/ray_weighting_cpu.cpp#L343) |
| Offset scan, truncated scan | yes | no extrapolating ramp filter | [src/leapctype.py#L5739](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5739); `scale_recon_shape` in [mbirtorch/tomography_model.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/tomography_model.py) |
| Single-slice reconstruction along x, y, or z | yes | no | [src/leapctype.py#L2504](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2504) |
| Separate filter and backproject steps | yes | yes | [src/leapctype.py#L2159](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2159); [mbirtorch/cone_beam.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/cone_beam.py) |

mbirtorch treats its direct reconstruction mainly as an initializer for VCD, and its docstrings call
a standalone direct reconstruction only approximate for nonuniform, limited-angle, or short scans.

### Iterative reconstruction algorithms

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Coordinate descent on a MAP objective | no | yes, VCD | `recon` in [mbirtorch/tomography_model.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/tomography_model.py) |
| Algebraic methods, SIRT and SART | yes | no | [src/leapctype.py#L3773](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3773) |
| Emission methods, MLEM and OSEM | yes | no | [src/leapctype.py#L3606](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3606) |
| Transmission statistical methods, RWLS and MLTR | yes | no | [src/leapctype.py#L4175](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4175) |
| Few-view methods, ASD-POCS and RDLS | yes | no | [src/leapctype.py#L3904](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3904) |
| View subsets to accelerate early iterations | yes, `numSubsets` | no | [src/leapctype.py#L3542](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3542) |
| Preconditioners | yes, three named choices | one fixed cone damping profile | [demo_leapctype/d26_preconditioners.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d26_preconditioners.py) |
| Stopping rule other than an iteration count | no | yes, relative change | `LEAP-inv` section 6.4; [mbirtorch/tomography_model.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/tomography_model.py) |
| Region-of-interest mask | arbitrary binary 3D mask, applied inside the operators | 2D region mask applied to every slice | [src/leapctype.py#L3378](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3378); [mbirtorch/vcd_utils.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/vcd_utils.py) |
| Proximal map for plug and play | no | yes, `prox_map`, and a MACE consensus loop on `mace_4d_dev` | [mbirtorch/tomography_model.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/tomography_model.py); [mbirtorch/mace.py](https://github.com/cabouman/mbirtorch/blob/8304b25ec4f1877a236946b809312779fae38917/mbirtorch/mace.py) |
| 4D reconstruction of a moving object | no | yes, `MACE4DModel` on `mace_4d_dev` | [mbirtorch/mace4d.py](https://github.com/cabouman/mbirtorch/blob/8304b25ec4f1877a236946b809312779fae38917/mbirtorch/mace4d.py) |

LEAP's algorithms stop only when the iteration count runs out, while mbirtorch stops on a measured
relative change.  The two packages also mean different things by a subset.  LEAP's ordered subsets
are subsets of projection views, while mbirtorch's multi-granular partitions are subsets of
reconstruction voxels whose every update uses all views.

### Regularizers and priors

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| qGGMRF prior | no | yes | [docs/source/theory.rst](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/docs/source/theory.rst) |
| Anisotropic total variation | yes | no | [src/leap_filter_sequence.py#L193](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py#L193) |
| Composable sequence of several priors | yes, `filterSequence` | no | [docs/source/filter_sequence.rst](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/filter_sequence.rst) |
| Histogram sparsity, azimuthal sparsity | yes | no | [src/leap_filter_sequence.py#L431](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py#L431) |
| Prior-image regularization | yes, an `f_0` argument | no | [src/leap_filter_sequence.py#L193](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py#L193) |
| Standalone denoisers | many, including bilateral and guided | qGGMRF denoiser, 3D median filter, and `denoise_stack` on `mace_4d_dev` | [docs/source/filters.rst](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/filters.rst); [mbirtorch/denoising.py](https://github.com/cabouman/mbirtorch/blob/8304b25ec4f1877a236946b809312779fae38917/mbirtorch/denoising.py) |
| Automatic regularization strength | no | yes, from `sharpness` and `snr_db` | [mbirtorch/tomography_model.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/tomography_model.py) |

LEAP gives the user several regularizers to combine and expects the user to choose the weights.  A
network-denoiser prior runs in mbirtorch through the MACE loop, not through the VCD prior term.

### Preprocessing and artifact correction

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Flat and dark correction, negative log | yes | yes | [src/leap_preprocessing_algorithms.py#L176](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L176); [mbirtorch/preprocess/utilities.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/preprocess/utilities.py) |
| Bad pixel and outlier correction | yes | yes | [src/leap_preprocessing_algorithms.py#L287](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L287); [mbirtorch/preprocess/utilities.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/preprocess/utilities.py) |
| Low-signal and high-energy outlier correction | yes | no | [src/leap_preprocessing_algorithms.py#L373](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L373) |
| Ring and stripe removal | three variants | three routines | [src/leap_preprocessing_algorithms.py#L506](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L506); [mbirtorch/preprocess/stripe.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/preprocess/stripe.py) |
| Detector deblur | yes, Wiener and Richardson-Lucy | no | [src/leap_preprocessing_algorithms.py#L439](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L439) |
| Scatter correction | yes, physics-based first order | no | [src/leapctype.py#L1473](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1473); `MT-inv` section 4 |
| Metal artifact reduction | sinogram replacement | weights plus a beam-hardening loop | [src/leapctype.py#L1343](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1343); [mbirtorch/tomography_model.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/tomography_model.py) |
| Beam hardening correction | yes, spectrum-based | yes, empirical curve fit | [docs/source/physics_based_preprocessing_algorithms.rst](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/physics_based_preprocessing_algorithms.rst); [mbirtorch/preprocess/utilities.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/preprocess/utilities.py) |
| Vendor scanner readers | none found | four formats | `LEAP-inv` section 11; [mbirtorch/preprocess/](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/preprocess/) |
| Zeiss Metrotom log and raw counts | no | no | section "The ORNL code" below, where the scripts mine the log themselves |

The two preprocessing sets have most operations in common, and differ in physics modeling and in
data ingestion.  The ORNL users apply LEAP's corrections before reconstructing with either package.

### Geometric calibration

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Center of rotation from the data | yes, `find_centerCol` | yes, `estimate_det_channel_offset`, conjugate-view method | [src/leapctype.py#L846](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L846); [mbirtorch/preprocess/geometry_calibration.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/preprocess/geometry_calibration.py) |
| Detector rotation from the data | yes, `estimate_tilt` | yes, `estimate_det_rotation` | [src/leapctype.py#L937](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L937); [mbirtorch/preprocess/geometry_calibration.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/preprocess/geometry_calibration.py) |
| Source offset `tau` | yes, `find_tau` | no | [src/leapctype.py#L892](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L892) |
| Rotation-direction check | no | yes, `check_rotation_direction` | [mbirtorch/preprocess/geometry_calibration.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/preprocess/geometry_calibration.py) |
| Data-consistency metric | yes | yes, `conjugate_difference` | [src/leapctype.py#L1021](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1021); [mbirtorch/preprocess/geometry_calibration.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/preprocess/geometry_calibration.py) |
| Inconsistency reconstruction | yes | no | [src/leapctype.py#L2845](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2845) |
| Joint search | yes, a two-parameter search | the documented alternating sequence, with a reconstruction-scored estimator planned | [src/leap_preprocessing_algorithms.py#L737](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L737); [docs/source/usr_preprocess.rst](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/docs/source/usr_preprocess.rst) |
| Parameter sweep, one slice per candidate | yes, eight parameters | yes, three: `det_channel_offset`, `det_row_offset`, `det_rotation` | [src/leap_preprocessing_algorithms.py#L872](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L872); `parameter_sweep` in [mbirtorch/preprocess/geometry_calibration.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/preprocess/geometry_calibration.py) |
| Calibration from a ball phantom | yes | no | [src/leap_preprocessing_algorithms.py#L1159](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L1159) |
| Resolution measurement, MTF | yes | no | [src/leap_preprocessing_algorithms.py#L1087](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L1087) |
| View alignment across views | no | yes, `align_sino_views` | [mbirtorch/preprocess/utilities.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/preprocess/utilities.py) |
| Applies the estimate to the model and sinogram | the user sets the parameters | yes, `apply_calibration` | [mbirtorch/preprocess/geometry_calibration.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/preprocess/geometry_calibration.py) |

mbirtorch's estimators accept a parallel-beam or a cone-beam scan over a full rotation.  They refuse
four kinds of input: a scan over less than a full rotation, a helical scan, a multiaxis scan, and a
sinogram already split across GPUs.  `parameter_sweep` accepts every scan the readers produce.

On synthetic data the two estimator sets were measured against each other at 512 and 1024 channels.
mbirtorch was the more accurate on offset, the two were comparable on rotation, and LEAP was the
faster of the two.  The measured values are these:

| Quantity | mbirtorch | LEAP |
| --- | --- | --- |
| Offset error, 512 channels | within 0.005 channels | 0.017 to 0.024 channels at true offsets 1.3 and -2.2; 0.003 channels or less at 0.0 and 7.5 |
| Offset error, 1024 channels | within 0.001 channels | as above |
| Rotation below half a pixel of edge displacement | 10 to 24 percent low | 10 to 18 percent high |
| Rotation above 1.34 pixels of edge displacement | within 0.5 percent | within 1 percent |
| Offset time | 0.3 to 4.0 s | 0.1 to 0.9 s |
| Rotation time | 2.1 to 11.0 s | 0.01 to 0.05 s |

One synthetic case failed.  At a true offset of 7.5 channels the mbirtorch estimate stopped at the
edge of its default search window and erred by 3.5 channels, while LEAP found the offset.  The
window now moves to center on the edge where the coarse minimum sits, at the same width, up to eight
times, and that change has not been re-measured against LEAP.

On real scans neither estimator measured the reference rotation.  LEAP's `estimate_tilt` returned
-1.029 degrees on an NSI scan.  Reconstructions of that scan put the rotation at the vendor's 0.167
degrees.  The `estimate_tilt` cost was monotone over the range -0.4 to 0.4 degrees, so its search
had no interior minimum to find.  The mbirtorch band estimator has a clear minimum on that scan, and
on that object the minimum sits at the wrong zero point of 0.047 degrees.  A fine sweep of slices
far from the central plane put the rotation of that scan near 0.15 degrees.

### Simulation and physics

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Voxelized phantoms | yes | yes, several generators | [src/leapctype.py#L7286](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7286); [mbirtorch/utilities.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/utilities.py) |
| Analytic ray-traced phantoms | yes, eight primitive shapes | no | [src/leapctype.py#L7304](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7304) |
| FORBILD head phantom | yes | no | [src/leapctype.py#L7357](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7357) |
| Ray oversampling for partial volume | yes | no | [demo_leapctype/d08_ray_tracing_simulation.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d08_ray_tracing_simulation.py) |
| Built-in noise model | no | no | `LEAP-inv` section 8; `MT-inv` section 7 |
| Source spectra and detector response | yes, via XrayPhysics | no | [docs/source/physics_based_preprocessing_algorithms.rst](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/physics_based_preprocessing_algorithms.rst) |
| Dual-energy decomposition | yes | no | [src/leapctype.py#L4829](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4829) |
| Hyperspectral neutron data support | no | yes | [docs/source/usr_hsnt.rst](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/docs/source/usr_hsnt.rst) |

An inverse crime is the use of the same discretized forward model to make the data and to
reconstruct it.  A study that commits one reports a lower error than the method would achieve on
real data.  LEAP can avoid it, because it can ray-trace an analytic phantom instead of
forward-projecting a voxelized one.

### Deep-learning integration and autograd

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Differentiable forward and back projection | yes | yes | [src/leaptorch.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py); [mbirtorch/autograd.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/autograd.py) |
| Gradient is the adjoint | yes | yes | [src/leaptorch.py#L31-L36](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py#L31-L36); [docs/source/usr_autograd.rst](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/docs/source/usr_autograd.rst) |
| `nn.Module` wrapper | yes, `Projector` | yes, `TorchProjector` | [src/leaptorch.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py); [mbirtorch/autograd.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/autograd.py) |
| Differentiable FBP | yes | no | [src/leaptorch.py#L145](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py#L145) |
| Batch dimension | yes, a Python loop per element | no | [src/leaptorch.py#L41-L45](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py#L41-L45) |
| Runs on GPU-resident tensors | yes, one GPU at a time | yes, one GPU | [demo_leapctype/d02_standard_geometries_torch.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d02_standard_geometries_torch.py); [docs/source/usr_autograd.rst](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/docs/source/usr_autograd.rst) |

Both packages expose a differentiable projector whose backward pass is the adjoint operator rather
than a traced graph.  A comment in LEAP's source marks its FBP backward as needing replacement.

### Compute, GPUs, and memory

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| CUDA | yes | yes | [src/CMakeLists.txt](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/CMakeLists.txt); [mbirtorch/tomography_model.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/tomography_model.py) |
| CPU | yes, OpenMP, with gaps | yes, every geometry | `LEAP-inv` section 10.2; `MT-inv` section 6 |
| Apple MPS | no | yes | `LEAP-inv` section 10.3; [mbirtorch/tomography_model.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/tomography_model.py) |
| AMD GPUs | listed as future work | untested | [README.md](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md); `MT-inv` section 12.2 |
| Multi-GPU | yes, streamed chunks | yes, sharded volume and sinogram | [src/tomographic_models.cpp#L803-L804](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L803-L804); [docs/source/dev_sharding_overview.rst](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/docs/source/dev_sharding_overview.rst) |
| Multi-GPU with GPU-resident input | no | yes | [src/leapctype.py#L39](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L39); [docs/source/usr_multi_gpu.rst](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/docs/source/usr_multi_gpu.rst) |
| Automatic chunking below GPU memory | yes, a halving loop | no, a manual band split today, made automatic by a later increment of the device memory tiers plan | [src/tomographic_models.cpp#L779-L787](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L779-L787); `recon_split_sino` in [mbirtorch/tomography_model.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/tomography_model.py) |
| Detector-row and slice range calculators | yes | no | [src/leapctype.py#L2962](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2962) |
| Memory cost computed before allocation | yes, an error message | yes, a memory ledger, and `get_memory_stats` now reports the allocator pool's peak and its unused part | [src/projectors.cpp#L66-L70](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors.cpp#L66-L70); [mbirtorch/_memory_ledger.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/_memory_ledger.py) and [mbirtorch/memory_stats.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/memory_stats.py) |
| GPU memory on four GPUs, ORNL scan | 22.2 GiB combined | 156.32 GiB combined before the September changes and 124.65 GiB after, or 116.68 GiB with the allocator setting | `surveys/leap_comparison/findings/ornl_reproduction.md`; `plans/device_memory_tiers/experiments/dm1_record.md` |
| Host memory on the same run | 175.4 GiB | 72.6 GiB | `surveys/leap_comparison/findings/ornl_reproduction.md` |
| Reduced precision, multi-node | no | no | `LEAP-inv` sections 4.5 and 10.5; [docs/source/usr_multi_gpu.rst](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/docs/source/usr_multi_gpu.rst) |

The two designs differ in where the arrays live.  LEAP owns no arrays, and it streams chunks of at
most 128 detector rows or 128 volume slices from host memory through the GPUs.  mbirtorch holds the
split problem resident on the GPUs for the whole run.  The price LEAP pays is host memory and host
arithmetic, and the price mbirtorch pays is GPU memory.

### Data I/O and visualization

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| File formats written | tif sequence, nrrd, npy | HDF5 | [src/leapctype.py#L6709](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6709); [mbirtorch/viewers/slice_figure.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/viewers/slice_figure.py) |
| General array formats read | tif sequence, nrrd, npy | npy, npz, HDF5 | [src/leapctype.py#L7092](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7092); [mbirtorch/viewers/slice_figure.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/viewers/slice_figure.py) |
| TIFF writing | yes | no | `LEAP-inv` section 11; `MT-inv` section 7 |
| DICOM | no | no | `LEAP-inv` section 11; `MT-inv` section 7 |
| Geometry parameters to a file | yes, a text file | no dedicated pair | [src/leapctype.py#L6695](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6695); `MT-inv` section 7 |
| Volume viewer | napari | a matplotlib slice viewer | [src/leapctype.py#L6255](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6255); [mbirtorch/viewers/slice_figure.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/viewers/slice_figure.py) |
| Geometry viewer | a static sketch, `sketch_system` | five panels, a view slider, a sinogram overlay on the detector face, a reconstruction silhouette, and a comparison of two geometries | [src/leapctype.py#L6279](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6279); [docs/source/usr_geometry_viewer.rst](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/docs/source/usr_geometry_viewer.rst) |
| Graphical user interface | a separate repository | none | [README.md](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md) |
| Bridges to other toolkits | TIGRE and LTT | none | [utils/](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/utils/) |

### API, documentation, tests, and packaging

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Shortest working example | about ten lines | two lines | [demo_leapctype/d01_standard_geometries.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d01_standard_geometries.py); [README.md](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/README.md) |
| Who owns the arrays | the caller allocates and passes | the package returns numpy arrays | [documentation/LEAP.tex](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex); [mbirtorch/__init__.py](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/mbirtorch/__init__.py) |
| Documentation pages | 18, plus a PDF manual | 31 | `LEAP-inv` section 13.3; counted under `docs/source/` at `efeca90` |
| Documentation version label | v1.4 | current | `LEAP-inv` section 21 |
| Demo scripts | 38, plus 5 for the torch interface | 11 | `LEAP-inv` section 13.3; counted under `demo/` at `efeca90` |
| Tests | 3 scripts | 988 test functions in 45 files | `LEAP-inv` section 13.3; counted under `tests/` at `efeca90` |
| Continuous integration | none | GitHub Actions, four Python versions | `LEAP-inv` section 13.3; [.github/workflows/ci.yml](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/.github/workflows/ci.yml) |
| Install without a compiler | no, source build or conda | yes, from PyPI | `LEAP-inv` sections 17 and 20; `MT-inv` section 1 |

Most of LEAP's 38 demo scripts carry an explanatory docstring, which makes them usable as teaching
material.  LEAP's three test files are scripts rather than a suite, nothing runs them automatically,
and the main script's geometry loop is disabled as checked in
([unitTests/unit_tests.py#L40](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/unitTests/unit_tests.py#L40)).

## Performance

Version 1 records LEAP's four published timings and mbirtorch's own regression numbers, and neither
set is repeated here.

### 1. Head-to-head on H100s

Both packages ran the same circular cone-beam problem at three sizes on one NVIDIA H100 80 GB.  Each
size N used N views over a full turn, an N by N flat detector, and an N by N by N volume.  The
phantom was the same array in both runs, LEAP 1.26 was built from source, mbirtorch 0.0.2 ran at
commit `26bd0ea`, and both used torch 2.13.0+cu130.  The multi-GPU comparison used four of the same
GPUs on one node.  Each projection and direct-reconstruction time below is the best of three timed
repeats after one warmup run, and the ten-iteration rows are single runs that charge mbirtorch's
one-time compilation to the reconstruction.  The source is
`surveys/leap_comparison/experiments/results/leap_benchmark_results.md`.

| N | Operation | LEAP time | mbirtorch time | LEAP GPU peak | mbirtorch GPU peak |
| --- | --- | --- | --- | --- | --- |
| 256 | forward projection | 0.02587 s | 0.03518 s | 0.79 GiB | 1.25 GiB |
| 256 | back projection | 0.01609 s | 0.02015 s | 0.86 GiB | 1.30 GiB |
| 256 | direct reconstruction | 0.02985 s | 0.02819 s | 0.88 GiB | 1.35 GiB |
| 256 | 10 iterations | 0.9658 s | 19.28 s | 1.63 GiB | 1.78 GiB |
| 512 | forward projection | 0.3997 s | 0.5479 s | 2.11 GiB | 3.97 GiB |
| 512 | back projection | 0.2530 s | 0.3070 s | 2.61 GiB | 4.23 GiB |
| 512 | direct reconstruction | 0.2960 s | 0.3364 s | 3.11 GiB | 5.28 GiB |
| 512 | 10 iterations | 12.56 s | 32.06 s | 8.74 GiB | 7.31 GiB |
| 1024 | forward projection | 6.314 s | 8.636 s | 12.61 GiB | 17.92 GiB |
| 1024 | back projection | 4.070 s | 4.837 s | 16.61 GiB | 20.86 GiB |
| 1024 | direct reconstruction | 4.314 s | 4.980 s | 20.61 GiB | 28.93 GiB |
| 1024 | 10 iterations | 192.7 s | 200.2 s | 65.62 GiB | 45.03 GiB |

The next table gives the steady-state cost of one iteration, taken from the third of three
reconstructions run in the same process.

| N | LEAP RWLS with TV | mbirtorch VCD with qGGMRF | LEAP : mbirtorch | mbirtorch first-run extra cost |
| --- | --- | --- | --- | --- |
| 256 | 0.08782 s/iteration | 0.34643 s/iteration | 1 : 3.94 | 10.28 s |
| 512 | 1.26005 s/iteration | 1.58913 s/iteration | 1 : 1.26 | 15.74 s |
| 1024 | 19.27966 s/iteration | 17.73758 s/iteration | 1 : 0.92 | 19.45 s |

The next table gives four-GPU times at N = 1024, with each package's one-GPU time for comparison.
LEAP's four-GPU reconstruction was slower than its one-GPU reconstruction.  LEAP's multi-GPU path
runs only for host-resident arrays, so its solver paid a host transfer on every call.

| Configuration | Forward projection | Back projection | Direct reconstruction | 10 iterations |
| --- | --- | --- | --- | --- |
| LEAP, one GPU | 6.314 s | 4.070 s | 4.314 s | 192.7 s |
| mbirtorch, one GPU | 8.636 s | 4.837 s | 4.980 s | 200.2 s |
| mbirtorch, four GPUs, automatic | 8.623 s | 4.841 s | 1.429 s | 109.03 s |
| mbirtorch, four GPUs, pinned | 2.189 s | 1.331 s | 1.432 s | 75.81 s |
| LEAP, four GPUs | 2.214 s | 1.645 s | 1.849 s | 313.75 s |

The per-GPU peaks for the same four-GPU runs are these, in MiB as the record reports them:

| Configuration | Forward projection, per-GPU peak (MiB) | 10 iterations, per-GPU peak (MiB) |
| --- | --- | --- |
| mbirtorch, four GPUs, automatic | 18349, 529, 529, 529 | 20189, 13525, 13525, 12977 |
| mbirtorch, four GPUs, pinned | 9439, 6269, 6269, 6269 | 18055, 13523, 13523, 12975 |
| LEAP, four GPUs | 1941, 1813, 1941, 1941 | 12827, 2261, 2777, 2521 |

These measurements support four findings:

- On one GPU LEAP's projectors were faster at every size, and direct reconstruction was close.
- The steady-state cost per iteration crosses over between N = 512 and N = 1024, where mbirtorch
  becomes the cheaper of the two at 17.73758 s against 19.27966 s.
- With mbirtorch pinned to four GPUs, the two forward projections were within 1.2 percent of each
  other, and mbirtorch was faster on back projection, direct reconstruction, and ten iterations.
  Under mbirtorch's automatic policy the two bare projection calls stayed on one GPU.
- The two designs hold different amounts of memory for the same operation.  For a bare forward
  projection LEAP's chunked path held at most 1941 MiB on any GPU against 6269 to 9439 MiB for
  mbirtorch pinned, and for ten iterations the busiest GPU held 12827 MiB against 18055 MiB.

The next table gives the correctness cross-checks, at N = 256 unless stated otherwise.  The N = 64
readings are in `surveys/leap_comparison/experiments/results/smoke_results.jsonl`.

| Check | mbirtorch | LEAP |
| --- | --- | --- |
| Forward projections agree between the packages, NRMSE | 0.052 percent | 0.052 percent |
| Direct reconstruction against the phantom, NRMSE | 0.06090 | 0.06468 |
| Adjoint relative difference | 1.240e-09 | 5.918e-06 |
| Adjoint relative difference at N = 64 | 4.068e-09 | 3.881e-04 |

Texture quantization is the mechanism that keeps LEAP's pair from being an exact transpose.  Both
LEAP cone-beam kernels read their input through a 3D texture with linear interpolation, and the CUDA
guide cited above holds the interpolation weight to 8 fractional bits.  The forward and back kernels
round different coordinates, so the pair is matched only to that precision.  LEAP's manual says the
same in its own words: the texture scheme is "very, very slightly less accurate" than the
rectangle-rectangle method.  Each reading is one random vector pair, and the size dependence of
LEAP's reading was not explained by any measurement.

### 2. Fixed image quality on noisy data

This study measured iterations to a fixed image quality on noisy data, using the timing study's
geometry and phantom rescaled to attenuation values of 0.02, 0.01, and 0.04 per mm.  Poisson noise
at 10000 counts per pixel was added to the sinogram, and both packages reconstructed from that same
noisy sinogram and the same transmission weights.  Each package started from its own direct
reconstruction, ran for exactly k iterations per point at its own best setting, and was scored by
NRMSE against the voxelized phantom inside the inscribed cylinder.  The noiseless sinogram was
forward projected with mbirtorch rather than LEAP.  The two forward projectors agree to 0.052
percent at N = 256, so neither package is reconstructing data only its own projector could have
made.  The source is `surveys/leap_comparison/findings/quality_results.md`.

The target at each size is 1.02 times the larger of the two packages' best NRMSE at that size, which
is a quality both packages demonstrably reach.  Warm time is k times the steady-state per-iteration
time plus a warm direct reconstruction.

| N | target NRMSE | package | first k at target | NRMSE there | measured time (s) | warm time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 0.04633 | LEAP | 100 | 0.04542 | 7.895 | 8.510 |
| 256 | 0.04633 | mbirtorch | 7 | 0.04489 | 12.970 | 2.609 |
| 512 | 0.03980 | LEAP | 100 | 0.03902 | 111.777 | 121.205 |
| 512 | 0.03980 | mbirtorch | 10 | 0.03887 | 20.145 | 16.982 |
| 1024 | 0.05788 | LEAP | 40 | 0.05675 | 703.057 | 753.106 |
| 1024 | 0.05788 | mbirtorch | 5 | 0.04898 | 127.916 | 95.376 |

A later run warmed each process before timing anything, which removes compilation from the clock.
The next table repeats the comparison on that warm clock, at the two sizes where the warm run
covered the full range of k.

| N | target NRMSE | package | first k at target | warm time (s) | ratio |
| --- | --- | --- | --- | --- | --- |
| 256 | 0.04633 | LEAP | 100 | 7.921 | |
| 256 | 0.04633 | mbirtorch | 7 | 2.350 | 3.4 |
| 512 | 0.03980 | LEAP | 100 | 112.851 | |
| 512 | 0.03980 | mbirtorch | 10 | 17.020 | 6.6 |

mbirtorch's default stopping rule reaches a comparable result without the hand-tuned k values above.
With `max_iterations=100` and the default 0.2 percent relative-change rule, it stopped at 9, 10, and
12 iterations at N = 256, 512, and 1024, with NRMSE of 0.04357, 0.03887, and 0.03392, all at or
below that size's target.  A slice figure at N = 512 compares the two packages at matched wall time,
where LEAP at k = 14 reached an NRMSE of 0.07191 in 16.806 s and mbirtorch at k = 10 reached 0.03887
in 16.928 s.

mbirtorch reached the target in 8 to 14 times fewer iterations than LEAP at every size.  On the
warm-process clock it reached the target 3.4 times faster at N = 256 and 6.6 times faster at
N = 512, and the N = 1024 ratio of 7.9 rests on a carried-over direct-reconstruction time rather
than a measured one.  Its best NRMSE was lower than LEAP's at every size, and its default stop rule
landed at or below the target every time.

Three caveats qualify the comparison.  It used one phantom at one noise level with one seed.  Both
packages' regularization settings were chosen by a sweep at N = 256 and applied unchanged at the
larger sizes, and mbirtorch's winning setting was sharpness -1 rather than its shipped default of
1.0.  LEAP was still descending at the last k measured at every size, so a longer LEAP run would
lower LEAP's best, tighten the target, and raise mbirtorch's iteration count.

### 3. The ORNL scan: memory and time on four H100s

The scan is one full-turn cone-beam scan of an Inconel additive-manufacturing sample, with 2132
views and a 1456 by 1840 detector at 0.127 mm.  The reconstruction is 1360 by 1360 by 1296 voxels of
0.0172847 mm, so a sinogram-shaped float32 array is 21.3 GiB and a volume is 8.9 GiB.  The harness
ran the collaborators' loader and their OGM2 loop unchanged, with mbirtorch at commit `41fca86` of
2026-09-12 and LEAP at the pinned commit.  The source is `surveys/leap_comparison/findings/ornl_reproduction.md`.

| quantity | mbirtorch | ORNL OGM2 loop |
| --- | --- | --- |
| FDK, first run | 25.9 s | 12.2 s |
| FDK, second run | 18.0 s | 12.3 s |
| MBIR, 15 iterations | 619.0 s | 829.8 s |
| time per iteration | 41.3 s, averaged over the run | 52.0 s, steady state |
| peak GPU memory per GPU, NVML | 39.7, 39.3, 39.3, 38.0 GiB | 18.9, 6.2, 6.2, 6.2 GiB |
| peak GPU memory, four GPUs combined | 156.4 GiB | 22.2 GiB |
| peak host memory | 72.6 GiB | 175.4 GiB |

The combined figure is the peak of the summed usage over time, read by the sampler at each instant.
mbirtorch's four GPUs peak together, and LEAP's do not, because the regularizer's two volumes sit on
one GPU while the others hold only chunks, so LEAP's combined peak is below the sum of its per-GPU
peaks.

The reproduction agrees with the collaborators' own table, which reported 143.97 GB and 23.96 GB of
combined GPU memory and 85.78 GB and 175.27 GB of host memory on four A100 40 GB GPUs.  The two
reconstructions agree to 5.7 percent NRMSE in the central slice.

The two per-iteration entries in the table are measured on different bases, and mbirtorch's is the
conservative one.  An OGM2 iteration is one forward projection, one back projection, one TV
gradient, and the host arithmetic, which on four H100s took 17.0 s, 10.8 s, 3.4 s, and about 21 s,
and the 52.0 s excludes the first iteration, which took 58.1 s.  An mbirtorch iteration updates
every voxel through its subsets, and the 41.3 s average includes the compilation, the error
sinogram, and the Hessian diagonal.  In a pinned 30-iteration run mbirtorch's steady-state cost was
35.8 s per iteration, with about 31 s of one-time cost (`plans/device_memory_tiers/experiments/dm1_record.md`).  mbirtorch's stop rule did
not fire on this scan, because the relative change was 0.84 percent after fifteen iterations against
the 0.2 percent threshold.  What an iteration achieves is a separate question.  This record does not
answer it.

The GPU memory mbirtorch reports has three layers, and only the first is the arrays it needs.

| devices | iterations | persistent set per GPU | peak allocated per GPU | peak reserved per GPU | NVML peak per GPU |
| --- | --- | --- | --- | --- | --- |
| 4 | 15 | 15.3 GiB | 29.0 GiB | 38.4 GiB | 39.7 GiB |
| 2 | 5 | 30.4 GiB | 57.9 GiB | 69.8 GiB | 71.1 GiB |

Changes committed on 2026-09-14 lowered the peak of arrays in use at no time cost.  The next table
is from `plans/device_memory_tiers/experiments/dm1_record.md`, section "Verification after the changes", with mbirtorch at commit
`23c4a43`.  The runs in this table pin the GPU count to four and disable the stop rule, so their
15-iteration time of 567.6 s is 51 s below the 619.0 s of the reproduction, which chose its GPUs by
its own search.

| quantity | before the changes | after, default allocator | after, with the allocator setting |
| --- | --- | --- | --- |
| 15 iterations | 567.6 s | 567.9 s | 568.2 s |
| peak allocated, largest GPU | 29.03 GiB | 24.89 GiB | 24.89 GiB |
| peak reserved, largest GPU | 38.45 GiB | 31.56 GiB | 28.56 GiB |
| NVML peak, four GPUs combined | 156.32 GiB | 124.65 GiB | 116.68 GiB |

LEAP holds little GPU memory because it owns no arrays.  The ORNL OGM2 loop keeps every array in
host memory, and each LEAP call splits the work into chunks of at most 128 detector rows or 128
volume slices.  The widest forward chunk is 3.79 GiB and the widest back-projection chunk is 5.13
GiB.  Adding a per-chunk reserve and the CUDA context gives the 6.2 GiB measured on the three
streaming GPUs.  The exception is the regularizer, whose gradient puts two whole volumes of 8.9 GiB
on the first GPU and accounts for the 18.9 GiB measured there.  The loop holds 175 GiB in host
memory, and its own arithmetic on 21 GiB arrays runs in single-threaded numpy and does not shrink
with more GPUs.  That arithmetic took about 18 s on two GPUs and 21 s on four, while the two
projections fell from 50 s to 28 s.

Chunking is what LEAP pays for holding little GPU memory.  On one GPU LEAP is compute bound, and a
projection's copies are about a tenth of the call.  On four GPUs the same two calls took 17.0 s and
10.5 s, which is 2.8 and 2.3 times faster for four times the GPUs, so about a third of a four-GPU
call is chunk overhead.  LEAP's back-projection kernel is about 1.5 times faster per GPU than
mbirtorch's, and mbirtorch's multi-GPU path carries no chunk overhead, which is why mbirtorch is
faster on four GPUs despite the slower kernel.

LEAP chunks only when it must.  On one GPU it does not chunk at all, and the same scan put 31.3 GiB
on the GPU during the loop and 52.6 GiB during the FDK.  At N = 1024 on one GPU the head-to-head
benchmark saw the same behavior, with LEAP's iterative peak at 65.62 GiB against mbirtorch's
45.03 GiB.

The direct reconstruction splits into parts that say where each package spends its time.

| part | mbirtorch | LEAP |
| --- | --- | --- |
| sinogram, host to GPUs | 2.2 s, one GPU at a time | inside each chunk, that chunk's rows of every view |
| filter | 0.27 s, on the GPUs | 1.5 s, measured on the whole host array |
| back projection | 9.2 s, kernel plus band reduce | 10.5 s, chunked, copies included |
| volume, GPUs to host | 7.1 to 8.0 s | inside each chunk, one 0.9 GiB slab |
| whole | 19.7 s | 12.2 s |

The back projection kernel was not the slow part.  The gather of the volume to the host was the slow
part, and it was rebuilt after this measurement.  On four GPUs that gather fell from 5.7 s to
0.24 s, a rate of 36 to 40 GB/s (`surveys/leap_comparison/findings/host_gather.md`).

### 4. Automatic device policy against a pinned layout

A separate cluster run made three ten-iteration reconstructions in each configuration at N = 1024 on
four GPUs.  The pinned configuration ran first, so any ordering advantage favored the automatic one.
In steady state the automatic policy cost 6.818 s per iteration against 6.595 s pinned, which is 3.4
percent more, so the earlier gap of 109.03 s against 75.81 s was almost entirely one-time cost.

### 5. How to read these numbers

The head-to-head subsection says what one projection and one iteration of each package's own
algorithm cost, and the fixed-quality subsection says how many iterations each package needs to
reach one image quality.  The ORNL subsection is the only measurement on a real scan.

### 6. Follow-up measurements

Five measurements would extend the conclusions above:

- A fixed-quality comparison against the ORNL OGM2 loop on the real scan, which would say what each
  iteration achieves rather than what it costs.
- The multi-slice fusion prototype on the ORNL scan.  Its run record is
  `surveys/leap_comparison/experiments/ornl/msf_ornl.md`, whose results section is
  pending.
- The same adjoint check on a LEAP build without textures, which would settle the attribution.  The
  `AMD` branch is one ("update kernensl for AMD (no texture memory GPU)", `LEAP-inv` line 416).
- A real scan with a known geometry error, reconstructed with both packages' calibration sets.
- A study at more than one noise level.

## The ORNL code: a survey of leapMBIR

The collaborators' repository is `leapMBIR`, read from a sibling checkout.  Its remote is
`https://github.com/aziabari/leapMBIR/`, and the surveyed commit is
`044f38ae2a6bc4509f565ba0384b426998bd43fc` of 2026-09-12, cited below as `leapMBIR/<path>:<line>`.
Commits run from 2025-11-14 to 2026-09-12 under two committer identities for Obaid Rahman.

| File | Lines | Purpose |
| --- | --- | --- |
| `recon_mbir_TipShoe_LEAP.py` | 250 | Reads a Volumax scan from JSON metadata and raw frames, then runs a LEAP FDK reconstruction. |
| `Metrotom_FDK_MBIR.py` | 114 | Zeiss Metrotom pipeline: read, FDK, scatter and beam-hardening correction, FDK again, then MBIR on 20 central detector rows. |
| `Metrotom_MBIR_AutoReg.py` | 96 | The same pipeline with an L-curve search for the TV weight before the full MBIR. |
| `LEAP_(MBIRtorch_vs_LEAP).py` | 148 | LEAP arm of the comparison: FDK plus MBIR iterations on the Inconel scan, with memory instrumentation. |
| `MBIRtorch_(MBIRtorch_vs_LEAP).py` | 144 | mbirtorch arm of the same comparison, with the same instrumentation. |
| `utils/Get_projection_parameters_standalone_fast.py` | 368 | Mines the Metrotom log, reads the raw projection file, and forms normalized projections and weights. |
| `utils/MBIR_LEAP.py` | 131 | First MBIR solver: Nesterov OGM2 with a scalar step size on LEAP's projectors and TV. |
| `utils/MBIR_LEAP_optim.py` | 177 | Rewritten solver: the same algorithm with a voxel-wise preconditioner and no heap allocation in the loop. |
| `utils/LEAP_MBIR_Autoreg.py` | 75 | L-curve search over 15 TV weights on 20 central detector rows. |
| `utils/LEAP_Scatter_BH_correct.py` | 218 | Physics-based scatter and beam-hardening correction using LEAP's `scatter_model` and XrayPhysics spectra. |
| `utils/align_projection_data.py` | 267 | Per-view sinogram alignment on OpenCV ECC, with the shift applied by JAX. |
| `utils/align_projection_data_pyTorch.py` | 425 | The same alignment reimplemented on PyTorch across several GPUs. |
| `utils/readOle_xtrm_distances.py` | 292 | Reads Xradia and Zeiss OLE files for metadata, angles, stage positions, and image stacks. |
| `utils/__init__.py` | 1 | Package marker. |

The pipeline is a single sequence of stages with no feedback between them.  A Zeiss Metrotom scan is
read by text-mining `0_reco_dll.log` for the geometry, the projection count, the volume size, and
the per-view angles and offsets
(`leapMBIR/utils/Get_projection_parameters_standalone_fast.py:34`).  Dark and bright frames are
averaged out of the head of the same raw file, and the weights are the dark-corrected counts.
Scatter and beam-hardening corrections are then applied with LEAP and XrayPhysics
(`leapMBIR/utils/LEAP_Scatter_BH_correct.py:47`).  An alignment routine ported from mbirjax exists
in two versions and is not called by any script here.  LEAP's `FBP` provides the starting image,
their own Nesterov OGM2 loop with LEAP's total-variation regularizer refines it
(`leapMBIR/utils/MBIR_LEAP_optim.py:62`), and a separate L-curve routine picks the TV weight from 15
candidates on a 20-row slab (`leapMBIR/utils/LEAP_MBIR_Autoreg.py:7`).

From LEAP they use these calls: the geometry setters, `FBP`, `project`, `backproject`, `TVcost`,
`TVgradient`, the allocation helpers, and the physics path of `scatter_model`,
`convert_to_modularbeam`, the resampling helpers, and `applyTransferFunction`.  From the XrayPhysics
library they use the spectrum, response, cross-section, and lookup-table calls, which include
`setBHlookupTable` and `setBHClookupTable`.

From mbirtorch they use four calls in one file: `mbirtorch.ConeBeamModel`, `cone_model.set_params`,
`cone_model.recon_fdk`, and `cone_model.recon`.  `set_params` receives the shape and pitches,
`snr_db=30`, `sharpness=1.0`, `positivity_flag` false, and `use_ror_mask` false
(`leapMBIR/MBIRtorch_(MBIRtorch_vs_LEAP).py:71` to `:75`), and the reconstruction call is
`recon(proj_data, weights=weight_data, max_iterations=NUM_ITER, init_recon=recon)` at line 120.  No
mbirtorch reader, no `gen_weights`, and no viewer is used.

They wrote the rest themselves, in four groups:

- ingestion: the Metrotom log mining, the Volumax acquisition parsing, the Xradia OLE reader, and
  the dark and bright field handling;
- preparation: the per-view offset correction, the detector binning, the volume padding rule, the
  cylindrical field-of-view mask, and the segmentation for the scatter model;
- reconstruction: the OGM2 loop, the voxel-wise preconditioner, the allocation-free update, and the
  L-curve regularization search;
- other: the ported alignment routine, its multi-GPU PyTorch rewrite, and the instrumentation.

The two comparison scripts reconstruct the same Inconel scan from the same corrected sinogram and
the same weights, with four GPUs made visible to each, and both start from their own direct
reconstruction.  The committed scripts have no version pins and set 100 iterations, while the
reported table used 15.  Several differences are visible in the code, and each one keeps the two
runs from measuring the same thing:

- The algorithms differ, so an iteration is not the same unit of work.
- The regularizers differ.  The author's comment at `leapMBIR/LEAP_(MBIRtorch_vs_LEAP).py:121`
  records that the TV weight may be stronger than mbirtorch's default prior.
- The stopping rules differ.  Their loop has no stopping test, and mbirtorch may stop early.
- The weight scaling differs.  Their solver divides the weights by their mean, and the mbirtorch arm
  receives the same weights unscaled.
- The detector pitch is assigned to opposite axes in the two arms, which matters if the pitch is not
  square.

Their instrumentation also bounds less than it appears to.  The GPU poll covers every visible GPU
rather than the process, the host figure is the peak resident memory since process start, and the
direct reconstruction falls outside the polling window in both arms.

What this shows about user needs:

- Users want automatic regularization.  They built an L-curve search over 15 TV weights, which costs
  15 slab reconstructions, because LEAP sets no prior strength on its own.
- Users want scanner ingestion.  They text-mine a Zeiss Metrotom log for geometry, which is fragile
  against any change in the log format, because no package gave them that reader.
- Users want physics corrections in the pipeline.  Scatter and beam-hardening correction through
  LEAP and XrayPhysics is the largest single block of outside code they use.
- Users want view alignment.  They ported an mbirjax alignment routine to LEAP and then rewrote it
  again on PyTorch across several GPUs, which is a large effort for one capability.
- Users want low GPU memory.  Their solver keeps every array in host memory and hands it to LEAP one
  call at a time, and the original docstring records that this trades GPU memory for data movement.
- Users want the device choice to be visible and overridable.  Their comment at
  `leapMBIR/MBIRtorch_(MBIRtorch_vs_LEAP).py:33` reads "mbirtorch's own "automatic" device count is
  a memory/workload heuristic and isn't guaranteed to use all of them, so use every visible GPU
  explicitly instead".  They wrote a function to force every visible GPU and left it disabled, so
  the benchmark ran on the automatic choice, which on this scan did choose all four GPUs.
- Users pad around artifacts they cannot diagnose.  Their rule adds 100 voxels in each dimension,
  because in their words "MBIR code needs an extended volume size to remove some artifacts (based on
  experiment)".

## High-value LEAP features missing in mbirtorch

### 1. Offset-scan and truncated-object support

LEAP handles two cases in which the object does not fit the detector
([src/leapctype.py#L5739](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5739)).
An offset scan puts the rotation axis near one detector edge and nearly doubles the field of view,
and a truncated scan is extrapolated off the detector edge for the ramp filter.  Without this
support the reconstruction is wrong rather than merely noisier.  This repository's flash remediation
work addressed the axial half of the same problem on real scans
(`../../archive/flash_remediation/flash_remediation_plan.md`), and mbirtorch warns on lateral
truncation and can enlarge the region with `scale_recon_shape`.

### 2. A low-memory execution mode

LEAP splits a projection into chunks of detector rows and a backprojection into chunks of volume
slices, then halves the chunk size until a chunk fits
([src/tomographic_models.cpp#L779-L787](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L779-L787)).
Its Python interface takes host arrays, so a large scan runs on small GPUs.  On the collaborators'
40 GB A100s mbirtorch held 35.6 and 35.4 GB per GPU, near their capacity.  The mode has a price,
because the ORNL OGM2 loop needs 175 GiB of host memory, so on a node with 126 GB per GPU it needs
two GPUs' worth of host memory to run.  mbirtorch's `recon_split_sino` is documented as only
approximately equal to `recon`, and in its device memory tiers plan Tier 0 is built and verified
except for its documentation increment while Tiers 1 and 2 are not started.

### 3. Scatter correction

LEAP offers a constant transmission shift and a physics-based first-order model that simulates
scatter through an object of a single material with variable density
([src/leapctype.py#L1473](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1473)).
This matters to cone-beam users scanning large or dense objects, where scatter is a dominant source
of cupping and streaks.  mbirtorch has none.  An implementation should start with the
constant-subtraction version, because the physics-based model needs a spectrum and so depends on
item 5.

### 4. Analytic ray-traced phantoms

LEAP builds a phantom from eight primitive shapes and can ray-trace it instead of forward-projecting
a voxelized version
([src/leapctype.py#L7304](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7304)).
That removes the inverse crime from a simulation study.  It matters to any user who validates a
method by simulation, including this repository's own studies, whose prior and fusion findings score
against a phantom forward-projected from its own voxelization.  An implementation adds a ray tracer
beside the phantom generators.

### 5. Polychromatic and dual-energy physics

LEAP models the x-ray spectrum through the companion library XrayPhysics, which gives it
beam-hardening correction for one or two materials, dual-energy decomposition, and conversion of a
low-energy and high-energy pair to electron density and effective atomic number
([src/leapctype.py#L4829](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4829)).
This matters to materials-characterization users and to anyone doing quantitative CT, and the ORNL
scripts use this path for every Metrotom scan they correct.  mbirtorch has an empirical
beam-hardening curve fit and an adaptive plastic-and-metal loop, and neither uses a spectrum.

### 6. Modular geometry with an arbitrary source position and detector pose per view

LEAP's `set_modularbeam` takes one three-vector per view for the source position, the detector
module center, the row direction, and the column direction
([src/leapctype.py#L638](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L638)).
Any scan whose source and detector positions can be listed is then describable without new code.
This ranks sixth because mbirtorch already covers laminography with its multi-axis parallel model,
and because LEAP's own demo warns that the modular projectors are slower and less accurate.  It is
the largest item on this list, because the separable-footprint factorization assumes an axis-aligned
rotation and a general pose needs a Joseph-style ray model.

### 7. The remaining calibration utilities

Four LEAP calibration utilities have no mbirtorch counterpart: `find_tau` for the source offset, the
ball-phantom least-squares fit, `inconsistencyReconstruction`, and the `MTF` measurement
([src/leapctype.py#L892](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L892),
[src/leap_preprocessing_algorithms.py#L1159](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L1159)).
This matters to users of real scanners whose geometry is not exactly known, and `MTF` is how a
resolution claim is defended in a paper.  mbirtorch now has the two estimators, the sweep, the
direction check, the consistency metric, and `apply_calibration`, and its plan of record builds a
reconstruction-scored estimator next.

### 8. Alternative iterative algorithms and a native total-variation prior

LEAP ships twelve iterative reconstruction functions across four families, and a `filterSequence`
that composes several priors inside any of its solvers
([docs/source/iterative_reconstruction.rst](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/iterative_reconstruction.rst)).
Two groups need this: users with emission or very-low-count data who need a likelihood a Gaussian
model does not provide, and users doing method comparisons who need baselines such as SIRT.  The
MACE loop on `mace_4d_dev` accepts any callable agent, so a total-variation or network prior can
enter mbirtorch through the loop today.  A native prior inside VCD needs a new surrogate, because
the VCD update divides by a sum of Hessians and total variation has none near zero.

### 9. A fan-beam geometry as its own model

LEAP has a fan-beam geometry type selectable as its own setting
([src/leapctype.py#L545](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L545)).
This matters to users with 2D fan datasets who want a named geometry rather than a workaround.  In
mbirtorch a fan problem is expressible as a cone-beam model with one detector row, because its
sinograms are always three-dimensional.  A fan-beam class would be small, because the horizontal fan
calculations already exist in an internal module.

### 10. Detector deblur

LEAP deconvolves a user-supplied detector blur kernel by Wiener deconvolution and by
Richardson-Lucy, the second of which preserves non-negativity
([src/leap_preprocessing_algorithms.py#L439](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L439)).
This matters to users whose detectors have significant optical or scintillator spread, such as the
Zeiss Xradia Ultra that mbirtorch already reads.  mbirtorch's nearest operations are defective-pixel
interpolation and zinger correction, and an implementation is one preprocessing function.

### Lower value

Several further LEAP capabilities are absent from mbirtorch and do not justify a subsection above:

- Low-signal and high-energy outlier correction, described for MV CT and neutron CT.
- Short-scan Parker weighting and six additional ramp filter orders.
- The detector-row and slice range calculators that tell a caller which rows a slab needs.
- `space_carving`, and an arbitrary binary volume mask applied inside the operators.
- Cropping that rewrites the geometry parameters, including the center column under an asymmetric
  crop.
- Frequency-domain fusion of two reconstructions to reduce cone-beam artifacts.

### Already covered

Several capabilities that LEAP advertises are already present in mbirtorch, in four groups:
acquisition, correction, algorithms and compute, and calibration.  The tables above give the rows.

## mbirtorch strengths relative to LEAP

### 1. Automatic sharding across GPUs with a memory ledger

Two rules choose the GPU count automatically, from measured per-count speeds and from the memory
each count requires, and a memory ledger prices every layout before the first large allocation.
A bare projector call uses one GPU until a reconstruction entry point runs the device policy and
widens the layout, so a script that only calls `forward_project` or `back_project` gets one GPU on a
four-GPU node.  The collaborators' doubt about the automatic device count is therefore correct for
bare projector calls, and on the ORNL scan the automatic policy did choose all four GPUs.  LEAP also
uses multiple GPUs, so the difference is in how, not whether.  LEAP streams chunks from host memory
through the GPUs, and mbirtorch keeps the split problem on them.

### 2. An adjoint pair exact to float32 rounding, checked automatically

mbirtorch's forward and back projectors are adjoint by construction, and the package measures that
property rather than claiming it.  A test checks that the inner product of a forward projection with
a sinogram equals the inner product of the volume with a back projection, at a relative tolerance of
1e-4.  Each hand-written Triton kernel is checked again against its compiled PyTorch equivalent at
run time.  Performance section 1 gives the readings for both packages.  LEAP also ships a matched
pair, whose kernels read their input through textures with rounded interpolation weights.

### 3. VCD with automatic parameters

mbirtorch's `recon` stops when the relative change between iterations falls below a threshold that
defaults to 0.2 percent, and it sets the noise and prior parameters from `sharpness` and `snr_db`.
The stop rule is qualified by the iteration cap.  `max_iterations` defaults to 15, so the cap often
binds before the rule fires.  The fixed-quality study raised the cap to 100, and on the ORNL scan
the rule did not fire, because the relative change was still 0.84 percent after fifteen iterations.
Performance section 2 gives the iteration counts.  LEAP's regularized weighted least squares is a
comparable objective, and no LEAP algorithm takes a tolerance argument.

### 4. A proximal map, a denoiser, a MACE loop, and the fusion prototype

mbirtorch exposes `prox_map`, which solves the proximal problem for the forward model, and a
standalone qGGMRF denoiser, and both accept and return data already split across GPUs.  On
`mace_4d_dev` the package also holds the MACE loop, whose agents are callables `agent(w, iteration)`
and which moves each agent's input toward the consensus of the agent outputs.  Three agents are
provided: `ForwardProxAgent`, `QGGMRFDenoiserAgent`, and `HyperplaneAgent`.  The multi-slice fusion
prototype runs three orientation network denoisers against the forward proximal map, and its code is
not in the released package.  Its demo-problem NRMSE was 0.0925, against 0.1244 for a
single-orientation prior, 0.1216 for the best postprocessing, and 0.363 for the standard
reconstruction.  On a real NSI scan it reconstructed a 626 by 626 by 467 volume at 102.94
micrometers in 799 s over 30 iterations, at a weighted sinogram residual of 0.0614 against 0.0657
for the standard reconstruction and 0.0656 for three-orientation postprocessing.  The record notes
that part of that difference is the fusion run's ninety extra warm-started proximal iterations, so
it treats the visual comparison as the real evaluation.  LEAP has no proximal map and no learned
prior.

### 5. Geometry calibration from the sinogram

The `geometry_calibration` module estimates the detector channel offset and the detector rotation
from conjugate views, checks the rotation direction, reconstructs one slice per candidate value, and
applies an accepted estimate to the model and the sinogram.  The documented workflow estimates the
offset, then the rotation at that offset, then the offset again, because the two are coupled.  On
four real scans from two scanners the offset agreed with the vendors' recorded values to within a
tenth of a channel.  The four scans cover two objects.  The rotation estimate's zero point depends
on the object, so the documentation tells users to prefer a vendor value.  The Geometric calibration
subsection gives the comparison with LEAP's estimators.

### 6. A geometry viewer

The geometry viewer draws what a model describes, in five panels: a 3D view, a top view, a side
view, the detector face, and a text panel of derived numbers.  A slider steps through the views, and
six toggles turn the source path, the 3D zoom, the angle-0 reference, the two overlays, and the
geometry comparison on and off.  The two overlays are one sinogram view painted on the detector face
and a reconstruction drawn as a silhouette in the volume box.  A comparison window lists every
difference between two geometries, which is how an estimated calibration is checked against a vendor
one.  `TomographyModel.project_points` is the map the panels share, and LEAP's counterpart is
`sketch_system`.

### 7. Vendor scanner readers and one-call model construction

mbirtorch reads four scan formats and returns a configured model from a single call to
`get_sino_and_model`.  The readers cover NorthStar Instruments, Zeiss Versa and Ultra, Zeiss
translation tomography, and ORNL HDF5.  They set the arbitrary length unit used internally and
record its size, so a reconstruction can be reported in physical units.  LEAP has no vendor reader
and expects the user to set the geometry from the scanner's metadata.  Neither package reads the
Zeiss Metrotom log format, which is why the ORNL scripts mine it themselves.

### 8. Installation and platform reach

mbirtorch installs from PyPI with no compiler and no CUDA toolkit required, and it selects its
backend automatically.  LEAP has no PyPI package, and its documented install compiles from a source
checkout.  A community conda-forge package `leapct` 1.26 exists for Linux and Windows only.

### 9. Automated testing, continuous integration, and documentation

The mbirtorch count of 988 test functions is of test function definitions, not of executed cases,
because many tests are parameterized.  Continuous integration runs the suite on pull requests across
four Python versions and builds the documentation with warnings treated as errors.  LEAP has more
teaching material, in 38 demo scripts with explanatory docstrings.

### 10. Capabilities LEAP does not have at all

mbirtorch models translation tomography, in which each view is a cone-beam projection of a
translated object, and multi-axis parallel beam, in which each view has both an azimuth and an
elevation angle.  It chooses which views to acquire with `get_opt_views`, which scores candidate
view sets against a reference object and runs a greedy search, while LEAP addresses the few-view
problem after acquisition.  It denoises hyperspectral neutron data with a non-negative matrix
factorization of the spectral axis.  LEAP covers laminography through its modular geometry, so
laminography is not unique to mbirtorch.

### 11. 4D reconstruction of a moving object

`MACE4DModel` on `mace_4d_dev` reconstructs one continuous scan of a moving object.  The scan is
divided into overlapping angular windows, one per time frame, and one volume is reconstructed per
frame.  The volumes are the consensus of four agents: the proximal map of each frame's data-fit term
and qGGMRF denoisers on the three sets of hyperplane volumes that hold the frame axis and two
spatial axes.  A frame-axis filter removes the modulation the overlapping windows imprint.  The
branch adds `mbirtorch/mace.py` at 873 lines, `mbirtorch/mace4d.py` at 701 lines, and 27 test
functions, over seven commits all dated 2026-09-14.  It has no documentation page, no demo, and no
H100 measurement, which its plan lists as stages 6, 7, and 8.

## Project health

| Item | LEAP | mbirtorch | Source |
| --- | --- | --- | --- |
| License | MIT | BSD 3-Clause | [LICENSE](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LICENSE); [LICENSE](https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/LICENSE) |
| Latest release | v1.26, 2024-12-14 | v0.0.2, 2026-08-21 | GitHub API and PyPI, read 2026-09-14 |
| Commits in the last 12 months | 0 on `main` | 212 on `greg_dev` | `LEAP-inv` section 15; counted at `efeca90` |
| Stars | 249 | 0 | GitHub API, read 2026-09-14 |
| Forks | 37 | 2 | GitHub API, read 2026-09-14 |
| Contributors | 3, with 430, 7, and 1 commits | 6, with 101 and 87 for the top two | `LEAP-inv` section 15; `MT-inv` section 1 |
| Open issues | 48 | 0 | GitHub API, read 2026-09-14 |
| Packaging channel | conda-forge, community maintained | PyPI | `LEAP-inv` section 20; `MT-inv` section 1 |
| Platforms | Linux and Windows with CUDA | CUDA, CPU, and Apple MPS | `LEAP-inv` sections 10.3 and 20.3; `MT-inv` section 6 |
| Last push | 2026-07-25 | 2026-09-14 | GitHub API, read 2026-09-14 |

LEAP is an established package whose released version has not changed since 2024-12-14.  Development
has continued on the unreleased `version_two` branch, whose last commit is dated 2026-07-25, so the
project is not abandoned.  A user installing LEAP today gets code from December 2024, and LEAP's
macOS support is a CPU-only source build with no released binary.

mbirtorch is about six weeks old as a public package and has no external issue history.  Its
repository was created on 2026-08-04, and its 212 commits and two releases all fall inside that
period.  The maintenance comparison therefore compares an established project with a new one.
Neither state is evidence about long-term support.

## Sources

The following sources support every claim above:

1. `surveys/leap_comparison/leap_comparison.md`, version 1 of this comparison; deleted 2026-09-18 (`git show cd4aac3:plans/features/leap_comparison/leap_comparison.md`)
2. `surveys/leap_comparison/leap_inventory.md`
3. `surveys/leap_comparison/mbirtorch_inventory.md`, at `26bd0ea`; deleted 2026-09-18 (`git show cd4aac3:plans/features/leap_comparison/leap_comparison_sources/mbirtorch_inventory.md`)
4. `https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/` , the LEAP base URL
5. `https://github.com/cabouman/mbirtorch/blob/efeca90ff0aa2aaf5326617ab899ad794af9db67/` , the mbirtorch base URL
6. `https://github.com/cabouman/mbirtorch/blob/8304b25ec4f1877a236946b809312779fae38917/` , the 4D branch base URL
7. [src/projectors_SF.cu](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors_SF.cu),
   [src/cuda_utils.cu](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/cuda_utils.cu),
   and [documentation/LEAP.tex](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex)
8. [unitTests/unit_tests.py#L40](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/unitTests/unit_tests.py#L40), LEAP's disabled geometry loop
9. `https://docs.nvidia.com/cuda/archive/10.0/pdf/CUDA_C_Programming_Guide.pdf` , PG-02829-001_v10.0 of October 2018, Appendix G section G.2, page 243
10. `surveys/leap_comparison/experiments/results/leap_benchmark_results.md`
11. `surveys/leap_comparison/experiments/results/smoke_results.jsonl`, the N = 64 readings
12. `surveys/leap_comparison/experiments/bench_leap_vs_mbirtorch.py`, with its batch files
13. `surveys/leap_comparison/findings/quality_results.md`
14. `surveys/leap_comparison/experiments/quality_leap_vs_mbirtorch.py`, with its batch files
15. `surveys/leap_comparison/experiments/results/`, the JSON results and the figures
16. `surveys/leap_comparison/findings/ornl_reproduction.md`
17. `surveys/leap_comparison/experiments/ornl/`, the reproduction harness and its records
18. `surveys/leap_comparison/findings/host_gather.md`
19. `plans/device_memory_tiers/experiments/dm1_record.md`
20. `plans/device_memory_tiers/plan.md`
21. `plans/geometric_calibration/experiments/closed/calibration_512_gautschi.md`
22. `plans/geometric_calibration/experiments/closed/real_scan_leap_tilt.md`
23. `plans/geometric_calibration/executive_summary_2026-09-05.md`; deleted 2026-09-18 (`git show cd4aac3:plans/features/geometric_calibration/executive_summary_2026-09-05.md`)
24. `plans/geometric_calibration/findings/increment_1_1_findings.md`
25. `plans/geometric_calibration/plan.md`
26. `plans/multi_slice_fusion/plan.md` and `plans/multi_slice_fusion/findings/multi_slice_fusion_findings.md`
27. `../../archive/flash_remediation/flash_remediation_plan.md`, the axial truncation work
28. `plans/mace4d/plan.md`
29. Section "The ORNL code" of this document, the survey of `leapMBIR`, whose remote is
    `https://github.com/aziabari/leapMBIR/` at commit `044f38ae2a6bc4509f565ba0384b426998bd43fc`
30. `https://api.github.com/repos/LLNL/LEAP` and `https://api.github.com/repos/cabouman/mbirtorch` , read 2026-09-14
31. `https://api.github.com/repos/LLNL/LEAP/releases` and `https://api.github.com/repos/LLNL/LEAP/branches/version_two` , read 2026-09-14
32. `https://pypi.org/pypi/mbirtorch/json`
33. `https://arxiv.org/abs/2307.05801` (Kim and Champley, "Differentiable Forward Projector for
    X-ray Computed Tomography", ICML workshop, 2023)
34. `https://arxiv.org/abs/2410.07552` (Champley and coauthors, "Methods for Few-View CT Image
    Reconstruction", 2024)
35. `https://leapct.readthedocs.io/` and `https://mbirtorch.readthedocs.io/`
36. `https://github.com/kylechampley/XrayPhysics` and `https://github.com/kylechampley/LEAPCT-UI-GUI`
