# LEAP vs mbirtorch: features and performance

Date: 2026-09-02. Versions compared: LEAP v1.26 at commit `0c8846f4`, and mbirtorch 0.0.2 at commit `26bd0ea`.

## Summary

LEAP and mbirtorch solve overlapping problems with different designs. LEAP is a CUDA library of CT algorithms. It covers many geometries, many direct and iterative reconstruction methods, and many correction and calibration utilities. mbirtorch is a model-based reconstruction package built on PyTorch. It has one main reconstruction algorithm, four geometries, automatic multi-GPU execution, and vendor scanner readers. LEAP covers more capabilities. mbirtorch covers fewer capabilities in more detail.

Both packages agree on the most important design decision. Both use a separable-footprint projector model. Both supply a forward projector and a back projector that are adjoint to each other. That adjoint property is what a long iterative loop requires for convergence.

Three terms recur below and are defined here. Sharding means splitting one reconstruction across several GPUs and running the pieces together. A memory model is code that computes the memory cost of a candidate layout before any large allocation, so a run that cannot fit fails immediately. Normalized RMSE (NRMSE) is the root-mean-square difference between two arrays divided by the root-mean-square value of the reference array.

The LEAP features that mbirtorch lacks are these, ranked by value to mbirtorch users:

1. Geometric calibration utilities that solve for center of rotation, detector tilt, and source offset.
2. Offset-scan and truncated-object support in both direct and iterative reconstruction.
3. Analytic ray-traced phantoms, which remove the inverse crime from simulation studies.
4. Scatter correction, from a constant transmission offset up to a physics-based first-order model.
5. Automatic splitting of a projection into pieces that each fit in one GPU's memory.
6. A modular geometry with an arbitrary source position and detector pose per view.
7. Alternative iterative algorithms, and a composable interface that accepts any prior.
8. Polychromatic and dual-energy physics, including spectra and two-material beam-hardening correction.
9. A fan-beam geometry as its own model.
10. Detector deblur by Wiener and Richardson-Lucy deconvolution.

mbirtorch's advantages over LEAP are these:

1. Automatic sharding of one reconstruction across several GPUs, guided by a memory model and by measured per-GPU speeds.
2. An exact adjoint pair, checked by an automated test suite and again by value checks at run time.
3. Multi-Granular Vectorized Coordinate Descent (VCD) with a qGGMRF prior, a relative-change stop rule, and automatic parameter selection.
4. A proximal map and a denoiser that both accept data already split across GPUs, which is what a plug-and-play prior loop needs.
5. Vendor scanner readers and one-call model construction.
6. Installation from PyPI with no compiler, and support for CUDA, CPU, and Apple MPS.
7. A test suite of 769 test functions plus continuous integration, against LEAP's three test scripts and no continuous integration.
8. Two geometries LEAP does not have, translation tomography and multi-axis parallel beam.
9. Automated view selection, which chooses which views to acquire.
10. Hyperspectral neutron tomography support.

mbirtorch reached a common NRMSE target in 8 to 14 times fewer iterations than LEAP, at all three sizes tested on noisy data. Once compilation is excluded, mbirtorch's warm time to that target is 7.1 times faster than LEAP's at N = 512 and 7.9 times faster at N = 1024. LEAP's single-GPU projectors were faster than mbirtorch's at every size in the cost-per-iteration benchmark. At N = 1024, LEAP's forward projection took 6.314 s against mbirtorch's 8.636 s. On four GPUs at N = 1024 the two forward projections were within one percent of each other, at 2.189 s for mbirtorch and 2.214 s for LEAP. Ten iterations took 75.81 s for mbirtorch on four GPUs, against LEAP's own best of 192.7 s on one GPU. The quality result is one phantom at one noise level, so it does not by itself show which package reconstructs better in general. Every number in this paragraph comes from `plans/experiments/features/leap_comparison/results/leap_benchmark_results.md` and `plans/experiments/features/leap_comparison/results/quality_results.md`.

---

## Scope and versions

This document compares the features and the recorded performance of the two packages. Its main purpose is to name the LEAP features that mbirtorch lacks and that would be worth building. Its secondary purpose is to name mbirtorch's advantages.

The evidence has three parts: two sourced inventories prepared in the same session, the LEAP source at the pinned commit, and the plan records in this repository. Every number below is copied from a named file or URL. Where a number does not exist in any source that was read, the text says so.

The LEAP version is v1.26, at commit `0c8846f42b2e59340d5559fc1271d590a292f9a0`, dated 2024-12-14. That commit is the released version, the tip of `main`, and what conda-forge packages.

An unreleased and unmerged `version_two` branch declares version 2.0 and last changed on 2026-07-25. Nothing here describes that branch, because it was not read.

The mbirtorch version is 0.0.2, at commit `26bd0ea`, dated 2026-08-27, on branch `greg_dev`.

---

## Feature comparison

References in this section are written against two pinned commits and two inventory files:

- LEAP file references link under `https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/`.
- mbirtorch file references link under `https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/`.
- `LEAP-inv` = `plans/features/leap_comparison_sources/leap_inventory.md`.
- `MT-inv` = `plans/features/leap_comparison_sources/mbirtorch_inventory.md`.

### Geometries

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Parallel and cone beam, helical, curved detector | yes | yes | [src/parameters.h#L620](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/parameters.h#L620); [mbirtorch/cone_beam.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/cone_beam.py) |
| Fan beam | yes, own type | no class | [src/leapctype.py#L545](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L545); `MT-inv` section 2 |
| Cone-parallel | yes | no | [src/leapctype.py#L431](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L431) |
| Modular, per-view source and detector pose | yes | no | [src/leapctype.py#L638](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L638) |
| Laminography | via modular beam | via multi-axis parallel | [demo_leapctype/d32_laminography.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d32_laminography.py); [mbirtorch/multiaxis_parallel.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/multiaxis_parallel.py) |
| Multi-axis parallel, per-view elevation | no | yes | [mbirtorch/multiaxis_parallel.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/multiaxis_parallel.py) |
| Translation tomography | no | yes, alpha | [mbirtorch/translation_model.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/translation_model.py) |
| Detector tilt as a model parameter | yes | no, corrected in preprocessing | [src/leapctype.py#L750](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L750); [mbirtorch/preprocess/utilities.py:265](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/utilities.py#L265) |
| Symmetric objects, attenuated Radon transform | yes | no | [src/leapctype.py#L5767](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5767), [#L5876](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5876) |

LEAP covers more scan geometries. In LEAP each geometry is selected by a runtime parameter rather than by a separate class. mbirtorch covers fewer geometries, and two of them have no LEAP counterpart. Translation tomography and multi-axis parallel beam with a per-view elevation angle are mbirtorch-only capabilities.

### Projector models and adjoints

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Model family | separable footprint | separable footprint | [documentation/LEAP.tex](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex); [mbirtorch/horizontal_fan.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/horizontal_fan.py) |
| Matched forward and back pair | yes, by default | yes, by construction | [LEAP_features.md](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LEAP_features.md); [docs/source/usr_autograd.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_autograd.rst) |
| Unmatched fast backprojector option | yes, `set_projector('VD')` | no | [src/leapctype.py#L5788](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5788) |
| Alternative model for hard cases | extended SF, Joseph for modular | one model per geometry | [src/projectors.cpp#L74-L80](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors.cpp#L74-L80) |
| Adjointness checked by an automated test | no | yes, at 1e-4 relative | `LEAP-inv` section 13.3; [tests/test_adjoint.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/tests/test_adjoint.py) |
| Implementation language | CUDA C plus OpenMP C++ | PyTorch plus Triton | `LEAP-inv` section 10.1; [docs/source/dev_projector_kernels.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/dev_projector_kernels.rst) |
| Precision | float32 only | float32 only | `LEAP-inv` section 4.5; `MT-inv` section 3 |

The two packages implement the same class of projector. LEAP adds a voxel-driven backprojector that is faster and is not the transpose of its forward projector. LEAP's matched property therefore holds only under the default setting. mbirtorch offers no unmatched option. Its test suite fails when the adjoint property does not hold.

### Direct reconstruction

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| FBP and FDK | yes, all geometries | yes, all four geometries | [src/leapctype.py#L2595](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2595); [mbirtorch/cone_beam.py:798](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/cone_beam.py#L798) |
| Ramp filter choices | 7 orders, 0 through 12 | 1, "ramp" only | [src/leapctype.py#L5831](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5831); [mbirtorch/tomography_utils.py:18](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_utils.py#L18) |
| Low-pass on the ramp filter | yes | no | [src/leapctype.py#L5854](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5854) |
| Short-scan Parker weighting | yes, applied automatically | no | [src/ray_weighting_cpu.cpp#L66](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/ray_weighting_cpu.cpp#L66); [mbirtorch/cone_beam.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/cone_beam.py) |
| Non-equispaced view weighting | yes | no | [src/ray_weighting_cpu.cpp#L343](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/ray_weighting_cpu.cpp#L343) |
| Offset scan, truncated scan | yes | no extrapolating ramp filter | [src/leapctype.py#L5739](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5739), [#L5721](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5721); mbirtorch warns on lateral truncation and can enlarge the reconstruction region, [mbirtorch/tomography_model.py:2117](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L2117) |
| Single-slice reconstruction along x, y, or z | yes | no | [src/leapctype.py#L2504](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2504) |
| Separate filter and backproject steps | yes | yes | [src/leapctype.py#L2159](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2159); [mbirtorch/cone_beam.py:661](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/cone_beam.py#L661) |

LEAP's direct reconstruction is far more complete than mbirtorch's. mbirtorch treats its direct reconstruction mainly as an initializer for VCD. The mbirtorch docstrings say that a standalone direct reconstruction is only approximate for nonuniform, limited-angle, or short scans.

### Iterative reconstruction algorithms

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Coordinate descent on a MAP objective | no | yes, VCD | [mbirtorch/tomography_model.py:3130](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L3130) |
| Algebraic methods, SIRT and SART | yes | no | [src/leapctype.py#L3773](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3773), [#L3798](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3798) |
| Emission methods, MLEM and OSEM | yes | no | [src/leapctype.py#L3606](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3606), [#L3684](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3684) |
| Transmission statistical methods, RWLS and MLTR | yes | no | [src/leapctype.py#L4175](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4175), [#L4561](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4561) |
| Few-view methods, ASD-POCS and RDLS | yes | no | [src/leapctype.py#L3904](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3904), [#L4399](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4399) |
| View subsets to accelerate early iterations | yes, `numSubsets` | no | [src/leapctype.py#L3542](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3542); [mbirtorch/vcd_utils.py:97](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/vcd_utils.py#L97) |
| Preconditioners | yes, three named choices | one fixed cone damping profile | [demo_leapctype/d26_preconditioners.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d26_preconditioners.py); [mbirtorch/cone_beam.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/cone_beam.py) |
| Stopping rule other than an iteration count | no | yes, relative change | `LEAP-inv` section 6.4; [mbirtorch/tomography_model.py:3130](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L3130) |
| Region-of-interest mask | arbitrary binary 3D mask, applied inside the operators | 2D region mask applied to every slice | [src/leapctype.py#L3378](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3378); [mbirtorch/vcd_utils.py:18](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/vcd_utils.py#L18) |
| Proximal map for plug and play | no | yes, `prox_map` | [mbirtorch/tomography_model.py:3284](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L3284) |

LEAP offers twelve iterative algorithms and mbirtorch offers one. LEAP's algorithms stop only when the iteration count runs out, because none of them takes a tolerance argument. mbirtorch stops on a measured relative change, and it supplies a proximal-map entry point that LEAP does not have.

The two packages also mean different things by a subset. LEAP's ordered subsets are subsets of projection views, used by OS-EM and OS-SART to accelerate early iterations. mbirtorch's multi-granular partitions are subsets of reconstruction voxels, and every subset update uses all views ([mbirtorch/vcd_utils.py:97](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/vcd_utils.py#L97)). View subsets are therefore not part of the VCD design, and mbirtorch has no counterpart to them.

### Regularizers and priors

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| qGGMRF prior | no | yes | [docs/source/theory.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/theory.rst) |
| Anisotropic total variation | yes | no | [src/leap_filter_sequence.py#L193](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py#L193) |
| Composable sequence of several priors | yes, `filterSequence` | no | [docs/source/filter_sequence.rst](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/filter_sequence.rst) |
| Histogram sparsity, azimuthal sparsity | yes | no | [src/leap_filter_sequence.py#L431](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py#L431), [#L511](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py#L511) |
| Prior-image regularization | yes, an `f_0` argument | no | [src/leap_filter_sequence.py#L193](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py#L193) |
| Standalone denoisers | many, including bilateral and guided | qGGMRF denoiser, 3D median filter | [docs/source/filters.rst](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/filters.rst); [mbirtorch/denoising.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/denoising.py) |
| Automatic regularization strength | no | yes, from `sharpness` and `snr_db` | [mbirtorch/tomography_model.py:2086-2186](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L2086-L2186) |

LEAP gives the user several regularizers to combine, and it expects the user to choose the weights. mbirtorch gives one prior. It sets the prior's parameters automatically from two user parameters, `sharpness` and `snr_db`.

Neural-network priors exist in mbirtorch only as research scripts under `experiments/drunet/`, and they are not part of the package.

### Preprocessing and artifact correction

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Flat and dark correction, negative log | yes | yes | [src/leap_preprocessing_algorithms.py#L176](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L176); [mbirtorch/preprocess/utilities.py:51](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/utilities.py#L51) |
| Bad pixel and outlier correction | yes | yes | [src/leap_preprocessing_algorithms.py#L287](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L287); [mbirtorch/preprocess/utilities.py:174](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/utilities.py#L174) |
| Low-signal and high-energy outlier correction | yes | no | [src/leap_preprocessing_algorithms.py#L373](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L373), [#L406](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L406) |
| Ring and stripe removal | three variants | three routines | [src/leap_preprocessing_algorithms.py#L506](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L506); [mbirtorch/preprocess/stripe.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/stripe.py) |
| Detector deblur | yes, Wiener and Richardson-Lucy | no | [src/leap_preprocessing_algorithms.py#L439](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L439), [#L466](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L466) |
| Scatter correction | yes, physics-based first order | no | [src/leapctype.py#L1473](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1473); `MT-inv` section 4 |
| Metal artifact reduction | sinogram replacement | weights plus a beam-hardening loop | [src/leapctype.py#L1343](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1343); [mbirtorch/tomography_model.py:420](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L420) |
| Beam hardening correction | yes, spectrum-based | yes, empirical curve fit | [docs/source/physics_based_preprocessing_algorithms.rst](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/physics_based_preprocessing_algorithms.rst); [mbirtorch/preprocess/utilities.py:1250](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/utilities.py#L1250) |
| Vendor scanner readers | none found | four formats | `LEAP-inv` section 11; [mbirtorch/preprocess/](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/) |

The two preprocessing sets have most operations in common. They differ in physics modeling and in data ingestion. LEAP is stronger on physics, because it has scatter correction and detector deblur. mbirtorch is stronger on data ingestion, because it reads four scanner formats and returns a configured model in one call.

### Geometric calibration

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Center of rotation from the data | yes, `find_centerCol` | view-offset estimator only | [src/leapctype.py#L846](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L846); [mbirtorch/preprocess/utilities.py:1072](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/utilities.py#L1072) |
| Source offset `tau` | yes, `find_tau` | no | [src/leapctype.py#L892](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L892) |
| Detector tilt | yes, `estimate_tilt` | no | [src/leapctype.py#L937](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L937) |
| Data-consistency metric for calibration | yes | no | [src/leapctype.py#L1021](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1021) |
| Reconstruction that is pure noise when the geometry is right | yes, `inconsistencyReconstruction` | no | [src/leapctype.py#L2845](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2845) |
| Joint search over two parameters | yes | no | [src/leap_preprocessing_algorithms.py#L737](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L737) |
| Parameter sweep, one slice per candidate | yes, eight parameters | no | [src/leap_preprocessing_algorithms.py#L872](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L872) |
| Calibration from a ball phantom | yes | no | [src/leap_preprocessing_algorithms.py#L1159](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L1159) |
| Resolution measurement, MTF | yes | no | [src/leap_preprocessing_algorithms.py#L1087](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L1087) |
| View alignment across views | no | yes, `align_sino_views` on top of the ECC estimator | [mbirtorch/preprocess/utilities.py:1205](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/utilities.py#L1205) |

This is the widest capability gap in the comparison. The table above lists nine LEAP calibration and measurement utilities and two mbirtorch ones. The mbirtorch inventory records the gap: mbirtorch has no source position solver, no tilt solver, and no magnification solver.

### Simulation and physics

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Voxelized phantoms | yes | yes, several generators | [src/leapctype.py#L7286](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7286); [mbirtorch/utilities.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/utilities.py) |
| Analytic ray-traced phantoms | yes, eight primitive shapes | no | [src/leapctype.py#L7304](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7304), [#L7261](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7261) |
| FORBILD head phantom | yes | no | [src/leapctype.py#L7357](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7357) |
| Ray oversampling for partial volume | yes | no | [demo_leapctype/d08_ray_tracing_simulation.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d08_ray_tracing_simulation.py) |
| Built-in noise model | no | no | `LEAP-inv` section 8; `MT-inv` section 7 |
| Source spectra and detector response | yes, via XrayPhysics | no | [docs/source/physics_based_preprocessing_algorithms.rst](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/physics_based_preprocessing_algorithms.rst) |
| Dual-energy decomposition | yes | no | [src/leapctype.py#L4829](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4829) |
| Hyperspectral neutron data support | no | yes | [docs/source/usr_hsnt.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_hsnt.rst) |

LEAP can simulate a scan without committing the inverse crime, because it can ray-trace an analytic phantom instead of forward-projecting a voxelized one. An inverse crime is the use of the same discretized forward model to make the data and to reconstruct it, which flatters the result. mbirtorch simulates only by forward-projecting a voxelized phantom. Neither package has a built-in noise model, and both leave noise to the calling script.

### Deep-learning integration and autograd

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Differentiable forward and back projection | yes | yes | [src/leaptorch.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py); [mbirtorch/autograd.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/autograd.py) |
| Gradient is the exact adjoint | yes | yes | [src/leaptorch.py#L31-L36](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py#L31-L36); [docs/source/usr_autograd.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_autograd.rst) |
| `nn.Module` wrapper | yes, `Projector` | yes, `TorchProjector` | [src/leaptorch.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py); [mbirtorch/autograd.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/autograd.py) |
| Differentiable FBP | yes | no | [src/leaptorch.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py) |
| Batch dimension | yes, a Python loop per element | no | [src/leaptorch.py#L41-L45](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py#L41-L45) |
| Runs on GPU-resident tensors | yes, one GPU at a time | yes, one GPU | [demo_leapctype/d02_standard_geometries_torch.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d02_standard_geometries_torch.py); [docs/source/usr_autograd.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_autograd.rst) |

Both packages expose a differentiable projector whose backward pass is the adjoint operator rather than a traced graph. LEAP adds a batch dimension and a differentiable FBP that mbirtorch does not have. A comment in LEAP's own source marks its FBP backward as needing replacement, so that path carries a caveat from its author ([src/leaptorch.py#L145](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leaptorch.py#L145)).

### Compute, GPUs, and memory

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| CUDA | yes | yes | [src/CMakeLists.txt](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/CMakeLists.txt); [mbirtorch/tomography_model.py:91](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L91) |
| CPU | yes, OpenMP, with gaps | yes, every geometry | `LEAP-inv` section 10.2; `MT-inv` section 6 |
| Apple MPS | no | yes | `LEAP-inv` section 10.3; [mbirtorch/tomography_model.py:91](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L91) |
| AMD GPUs | listed as future work | untested | [README.md](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md); `MT-inv` section 12.2 |
| Multi-GPU | yes, streamed chunks | yes, sharded volume and sinogram | [src/tomographic_models.cpp#L803-L804](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L803-L804); [docs/source/dev_sharding_overview.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/dev_sharding_overview.rst) |
| Multi-GPU with GPU-resident input | no | yes | [src/leapctype.py#L39](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L39); [docs/source/usr_multi_gpu.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_multi_gpu.rst) |
| Automatic chunking below GPU memory | yes, a halving loop | no, a manual band split instead | [src/tomographic_models.cpp#L779-L787](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L779-L787); [mbirtorch/tomography_model.py:372](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L372) |
| Detector-row and slice range calculators | yes | no | [src/leapctype.py#L2962](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2962), [#L3046](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3046) |
| Memory cost computed before allocation | yes, an error message | yes, a memory model | [src/projectors.cpp#L66-L70](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/projectors.cpp#L66-L70); [mbirtorch/_memory_ledger.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/_memory_ledger.py) |
| Reduced precision, multi-node | no | no | `LEAP-inv` sections 4.5 and 10.5; [docs/source/usr_multi_gpu.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_multi_gpu.rst) |

The two memory strategies differ in kind. LEAP repeatedly halves the size of each piece of work until a piece fits in GPU memory. A large problem then runs slowly on one GPU rather than failing. mbirtorch computes the memory cost of the whole layout before it allocates. It refuses the run when no layout fits, so the user gets an immediate message instead of a failure part way through the run. The two multi-GPU designs also differ. mbirtorch holds one persistent split of the problem across the GPUs. LEAP instead streams detector-row or slice chunks of the same problem from host memory through the GPUs.

### Data I/O and visualization

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| File formats written | tif sequence, nrrd, npy | HDF5 | [src/leapctype.py#L6709](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6709); [mbirtorch/viewer.py:556](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/viewer.py#L556) |
| File formats read | tif sequence, nrrd, npy | npy, npz, HDF5 | [src/leapctype.py#L7092](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7092); [mbirtorch/viewer.py:66](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/viewer.py#L66) |
| TIFF writing | yes | no | `LEAP-inv` section 11; `MT-inv` section 7 |
| DICOM | no | no | `LEAP-inv` section 11; `MT-inv` section 7 |
| Geometry parameters to a file | yes, a text file | no dedicated pair | [src/leapctype.py#L6695](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6695); `MT-inv` section 7 |
| Volume viewer | napari | a matplotlib slice viewer | [src/leapctype.py#L6255](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6255); [mbirtorch/viewer.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/viewer.py) |
| Geometry sketch | yes | no | [src/leapctype.py#L6279](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L6279) |
| Graphical user interface | a separate repository | none | [README.md](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md) |
| Bridges to other toolkits | TIGRE and LTT | none | [utils/](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/utils/) |

LEAP writes more interchange formats, and it can draw the scan geometry. mbirtorch ships a slice viewer. It shows side-by-side volumes, difference images, and region statistics. Neither package reads DICOM.

### API, documentation, tests, and packaging

| Capability | LEAP | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Shortest working example | about ten lines | two lines | [demo_leapctype/d01_standard_geometries.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d01_standard_geometries.py); [README.md](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/README.md) |
| Who owns the arrays | the caller allocates and passes | the package returns numpy arrays | [documentation/LEAP.tex](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex); [mbirtorch/__init__.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/__init__.py) |
| Documentation pages | 18, plus a PDF manual | 30 | `LEAP-inv` section 13.3; `MT-inv` section 8 |
| Documentation version label | v1.4 | current | `LEAP-inv` section 21 |
| Demo scripts | 38, plus 5 for the torch interface | 9 | `LEAP-inv` section 13.3; [demo/](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/demo/) |
| Tests | 3 scripts | 769 test functions in 37 files | `LEAP-inv` section 13.3; `MT-inv` section 8 |
| Continuous integration | none | GitHub Actions, four Python versions | `LEAP-inv` section 13.3; [.github/workflows/ci.yml](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/.github/workflows/ci.yml) |
| Install without a compiler | no, source build or conda | yes, from PyPI | `LEAP-inv` sections 17 and 20; `MT-inv` section 1 |

LEAP has more demonstration material and mbirtorch has more automated checking. LEAP's 38 demo scripts each carry an explanatory docstring. These docstrings make the demos usable as teaching material. LEAP has no continuous integration. In its main unit-test script as checked into the repository, the list of geometries to test is empty ([unitTests/unit_tests.py#L40](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/unitTests/unit_tests.py#L40)). No automated test therefore covers LEAP's released code. LEAP's documentation site is labeled v1.4 while the code is 1.26, and its technical manual is labeled version 1.1 (`LEAP-inv` section 21).

---

## Performance

### LEAP's published numbers

LEAP has published exactly four timed configurations. They appear in Table 1 of Kim and Champley, "Differentiable Forward Projector for X-ray Computed Tomography", arXiv:2307.05801. They are forward projection only, on one NVIDIA Tesla P100 with 16 GB. The comparison partner is LTT, which is LLNL's own closed-source package. The paper reports two times in each table entry, one before and one after the transfer between host and device.

| Geometry and size | LEAP time | LEAP memory | LTT time |
| --- | --- | --- | --- |
| Parallel, 512^3 image, 180 projections | 0.5 then 1.8 s | 1.5 GB | 4.2 s |
| Parallel, 1024^3 image, 720 projections | 11.5 then 15.4 s | 8 GB | 17.4 s |
| Cone, 512^3 image, 180 projections | 1.4 then 2.8 s | 1.5 GB | 4.5 s |
| Cone, 1024^3 image, 720 projections | 37.1 then 39.2 s | 11.1 GB | 38.9 s |

The source for the whole table is arXiv:2307.05801, Table 1, as recorded in `LEAP-inv` section 23. The angular range is 180 degrees for parallel beam and 360 degrees for cone beam. No CPU model is stated, and no detector size is stated beyond the image dimension.

Every other LEAP performance claim is unquantified. The README states that most algorithms run as fast or faster than other popular CT reconstruction packages, and it names no benchmark ([README.md](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md)). The feature list repeats that claim for the multi-GPU and multi-core CPU implementations ([LEAP_features.md](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LEAP_features.md)). The repository states that voxel-driven backprojection of cone-beam data is about twice as fast as separable-footprint backprojection ([results/SF_vs_VD.md](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/results/SF_vs_VD.md)). That statement names no hardware and no problem size. The repository ships a benchmark script ([demo_leapctype/d99_speedTest.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d99_speedTest.py)). It records no output from that script. There are no published backprojection times, no published FBP times, no published iterative reconstruction times, and no published multi-GPU scaling numbers.

The README also claims that a LEAP walnut reconstruction has 1.7 times higher signal-to-noise ratio than an ASTRA one ([README.md](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/README.md)). No reconstruction parameters, ASTRA version, or definition of the ratio is published for that figure. The supporting document in the same repository describes a different experiment. In that experiment the separable-footprint result scores 43.6 and the voxel-driven result scores 25.7 at over-sized voxels, with noise added to the projection data ([results/SF_vs_VD.md](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/results/SF_vs_VD.md)). The scores are signal-to-noise ratios. That document also says the voxel-driven result has higher resolution. The comparison is therefore a trade between resolution and noise, and not an improvement in both.

### mbirtorch's recorded numbers

mbirtorch's latest regression run is on NVIDIA H100 80GB HBM3 GPUs. It is recorded in `mbirtorch_metrics`, a sibling repository of this one. The source file is `mbirtorch_metrics/results/gpu/prerelease/regression_gpu_20260827T175529Z_26bd0ea9_table.yaml`, measured 2026-08-30 at commit `26bd0ea9` with mbirtorch 0.0.2. Sizes are sinogram shapes written as views by detector rows by detector channels.

| Geometry | Operation | Size | 1 GPU | 2 GPUs | 4 GPUs |
| --- | --- | --- | --- | --- | --- |
| parallel | forward | 1024x1008x992 | 2,529.7 ms | 1,236.2 ms | 630.7 ms |
| parallel | back | 1024x1008x992 | 778.3 ms | 414.5 ms | 245.2 ms |
| parallel | VCD, 3 iterations | 1024x1008x992 | 18,976.3 ms | 10,272.6 ms | 5,879.2 ms |
| cone | forward | 1024x1008x992 | 8,000.0 ms | 4,029.2 ms | 2,085.5 ms |
| cone | back | 1024x1008x992 | 4,546.5 ms | 2,353.6 ms | 1,256.1 ms |
| cone | VCD, 3 iterations | 1024x1008x992 | 59,178.1 ms | 31,038.9 ms | 17,344.2 ms |
| parallel | forward | 512x448x384 | 87.6 ms | 47.8 ms | 27.9 ms |
| cone | forward | 512x448x384 | 307.4 ms | 154.5 ms | 78.3 ms |

The source for every row above is the regression table file named in the paragraph before it. Each row with a 1024-view sinogram is a single trial, and so is every VCD row at every size, because the harness sets one trial for that operation. A single trial gives no estimate of the variation between repeats.

That file also records four-GPU speedups of 4.01 for the parallel forward projection, 3.84 for the cone forward projection, and 3.41 for the cone VCD row. Several parallel rows with a 1024-view sinogram carry a "throttled" flag. The harness sets that flag when nvidia-smi reports an active thermal or power throttle reason for a GPU, or when a GPU core reaches 85 C, or when its HBM reaches 95 C, and the dashboard guide says such a point's timing is unreliable (`mbirtorch_metrics/tooling/scaling_tests/scaling_common.py:246`; `mbirjax_metrics/tooling/dashboard/template.html:71`). Those speedups carry the same caveat.

A larger run measured a 2048-view sinogram of shape (2048, 2016, 1984), reconstructed to a volume of shape (1984, 1984, 2016), on a four-H100 node on 2026-08-17. Three iterations took 420 s for cone beam and 216 s for parallel beam on four GPUs, with per-GPU peak memory of 43.1 to 46.3 GiB for the cone configurations. The tables are in `plans/torch_port/active/multigpu_findings.md` section 1.20, and the run detail is in `.../plans/experiments/torch_port/mg19_two_k_baselines.md`.

One earlier record in this repository normalized LEAP against mbirjax rather than mbirtorch. The file is `plans/projector_kernels/headroom_appendices/appendix_ct_kernel_practice.md`, dated 2026-07-12. It placed mbirjax's parallel forward projection at 40 G voxel-view updates per second per TB/s of memory bandwidth, and LEAP's at 92 G. It described mbirjax as about 2.3 times slower than a state-of-the-art band of 80 to 120 G. Those figures are the appendix author's own normalization of published times rather than published throughputs, and the appendix marks them as estimates pending verification.

Those figures are also stale for mbirtorch. On one GPU, a three-iteration parallel-beam reconstruction with a 1024-view sinogram took 94.0 s of wall-clock time before the hand-written kernels and 21.26 s after them, recorded in `plans/torch_port/active/execution_overview.md`. That 21.26 s reading and the 18,976.3 ms in the regression table above are different runs under different measurement protocols, which the same file says differ by up to 15 percent.

### Head-to-head measurements on H100s

Both packages ran the same circular cone-beam problem at three sizes on one NVIDIA H100 80 GB. Each size N used N views over a full turn, an N by N flat detector, and an N by N by N volume. The three sizes were N = 256, 512, and 1024. The phantom was the same array in both runs. LEAP 1.26 was built from source and mbirtorch 0.0.2 ran at commit `26bd0ea`. Both used torch 2.13.0+cu130. The multi-GPU comparison below used four of the same GPUs on one node.

With the detector and view conventions aligned, the two forward projections agree to 0.052 percent NRMSE. These results indicate that both packages were timed on the same computation.

The next table gives forward projection, back projection, direct reconstruction, and a ten-iteration reconstruction on one GPU. Each projection and direct-reconstruction time is the best of three timed repeats after one warmup run. The ten-iteration rows are single runs:

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

The source for the table is `plans/experiments/features/leap_comparison/results/leap_benchmark_results.md`. The GPU peak is the maximum GPU memory in use during the measurement, so it includes the CUDA context. The ten-iteration rows are first-run times, so they charge mbirtorch's one-time compilation to the reconstruction.

The next table gives the steady-state cost of one iteration, taken from the third of three reconstructions run in the same process:

| N | LEAP RWLS with TV | mbirtorch VCD with qGGMRF | LEAP : mbirtorch | mbirtorch first-run extra cost |
| --- | --- | --- | --- | --- |
| 256 | 0.08782 s/iteration | 0.34643 s/iteration | 1 : 3.94 | 10.28 s |
| 512 | 1.26005 s/iteration | 1.58913 s/iteration | 1 : 1.26 | 15.74 s |
| 1024 | 19.27966 s/iteration | 17.73758 s/iteration | 1 : 0.92 | 19.45 s |

The source for the table is the same results file. The extra cost is the difference between mbirtorch's first reconstruction and its third, and it is a `torch.compile` cost paid once per inductor cache directory rather than once per machine. The benchmark used a cold cache, so it charges the full amount. This repository records a smaller repeat cost for a warm cache. A new process with a full cache ran a three-iteration parallel reconstruction in 26.02 s, against 21.17 s for an in-process repeat (`plans/torch_port/active/multigpu_findings.md` section 1.48).

These are two different algorithms with different priors. The numbers are therefore a cost per iteration only. They say nothing about image quality, and nothing about how many iterations either algorithm needs.

The next table gives four-GPU times at N = 1024, with each package's one-GPU time for comparison:

| Configuration | Forward projection | Back projection | Direct reconstruction | 10 iterations |
| --- | --- | --- | --- | --- |
| LEAP, one GPU | 6.314 s | 4.070 s | 4.314 s | 192.7 s |
| mbirtorch, one GPU | 8.636 s | 4.837 s | 4.980 s | 200.2 s |
| mbirtorch, four GPUs, automatic | 8.623 s | 4.841 s | 1.429 s | 109.03 s |
| mbirtorch, four GPUs, pinned | 2.189 s | 1.331 s | 1.432 s | 75.81 s |
| LEAP, four GPUs | 2.214 s | 1.645 s | 1.849 s | 313.75 s |

The next table gives each four-GPU configuration's speedup against its own one-GPU time:

| Configuration | Forward projection | Back projection | Direct reconstruction | 10 iterations |
| --- | --- | --- | --- | --- |
| mbirtorch, four GPUs, automatic | 1.00x | 1.00x | 3.48x | 1.84x |
| mbirtorch, four GPUs, pinned | 3.95x | 3.63x | 3.48x | 2.64x |
| LEAP, four GPUs | 2.85x | 2.47x | 2.33x | 0.61x |

The source for both tables is the same results file. Each speedup is measured against a different baseline, so the two packages' ratios are not directly comparable to each other; the absolute times above them are.

Two rows show no speedup, each for its own reason. mbirtorch's automatic policy chose one GPU at the first `forward_project` call, so the bare forward and back projections left three GPUs idle. The pinned configuration called `configure_devices` and spread that work across all four. LEAP's four-GPU reconstruction was slower than its one-GPU reconstruction, because `leapctype` splits work across GPUs only for arrays that start on the host. Every projection call in that configuration therefore paid a host-to-device and a device-to-host copy.

The next table gives the correctness cross-checks, at N = 256 unless stated otherwise:

| Check | mbirtorch | LEAP |
| --- | --- | --- |
| Forward projections agree between the packages, NRMSE | 0.052 percent | 0.052 percent |
| Direct reconstruction against the phantom, NRMSE | 0.06090 | 0.06468 |
| Adjoint relative difference | 1.240e-09 | 5.918e-06 |
| Adjoint relative difference at N = 64 | 4.068e-09 | 3.881e-04 |

The source for the table is the same results file and the smoke-pass records beside it. These measurements support the following findings:

- LEAP's forward and back projectors were faster than mbirtorch's at every size.
- Direct reconstruction was close at every size, and mbirtorch was faster at N = 256.
- The steady-state cost per iteration crosses over between N = 512 and N = 1024. LEAP is cheaper at N = 512, at 1.26005 s against 1.58913 s. mbirtorch is cheaper at N = 1024, at 17.73758 s against 19.27966 s.
- mbirtorch's steady-state iteration cost was 0.34643 s at N = 256, against a forward projection of 0.03518 s plus a back projection of 0.02015 s, and 17.73758 s at N = 1024 against 8.636 s plus 4.837 s. These results indicate a fixed per-iteration overhead that amortizes as the problem grows, rather than better algorithmic scaling.
- mbirtorch held more GPU memory for every projection and direct reconstruction, and less for every ten-iteration reconstruction. At N = 1024 LEAP's reconstruction peaked at 65.62 GiB of a 79.65 GiB card, which is the closest anything measured came to filling it.
- On four GPUs the two forward projections were within one percent of each other, at 2.189 s and 2.214 s, and mbirtorch was faster on back projection and direct reconstruction. LEAP reached its forward-projection time while paying host transfers that mbirtorch did not pay, so the comparison mixes two causes.
- mbirtorch's projectors agree with each other to float32 summation noise, at 1.240e-09. LEAP's disagree by 5.918e-06 at N = 256 and by 3.881e-04 at N = 64. Accumulation noise falls as a problem shrinks, so a reading that grows at the smaller size argues the difference is not summation order. The test is one random vector pair per package, and LEAP ran with its default matched separable-footprint projector.

Four conditions limit what the benchmark can say. The phantom is three nested spheres and carries no noise, and both packages ran with uniform weights, so neither statistical model was exercised. mbirtorch ran with `stop_threshold_change_pct=0.0`, which switches off the stopping rule listed as strength 3 below. LEAP ran with its default matched projector rather than the voxel-driven backprojector its own repository says is about twice as fast.

The benchmark leaves the following unmeasured:

- image quality at matched regularization strength;
- the number of iterations either package needs to converge;
- CPU performance;
- curved detectors;
- helical scans;
- short scans.

### Fixed image quality on noisy data

This study measured iterations to a fixed image quality on noisy data, using the timing study's geometry and phantom positions rescaled to attenuation values of 0.02, 0.01, and 0.04 per mm. Poisson noise at 10000 counts per pixel was added to the sinogram, and both libraries reconstructed from that same noisy sinogram and the same transmission weights. Each library started from its own direct reconstruction, then ran for exactly k iterations per point, scored by NRMSE against the voxelized phantom inside the inscribed cylinder. mbirtorch, not LEAP, forward projected the sinogram, which is an inverse crime, but a symmetric one, because the two projectors agree to 0.052 percent. The source for this subsection and the next is `plans/experiments/features/leap_comparison/results/quality_results.md`.

A parameter sweep at N = 256 found each library's best setting, confirmed against boundary probes so that neither winner sits at the edge of its grid:

**mbirtorch**, sharpness at snr_db 30:

| setting | best NRMSE | at k |
| --- | --- | --- |
| sharpness -3 (probe) | 0.05231 | 15 |
| sharpness -2 (probe) | 0.04442 | 20 |
| sharpness -1 (winner) | 0.04226 | 100 |

**LEAP**, RWLS with TV, delta 0.001:

| TV weight | best NRMSE |
| --- | --- |
| 3 (probe) | 0.05423 |
| 10 (winner) | 0.04542 |
| 30 (probe) | 0.04780 |

Both winners use three to four times more smoothing than each library's own starting point. mbirtorch's shipped defaults are sharpness 1.0 and snr_db 30.0, and the TV weight used in the earlier timing study was 1.

The next table gives, for each size, the target NRMSE and the first iteration count k at which each library reached it. The target is 1.02 times the larger of the two libraries' best NRMSE at that size. That is a quality both libraries demonstrably reach. Warm time is k times the steady-state per-iteration time, plus a warm direct-reconstruction time carried over from the timing study.

| N | target NRMSE | library | first k at target | NRMSE there | measured time (s) | warm time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 0.04633 | LEAP | 100 | 0.04542 | 7.895 | 8.510 |
| 256 | 0.04633 | mbirtorch | 7 | 0.04489 | 12.970 | 2.609 |
| 512 | 0.03980 | LEAP | 100 | 0.03902 | 111.777 | 121.205 |
| 512 | 0.03980 | mbirtorch | 10 | 0.03887 | 20.145 | 16.982 |
| 1024 | 0.05788 | LEAP | 40 | 0.05675 | 703.057 | 753.106 |
| 1024 | 0.05788 | mbirtorch | 5 | 0.04898 | 127.916 | 95.376 |

Two more measurements complete the comparison: each library's direct-reconstruction NRMSE on the noisy data, and its steady-state cost per iteration in this study.

| N | LEAP `FBP` NRMSE | mbirtorch `recon_fdk` NRMSE |
| --- | --- | --- |
| 256 | 0.09987 | 0.11237 |
| 512 | 0.11810 | 0.14153 |
| 1024 | 0.15638 | 0.19245 |

| N | LEAP (s/iteration) | mbirtorch (s/iteration) |
| --- | --- | --- |
| 256 | 0.08480 | 0.36869 |
| 512 | 1.20909 | 1.66453 |
| 1024 | 18.71982 | 18.07920 |

mbirtorch's default stopping rule reaches a comparable result on its own, without the hand-tuned k values used above. With `max_iterations=100` and the default 0.2 percent relative-change rule, it stopped at 9, 10, and 12 iterations at N = 256, 512, and 1024, with NRMSE of 0.04357, 0.03887, and 0.03392. All three values were at or below that size's target.

These results support five findings:

- mbirtorch reached the target in 8 to 14 times fewer iterations than LEAP, at every size.
- Warm time to target favors mbirtorch by 3.3 times at N = 256, 7.1 times at N = 512, and 7.9 times at N = 1024.
- Measured time, which includes compilation, is 1.6 times slower for mbirtorch at N = 256 and 5.5 times faster at N = 512 and N = 1024.
- mbirtorch's best NRMSE was lower than LEAP's at every size.
- The default stop rule stopped at or below the target every time.

Six caveats qualify this comparison:

- LEAP was still descending at the last k measured at every size, so the target is set by where measurement stopped, and LEAP's best NRMSE may lie beyond it.
- The TV sweep was a ten-point grid at N = 256, applied unchanged at the larger sizes.
- RWLS has no `numSubsets` argument, so ordered-subsets acceleration was not available for this algorithm.
- No warm direct reconstruction was measured in this study.
- N = 1024 has four k values and one steady-state run.
- The comparison used one phantom, one noise draw, and one noise level.

### Automatic device policy against a pinned layout

A separate job repeated the four-GPU comparison at N = 1024 to separate a one-time cost from a persistent one. The head-to-head subsection above reports ten iterations taking 109.03 s under the automatic policy against 75.81 s pinned to four devices. In that comparison, the automatic arm ran first and paid the cold compilation, while the pinned arm did not. This job ran three ten-iteration reconstructions per arm in one process, and it ran the pinned arm first. Any ordering advantage therefore now favors the automatic arm.

| arm | run 1 (s) | run 2 (s) | run 3 (s) | steady state (s/iteration) |
| --- | --- | --- | --- | --- |
| pinned to four devices | 99.233 | 66.046 | 65.954 | 6.595 |
| automatic policy | 77.821 | 68.246 | 68.183 | 6.818 |

In steady state the automatic policy costs 3.4 percent more per iteration than the pinned layout, 6.818 s against 6.595 s. The earlier gap of 109.03 s against 75.81 s was therefore almost entirely one-time cost, not a persistent penalty of the automatic policy.

The mechanism is in the projector code. `forward_project` and `back_project` do not call the device policy. A bare projector call therefore runs on the model's current layout, and that layout is one device for a newly constructed model ([mbirtorch/tomography_model.py:1906](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L1906), [mbirtorch/tomography_model.py:1940](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L1940)). The policy widens the layout to four GPUs only at the first reconstruction entry point, such as `recon_fdk` in `cone_beam.py` ([mbirtorch/cone_beam.py:783](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/cone_beam.py#L783)). That call prints a message once: "Using 4 CUDA device(s) for this reconstruction (was 1)".

### How to read these numbers

The five subsections above answer different questions. Only the head-to-head, fixed-quality, and device-policy subsections compare the two packages directly on the same hardware. LEAP's published numbers and mbirtorch's recorded numbers were taken on different hardware, at different sizes, and for different operations. Those two sets therefore cannot be compared with each other. LEAP's are from a Tesla P100 released in 2016, and mbirtorch's are from an H100. LEAP quotes an image dimension of 1024^3 with 720 views, while mbirtorch quotes a sinogram of 1024 views by 1008 rows by 992 channels. The head-to-head subsection removes those mismatches by running both packages on one machine at one geometry.

The head-to-head numbers still answer a narrow question. They say which package computes one projection faster at one size. They also say what one iteration of each package's own algorithm costs. They do not say which package produces a better image, how many iterations each needs, or how each behaves on a real scan with imperfect geometry. Reconstruction quality and iteration count are the quantities a user cares about. The fixed-quality subsection above measures both, for one phantom at one noise level.

### Follow-up measurements

Three measurements would extend the conclusions above.

A wider TV sweep at each size, and a best-setting run of LEAP's other algorithms such as ASD-POCS and SART, would show whether TV with RWLS is LEAP's strongest choice on this problem.

A real scan with imperfect geometry would test both packages under calibration errors that a synthetic phantom does not have.

A study at more than one noise level would show whether the iteration gap and the smoothing-strength result above hold beyond 10000 counts per pixel.

The check of mbirtorch's automatic device policy for bare projector calls is done. Its steady-state penalty against a pinned four-device layout was 3.4 percent at N = 1024 (`plans/experiments/features/leap_comparison/results/quality_results.md`).

---

## High-value LEAP features missing in mbirtorch

### 1. Geometric calibration utilities

LEAP solves for scan geometry from the projection data. It provides `find_centerCol` for the center of rotation, `find_tau` for the source offset, and `estimate_tilt` for detector rotation about the optical axis ([src/leapctype.py#L846](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L846), [#L892](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L892), [#L937](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L937)). Four more utilities build on those three:

- a data-consistency cost for axial flat-panel cone beam ([src/leapctype.py#L1021](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1021));
- a joint two-parameter search ([src/leap_preprocessing_algorithms.py#L737](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L737));
- a parameter sweep that reconstructs one slice per candidate value ([src/leap_preprocessing_algorithms.py#L872](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L872));
- a ball-phantom least-squares fit ([src/leap_preprocessing_algorithms.py#L1159](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L1159)).

LEAP also supplies the score function those sweeps need. `inconsistencyReconstruction` replaces the ramp filter with a derivative and returns pure noise when the geometry is correct. LEAP's own docstring calls it a robust way to find the center column or estimate detector tilt ([src/leapctype.py#L2845](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2845)).

This matters most to users of real scanners whose geometry is not exactly known. A wrong center of rotation produces obvious artifacts, and a wrong tilt produces subtle ones. mbirtorch's own FAQ tells a user with a blurry reconstruction to adjust `det_channel_offset` by hand, which is a manual version of this search.

mbirtorch has two related routines: a view-offset estimator ([mbirtorch/preprocess/utilities.py:1072](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/utilities.py#L1072)) and a view-alignment routine ([mbirtorch/preprocess/utilities.py:1205](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/utilities.py#L1205)). Both already reconstruct, forward project, high-pass filter, and estimate shifts, so the scaffolding exists. What is missing is a solver that turns those shifts into a single global geometry parameter.

One design constraint decides the implementation. `det_channel_offset` is declared with the flag that forces a projector recompile when it changes ([mbirtorch/_utils.py:77](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/_utils.py#L77)). A loop that calls `recon_direct` at each candidate offset would therefore pay a `torch.compile` rebuild per candidate, and it would exhaust the recompile budget in `mbirtorch/projectors.py`. A design that scores candidates by resampling the sinogram, rather than by changing the model parameter, avoids that cost.

### 2. Offset-scan and truncated-object support

LEAP handles two cases in which the object does not fit the detector. An offset scan places the rotation axis near one edge of the detector and nearly doubles the field of view ([src/leapctype.py#L5739](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5739)). A truncated scan is handled by extrapolating the signal off the detector edge rather than zero-padding it before the ramp filter ([src/leapctype.py#L5721](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5721)), and `set_diameterFOV` then reconstructs past the detector.

LEAP also handles the harder iterative case. Its long-object demo states the trap directly: direct reconstruction can reconstruct a reduced region of interest, but iterative reconstruction requires every region a ray passes through to be included ([demo_leapctype/d34_iterative_reconstruction_long_object.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d34_iterative_reconstruction_long_object.py)). The method reconstructs the slabs above and below the region analytically, forward projects them, and subtracts them from the data. The supporting primitive is `sliceRangeNeededForProjection` ([src/leapctype.py#L3046](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3046)).

This matters to industrial users scanning wide or tall parts, and to synchrotron users doing local tomography. It is a correctness trap rather than a quality question, because including too few slices changes the answer.

mbirtorch has partial coverage. It warns when it detects lateral field-of-view truncation and documents enlarging the reconstruction region with `scale_recon_shape` ([mbirtorch/tomography_model.py:2117](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L2117)). This repository's `flash_remediation` program addresses the axial half of the same problem, and its plan records the cone per-end axial extension as implemented and validated on real scans (`plans/flash_remediation/flash_remediation_plan.md`). What is absent is an extrapolating ramp filter, an offset-scan weighting, and the subtract-the-caps method for iterative reconstruction.

An implementation has two independent parts. The extrapolating filter and the offset-scan weighting are changes to the direct-reconstruction filter in `mbirtorch/tomography_utils.py`. The long-object method needs no new projector, because it composes an existing direct reconstruction, an existing forward projection, and a subtraction.

### 3. Analytic ray-traced phantoms

LEAP builds a phantom from geometric primitives and can ray-trace it instead of forward-projecting a voxelized version ([src/leapctype.py#L7304](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7304), [#L7261](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7261)). Eight primitive types are available, each accepting a rotation matrix and clipping planes. LEAP also ships the FORBILD head phantom ([src/leapctype.py#L7357](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7357)). It supports ray oversampling, which averages in transmission space to model partial-volume effects ([demo_leapctype/d08_ray_tracing_simulation.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d08_ray_tracing_simulation.py)).

This matters to this repository's own research. The `nn_priors` findings evaluate a DRUNet prior and a MACE loop by NRMSE against a phantom that was forward-projected from its own voxelization, which is the inverse crime by construction. `multi_slice_fusion_findings.md` names one phantom and one noise draw as its weakest caveat. An analytic phantom would make those comparisons defensible under review. It would also have improved the benchmark above, whose direct-reconstruction cross-check of 0.06090 against 0.06468 is measured against a voxelized sphere, so part of that error is voxelization rather than reconstruction.

mbirtorch simulates only by forward-projecting a voxelized phantom, and its public generators cover two Shepp-Logan variants, a cube, dots, and text ([mbirtorch/utilities.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/utilities.py)). An implementation would add an analytic ray tracer next to the phantom generators in `mbirtorch/utilities.py`. It requires no change to the projectors, because the ray tracer computes line integrals through analytic shapes directly. The work is one intersection routine per primitive type.

### 4. Scatter correction

LEAP offers two scatter corrections at very different costs. `transmission_shift` subtracts a constant in transmission space ([src/leap_preprocessing_algorithms.py#L686](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L686)). The physics-based model simulates first-order scatter through an object of a single material with variable density ([src/leapctype.py#L1473](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1473)). Its stated limits are a volume no larger than 200^3, projections no larger than 256^2, and a spectrum of at most 20 samples.

This matters to cone-beam users scanning large or dense objects, where scatter is a dominant source of cupping and streaks. mbirtorch has no scatter correction. The mbirtorch inventory records a search of the package and the documentation that found none, and that search was by keyword rather than exhaustive.

An implementation should start with the constant-subtraction version. A constant transmission offset is a few lines in `mbirtorch/preprocess/utilities.py`, and it removes most of the scatter when the scatter is close to constant across the detector. That is the least work of anything on this list. The physics-based model needs a spectrum, so it depends on item 8 below.

### 5. Automatic splitting of a projection to fit one GPU

LEAP splits a projection into chunks of detector rows and a backprojection into chunks of volume z-slices. It then halves the chunk size in a loop until a chunk fits ([src/tomographic_models.cpp#L779-L787](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L779-L787), [#L1108-L1121](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L1108-L1121)). Its feature list states the consequence: LEAP's algorithms are not limited by the amount of GPU memory ([LEAP_features.md](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LEAP_features.md)). The iterative algorithms get the same behavior, because they are Python loops over the split projectors.

Two open LEAP issues qualify that claim (`LEAP-inv` section 10.6). One reports GPU memory that grows under repeated FBP calls, and the other is titled "cuda memory leak". This matters to a user with one GPU and a large scan, which is the PyPI and Apple audience rather than the four-GPU production work. It is the difference between a slow run and no run. mbirtorch's closest capability is `recon_split_sino` ([mbirtorch/tomography_model.py:372](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L372)). It splits the detector rows into overlapping bands and combines the reconstructed bands into one volume, and it is documented as only approximately equal to `recon`. The automatic path instead computes the memory cost of every candidate layout. When no layout fits, it fails immediately and names the shortfall.

An implementation would add a pixel-axis split inside `mbirtorch/projectors.py`, which is the axis a TODO comment in that file names and marks as near-term. The package already has two of the pieces this needs: the memory model computes the per-GPU peak, and the projector code already splits the work over views. The TODO records three costs: the forward projection must sum partial sinograms over pixel batches, the back projection must combine per-batch outputs inside the view sum, and the two batch sizes must be chosen together. One limit is worth stating plainly. The projectors and the direct reconstruction can be split this way, and the VCD solver cannot, because it keeps one error sinogram resident and updates it in place across the whole subset loop. Splitting the solver means recomputing that per chunk or accepting a different fixed point, which is what `recon_split_sino` already does.

### 6. Modular geometry with an arbitrary source position and detector pose per view

LEAP's `set_modularbeam` takes four arrays of shape (numAngles, 3), so each array holds one three-vector per view ([src/leapctype.py#L638](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L638)). The four arrays give the source position, the detector module center, the row direction, and the column direction. Any scan whose source and detector positions can be listed can then be described without new code. LEAP uses this one geometry for four applications:

- laminography ([demo_leapctype/d32_laminography.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d32_laminography.py));
- detector dithering ([demo_leapctype/d25_detector_dithering.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d25_detector_dithering.py));
- circle-plus-line trajectories ([demo_leapctype/d24_circle_plus_line.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d24_circle_plus_line.py));
- multi-source flash radiography ([demo_leapctype/d05_3DflashCT.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d05_3DflashCT.py)).

This ranks sixth rather than higher for two reasons. mbirtorch already covers laminography with its multi-axis parallel model, and the other three applications are outside this user base. LEAP's own demo also warns that data fitting a standard geometry should use it, because the modular-beam projectors are not as fast and not as accurate ([demo_leapctype/d05_3DflashCT.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d05_3DflashCT.py)).

mbirtorch's closest capability is its multi-axis parallel model, which allows a per-view azimuth and elevation ([mbirtorch/multiaxis_parallel.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/multiaxis_parallel.py)). That covers tilted-axis scans but not an arbitrary source position. An implementation would be a fifth model class, and it is the largest item on this list. All four current geometries share a separable-footprint factorization, and that factorization assumes an axis-aligned rotation. A general per-view pose therefore needs one of two changes: a Joseph-style ray model, or a restriction to nearly aligned poses. LEAP made the first change. It uses a matched Joseph projector when the detector columns are far from the z axis ([documentation/LEAP.tex](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/documentation/LEAP.tex)). The surrounding surface is larger than the model class itself. A new geometry supplies about seven hooks on `TomographyModel` and starts with no Triton kernels of its own. It also needs entries in four other places: the per-geometry kernel checks, the memory model's geometry branches, its own test file, and its own golden values ([docs/source/dev_api.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/dev_api.rst)).

### 7. Alternative iterative algorithms and a composable prior interface

LEAP ships twelve iterative reconstruction functions in Python on top of its projectors ([docs/source/iterative_reconstruction.rst](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/iterative_reconstruction.rst)). They span four families:

- algebraic methods, SIRT and SART;
- emission methods, MLEM and OSEM;
- transmission statistical methods, RWLS and MLTR;
- few-view methods, RDLS and ASD-POCS.

LEAP also ships the design worth copying. `filterSequence` is an append-able list of prior objects that any of its solvers accepts, each prior carries its own weight, and priors that are not differentiable are legal when routed to ASD-POCS ([docs/source/filter_sequence.rst](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/filter_sequence.rst)). Four differentiable priors ship with it: anisotropic total variation, an Lp norm on a filtered difference, histogram sparsity, and azimuthal sparsity ([src/leap_filter_sequence.py#L193](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_filter_sequence.py#L193)).

Two groups need this. Users with emission or very-low-count data need a Poisson or transmission likelihood that a Gaussian model does not provide, which is the real gap for photon-starved neutron data. Users doing method comparisons need baselines such as SIRT, and an mbirtorch user must install another package to get one.

mbirtorch has VCD with a qGGMRF prior, plus `prox_map` for plug-and-play priors ([mbirtorch/tomography_model.py:3130](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L3130), [:3284](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L3284)). Its `gen_weights` covers unweighted, transmission, transmission-root, and emission cases ([docs/source/theory.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/theory.rst)).

An implementation should not start with a native total-variation prior. There is no general prior interface in mbirtorch today: the prior is a two-branch choice inside one worker function, qGGMRF when no proximal input is given and the proximal map otherwise ([mbirtorch/tomography_model.py:2553](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L2553)). The VCD update also divides by a sum of forward and prior Hessians, so a prior with no positive quadratic surrogate has nothing to contribute to that denominator. Total variation's derivative ratio grows without bound as a voxel difference approaches zero, so a native total-variation prior needs a new surrogate rather than a new branch. The realistic route for a total-variation-like prior or a learned prior is `prox_map` and the plug-and-play loop, which already exists. Making the prior a parameter, as `filterSequence` does, is the architectural change that would generalize it.

### 8. Polychromatic and dual-energy physics

LEAP models the x-ray spectrum through the companion package XrayPhysics. With it, LEAP performs three operations:

- single-material and two-material beam-hardening correction;
- dual-energy decomposition;
- conversion of a low-energy and high-energy pair to electron density and effective atomic number ([src/leapctype.py#L4829](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4829)).

It also models the heel effect through a takeoff-angle lookup table ([src/leapctype.py#L4765](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L4765)). This matters to materials-characterization users and to anyone doing quantitative CT. Effective atomic number and electron density are the outputs those users want, and neither can be obtained from a single-energy reconstruction.

mbirtorch's closest capability is empirical beam-hardening correction. It fits a correction curve to the data and applies it, with a matching inverse ([mbirtorch/preprocess/utilities.py:1250](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/utilities.py#L1250)). It also has an adaptive plastic-and-metal beam-hardening loop ([mbirtorch/tomography_model.py:420](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L420)). Neither uses a spectrum, and together they already handle the dominant artifact. An implementation needs a spectral model before anything else. The smallest amount of new work is to depend on an existing cross-section and spectrum library rather than write one. LEAP does this with XrayPhysics. Dual-energy decomposition would then be a preprocessing function on a pair of sinograms. It would require no change to the projectors.

### 9. A fan-beam geometry as its own model

LEAP has a fan-beam geometry type, selectable as its own setting ([src/leapctype.py#L545](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L545)).

This matters to users with 2D fan datasets who want a named geometry rather than a workaround. In mbirtorch a fan problem is expressible as a cone-beam model with one detector row, because mbirtorch sinograms are always three-dimensional ([docs/source/quick_start.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/quick_start.rst)). An internal module named `horizontal_fan.py` holds the shared trapezoidal footprint calculations. It is not a geometry class.

A fan-beam class would be a small implementation. The horizontal fan calculations already exist, and a fan-beam model needs no vertical fan calculation at all. It would still carry its own test file, its own golden values, and a device-policy entry, and it would start with no Triton kernels. LEAP's cone-parallel geometry and its rebinning between geometries are lower value for this user base, because their stated purpose is medical CT data ([src/leapctype.py#L431](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L431), [#L1270](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L1270)).

### 10. Detector deblur

LEAP deconvolves a user-supplied detector blur kernel in two ways. One is Wiener deconvolution and the other is Richardson-Lucy, which preserves non-negativity ([src/leap_preprocessing_algorithms.py#L439](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L439), [#L466](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L466)).

This matters to users whose detectors have significant optical or scintillator spread. On such a detector the finest resolvable feature is larger than one pixel. The Zeiss Xradia Ultra, which mbirtorch already reads, is optics-limited in exactly this way. mbirtorch has no detector deblur, and its nearest preprocessing operations are defective-pixel interpolation and zinger correction ([mbirtorch/preprocess/utilities.py:174](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/utilities.py#L174), [:1582](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/utilities.py#L1582)).

An implementation is a self-contained preprocessing function. Wiener deconvolution of each view against a supplied 2D frequency response is three steps: a fast Fourier transform, a division, and an inverse transform. Those three steps fit the existing batched-view pipeline in `mbirtorch/preprocess/pipeline.py`, provided the frequency response is held as a NumPy array so the multi-GPU path can move it per batch.

### Lower value

Several further LEAP capabilities are absent from mbirtorch and do not justify a subsection above. They are listed here with a source:

- Low-signal and high-energy outlier correction, described for MV CT and neutron CT, where outliers affect a larger neighborhood ([src/leap_preprocessing_algorithms.py#L373](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L373), [#L406](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L406)).
- Short-scan Parker weighting and six additional ramp filter orders, which improve a direct reconstruction that mbirtorch uses mainly as an initializer ([src/ray_weighting_cpu.cpp#L66](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/ray_weighting_cpu.cpp#L66), [src/leapctype.py#L5831](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L5831)). A local-spacing or channel weight fits the existing view-independent `row_weight` hook in `mbirtorch/tomography_utils.py`; a Parker weight is view-dependent and does not.
- The detector-row and slice range calculators that tell a caller exactly which rows a slab needs and which slices the data touches ([src/leapctype.py#L2962](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L2962), [#L3046](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3046)). They are the primitives a band split or a chunk loop needs.
- `space_carving`, which estimates object support from thresholded projections, and an arbitrary binary volume mask applied inside the operators ([src/leapctype.py#L3572](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3572), [#L3378](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L3378)).
- Cropping that rewrites the geometry parameters, including the center column under an asymmetric crop, with a disk-side slab loop ([demo_leapctype/d13_cropping_subchunking.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d13_cropping_subchunking.py)).
- Frequency-domain fusion of two reconstructions to reduce cone-beam artifacts, which is the missing-cone problem that laminography shows in its most severe form ([demo_leapctype/d33_reducingConeBeamArtifacts.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d33_reducingConeBeamArtifacts.py)).
- `MTF`, which measures resolution from a radial edge and is how a resolution claim is defended in a paper ([src/leap_preprocessing_algorithms.py#L1087](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leap_preprocessing_algorithms.py#L1087)).

### Already covered

Several capabilities that LEAP advertises are already present in mbirtorch, so they are not listed above. They fall into three groups:

- acquisition: helical cone-beam scanning, curved detectors, and non-uniform angular spacing;
- correction: ring and stripe removal, bad-pixel and outlier correction, metal artifact reduction, and empirical beam-hardening correction;
- algorithms and compute: a matched forward and back projector pair, differentiable projectors with an adjoint backward pass, a non-negativity constraint, multi-GPU execution, and a memory cost computed before the first large allocation.

Two capabilities are near matches rather than equivalents. mbirtorch restricts the reconstruction with a two-dimensional region mask applied identically to every slice ([mbirtorch/vcd_utils.py:18](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/vcd_utils.py#L18)), while LEAP takes an arbitrary binary 3D mask and applies it inside the operators. mbirtorch's multi-granular partitions divide voxels, while LEAP's ordered subsets divide views, so they are different algorithms rather than the same one under two names.

Two further items belong here because neither package has them, so neither is a LEAP advantage. Neither package supports float16 or bfloat16 (`LEAP-inv` section 4.5; `MT-inv` section 3). Neither package reads DICOM.

---

## mbirtorch strengths relative to LEAP

### 1. Automatic sharding across GPUs with a memory model

mbirtorch shards a single reconstruction across GPUs, splitting the volume by slice and the sinogram by view ([docs/source/dev_sharding_overview.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/dev_sharding_overview.rst)). Two rules choose the GPU count automatically. The first rule uses a measured minimum speed for each GPU count. The second rule allows a slower GPU count when the smaller memory per GPU requires it ([mbirtorch/_widening_floors.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/_widening_floors.py)). A memory model computes the memory cost of each candidate layout before the first large allocation ([mbirtorch/_memory_ledger.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/_memory_ledger.py)).

LEAP also uses multiple GPUs, so this is a difference of kind rather than a capability LEAP lacks. LEAP streams detector-row or slice chunks from host memory through the GPUs ([src/tomographic_models.cpp#L803-L804](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L803-L804), and the equivalent loops in its other multi-GPU functions). Its multi-GPU path requires the ZYX volume order ([src/tomographic_models.cpp#L737](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/tomographic_models.cpp#L737)), and it requires the data to start on the host ([src/leapctype.py#L39](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L39)).

The head-to-head benchmark measured both designs at N = 1024. mbirtorch on four GPUs was faster than LEAP on four GPUs for back projection, direct reconstruction, and ten iterations, and the two were within one percent for forward projection.

The same benchmark qualifies the automatic part of this strength, which is the automatic choice of GPU count. A bare projector call uses one device until a reconstruction entry point runs the device policy and widens the layout. A script that only calls `forward_project` or `back_project` therefore gets one GPU on a four-GPU node. Once the layout is settled, the automatic choice costs little: its steady-state penalty against a layout pinned to four devices was 3.4 percent at N = 1024 (`plans/experiments/features/leap_comparison/results/quality_results.md`).

### 2. An exact adjoint pair, checked automatically

mbirtorch's forward and back projectors are adjoint by construction. The package measures that property rather than claiming it. The test checks that the inner product of a forward projection with a sinogram equals the inner product of the volume with a back projection ([tests/test_adjoint.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/tests/test_adjoint.py)). The relative tolerance is 1e-4. Each hand-written Triton kernel is checked again against its compiled PyTorch equivalent at run time, per kernel and per GPU, before it is used ([mbirtorch/kernel_availability.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/kernel_availability.py)).

The head-to-head benchmark measured the property on both packages. At N = 256 the adjoint relative difference was 1.240e-09 for mbirtorch and 5.918e-06 for LEAP. At N = 64 it was 4.068e-09 for mbirtorch and 3.881e-04 for LEAP. The reading grows for LEAP as the problem shrinks, which is the opposite of how accumulation noise behaves, so summation order does not explain it. Each reading is one random vector pair.

LEAP also ships matched projector pairs, and its feature list names them first ([LEAP_features.md](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LEAP_features.md)). The difference is verification, because LEAP has no automated test of the property.

### 3. Model-based reconstruction with automatic parameters

mbirtorch's `recon()` minimizes a data term plus a qGGMRF prior by Multi-Granular Vectorized Coordinate Descent ([mbirtorch/tomography_model.py:3130](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L3130)). It stops when the relative change between iterations falls below a threshold that defaults to 0.2 percent, rather than when a fixed iteration count runs out. It sets the noise and prior parameters automatically from two user parameters, `sharpness` and `snr_db` ([mbirtorch/tomography_model.py:2086-2186](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L2086-L2186)).

The stopping rule is qualified by the default iteration cap. `max_iterations` defaults to 15, this repository's partition-sequence plan recommends raising it to roughly 25 to 50, and its flash-remediation study records about 20 iterations to reach the 0.2 percent stop. The cap therefore often binds before the stopping rule fires. In the fixed-quality noisy-data study, with `max_iterations` raised to 100, the rule stopped at 9, 10, and 12 iterations at N = 256, 512, and 1024, at or below that study's target every time (`plans/experiments/features/leap_comparison/results/quality_results.md`).

LEAP has regularized weighted least squares, which is a comparable objective. The two packages therefore differ in how they stop and how they set parameters, not in the objective. No LEAP algorithm takes a tolerance argument, and convergence is controlled only by `numIter` (`LEAP-inv` section 6.4). LEAP also asks the user to choose regularizer weights directly. mbirtorch derives those weights instead.

### 4. A proximal map and a denoiser that accept split data

mbirtorch exposes `prox_map`, which solves the proximal problem for the forward model ([mbirtorch/tomography_model.py:3284](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L3284)). A prior supplied by the caller can then control the reconstruction. mbirtorch also ships a standalone qGGMRF denoiser ([mbirtorch/denoising.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/denoising.py)). Both accept and return data already split across GPUs, through `output_sharded=True` ([docs/source/usr_multi_gpu.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_multi_gpu.rst)). A plug-and-play loop can therefore run without copying to the host between steps.

This is the capability behind this repository's live research program. The `nn_priors` records run a MACE loop with a DRUNet prior on `prox_map` and the qGGMRF denoiser.

LEAP has no proximal-map entry point. Its counterpart is `filterSequence`, which composes priors inside LEAP's own solvers rather than exposing the proximal operator ([docs/source/filter_sequence.rst](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/docs/source/filter_sequence.rst)).

### 5. Vendor scanner readers and one-call model construction

mbirtorch reads four scan formats and returns a configured model from a single call to `get_sino_and_model` ([docs/source/usr_preprocess.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_preprocess.rst)). The four readers are NorthStar Instruments, Zeiss Versa and Ultra, Zeiss translation tomography, and ORNL HDF5, the last of which is documented as the PYMBIR functions. Readers set the arbitrary length unit (ALU) used internally and record its physical size, so a reconstruction can be reported in physical units ([docs/source/unit_conversion.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/unit_conversion.rst)).

LEAP has no vendor reader. It reads TIFF sequences, NRRD, and NumPy files, and it expects the user to set the geometry parameters from the scanner's own metadata ([src/leapctype.py#L7092](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/src/leapctype.py#L7092)). LEAP does provide geometry bridges to TIGRE and to LTT, which mbirtorch does not ([utils/](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/utils/)).

### 6. Installation and platform reach

mbirtorch installs from PyPI as package `mbirtorch` at version 0.0.2, with no compiler and no CUDA toolkit required (`MT-inv` section 1, read from https://pypi.org/pypi/mbirtorch/json). It runs on CUDA, on CPU, and on Apple MPS. It selects the backend automatically ([mbirtorch/tomography_model.py:91](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L91)).

LEAP has no PyPI package at all, verified against the PyPI API for four candidate names (`LEAP-inv` section 20.1). Its documented install compiles from a source checkout, and its wiki asks for CMake 3.23.3 or newer and CUDA 11.7 or newer (`LEAP-inv` sections 10.1 and 17). A community conda-forge package `leapct` 1.26 exists for Linux and Windows only, and its recipe skips non-CUDA builds (`LEAP-inv` section 20.2). LEAP's macOS support is a CPU-only source build with no released binary, and nothing in its repository mentions Apple Silicon or Metal (`LEAP-inv` section 10.3).

### 7. Automated testing, continuous integration, and documentation

mbirtorch has 769 test functions across 37 files in `tests/`, plus 13 more test functions in `ci/` (`MT-inv` section 8). That is a count of test function definitions rather than of executed cases, because many are parameterized. Continuous integration runs the suite on pull requests across Python 3.11 through 3.14, and it builds the documentation with warnings treated as errors ([.github/workflows/ci.yml](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/.github/workflows/ci.yml)). The documentation is 30 pages published at https://mbirtorch.readthedocs.io/.

LEAP has three test scripts and no continuous integration (`LEAP-inv` section 13.3). It has 38 demo scripts with explanatory docstrings against mbirtorch's nine, so LEAP has more teaching material.

### 8. Two geometries LEAP does not have

mbirtorch models translation tomography, in which each view is a cone-beam projection of a translated object and the view parameter is a translation vector ([mbirtorch/translation_model.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/translation_model.py)). It also models multi-axis parallel beam. In that model each view has both an azimuth angle and an elevation angle ([mbirtorch/multiaxis_parallel.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/multiaxis_parallel.py)). Parallel-beam laminography is the constant-elevation case of the second model.

LEAP covers laminography through its modular geometry rather than a dedicated model ([demo_leapctype/d32_laminography.py](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/demo_leapctype/d32_laminography.py)). Laminography is therefore not unique to mbirtorch, and translation tomography has no LEAP counterpart. The mbirtorch translation model is labeled experimental and in alpha testing ([docs/source/usr_translation_model.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_translation_model.rst)).

### 9. Automated view selection

mbirtorch chooses which views to acquire. `get_opt_views` scores candidate view sets against a reference object and runs a greedy search over angle subsets ([mbirtorch/vcls.py:97](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/vcls.py#L97), [docs/source/usr_vcls.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_vcls.rst)).

LEAP has no counterpart, and it addresses the few-view problem after acquisition, with reconstruction algorithms such as RDLS and with sparsity regularizers, and its authors wrote a paper on those methods (arXiv:2410.07552). Choosing the views is the complementary half of the same problem.

### 10. Hyperspectral neutron tomography

mbirtorch denoises hyperspectral neutron data with a non-negative matrix factorization of the spectral axis ([mbirtorch/hsnt.py:14](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/hsnt.py#L14), [docs/source/usr_hsnt.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_hsnt.rst)). The method is published as Chowdhury and coauthors, IEEE Transactions on Computational Imaging, volume 11, pages 663 to 677, 2025.

LEAP has no hyperspectral support, and this capability serves the neutron user base that mbirtorch's ORNL reader also serves.

---

## Project health

| Item | LEAP | mbirtorch | Source |
| --- | --- | --- | --- |
| License | MIT | BSD 3-Clause | [LICENSE](https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/LICENSE); [LICENSE](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/LICENSE) |
| Latest release | v1.26, 2024-12-14 | v0.0.2, 2026-08-21 | `LEAP-inv` section 15; `MT-inv` section 1 |
| Commits in the last 12 months | 0 on `main` | 198 on `greg_dev` | `LEAP-inv` section 15; `MT-inv` section 1 |
| Stars | 249 | 0 | https://api.github.com/repos/LLNL/LEAP and https://api.github.com/repos/cabouman/mbirtorch , read 2026-09-03 |
| Contributors | 3, with 430, 7, and 1 commits | 6, with 101 and 87 for the top two | `LEAP-inv` section 15; `MT-inv` section 1 |
| Open issues | 45 open of 197 total | 0 of 0 | https://api.github.com/search/issues , read 2026-09-03 |
| Packaging channel | conda-forge, community maintained | PyPI | `LEAP-inv` section 20; `MT-inv` section 1 |
| Platforms | Linux and Windows with CUDA | CUDA, CPU, and Apple MPS | `LEAP-inv` sections 10.3 and 20.3; `MT-inv` section 6 |
| Maintenance status | static since 2024-12-14 | last push 2026-08-29 | `LEAP-inv` section 15; `MT-inv` section 1 |

LEAP is an established package whose released version has not changed in about twenty months. Development has continued on an unreleased `version_two` branch as recently as 2026-07-25. Those commits indicate that the project is not abandoned. Its maintainer answered issues through 2026-01. A user installing LEAP today gets code from December 2024, and macOS support is a CPU-only source build with no released binary.

The star and open-issue rows were read from the GitHub API on 2026-09-03. LEAP's issue counts had each risen by one since the inventory read them on 2026-09-02, and its star count was unchanged.

mbirtorch is about one month old as a public package and has no external issue history. Its repository was created on 2026-08-04, and its 198 commits and two releases all fall within that month. The maintenance comparison therefore compares an established project with a new one. Neither state is evidence about long-term support.

---

## Sources

The following sources support every claim above:

1. `plans/features/leap_comparison_sources/leap_inventory.md`, the sourced LEAP inventory, copied into this repository
2. `plans/features/leap_comparison_sources/mbirtorch_inventory.md`, the sourced mbirtorch inventory, copied into this repository
3. https://github.com/LLNL/LEAP/tree/0c8846f42b2e59340d5559fc1271d590a292f9a0 , the LEAP v1.26 tree that every LEAP reference above is pinned to
4. `https://github.com/LLNL/LEAP/blob/0c8846f42b2e59340d5559fc1271d590a292f9a0/` , the base URL that LEAP file references are written against
5. `https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/` , the base URL that mbirtorch file references are written against
6. https://github.com/cabouman/mbirtorch/tree/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2 , the mbirtorch source at the compared commit
7. `plans/projector_kernels/headroom_appendices/appendix_ct_kernel_practice.md`
8. `plans/torch_port/open_items_v5.md`
9. `plans/torch_port/port_plan.md`
10. `plans/torch_port/closed/current_plans.md`
11. `plans/torch_port/active/execution_overview.md`
12. `plans/torch_port/active/multigpu_findings.md`
13. `plans/experiments/torch_port/mg19_two_k_baselines.md`
14. `mbirtorch_metrics/results/gpu/prerelease/regression_gpu_20260827T175529Z_26bd0ea9_table.yaml`
15. `plans/experiments/features/leap_comparison/results/leap_benchmark_results.md`, the head-to-head benchmark results
16. `plans/experiments/features/leap_comparison/bench_leap_vs_mbirtorch.py`, the benchmark harness
17. `plans/experiments/features/leap_comparison/leap_cmp_gautschi.sbatch`, the single-GPU job at all three sizes
18. `plans/experiments/features/leap_comparison/leap_cmp_repeat_gautschi.sbatch`, the repeated reconstructions at N = 256
19. `plans/experiments/features/leap_comparison/leap_cmp_repeat2_gautschi.sbatch`, the repeated reconstructions at N = 512 and N = 1024
20. `plans/experiments/features/leap_comparison/leap_cmp_multigpu_gautschi.sbatch`, the four-GPU job at N = 1024
21. `plans/nn_priors/multi_slice_fusion_findings.md` and `plans/flash_remediation/flash_remediation_plan.md`
22. https://github.com/LLNL/leap
23. https://arxiv.org/abs/2307.05801 (Kim and Champley, "Differentiable Forward Projector for X-ray Computed Tomography", ICML workshop, 2023)
24. https://arxiv.org/abs/2410.07552 (Champley and coauthors, "Methods for Few-View CT Image Reconstruction", 2024)
25. https://leapct.readthedocs.io/
26. https://github.com/kylechampley/XrayPhysics
27. https://github.com/kylechampley/LEAPCT-UI-GUI
28. https://pypi.org/pypi/mbirtorch/json
29. https://mbirtorch.readthedocs.io/
30. https://software.llnl.gov/news/2024/01/07/leap-new/
31. `plans/experiments/features/leap_comparison/quality_leap_vs_mbirtorch.py`, the noisy image-quality benchmark harness
32. `plans/experiments/features/leap_comparison/quality_gautschi.sbatch`, the parameter sweep, curves, and default-stop job at all three sizes
33. `plans/experiments/features/leap_comparison/quality_probe_gautschi.sbatch`, the boundary-probe job at N = 256
34. `plans/experiments/features/leap_comparison/leap_cmp_autopin_gautschi.sbatch`, the automatic-against-pinned device-policy job at N = 1024
35. `plans/experiments/features/leap_comparison/results/quality_results.md`, the noisy image-quality results
36. `plans/experiments/features/leap_comparison/results/quality_nrmse_vs_time_256.png`, `_512.png`, and `_1024.png`, the NRMSE-against-wall-time curves at each size

Three earlier records in this repository mention LEAP, and all three are cited above.

The kernel survey at source 7, dated 2026-07-12, compares projector implementations across ASTRA, TIGRE, LEAP, CTorch, and svmbir, and it normalizes published throughputs by memory bandwidth. Its mbirjax numbers predate mbirtorch's hand-written kernels and must not be quoted as current mbirtorch performance.

A proposed compatibility wrapper for other packages appears in sources 8, 9, and 10. It would have eased a transition for users of LEAP and svmbir, and it was closed on 2026-08-19 in favor of the one-call functional interface. The recorded reason is that LEAP already presents a PyTorch interface. A wrapper on mbirtorch would therefore need less code than the same wrapper on mbirjax.
