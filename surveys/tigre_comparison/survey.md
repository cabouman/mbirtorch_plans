# TIGRE vs mbirtorch: features and performance

Date: 2026-09-03. Versions compared: TIGRE at commit `51ae1a02` (master, 2026-09-02, 77 commits past tag v3.1.3), and mbirtorch 0.0.2 at commit `26bd0ea`.

## Summary

TIGRE and mbirtorch overlap in purpose and differ in design. TIGRE is a CUDA library of cone-beam CT algorithms with MATLAB and Python front ends. Its stated focus is iterative algorithms, and it ships about two dozen of them over one flexible geometry object. mbirtorch is a model-based reconstruction package built on PyTorch. It has one main reconstruction algorithm, four geometry classes, automatic multi-GPU sharding, and vendor scanner readers. TIGRE covers more algorithms and more trajectories. mbirtorch covers its narrower scope with statistical modeling, automation, and testing that TIGRE does not have.

The deepest technical difference is the projector pair. mbirtorch's forward and back projectors are exact transposes of each other, and a test suite checks that property. TIGRE's default pair is unmatched: the forward projector is ray-driven and the backprojector is voxel-driven with FDK weights. TIGRE's own documentation calls its alternative backprojector weighting "mathematically very close to the transpose", and its Krylov solvers carry restart logic for the divergence that the mismatch can cause.

Three terms recur below and are defined here. Sharding means splitting one reconstruction across several GPUs and running the pieces together. A memory model is code that computes the memory cost of a candidate layout before any large allocation, so a run that cannot fit fails immediately. An unmatched projector pair is a forward projector and a backprojector that are not transposes of one another.

The TIGRE features that mbirtorch lacks are these, ranked by value to mbirtorch users:

1. A suite of classical iterative algorithms behind one interface, usable as baselines: SIRT, OS-SART, CGLS and other Krylov solvers, the ASD-POCS family, FISTA, and MLEM.
2. Prior-image constrained reconstruction, PICCS, which reconstructs sparse-view data against a prior image.
3. A per-view flexible geometry: three Euler angles per view, and per-view distances and offsets, with helpers for tomosynthesis and arbitrary source paths.
4. Offset-detector weighting and short-scan Parker weighting in direct reconstruction.
5. Automatic splitting of any projection or backprojection into pieces that fit in GPU memory.
6. Scatter correction, in the form of a kernel-based estimator inside the Varian pipeline.
7. A built-in noise simulator with GPU Poisson and Gaussian noise.
8. Six vendor data loaders whose vendor set barely overlaps mbirtorch's four.
9. A per-iteration image-quality hook that logs RMSE, SSIM, and other metrics during a run.
10. A geometry sketch plot that draws the source, detector, and volume for a given view.

mbirtorch's advantages over TIGRE are these:

1. An exact adjoint projector pair, checked by an automated test suite and again at run time.
2. A statistical objective with measurement weights and a qGGMRF prior, automatic parameter selection, and a relative-change stopping rule.
3. Multi-GPU sharding that keeps data resident on the GPUs, guided by a memory model; TIGRE moves host arrays in and out on every call.
4. Native PyTorch integration whose gradient is the exact adjoint; TIGRE's torch wrapper copies through host memory on every call and back-projects its gradient with FDK weights.
5. Installation from PyPI with no compiler, and support for CUDA, CPU, and Apple MPS; TIGRE requires nvcc and an NVIDIA GPU.
6. A proximal map and a sharded qGGMRF denoiser for plug-and-play priors.
7. A test suite of 769 test functions with continuous integration; TIGRE's CI builds wheels but runs no tests.
8. Standalone preprocessing corrections, including beam hardening, metal artifact weights, and stripe removal; TIGRE's corrections live inside one vendor loader.
9. HDF5 export and physical-unit bookkeeping; TIGRE ships no file writers at all.
10. A documentation site of 30 pages; TIGRE documents itself through demos, and its readthedocs configuration points at a directory that does not exist in the tree.

No head-to-head timing has been run yet. A benchmark harness and two cluster job scripts were prepared in `plans/experiments/features/tigre_comparison/`, following the LEAP comparison's protocol, and this session had no cluster access to run them. The published TIGRE numbers below are from a Tesla K40, GTX 1080 Ti cards, and an RTX 4070. The recorded mbirtorch numbers are from H100 cards. Numbers from different hardware cannot be compared with each other, so this document makes no cross-package speed claim.

---

## Scope and versions

This document compares the features and the recorded performance of the two packages. Its main purpose is to name the TIGRE features that mbirtorch lacks and that would be worth building. Its secondary purpose is to name mbirtorch's advantages.

The evidence has three parts: a sourced TIGRE inventory prepared in this session, the mbirtorch inventory prepared for the LEAP comparison, and the published TIGRE papers. Every number below is copied from a named file or URL. Where a number does not exist in any source that was read, the text says so.

The TIGRE version is the master branch at commit `51ae1a02e070888c8aca481e41157d54b20f692f`, dated 2026-09-02. That commit is 77 commits past the last release tag, v3.1.3 (2026-03-06), and its `pyproject.toml` still declares version 3.1.3. The tip was compared rather than the tag because TIGRE's development lands on master continuously, and the tip includes recent work such as the automatic FISTA step-size estimate.

The mbirtorch version is 0.0.2, at commit `26bd0ea`, dated 2026-08-27, on branch `greg_dev`. This is the same mbirtorch pin as the LEAP comparison (`plans/features/leap_comparison/leap_comparison.md`), so the mbirtorch columns of the two documents describe the same code.

TIGRE has two language front ends over one CUDA core. The comparison target here is the Python package, pytigre. Capabilities that exist only on the MATLAB side are marked, because a Python user does not get them.

---

## Feature comparison

References in this section are written against two pinned commits and two inventory files:

- TIGRE file references link under `https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/`.
- mbirtorch file references link under `https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/`.
- `TG-inv` = `plans/features/tigre_comparison/tigre_comparison_sources/tigre_inventory.md`.
- `MT-inv` = `plans/features/leap_comparison/leap_comparison_sources/mbirtorch_inventory.md`.

### Geometries

| Capability | TIGRE | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Cone beam, circular | yes | yes | [Python/tigre/utilities/geometry.py#L38-L39](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/geometry.py#L38-L39); [mbirtorch/cone_beam.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/cone_beam.py) |
| Parallel beam | yes, `mode="parallel"` | yes, own class | `TG-inv` geometries section 1; [mbirtorch/parallel_beam.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/parallel_beam.py) |
| Helical | via per-view `offOrigin`, iterative only | yes, in the cone model | [Python/demos/d13_HelicalGeometry.py#L39-L41](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/demos/d13_HelicalGeometry.py#L39-L41); `MT-inv` section 2 |
| Fan beam | 2D cone with one detector row | cone model with one detector row | [Python/tigre/utilities/geometry_default.py#L78-L80](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/geometry_default.py#L78-L80); `MT-inv` section 2 |
| Three Euler angles per view | yes, ZYZ | no | [Python/demos/d18_ArbitraryAxisOfRotation.py#L9-L10](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/demos/d18_ArbitraryAxisOfRotation.py#L9-L10) |
| Per-view source and detector distances and offsets | yes: `DSD`, `DSO`, `COR`, `offOrigin`, `offDetector`, `rotDetector` | no; per-view angles only, plus per-view elevation in one model | [geometry.py#L80-L105](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/geometry.py#L80-L105); [mbirtorch/multiaxis_parallel.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/multiaxis_parallel.py) |
| Tomosynthesis and static-detector scans | yes, geometry helpers plus a DBT demo | no | [Python/tigre/utilities/common_geometry.py#L6-L25](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/common_geometry.py#L6-L25) |
| Arbitrary source and detector pose per view | yes, `ArbitrarySourceDetMoveGeo` | no | [common_geometry.py#L60](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/common_geometry.py#L60) |
| Curved detector | no, flatten-and-resample utility | yes, in the cone model | [Python/tigre/utilities/curved_detector.py#L4](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/curved_detector.py#L4); `MT-inv` section 2 |
| Translation tomography | expressible via per-view `offOrigin`; no dedicated model | yes, alpha | [geometry.py#L83-L87](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/geometry.py#L83-L87); [mbirtorch/translation_model.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/translation_model.py) |

The two geometry designs place their flexibility differently. TIGRE has one geometry object, and almost every parameter of it can vary per view. mbirtorch has four geometry classes, and each class fixes its trajectory family. TIGRE's design covers more scan types with no new code. mbirtorch's design lets each class carry its own direct reconstruction, memory model entries, and tests.

### Projector models and adjoints

| Capability | TIGRE | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Forward model | ray-driven: Siddon (default) or interpolated sampling | separable footprint | [Python/tigre/utilities/Ax.py#L29](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/Ax.py#L29); [docs/source/dev_projector_kernels.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/dev_projector_kernels.rst) |
| Back model | voxel-driven, FDK weights (default) or "matched" weights | transpose of the forward model | [Python/tigre/utilities/Atb.py#L9](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/Atb.py#L9); [docs/source/usr_autograd.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_autograd.rst) |
| Default pair is matched | no | yes, by construction | `TG-inv` projectors section 5 |
| Exactness of the matched option | "mathematically very close to the transpose"; named pseudo-matched in the source | exact transpose | [Python/demos/d12_ProjectionOperations.py#L7-L10](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/demos/d12_ProjectionOperations.py#L7-L10); [MATLAB/Utilities/cuda_interface/Atb_mex.cpp#L104](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/MATLAB/Utilities/cuda_interface/Atb_mex.cpp#L104) |
| Adjointness checked by an automated test | no test found | yes, at 1e-4 relative | `TG-inv` projectors section 5; [tests/test_adjoint.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/tests/test_adjoint.py) |
| Hardware texture interpolation | yes | no | [Common/CUDA/voxel_backprojection.cu#L799](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Common/CUDA/voxel_backprojection.cu#L799) |
| Implementation language | CUDA C, Cython bindings | PyTorch plus Triton | `TG-inv` projectors section 7; [docs/source/dev_projector_kernels.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/dev_projector_kernels.rst) |
| Precision | float32 only | float32 only | `TG-inv` projectors section 6; `MT-inv` section 3 |
| Arrays live | on the host; each call copies in and out | on the GPUs, optionally sharded | [Python/tigre/utilities/Ax.py#L31-L41](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/Ax.py#L31-L41); [docs/source/usr_multi_gpu.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_multi_gpu.rst) |

The mismatch of TIGRE's pair is visible in TIGRE's own code. Its CGLS tracks the residual and restarts when the residual rises, and the comment names "the mismatch of the backprojection w.r.t the real adjoint" as one suspected cause ([MATLAB/Algorithms/CGLS.m#L88-L93](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/MATLAB/Algorithms/CGLS.m#L88-L93)). TIGRE also ships AB-GMRES and BA-GMRES, which it describes as "a stable Krylov method for when the backprojector is not adjoint" ([MATLAB/Algorithms/BA_GMRES.m#L4](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/MATLAB/Algorithms/BA_GMRES.m#L4)). mbirtorch needs neither mechanism, because its pair is a true transpose.

### Direct reconstruction

| Capability | TIGRE | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| FDK for cone, FBP for parallel | yes | yes, for all four geometries | [Python/tigre/algorithms/single_pass_algorithms.py#L12](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/algorithms/single_pass_algorithms.py#L12), [#L208](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/algorithms/single_pass_algorithms.py#L208); [mbirtorch/cone_beam.py:798](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/cone_beam.py#L798) |
| Ramp filter choices | 5: ram-lak, shepp-logan, cosine, hamming, hann | 1, "ramp" only | [Python/tigre/utilities/filtering.py#L71-L83](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/filtering.py#L71-L83); [mbirtorch/tomography_utils.py:18](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_utils.py#L18) |
| Short-scan Parker weighting | MATLAB FDK: automatic; Python: function exists but FDK hardcodes it off | no | [MATLAB/Algorithms/FDK.m#L153-L158](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/MATLAB/Algorithms/FDK.m#L153-L158); [Python/tigre/algorithms/single_pass_algorithms.py#L199](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/algorithms/single_pass_algorithms.py#L199) |
| Offset-detector (Wang) weighting | yes, on by default in FDK, with zero padding | no | [single_pass_algorithms.py#L76-L130](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/algorithms/single_pass_algorithms.py#L76-L130) |
| Truncation extrapolation | no | no extrapolating filter; warns and can enlarge the region | `TG-inv` direct-recon section 5; [mbirtorch/tomography_model.py:2117](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L2117) |
| Helical direct reconstruction | no | no | `TG-inv` geometries section 4; `MT-inv` section 4 |

TIGRE's direct reconstruction is a means to start its iterative algorithms, as mbirtorch's is for VCD, and both packages say so in their demos and docstrings. TIGRE's version is stronger on scan protocol support. It has five filters, offset-detector weighting, and, on the MATLAB side, automatic short-scan weighting. The Python side hardcodes Parker weighting off in both entry points, and the filtering module carries the comment "TODO: Fix parker" ([Python/tigre/utilities/filtering.py#L14](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/filtering.py#L14)).

### Iterative reconstruction algorithms

| Capability | TIGRE | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Coordinate descent on a MAP objective | no | yes, VCD | [mbirtorch/tomography_model.py:3130](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L3130) |
| Algebraic methods | SART, SIRT, OS-SART, plus TV variants | no | [Python/tigre/algorithms/__init__.py#L5-L12](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/algorithms/__init__.py#L5-L12) |
| Krylov methods | CGLS, LSQR, LSMR, hybrid LSQR, IRN-TV-CGLS, hybrid fLSQR-TV, AB-GMRES, BA-GMRES | no | [__init__.py#L13-L20](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/algorithms/__init__.py#L13-L20) |
| TV-constrained few-view methods | ASD-POCS, OS-ASD-POCS, AwASD-POCS, PCSD variants | no | [__init__.py#L21-L28](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/algorithms/__init__.py#L21-L28) |
| Prior-image method | PICCS and OS-PICCS | no | [Python/tigre/algorithms/pocs_algorithms.py#L325-L326](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/algorithms/pocs_algorithms.py#L325-L326) |
| Proximal gradient | FISTA and ISTA with a TV proximal step | plug-and-play via `prox_map` | [Python/tigre/algorithms/ista_algorithms.py#L208](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/algorithms/ista_algorithms.py#L208); [mbirtorch/tomography_model.py:3284](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L3284) |
| Emission method | MLEM (Python); OSEM is MATLAB-only | no, emission weights only | [Python/tigre/algorithms/statistical_algorithms.py#L28-L50](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/algorithms/statistical_algorithms.py#L28-L50); `MT-inv` section 4 |
| Measurement noise weights in the objective | no | yes, `weights` and `gen_weights` | `TG-inv` iterative-algos section 1; [docs/source/theory.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/theory.rst) |
| Stopping rule other than an iteration count | only the ASD-POCS family's constraint test | yes, relative change, default 0.2 percent | [Python/tigre/algorithms/pocs_algorithms.py#L188](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/algorithms/pocs_algorithms.py#L188); [mbirtorch/tomography_model.py:3130](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L3130) |
| Ordered subsets of views | yes, with four ordering strategies | no; voxel partitions instead | [Python/tigre/utilities/order_subsets.py#L41-L52](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/order_subsets.py#L41-L52); [mbirtorch/vcd_utils.py:97](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/vcd_utils.py#L97) |
| Per-iteration quality metrics | yes: RMSE, nRMSE, CC, MSSIM, UQI, SSD | no | [Python/tigre/utilities/Measure_Quality.py#L26-L33](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/Measure_Quality.py#L26-L33) |
| Automatic step size | FISTA estimates its Lipschitz constant by a power method | not applicable; VCD needs no step size | [ista_algorithms.py#L130-L160](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/algorithms/ista_algorithms.py#L130-L160) |

TIGRE offers about 24 iterative algorithms in Python and mbirtorch offers one. Nearly all TIGRE algorithms run for a fixed iteration count; the exceptions are the ASD-POCS family's constraint-based exit and the Krylov solvers' divergence guard. mbirtorch stops on a measured relative change. TIGRE's data models are unweighted or ray-length-weighted least squares, plus MLEM's Poisson model. mbirtorch's data model takes per-measurement noise weights, which none of TIGRE's least-squares algorithms accept.

### Regularizers and priors

| Capability | TIGRE | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| qGGMRF prior | no | yes | `TG-inv` regularizers section 1; [docs/source/theory.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/theory.rst) |
| Total variation | yes, three CUDA kernels: gradient descent, adaptive-weighted, primal-dual denoiser | no | `TG-inv` regularizers section 1 |
| Prior-image regularization | yes, PICCS | no | [Common/CUDA/PICCS.cu#L274](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Common/CUDA/PICCS.cu#L274) |
| Standalone denoiser | TV denoiser, `im3ddenoise` | qGGMRF denoiser, 3D median filter | [Python/tigre/utilities/im_3d_denoise.py#L7-L26](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/im_3d_denoise.py#L7-L26); [mbirtorch/denoising.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/denoising.py) |
| Plug-and-play prior interface | no; the TV proximal step is hardcoded | yes, `prox_map` | `TG-inv` regularizers section 6; [mbirtorch/tomography_model.py:3284](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L3284) |
| Automatic regularization strength | MATLAB hybrid solvers only, by discrepancy principle or GCV | yes, from `sharpness` and `snr_db` | [MATLAB/Algorithms/hybrid_LSQR.m#L48-L56](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/MATLAB/Algorithms/hybrid_LSQR.m#L48-L56); [mbirtorch/tomography_model.py:2086-2186](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L2086-L2186) |

TIGRE's only prior family is total variation, in three variants plus the PICCS prior-image form. Its TV strength parameters are set by hand, and its Python hybrid solvers take a fixed lambda while the MATLAB versions can choose lambda per iteration. mbirtorch's only native prior is qGGMRF, set automatically from two user parameters, and its plug-and-play route accepts any denoiser. Neither package ships a learned prior.

### Preprocessing, artifact correction, and data loaders

| Capability | TIGRE | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Vendor loaders (Python) | 6: Nikon, Bruker/SkyScan, YXLON, Diondo, DXChange, Varian | 4: NSI, Zeiss Versa/Ultra, Zeiss translation, ORNL HDF5 | [Python/tigre/utilities/io/__init__.py#L1-L6](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/io/__init__.py#L1-L6); [mbirtorch/preprocess/](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/) |
| DICOM | MATLAB only, Philips C-arm | no | [MATLAB/Utilities/IO/Dicom/dicomDataLoader.m#L1](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/MATLAB/Utilities/IO/Dicom/dicomDataLoader.m#L1); `MT-inv` section 7 |
| Loader output | `(projections, geometry, angles)` | sinogram plus a configured model, one call | `TG-inv` preprocess section 1; [docs/source/usr_preprocess.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_preprocess.rst) |
| Scatter correction | yes, kernel-based, inside the Varian loader | no | [Python/tigre/utilities/io/varian/scatter.py#L354](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/io/varian/scatter.py#L354); `MT-inv` section 4 |
| Beam hardening correction | MATLAB Varian pipeline only | yes, empirical curve fit, standalone | [MATLAB/Utilities/IO/VarianCBCT/BHCorrection.m#L1-L6](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/MATLAB/Utilities/IO/VarianCBCT/BHCorrection.m#L1-L6); [mbirtorch/preprocess/utilities.py:1250](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/utilities.py#L1250) |
| Metal artifact reduction | none found | weights plus a beam-hardening loop | grep for "metal" over the package found nothing; [mbirtorch/tomography_model.py:420](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L420) |
| Ring or stripe removal | a median filter inside the Varian loader | three standalone routines | [Python/tigre/utilities/io/VarianDataLoader.py#L66-L72](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/io/VarianDataLoader.py#L66-L72); [mbirtorch/preprocess/stripe.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/stripe.py) |
| Bad pixel and outlier correction | no standalone routine found | yes | `TG-inv` preprocess section 3; [mbirtorch/preprocess/utilities.py:174](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/utilities.py#L174) |
| Dual-energy decomposition | MATLAB only | no | [MATLAB/Utilities/DE/DeDecompose.m#L1-L3](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/MATLAB/Utilities/DE/DeDecompose.m#L1-L3) |

The two packages organize preprocessing differently. mbirtorch's corrections are standalone functions that apply to any dataset. TIGRE's corrections are embedded in its Varian clinical pipeline, so a Nikon or Bruker user cannot call the scatter correction on their data without extracting the code. TIGRE's loader list and mbirtorch's loader list cover disjoint vendors, except that both read a synchrotron HDF5 layout.

### Geometric calibration

| Capability | TIGRE | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Center-of-rotation estimation from data | MATLAB only, `computeCOR` | view-offset estimator | [MATLAB/Utilities/computeCOR.m#L1-L5](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/MATLAB/Utilities/computeCOR.m#L1-L5); [mbirtorch/preprocess/utilities.py:1072](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/utilities.py#L1072) |
| COR as a geometry parameter | yes, per view | via detector offset | `TG-inv` geometries section 3; `MT-inv` section 2 |
| View alignment across views | no | yes, `align_sino_views` | [mbirtorch/preprocess/utilities.py:1205](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/preprocess/utilities.py#L1205) |
| Tilt, source-offset, or ball-phantom solvers | no | no | `TG-inv` preprocess section 4; `MT-inv` section 5 |

Neither package solves for scan geometry beyond the center of rotation. This is unlike LEAP, whose calibration suite was the widest gap in the LEAP comparison. TIGRE expects the vendor metadata to be right, and its loaders read the vendor's own center-of-rotation value with a warning that the sign convention may be wrong ([Python/tigre/utilities/io/DiondoDataLoader.py#L107-L115](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/io/DiondoDataLoader.py#L107-L115)).

### Simulation and physics

| Capability | TIGRE | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Voxelized phantoms | 3D Shepp-Logan in three variants, plus a CT head | several generators | [Python/tigre/utilities/sl3d.py#L18-L25](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/sl3d.py#L18-L25); [mbirtorch/utilities.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/utilities.py) |
| Analytic ray-traced phantoms | no | no | `TG-inv` preprocess section 6; `MT-inv` section 7 |
| Built-in noise model | yes: GPU Poisson plus Gaussian, `CTnoise.add` | no | [Python/tigre/utilities/CTnoise.py#L4-L25](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/CTnoise.py#L4-L25); `MT-inv` section 7 |
| Proton CT | yes, projectors and a demo | no | `TG-inv` identity section 4; [Frontispiece/pCT_INSTRUCTIONS.md](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Frontispiece/pCT_INSTRUCTIONS.md) |
| Hyperspectral neutron support | no | yes | [docs/source/usr_hsnt.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_hsnt.rst) |

Both packages simulate by forward-projecting a voxelized phantom, so both commit the inverse crime in simulation studies. An inverse crime is the use of the same discretized forward model to make the data and to reconstruct it, which flatters the result. TIGRE adds a noise step that mbirtorch leaves to the calling script.

### Deep-learning integration and autograd

| Capability | TIGRE | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Differentiable projectors | yes, `torch.autograd.Function` wrappers | yes, native | [Python/tigre/utilities/pytorch_bindings.py#L9](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/pytorch_bindings.py#L9); [mbirtorch/autograd.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/autograd.py) |
| Gradient operator | FDK-weighted backprojector, i.e. not the adjoint | the exact adjoint | [pytorch_bindings.py#L62-L64](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/pytorch_bindings.py#L62-L64); [docs/source/usr_autograd.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_autograd.rst) |
| Tensors stay on the GPU | no; every call copies to numpy and back | yes | [pytorch_bindings.py#L39](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/pytorch_bindings.py#L39) |
| Batch dimension | yes, a Python loop per element | no | [pytorch_bindings.py#L34-L42](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/pytorch_bindings.py#L34-L42) |
| Committed test runs | the binding test imports a module that does not exist in the tree | autograd covered by the test suite | [Python/tigre/tests/utilities/pytorch_bindings_test.py#L9](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/tests/utilities/pytorch_bindings_test.py#L9); `MT-inv` section 8 |

TIGRE's torch wrapper makes its projectors usable inside a network, and a learned-gradient demo trains against them. Two costs qualify it. Every call detaches the tensor, copies it to host numpy, runs TIGRE, and copies the result back, so gradients flow but the GPU-to-GPU path does not exist. The backward pass calls `Atb` with its default weighting, so the gradient is computed with the FDK-weighted backprojector rather than the transpose.

### Compute, GPUs, and memory

| Capability | TIGRE | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| CUDA | yes, required | yes | `TG-inv` identity section 7; [mbirtorch/tomography_model.py:91](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L91) |
| CPU | no fallback of any kind | yes, every geometry | `TG-inv` identity section 7; `MT-inv` section 6 |
| Apple MPS | no | yes | `TG-inv` identity section 6; [mbirtorch/tomography_model.py:91](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L91) |
| Multi-GPU | yes: views split for projection, slabs for backprojection | yes, sharded volume and sinogram | `TG-inv` projectors section 10; [docs/source/dev_sharding_overview.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/dev_sharding_overview.rst) |
| Data placement across calls | host memory; each call re-uploads | persistent GPU shards | `TG-inv` projectors section 1; [docs/source/usr_multi_gpu.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_multi_gpu.rst) |
| Automatic chunking below GPU memory | yes: free-memory query, then z-axis splits | no; a manual band split instead | [Common/CUDA/voxel_backprojection.cu#L714-L753](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Common/CUDA/voxel_backprojection.cu#L714-L753); [mbirtorch/tomography_model.py:372](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L372) |
| Memory cost computed before allocation | per call, from free memory at that moment | per model, from a memory model | [voxel_backprojection.cu#L992-L1008](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Common/CUDA/voxel_backprojection.cu#L992-L1008); [mbirtorch/_memory_ledger.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/_memory_ledger.py) |
| Behavior on a busy GPU | hard error when a GPU is over half occupied | the layout accounts for free memory | [voxel_backprojection.cu#L1001-L1002](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Common/CUDA/voxel_backprojection.cu#L1001-L1002); [mbirtorch/_memory_ledger.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/_memory_ledger.py) |
| Mixed GPU models in one run | warned against; the first GPU's limits are assumed for all | supported; per-device memory is measured | [Common/CUDA/Siddon_projection.cu#L283-L285](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Common/CUDA/Siddon_projection.cu#L283-L285); [docs/source/usr_multi_gpu.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/usr_multi_gpu.rst) |

The two memory strategies mirror the LEAP comparison's finding, with TIGRE on LEAP's side of the split. TIGRE chunks each call so that any problem runs, however slowly, on whatever memory is present. mbirtorch prices the whole layout before allocating and refuses runs that do not fit. TIGRE's per-call design also means every operator call pays host transfers, because no array survives on the GPU between calls. mbirtorch keeps the sinogram and volume resident across the whole iterative loop.

### Data I/O and visualization

| Capability | TIGRE | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| File writers for results | none | HDF5 | `TG-inv` preprocess section 8; [mbirtorch/viewer.py:556](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/viewer.py#L556) |
| Volume viewer | animated slice loop with GIF export | matplotlib slice viewer with side-by-side and difference views | [Python/tigre/utilities/visualization/plotimg.py#L18-L27](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/visualization/plotimg.py#L18-L27); [mbirtorch/viewer.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/viewer.py) |
| Geometry sketch | yes, `plot_geometry` and an animated trajectory plot | no | [Python/tigre/utilities/visualization/plot_geometry.py#L65](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/visualization/plot_geometry.py#L65) |
| Sinogram viewer | yes, `plotSinogram` | via the slice viewer | [Python/tigre/utilities/visualization/plotproj.py#L234](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/visualization/plotproj.py#L234) |

TIGRE ships no way to save a reconstruction to disk; the user calls numpy or h5py directly. Its plotting set is broader than mbirtorch's in one respect, the geometry sketch, which draws the scan setup and is useful when debugging an exotic trajectory.

### API, documentation, tests, and packaging

| Capability | TIGRE | mbirtorch | Notes/source |
| --- | --- | --- | --- |
| Shortest working example | five calls: geometry, angles, `Ax`, `fdk`, `ossart` | two lines | [Python/example.py#L27-L42](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/example.py#L27-L42); [README.md](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/README.md) |
| Documentation | demos as documentation, by the project's own statement | 30 pages on readthedocs | [README.md#L96](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/README.md#L96); `MT-inv` section 8 |
| Demo scripts | 24 Python, 27 MATLAB | 9 | `TG-inv` api-docs section 3; [demo/](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/demo/) |
| Tests | about 21 test functions plus a shell-driven RMSE matrix; one file is Python 2 and cannot run | 769 test functions in 37 files | `TG-inv` api-docs section 5; `MT-inv` section 8 |
| Continuous integration runs tests | no; CI builds wheels and an sdist only | yes, four Python versions | [.github/workflows/build.yml](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/.github/workflows/build.yml); [.github/workflows/ci.yml](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/.github/workflows/ci.yml) |
| Install without a compiler | no; pip install compiles CUDA with nvcc | yes, from PyPI | [Frontispiece/python_installation.md#L41-L43](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Frontispiece/python_installation.md#L41-L43); `MT-inv` section 1 |
| PyPI availability | none current: `pytigre` last released 2019, and the name `tigre` on PyPI is an unrelated genomics tool | yes, `mbirtorch` 0.0.2 | `TG-inv` web section 2 and addendum; `MT-inv` section 1 |
| Conda availability | ccpi channel, 3.1.3, Linux and Windows | no | `TG-inv` web section 2 |

TIGRE's installation story is its weakest point for a Python user. The pip route compiles CUDA from source and requires nvcc, the PyPI names are stale or taken by another project, and the maintained binary channel is community conda. Its documentation is its demos, which are numerous and well commented, but there is no API reference; the readthedocs site renders about ten of the demos, and the repository's readthedocs configuration points at a `Python/docs/` directory that does not exist in the tree (`TG-inv` api-docs section 4).

---

## Performance

### TIGRE's published numbers

TIGRE's repository contains no benchmark scripts and no tabulated timings; a repository-wide search for "benchmark" found nothing (`TG-inv` performance section 1). The quantitative record lives in three papers and one tuning document.

The 2016 paper measured projection speed on a Tesla K40 with MATLAB R2014b (Biguri, Dosanjh, Hancock, Soleimani, Biomedical Physics & Engineering Express 2(5) 055010, 2016). A single Siddon forward projection of a 512-cube onto a 512 by 512 detector took about 10 ms. The interpolated projector was about 4 times slower than Siddon. The largest case reported was about 1 s per projection for a 1024-cube with a 1024 by 1024 detector. A 512-cube reconstruction with 15 CGLS iterations took 4 min 41 s on that machine, as restated in the later multi-GPU paper.

The 2020 multi-GPU paper measured scaling on GTX 1080 Ti cards, 11 GiB each (Biguri and coauthors, Journal of Parallel and Distributed Computing, 2020; arXiv:1905.03748). The same 512-cube, 15-iteration CGLS reconstruction took 1 min 01 s on one GTX 1080 Ti. Speedups approached the theoretical 50, 33, and 25 percent of single-GPU time for 2, 3, and 4 GPUs at large sizes, with forward projection scaling best and backprojection worse. Two large real cases anchor the memory claim: a 3340 by 3340 by 900 voxel reconstruction, a 40 GB volume from 29 GB of projections, ran 30 CGLS iterations in 4 h 21 min on two GTX 1080 Ti cards, and a fossil dataset ran 50 OS-SART iterations in 6 h 40 min on the same machine. The timings include host transfers, which is inherent to TIGRE's design.

The TIGRE v3 paper reports one wall time (Biguri and coauthors, Engineering Research Express 7, 015011, 2025; arXiv:2412.10129). On an RTX 4070 desktop, a Varian head scan of 493 projections at 1024 by 768, reconstructed to 512 by 512 by 362, took a few seconds for FDK and under 5 minutes each for 50 iterations of OS-SART and of OS-ASD-POCS. The paper's other four experiments report no timings.

The tuning document adds one in-repository number: a backprojection of a 512-cube from 500 projections of size 1000 by 1000 takes approximately 1 s on a GTX 1080 Ti ([Frontispiece/Tune_TIGRE.md#L9](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Frontispiece/Tune_TIGRE.md#L9)).

### mbirtorch's recorded numbers

mbirtorch's latest regression run is on NVIDIA H100 80GB HBM3 GPUs, recorded in `mbirtorch_metrics/results/gpu/prerelease/regression_gpu_20260827T175529Z_26bd0ea9_table.yaml`, measured 2026-08-30 at commit `26bd0ea9`. Sizes are sinogram shapes written as views by detector rows by detector channels.

| Geometry | Operation | Size | 1 GPU | 2 GPUs | 4 GPUs |
| --- | --- | --- | --- | --- | --- |
| cone | forward | 1024x1008x992 | 8,000.0 ms | 4,029.2 ms | 2,085.5 ms |
| cone | back | 1024x1008x992 | 4,546.5 ms | 2,353.6 ms | 1,256.1 ms |
| cone | VCD, 3 iterations | 1024x1008x992 | 59,178.1 ms | 31,038.9 ms | 17,344.2 ms |
| cone | forward | 512x448x384 | 307.4 ms | 154.5 ms | 78.3 ms |

The caveats on these rows are recorded in the LEAP comparison and are not repeated here: the 1024-view rows are single trials, and some parallel-beam rows in the same file carry a thermal-throttle flag (`plans/features/leap_comparison/leap_comparison.md`, "mbirtorch's recorded numbers").

### No head-to-head has been run

The TIGRE and mbirtorch numbers above are from different decades of hardware, different problem sizes, and different operations. They support no cross-package speed conclusion. The nearest same-hardware anchor is indirect and weak: the LEAP comparison measured mbirtorch against LEAP on H100s, and no measurement connects LEAP and TIGRE on shared hardware either.

A head-to-head harness is ready. `plans/experiments/features/tigre_comparison/bench_tigre_vs_mbirtorch.py` reuses the LEAP benchmark's protocol: the same sphere phantom, the same cone geometry at N = 256, 512, and 1024 on one H100, timed forward and back projection, FDK, ten fixed iterations, adjoint checks, and cross-checks of the two forward projections. Its mbirtorch mode is identical to the LEAP harness's mbirtorch mode, so one run of the TIGRE side would make all three packages comparable at those sizes. The TIGRE side times both of TIGRE's forward models, both of its backprojector weightings, and OS-SART and CGLS as the iterative arms. Two job scripts, `tigre_cmp_gautschi.sbatch` and `tigre_cmp_multigpu_gautschi.sbatch`, cover the single-GPU sizes and a four-GPU arm at N = 1024. The harness compiles but has not executed, because this session had no cluster access; a smoke pass at N = 64 should precede the full sizes.

Two design facts will shape the eventual numbers and are worth stating in advance. TIGRE's times will include host transfers on every call, because its interface is host arrays in and host arrays out. And TIGRE's iterative arms solve different objectives than VCD, so per-iteration cost will need the same interpretation care as in the LEAP comparison: cost per iteration says nothing about iterations to a given quality. A fixed-quality study on noisy data, like the LEAP comparison's, is the follow-up that would answer the question a user cares about, and TIGRE's OS-ASD-POCS would be its natural arm.

---

## High-value TIGRE features missing in mbirtorch

### 1. A suite of classical iterative algorithms as baselines

TIGRE ships about 24 iterative algorithms behind one shared interface: the SART family, eight Krylov solvers, the ASD-POCS family, FISTA and ISTA, and MLEM (`TG-inv` iterative-algos sections 2 through 6). The shared base class gives every one of them the same initialization options, ordered-subset machinery, non-negativity clip, and per-iteration quality hook ([Python/tigre/algorithms/iterative_recon_alg.py#L19-L23](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/algorithms/iterative_recon_alg.py#L19-L23)).

This matters for method comparisons. TIGRE is the standard baseline toolbox in the CBCT literature, so a paper that evaluates a new mbirtorch-based method will be asked how it compares against OS-SART, CGLS, or ASD-POCS. Today that comparison requires installing TIGRE alongside mbirtorch and aligning the two geometry conventions by hand, which is exactly what the unexecuted benchmark harness in this repository does.

mbirtorch's counterpart is VCD plus `prox_map`, which is one algorithm and one extension point. An implementation of baselines inside mbirtorch would not copy TIGRE's list. SIRT and OS-SART are each a few lines over the existing forward and back projectors, and CGLS is a short loop that mbirtorch's exact adjoint makes safe, whereas TIGRE must guard it against divergence. Those three would cover most reviewer requests.

### 2. Prior-image constrained reconstruction, PICCS

TIGRE implements PICCS and an ordered-subsets variant. The algorithm takes a prior image and minimizes a weighted sum of the TV of the image and the TV of the difference from the prior, subject to data consistency; the weight `prior_ratio` defaults to 0.2 ([Python/tigre/algorithms/pocs_algorithms.py#L325-L326](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/algorithms/pocs_algorithms.py#L325-L326)), with a dedicated CUDA kernel ([Common/CUDA/PICCS.cu#L274](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Common/CUDA/PICCS.cu#L274)).

This matters directly to this group's research programs. The 4D reconstruction work reconstructs sparse-view gated frames for which a full-data reconstruction exists as a natural prior, and the single-view program has a steady-state CBCT scan as its stated prior. PICCS is the standard published baseline for exactly that setting. mbirtorch has no prior-image term. Its nearest route is the plug-and-play loop through `prox_map`, where the prior enters as a denoiser rather than as a difference-from-prior penalty.

An implementation inside mbirtorch's native machinery would need a new prior term in the VCD update, which the LEAP comparison's item 7 already scoped as the hard route. The cheap route is a proximal-map loop whose agent is a TV-of-difference denoiser against the prior image. That route needs no change to mbirtorch itself and could be validated against TIGRE's PICCS on the same data.

### 3. Per-view flexible geometry

TIGRE's geometry lets almost every parameter vary per view: the three ZYZ Euler angles, the source and detector distances, the center-of-rotation offset, the volume offset, the detector offset, and the detector rotation (`TG-inv` geometries sections 2 and 3). Helper functions compose these into tomosynthesis, static-detector, and fully arbitrary source-path scans ([Python/tigre/utilities/common_geometry.py#L6-L60](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/common_geometry.py#L6-L60)), and demos cover helical, offset, arbitrary-axis, and breast-tomosynthesis cases.

This is TIGRE's equivalent of LEAP's modular geometry, ranked high for the same reason: a lab whose scanner does anything beyond a circular orbit can describe the scan without new code. mbirtorch's cone model supports a per-view angle list and helical translation, and its multi-axis parallel model adds a per-view elevation. A per-view detector pose or source distance has no mbirtorch expression today.

The implementation constraint is the same one the LEAP comparison recorded: mbirtorch's separable-footprint factorization assumes an axis-aligned rotation, so a general per-view pose needs either a new ray model or a restriction to nearly aligned poses. This item is the largest on the list, and the LEAP comparison's item 6 discussion applies unchanged.

### 4. Offset-detector and short-scan weighting in direct reconstruction

TIGRE's FDK applies Wang redundancy weights for laterally offset detectors by default, with zero padding sized to the offset ([Python/tigre/algorithms/single_pass_algorithms.py#L76-L130](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/algorithms/single_pass_algorithms.py#L76-L130)). The MATLAB FDK also detects short scans and applies Parker weights automatically ([MATLAB/Algorithms/FDK.m#L153-L158](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/MATLAB/Algorithms/FDK.m#L153-L158)); the Python side ships the Parker function but hardcodes it off.

Offset-detector scans nearly double the field of view and are common on industrial and clinical CBCT systems. mbirtorch has neither weighting. Its direct reconstruction docstrings already say the result is approximate for short scans, and its iterative solver tolerates redundancy imperfectly weighted data better than FDK does, so the direct reconstruction used as the VCD initializer is where the gap shows.

Both weights are view- or channel-dependent factors applied before the ramp filter. The Wang weight is a per-channel sine ramp and fits mbirtorch's filter path in `mbirtorch/tomography_utils.py`. The Parker weight is view-dependent, which the LEAP comparison already noted does not fit the existing view-independent hook; it needs a small extension of that hook.

### 5. Automatic splitting to fit GPU memory

TIGRE queries free GPU memory at every call, keeps a 5 percent margin, and splits the volume along z and the projection set into chunks until the pieces fit ([Common/CUDA/voxel_backprojection.cu#L714-L753](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Common/CUDA/voxel_backprojection.cu#L714-L753)). The multi-GPU paper demonstrates the consequence: a 40 GB volume reconstructed iteratively on two 11 GiB cards (arXiv:1905.03748). The same mechanism serves every algorithm, because the algorithms are loops over the split operators.

mbirtorch's counterpart is `recon_split_sino`, a manual band split documented as only approximately equal to `recon` ([mbirtorch/tomography_model.py:372](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L372)). The LEAP comparison ranked the equivalent LEAP capability fifth and scoped the implementation: a pixel-axis split inside `mbirtorch/projectors.py`, which a TODO in that file already names. That scoping applies here unchanged, including its limit: the projectors and direct reconstruction can split this way, and the VCD solver cannot without changing its fixed point, because it keeps one error sinogram resident.

### 6. Scatter correction

TIGRE's Varian pipeline includes detector point-scatter correction and a kernel-based object-scatter estimator, on by default for that loader, implementing the adaptive scatter kernel superposition method of Sun and Star-Lack ([Python/tigre/utilities/io/varian/scatter.py#L354](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/io/varian/scatter.py#L354)). The correction is tied to the Varian loader rather than exposed as a general function.

mbirtorch has no scatter correction of any kind, which the LEAP comparison ranked fourth among LEAP's advantages. Scatter is a dominant artifact source for the wide cone angles and dense objects of industrial CBCT. TIGRE's version is less general than LEAP's physics-based model but is a working reference implementation of the kernel method. The cheap first step recorded in the LEAP comparison stands: a constant transmission offset in `mbirtorch/preprocess/utilities.py`, then a kernel estimator patterned on TIGRE's, which needs no spectrum.

### 7. Built-in noise simulation

TIGRE simulates measurement noise in one call: `CTnoise.add(projections, Gaussian, Poisson)` applies Poisson and Gaussian noise with a CUDA random-number kernel ([Python/tigre/utilities/CTnoise.py#L4-L25](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/CTnoise.py#L4-L25)). Neither LEAP nor mbirtorch has a built-in noise model; every mbirtorch simulation study in this repository writes its own Poisson block, including the LEAP quality benchmark.

The value is standardization rather than capability. A `add_ct_noise(sinogram, counts_per_pixel, gaussian_sigma)` in `mbirtorch/utilities.py` would be a small function, would remove a copy-pasted block from every experiment script, and would fix the counts convention in one place.

### 8. More vendor data loaders

TIGRE's six Python loaders read Nikon, Bruker SkyScan, YXLON, Diondo, DXChange synchrotron HDF5, and Varian clinical CBCT data (`TG-inv` preprocess section 1). mbirtorch's four readers cover NSI, Zeiss Versa and Ultra, Zeiss translation, and ORNL HDF5. The two sets overlap only in synchrotron-style HDF5.

The value of each loader is proportional to the user base that owns that scanner, so this item is a menu rather than a single feature. Nikon and Bruker are the most common lab micro-CT vendors in TIGRE's set. TIGRE's loaders are plain metadata parsers over PIL and numpy, and each returns projections, geometry, and angles, so porting one to an mbirtorch `get_sino_and_model` entry is mechanical.

### 9. A per-iteration quality hook

Every TIGRE algorithm accepts `Quameasopts` and then logs RMSE, normalized RMSE, correlation, SSIM, UQI, or squared distance between consecutive iterates at every iteration ([Python/tigre/utilities/Measure_Quality.py#L26-L33](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/Measure_Quality.py#L26-L33)). The MATLAB side can compute the same metrics against a supplied ground truth instead ([MATLAB/Algorithms/SART.m#L67-L71](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/MATLAB/Algorithms/SART.m#L67-L71)).

mbirtorch records its own relative-change statistic for the stopping rule, and nothing else per iteration. Convergence studies in this repository, including the LEAP quality benchmark, re-run the solver at increasing iteration counts to trace quality against iterations, which multiplies the compute by the number of trace points. A callback hook on `recon` that receives the iterate would produce the same curves in one run.

### 10. A geometry sketch

`plot_geometry(geo, angle)` draws the source, detector, volume, and axes for a chosen view, and `plot_angles` animates the trajectory ([Python/tigre/utilities/visualization/plot_geometry.py#L65](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/visualization/plot_geometry.py#L65)). LEAP has the same capability, and the LEAP comparison listed it without ranking it. It earns a rank here because both other packages have it and mbirtorch does not, and because a wrong sign in a detector offset is far faster to see in a sketch than in a blurry reconstruction.

### Lower value

Several further TIGRE capabilities are absent from mbirtorch and do not justify a subsection:

- The interpolated forward projector as a second ray model, with `geo.accuracy` controlling the sample rate ([Python/tigre/utilities/Ax.py#L29](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/Ax.py#L29)). mbirtorch deliberately has one model per geometry.
- Ordered-subset ordering strategies, including a greedy angular-distance order ([Python/tigre/utilities/order_subsets.py#L58-L76](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/order_subsets.py#L58-L76)). These attach to view-subset algorithms, which mbirtorch does not have.
- The curved-detector flattening utility ([Python/tigre/utilities/curved_detector.py#L4](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/curved_detector.py#L4)); mbirtorch models curved detectors natively.
- Proton CT projectors and demos, which serve a user base this group does not have.
- The FISTA automatic Lipschitz estimate by power method ([ista_algorithms.py#L140-L160](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/algorithms/ista_algorithms.py#L140-L160)); VCD has no step size to estimate.

Three capabilities would rank higher were they not MATLAB-only, since a pytigre user does not get them either: the DICOM loader, the dual-energy decomposition module, and the `computeCOR` center-of-rotation estimator (`TG-inv` preprocess sections 1, 3, and 4).

### Already covered

Several capabilities TIGRE advertises are already present in mbirtorch: helical cone beam, curved detectors, non-uniform angular spacing, multi-GPU execution, differentiable projectors, non-negativity constraints, a head-phantom-style set of test volumes, and Shepp-Logan phantom generation. Two near matches deserve a note. TIGRE's per-view `offOrigin` can express a translated-object scan, so mbirtorch's translation model is a dedicated implementation of something TIGRE can emulate. TIGRE's 2D mode covers fan-beam data the same way mbirtorch does, as a one-row cone geometry, and neither package has a dedicated fan-beam class.

---

## mbirtorch strengths relative to TIGRE

### 1. An exact adjoint pair, checked automatically

mbirtorch's back projector is the transpose of its forward projector, the test suite checks the inner-product identity at 1e-4 relative tolerance ([tests/test_adjoint.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/tests/test_adjoint.py)), and each Triton kernel is checked against its compiled PyTorch equivalent at run time ([mbirtorch/kernel_availability.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/kernel_availability.py)).

TIGRE's default pair is unmatched by design, its best option is named pseudo-matched in its own source ([MATLAB/Utilities/cuda_interface/Atb_mex.cpp#L104](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/MATLAB/Utilities/cuda_interface/Atb_mex.cpp#L104)), and no adjointness test exists in the repository (`TG-inv` projectors section 5). The consequences appear in TIGRE's solvers: CGLS restarts on divergence, and two GMRES variants exist to remain stable under a non-adjoint backprojector. For a long iterative loop, and for any use of the projectors as autograd operators, the exact pair is the property that guarantees convergence and correct gradients.

### 2. A statistical objective with automatic parameters and a stopping rule

mbirtorch minimizes a weighted data term plus a qGGMRF prior, sets the noise and prior parameters from `sharpness` and `snr_db`, and stops on a relative-change threshold ([mbirtorch/tomography_model.py:3130](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L3130)). TIGRE's least-squares algorithms take no measurement weights, their TV strengths are set by hand, and all but the ASD-POCS family run to a fixed iteration count (`TG-inv` iterative-algos sections 1 and 9). The practical difference is who does the tuning. The LEAP quality study measured what this automation is worth on that comparison: the default stopping rule ended runs at 9 to 12 iterations at or below the quality target at every size. The same study has not been run against TIGRE.

### 3. GPU-resident multi-GPU sharding with a memory model

mbirtorch holds the volume and sinogram sharded across GPUs for the life of a model, chooses the GPU count from measured floors and a memory model, and accepts and returns sharded tensors ([docs/source/dev_sharding_overview.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/dev_sharding_overview.rst)). TIGRE's interface is host arrays on every call. Its multi-GPU path splits each call internally and streams chunks through the cards, and page-locks host memory to overlap the copies ([Common/CUDA/Siddon_projection.cu#L343-L349](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Common/CUDA/Siddon_projection.cu#L343-L349)). Every TIGRE iteration therefore moves the projections and the volume over PCIe at least twice. The LEAP comparison measured the same architectural difference at N = 1024, where LEAP's four-GPU reconstruction was slower than its one-GPU reconstruction for this reason; whether TIGRE shows the same behavior is one of the questions the prepared benchmark would answer.

### 4. Native PyTorch integration with exact gradients

mbirtorch's `TorchProjector` runs on GPU-resident tensors, and its backward pass is the exact adjoint ([mbirtorch/autograd.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/autograd.py)). TIGRE's torch wrapper copies every tensor to host numpy and back per call, loops over batch elements in Python, and computes its gradient with the FDK-weighted backprojector rather than the transpose ([Python/tigre/utilities/pytorch_bindings.py#L39](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/pytorch_bindings.py#L39), [#L62-L64](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/utilities/pytorch_bindings.py#L62-L64)). The committed test for those bindings imports a module that does not exist in the tree, so the bindings are untested as shipped ([Python/tigre/tests/utilities/pytorch_bindings_test.py#L9](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/Python/tigre/tests/utilities/pytorch_bindings_test.py#L9)). For training loops, both the transfer cost and the inexact gradient matter.

### 5. Installation and platform reach

mbirtorch installs from PyPI with no compiler and runs on CUDA, CPU, and Apple MPS (`MT-inv` sections 1 and 6). TIGRE requires an NVIDIA GPU and has no CPU code path at all: the build fails without nvcc, and the import fails without the compiled extensions (`TG-inv` identity section 7). TIGRE's current binary route is the community ccpi conda channel for Linux and Windows; PyPI offers only the 2019 `pytigre` sdist, and the name `tigre` on PyPI belongs to an unrelated genomics tool (`TG-inv` addendum). A student on a laptop can run mbirtorch and cannot run TIGRE unless the laptop has an NVIDIA GPU and a CUDA toolchain.

### 6. A proximal map and a sharded denoiser for plug-and-play priors

mbirtorch exposes `prox_map` and a standalone qGGMRF denoiser, both accepting data already split across GPUs ([mbirtorch/tomography_model.py:3284](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/tomography_model.py#L3284); [mbirtorch/denoising.py](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/mbirtorch/denoising.py)). TIGRE has no proximal-map entry point and no swappable denoiser hook; its FISTA hardcodes the TV proximal step, and its MATLAB denoiser errors on any method other than TV (`TG-inv` regularizers section 6). A learned prior therefore has a supported route in mbirtorch and none in TIGRE.

### 7. Automated testing and continuous integration

mbirtorch has 769 test functions across 37 files, run in CI on four Python versions with documentation warnings treated as errors (`MT-inv` section 8). TIGRE has about 21 Python test functions plus a shell-driven RMSE matrix that requires a GPU, one committed test file that is Python 2 and cannot run, and no MATLAB tests; its CI builds wheels and an sdist and executes no tests (`TG-inv` api-docs sections 5 and 6). TIGRE's demos are good teaching material, and 24 Python demos against mbirtorch's 9 is a real TIGRE advantage, but demos are not checks.

### 8. Standalone preprocessing corrections

mbirtorch's beam-hardening correction, metal-artifact weighting loop, stripe removal, and outlier correction are package functions that apply to any dataset (`MT-inv` section 5). TIGRE's corrections are welded to the Varian loader, its Python side has no beam-hardening correction at all, and no metal-artifact code exists in the package (`TG-inv` preprocess section 3; grep recorded above). For the industrial scans this group works with, the corrections are the pipeline.

### 9. Result export and unit bookkeeping

mbirtorch writes HDF5 with geometry metadata, and its readers record the physical size of the internal length unit so results can be reported in physical units ([docs/source/unit_conversion.rst](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/docs/source/unit_conversion.rst)). TIGRE ships no file writers; a search of both sides found none (`TG-inv` preprocess section 8). TIGRE works in millimeters throughout, which is simple and sufficient, but the user assembles their own output pipeline.

### 10. An API reference

mbirtorch has 30 documentation pages with an API reference built from docstrings and checked in CI. TIGRE states that its demos are its documentation ([README.md#L96](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/README.md#L96)); its readthedocs site renders about ten demos, and the repository's readthedocs configuration points at a Sphinx directory that does not exist in the tree (`TG-inv` api-docs section 4). A user who wants the semantics of an argument reads TIGRE's source.

---

## Project health

| Item | TIGRE | mbirtorch | Source |
| --- | --- | --- | --- |
| License | BSD 3-Clause | BSD 3-Clause | [LICENSE.txt](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/LICENSE.txt); [LICENSE](https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/LICENSE) |
| Institutions | University of Bath and CERN | Purdue University | [README.md#L128](https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/README.md#L128); `MT-inv` section 1 |
| First release | 2016 | 2026 | `TG-inv` identity section 3; `MT-inv` section 1 |
| Latest release | v3.1.3, 2026-03-06; master is 77 commits ahead | v0.0.2, 2026-08-21 | `TG-inv` identity section 3; `MT-inv` section 1 |
| Last push | 2026-09-02 | 2026-08-29 | `TG-inv` web section 1; `MT-inv` section 1 |
| Stars | 801 | 0 | GitHub API, read 2026-09-03 |
| Contributors | 13 credited; top committer has 709 commits | 6 | `TG-inv` web section 1; `MT-inv` section 1 |
| Issues | 472 total, 83 open | 0 of 0 | GitHub API search, read 2026-09-03 |
| Binary distribution | ccpi conda channel, Linux and Windows | PyPI | `TG-inv` web section 2; `MT-inv` section 1 |

TIGRE is a mature, active, community-maintained project. It has ten years of releases, hundreds of closed issues, current commits, and multiple institutional contributors. mbirtorch is one month old as a public package. The comparison is therefore between an established toolbox with organic growth and a new package with systematic engineering. TIGRE's age shows in both directions: broad capability and a large user base on one side, and stale packaging metadata, a stale changelog, dead test files, and MATLAB-to-Python porting gaps on the other.

---

## Sources

The following sources support every claim above:

1. `plans/features/tigre_comparison/tigre_comparison_sources/tigre_inventory.md`, the sourced TIGRE inventory prepared for this comparison
2. `plans/features/leap_comparison/leap_comparison_sources/mbirtorch_inventory.md`, the sourced mbirtorch inventory, shared with the LEAP comparison
3. https://github.com/CERN/TIGRE/tree/51ae1a02e070888c8aca481e41157d54b20f692f , the TIGRE tree that every TIGRE reference above is pinned to
4. `https://github.com/CERN/TIGRE/blob/51ae1a02e070888c8aca481e41157d54b20f692f/` , the base URL that TIGRE file references are written against
5. `https://github.com/cabouman/mbirtorch/blob/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2/` , the base URL that mbirtorch file references are written against
6. https://github.com/cabouman/mbirtorch/tree/26bd0ea988bd83e99e8e4cbe2fa8223ac4d104d2 , the mbirtorch source at the compared commit
7. `plans/features/leap_comparison/leap_comparison.md`, whose mbirtorch columns share this document's mbirtorch pin
8. Biguri, Dosanjh, Hancock, Soleimani, "TIGRE: a MATLAB-GPU toolbox for CBCT image reconstruction", Biomedical Physics & Engineering Express 2(5) 055010, 2016, https://iopscience.iop.org/article/10.1088/2057-1976/2/5/055010
9. Biguri and coauthors, "Arbitrarily large iterative tomographic reconstruction on multiple GPUs using the TIGRE toolbox", Journal of Parallel and Distributed Computing, 2020, https://arxiv.org/abs/1905.03748
10. Biguri and coauthors, "TIGRE v3: Efficient and easy to use iterative computed tomographic reconstruction toolbox for real datasets", Engineering Research Express 7, 015011, 2025, https://arxiv.org/abs/2412.10129
11. https://api.github.com/repos/CERN/TIGRE and the GitHub issue-search API, read 2026-09-03
12. https://pypi.org/pypi/pytigre/json and https://pypi.org/pypi/tigre/json , read 2026-09-03
13. https://anaconda.org/ccpi/tigre , read 2026-09-03
14. https://tigre.readthedocs.io/en/latest/ , read 2026-09-03
15. `mbirtorch_metrics/results/gpu/prerelease/regression_gpu_20260827T175529Z_26bd0ea9_table.yaml`, mbirtorch's recorded H100 numbers
16. `plans/experiments/features/tigre_comparison/bench_tigre_vs_mbirtorch.py`, the prepared and unexecuted benchmark harness
17. `plans/experiments/features/tigre_comparison/tigre_cmp_gautschi.sbatch` and `tigre_cmp_multigpu_gautschi.sbatch`, the prepared job scripts
18. `plans/projector_kernels/headroom_appendices/appendix_ct_kernel_practice.md`, the 2026-07-12 kernel survey, which classifies TIGRE's projector taxonomy but contains no TIGRE throughput measurement

One earlier record in this repository mentions TIGRE: the kernel survey at source 18 lists TIGRE's forward and back projector types in its taxonomy table. It contains normalized throughput numbers for other packages and none for TIGRE, so no number from it appears above.
