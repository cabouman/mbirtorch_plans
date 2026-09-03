# MBIRTorch inventory (for comparison against LEAP)

Prepared 2026-09-02. Everything below was read in this session from local files.
Every bullet names its source path in parentheses. Numbers are copied verbatim
from the file named beside them; where a number is not recorded anywhere I read,
the bullet says "not recorded".

Package under review: `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch`,
branch `greg_dev`, commit `26bd0ea`, dated 2026-08-27 (git status at session start).

---

## 1. Identity, license, release history

- Name: mbirtorch. Description: "High-performance tomographic reconstruction using PyTorch"
  (`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/pyproject.toml`).
- Version 0.0.2 (`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/mbirtorch/__init__.py`, `__version__ = "0.0.2"`).
- License: BSD 3-Clause, "Copyright (c) 2024, Charles A. Bouman and Gregery T. Buzzard"
  (`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/LICENSE`).
- Requires Python >= 3.11 and torch >= 2.13
  (`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/pyproject.toml`).
- Runtime dependencies: torch, numpy, matplotlib, h5py, scipy, tifffile, tqdm,
  pywavelets, osqp, opencv-python, olefile, gdown, scikit-learn, psutil
  (`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/pyproject.toml`).
- On PyPI: package `mbirtorch`, latest version 0.0.2, releases `['0.0.1', '0.0.2']`,
  requires_python `>=3.11` (read from https://pypi.org/pypi/mbirtorch/json in this session).
- Git tags and dates: `v0.0.1rc1` 2026-08-12, `v0.0.1rc2` 2026-08-12, `v0.0.1` 2026-08-13,
  `v0.0.2` 2026-08-21 (`git for-each-ref` in the mbirtorch repo).
- GitHub releases: MBIRTorch v0.0.2 (2026-08-21T23:17:57Z), v0.0.1 (2026-08-13T18:18:04Z),
  plus two pre-releases (`gh release list -R cabouman/mbirtorch`).
- Repository is public, created 2026-08-04, last pushed 2026-08-29
  (`gh repo view cabouman/mbirtorch`). So the package is about one month old.
- Total commits on the current branch: 198. Contributors by commit count:
  Greg Buzzard 101, Charles Bouman 87, Charles A Bouman 7, gbuzzard 2, buzzard 1,
  github-actions[bot] 1 (`git shortlog -sn --all`).
- Credited development team: Charles A. Bouman, Gregery T. Buzzard. Sponsors listed:
  Eli Lilly Company, Oak Ridge National Laboratory, The Showalter Trust
  (`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/docs/source/credits.rst`).
- MBIRTorch is described as a PyTorch port of MBIRJAX
  (`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/docs/source/credits.rst`).
- GitHub issue tracker is enabled but contains zero issues, open or closed
  (`gh issue list -R cabouman/mbirtorch --state all`). So there is no public issue backlog.

---

## 2. Geometries supported

- Four geometry model classes are exported: `ParallelBeamModel`, `ConeBeamModel`,
  `TranslationModel`, `MultiAxisParallelModel` (and the alias `MultiAxisParallelBeamModel`)
  (`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/mbirtorch/__init__.py`).
- Parallel beam. Detector row r maps to recon slice r. Default
  `delta_voxel = delta_det_channel = 1 ALU`; slice spacing is fixed equal to detector row
  spacing (`docs/source/usr_parallel_beam_model.rst`).
- Cone beam. Constructor takes `sinogram_shape`, `angles`, `source_detector_dist`,
  `source_iso_dist`, and optionally `helical_z_shifts` and `use_curved_detector`
  (`mbirtorch/cone_beam.py` lines 302-338).
  - Flat panel is the default; `use_curved_detector=True` gives "a cylindrical detector of
    radius source_detector_dist" (`mbirtorch/cone_beam.py`).
  - Helical scanning is supported through a per-view z shift array; `helical_z_shifts=None`
    means a circular scan (`mbirtorch/cone_beam.py`).
- Translation mode (translation computed tomography, TCT). Each view is a cone beam
  projection of a translated object; the view parameter is a per-view `(t_x, t_y, t_z)`
  translation. Described as "an experimental tomography model in alpha testing"
  (`docs/source/usr_translation_model.rst`; `mbirtorch/translation_model.py` docstring).
- Multi-axis parallel beam. Each view carries an azimuth and an elevation (tilt) angle;
  `angles` is a `(num_views, 2)` array. "Parallel beam laminography is a special case of
  this geometry when there is a constant tilt for all views." At elevation 0 it is
  mathematically equivalent to parallel beam
  (`docs/source/usr_multiaxis_parallel_beam_model.rst`; `mbirtorch/multiaxis_parallel.py`).
  - A warning fires if any elevation exceeds 45 degrees: "This may degrade approximation
    quality" (`mbirtorch/multiaxis_parallel.py` line ~246).
- Laminography: supported only as the constant-tilt case of multi-axis parallel beam.
  There is no separate laminography class (`docs/source/usr_multiaxis_parallel_beam_model.rst`).
- Fan beam: there is no fan-beam model class. `mbirtorch/horizontal_fan.py` is an internal
  shared helper holding the trapezoid "horizontal fan" math that all four geometries use;
  it is not a geometry, is not exported in `__all__`, has no model class, and is not
  separately documented (`mbirtorch/horizontal_fan.py` docstring;
  `mbirtorch/__init__.py` `__all__`). Its three functions (`tap_weights`,
  `fan_forward_batch`, `fan_back_batch`) are imported by parallel_beam.py, cone_beam.py,
  multiaxis_parallel.py, and translation_model.py.
- 2D reconstruction: there is no dedicated 2D API. Sinograms are always 3D
  `(num_views, num_det_rows, num_det_channels)`, so a 2D problem is a one-detector-row
  sinogram (`docs/source/quick_start.rst`). The slice viewer accepts 2D arrays and
  promotes them to 3D for display (`mbirtorch/view_utils.py`).
- Modular / arbitrary geometry: no runtime arbitrary-geometry description (no per-view
  source/detector matrices). New geometries are added by subclassing `TomographyModel`
  and writing a forward body and a back body plus a small amount of plumbing; there is a
  developer page and a code skeleton for this
  (`docs/source/dev_api.rst`; `docs/source/_static/new_model_template.py`).
  The overview says "new geometries can be added by constructing a new class with the
  associated sparse forward and back projection code" (`docs/source/overview.rst`).
- Offset / half-scan handling: `det_channel_offset` sets the center-of-rotation offset in
  ALU and `det_row_offset` the source-to-detector row offset
  (`docs/source/usr_parameters.rst`). There is no Parker weighting or short-scan
  redundancy weighting: "FDK assumes equally spaced views over the full angular range and
  applies no short-scan redundancy weighting" (`mbirtorch/cone_beam.py`, recon_fdk
  docstring, line ~772). The same caveat is stated for the shared direct filter: "The
  pi / num_views factor assumes equally spaced views over the full angular range; for
  nonuniform, limited-angle, or short scans a standalone direct recon is only approximate
  -- prefer `recon()`" (`mbirtorch/tomography_model.py` line ~2316).
- Detector tilt / rotation: there is no model parameter for detector tilt. Detector
  rotation is handled in preprocessing instead, by resampling the sinogram:
  `mbirtorch.preprocess.correct_det_rotation(sino, det_rotation=0.0, ...)`
  (`mbirtorch/preprocess/utilities.py` line 265). The NSI reader computes a detector
  rotation from the scanner file with `calc_det_rotation`
  (`mbirtorch/preprocess/nsi.py` line 616).
- Recon slice offset: `recon_slice_offset` shifts the region of reconstruction up or down
  for cone beam (`docs/source/demos_and_faqs.rst`, FAQ on shifting the ROR).

---

## 3. Projector models and kernels

- Model: separable footprint. Projection is factored into a horizontal fan and (for
  cone, translation, multi-axis) a vertical fan, each applying a trapezoid overlap rule.
  The shared horizontal weight rule is
  `A = weight_scale * clip((W_p_c + 1) / 2 - |n_p - n|, 0, min(1, W_p_c))`,
  "the trapezoid overlap of the projected voxel with detector cell n"
  (`mbirtorch/horizontal_fan.py` docstring).
  The cone vertical weight rule is
  `A = clip((W_p_r + 1) / 2 - |m_p - m|, 0, min(1, W_p_r)) / cos_phi`
  (`mbirtorch/cone_beam.py` docstring).
- Per geometry:
  - Parallel beam: horizontal fan only; the detector row axis rides through unchanged
    (`mbirtorch/parallel_beam.py` docstring).
  - Cone beam: two separable fans; the horizontal fan is parallel beam's with per-pixel
    magnification, the vertical fan maps each slice to a range of detector rows via an
    affine pair (m0, W_p_r) (`mbirtorch/cone_beam.py` docstring).
  - Translation: two separable fans mirroring cone's; the in-plane coordinates are shifted
    rather than rotated, and t_z plays cone's helical-z role
    (`mbirtorch/translation_model.py` docstring).
  - Multi-axis parallel: two separable fans; the detector row coordinate is
    `v = z*cos(el) + y*sin(el)`, and the vertical weight is "a pure interpolation weight
    with a per-view mass-conserving amplitude" whose footprint is the largest of the
    voxel's three projected edges (`mbirtorch/multiaxis_parallel.py` docstring).
- Exact adjoint: yes, by construction. "The forward vertical fan is formulated from the
  DETECTOR side ... matching the back projector by construction so the pair stays exactly
  adjoint" (`mbirtorch/cone_beam.py` docstring). "The forward and back projectors are an
  exact adjoint pair by construction, so each is the correct autograd backward of the
  other" (`docs/source/usr_autograd.rst`). There is an adjointness gate test
  `<Ax, y> == <x, A'y>` with tolerance 1e-4 relative
  (`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/tests/test_adjoint.py`).
- Two kernel layers (`docs/source/dev_projector_kernels.rst`):
  1. "Torch bodies": ordinary PyTorch compiled with `torch.compile`. They run everywhere
     (CPU, MPS, any GPU, every geometry) and are the permanent fallback and the value
     reference.
  2. Hand-written Triton kernels on CUDA. Six kernel slots exist: cone back, cone forward,
     parallel back, parallel forward, multi-axis back, multi-axis forward. Translation has
     no Triton kernel ("translation has none").
- Triton kernel modules: `mbirtorch/triton_cone.py`, `mbirtorch/triton_parallel.py`,
  `mbirtorch/triton_multiaxis.py`. Triton compiles from Python at import time, so there
  is no separate CUDA build step (`docs/source/dev_projector_kernels.rst`).
- Kernel availability is decided by two automatic gates: a process-wide Triton probe that
  compiles and runs a trivial kernel, and a per-kernel per-device value self-check against
  the torch body on a tiny problem, with a 1e-4 relative contract
  (`mbirtorch/kernel_availability.py`; `docs/source/dev_projector_kernels.rst`).
- The parallel forward has a special "sort, then multiply" kernel: it argsorts each view's
  pixels by detector channel, accumulates a tile into a narrow channel window with a
  full-precision matrix multiply, and scatters only the finished window; a tile whose
  sorted span exceeds the window falls back tap by tap
  (`docs/source/dev_projector_kernels.rst`).
- Kernel width arguments are rounded up to a multiple of 16 because Triton compiles a
  faster specialized kernel for multiples of 16
  (`mbirtorch/_utils.py`, `padded_kernel_width`; `docs/source/dev_projector_kernels.rst`).
- Kill switches: `MBIRTORCH_DISABLE_TRITON=1` disables every hand-written kernel;
  `MBIRTORCH_SORTED_FORWARD=0` runs the per-tap parallel forward instead of the sorted one
  (`docs/source/dev_projector_kernels.rst`).
- Precision: float32 only. `_F32 = torch.float32` is used throughout the geometry modules
  and projectors (`mbirtorch/projectors.py`, `mbirtorch/parallel_beam.py`,
  `mbirtorch/cone_beam.py`, `mbirtorch/multiaxis_parallel.py`,
  `mbirtorch/translation_model.py`). A search for `bfloat16` and `float16` across
  `mbirtorch/*.py` returned no hits; float64 appears only in hsnt (numpy stability casts)
  and in comments. So there is no reduced-precision or mixed-precision option, and no
  double-precision projector.
- Voxel shape: voxels can be non-cubic. `delta_voxel` sets the column spacing;
  `voxel_row_aspect` sets the ratio of row spacing to column spacing and
  `voxel_slice_aspect` the ratio of slice spacing to column spacing, both defaulting to
  1.0 (`docs/source/usr_parameters.rst`; `mbirtorch/_utils.py` defaults).
  Parallel beam refuses a non-unit slice aspect: "Setting voxel slice aspect ratio is not
  supported for ..." (`mbirtorch/parallel_beam.py` line 223).
- Recon shape and offset control: `recon_shape` (a 3-tuple of rows, cols, slices),
  `scale_recon_shape(row_scale, col_scale, slice_scale)`, `auto_set_recon_geometry()`,
  `recon_slice_offset`, and the cone-only `axial_pad_fraction`
  (`docs/source/usr_parameters.rst`; `docs/source/advanced_features.rst`;
  `mbirtorch/tomography_model.py` line 3429).
- Region-of-reconstruction mask: `use_ror_mask` defaults to True, giving an elliptical
  mask inscribed in the region of reconstruction; False reconstructs the whole row-column
  space; a user-supplied 2D 0/1 array is also accepted
  (`docs/source/usr_parameters.rst`).
- Driver structure: `mbirtorch/projectors.py` owns the view-batch loop, the transient
  memory budget, per-device compiled instances, and a process-wide compile lock. The
  drivers tile over views only; two-axis (view and pixel) tiling is designed for but not
  built (`mbirtorch/projectors.py` docstring and the TODO at line 340).
- Measured compile wins quoted in the code: "1.7-3.6x (CPU), 5-17x (MPS), and 2.6-22x
  (CUDA), with the fan chain's peak-memory transients collapsing 6-41x"
  (`mbirtorch/projectors.py` lines ~32-36). The docs page states the same as "1.7x on CPU
  to 22x on CUDA" (`docs/source/dev_projector_kernels.rst`).
- Measured kernel-width penalty: "a cone band sweep read a 2.44x penalty across the
  divisibility boundary before the padding and 1.06x after it; the discarded lanes cost
  1.6 to 3 percent", measured 2026-08-18 (`docs/source/dev_projector_kernels.rst`).
- Measured recompile-budget cost: with torch's default budget, "the multi-axis and
  translation back bodies filled it at several two-device problem sizes, and their
  remaining calls ran at 5 to 11 times the compiled device time", measured 2026-08-19 on
  two H100s (`docs/source/dev_projector_kernels.rst`; `mbirtorch/projectors.py`).
- Measured multiaxis width-padding cost: passing the real detector row count as the mask
  bound "cost a factor of 3.1 at every row count that was not a multiple of 16 (measured
  2026-08-24)" (`mbirtorch/_utils.py`, `padded_kernel_width` docstring).
- The multi-axis Triton kernel pair is selected on the correctness gates alone; its tile
  constants were adopted from the cone kernels rather than swept, and no composed
  performance measurement has been made for that geometry
  (`docs/source/dev_projector_kernels.rst`; `mbirtorch/multiaxis_parallel.py`).
- Autograd: `mbirtorch.autograd` exposes `forward_project_differentiable`,
  `back_project_differentiable`, and a `TorchProjector` `nn.Module` wrapper. They run on a
  single device only: "configure the model with `model.configure_devices(1)` before using
  them in training" (`docs/source/usr_autograd.rst`; `mbirtorch/autograd.py`).

---

## 4. Reconstruction algorithms

- Primary method: `TomographyModel.recon()`, Multi-Granular Vectorized Coordinate Descent
  (VCD) minimizing `f(x) + h(x)` with a qGGMRF prior
  (`docs/source/theory.rst`; `mbirtorch/tomography_model.py` line 3130).
- `recon()` signature and defaults: `max_iterations=15`, `stop_threshold_change_pct=0.2`,
  `first_iteration=0`, `logfile_path='~/.mbirtorch/logs/recon.log'`, `print_logs=True`,
  `output_sharded=False` (`mbirtorch/tomography_model.py` line 3130).
- Stop criterion: "stop when `100 * ||delta_recon||_1 / ||recon||_1` between iterations
  drops below this value. Defaults to 0.2; set 0 to guarantee exactly max_iterations"
  (`mbirtorch/tomography_model.py`, recon docstring).
- Initialization: if `init_recon` is None, `recon_direct` is called with default arguments
  (that is, FBP for parallel, FDK for cone and translation, stacked 2D FBP for multi-axis)
  (`mbirtorch/tomography_model.py`, recon docstring).
- Restart / checkpoint: setting `first_iteration` to the number of completed iterations
  and `init_recon` to the previous output continues the same partition sequence
  (`mbirtorch/tomography_model.py`, recon docstring).
- Reproducibility: pixel partitions come from numpy's global RNG, so results vary run to
  run unless `np.random.seed(seed)` is called; results also differ slightly with device
  count (`mbirtorch/tomography_model.py`, recon docstring; `docs/source/usr_multi_gpu.rst`).
  The recorded device-count difference "falls from 6.1e-3 at 3 iterations to 8.8e-4 at 10"
  (`docs/source/usr_multi_gpu.rst`).
- Returned object: `(recon, recon_dict)` where recon_dict holds 'recon_params'
  (per-iteration traces and settings), 'recon_log', 'notes', and 'model_params'
  (`mbirtorch/tomography_model.py`, recon docstring). Per-iteration traces include
  `fm_rmse`, `stop_threshold_change_pct`, `alpha_values`, and `delta_norm_per_slice`
  (`mbirtorch/tomography_model.py`; `mbirtorch/_utils.py` `recon_param_names`).
- Prior: qGGMRF with a 4-point 2D in-plane neighborhood plus a 2-point slice neighborhood
  (`docs/source/theory.rst`). Parameter defaults: `p = 2.0`, `q = 1.2`, `T = 1.0`,
  `sigma_x = 1.0`, `qggmrf_nbr_wts = [1.0, 1.0, 1.0]` (`mbirtorch/_utils.py`).
  Note that the theory page writes the shape constraint as `p < q = 2.0` while the code
  defaults are `p = 2.0, q = 1.2`; the two use the symbols in opposite roles.
- Meta-parameters: `sharpness` (default 1.0, controls `sigma_x`) and `snr_db`
  (default 30.0, controls `sigma_y`) (`docs/source/usr_parameters.rst`;
  `mbirtorch/_utils.py`).
- Other reconstruction parameters: `positivity_flag` (default False), `max_alpha`
  (default 1.5, limits the VCD step size), `verbose` (default 1),
  `auto_regularize_flag` (default True), `use_ror_mask` (default True)
  (`mbirtorch/_utils.py`; `docs/source/usr_parameters.rst`).
- Automatic parameter selection: `auto_set_regularization_params`, `auto_set_sigma_y`,
  `auto_set_sigma_x`, `auto_set_sigma_prox` (`mbirtorch/tomography_model.py`
  lines 2086-2188). The overview claims "built-in automatic parameter selection, so it
  will produce a good reconstruction the first time" (`docs/source/overview.rst`).
- VCD partition schedule defaults: `granularity = [1, 2, 4, 8, 16, 32, 64, 128, 128, 128,
  128]` and `partition_sequence = [2, 4, 6] + [7, 8, 9, 10] * 25`, described in the source
  as "4 independent 128-subset partitions, cycled after warmup (covers 103 iterations;
  last entry repeats after that)" (`mbirtorch/_utils.py`).
- Direct (non-iterative) reconstructions:
  - `ParallelBeamModel.recon_fbp(sinogram, filter_name="ramp")` and `fbp_filter`
    (`mbirtorch/parallel_beam.py` lines 294, 321).
  - `ConeBeamModel.recon_fdk(sinogram, filter_name="ramp")` and `fdk_filter`; FDK is
    "standard filtering, then the exact adjoint of the forward projector as the
    backprojection" (`mbirtorch/cone_beam.py` line 753).
  - `TranslationModel.recon_fdk` / `fdk_filter` (`mbirtorch/translation_model.py`
    lines 397, 432).
  - `MultiAxisParallelModel.recon_fbp`, described as "stacked 2-D FBP"
    (`mbirtorch/multiaxis_parallel.py` lines 409-474).
  - A geometry-neutral `recon_direct` on every model dispatches to the geometry's own
    direct method (`mbirtorch/tomography_model.py` line 342).
- Filters available: exactly one. `generate_direct_recon_filter` has
  `supported_filters = ["ramp"]` and raises on anything else
  (`mbirtorch/tomography_utils.py` line 18). There is no Shepp-Logan, Hann, Hamming,
  cosine, or user-supplied filter.
- Helical FDK: `ConeBeamModel.helical_fdk_z_weight(recon, sinogram)` applies a z weighting
  when the helical z shifts span a nonzero range; the docstring notes FDK "for helical
  scans it is approximate regardless" (`mbirtorch/cone_beam.py` lines 697, 773).
- Cone-beam DC damping preconditioner: a slice damping profile
  `(a, b, p, c) = (0.25, 100.0, 0.7, 0.5)` (called the "C4" preconditioner) is ON by
  default and is not a public parameter; it changes only the trajectory, not the MAP fixed
  point (`mbirtorch/cone_beam.py` lines ~40-48).
- Proximal map / plug-and-play: `TomographyModel.prox_map(prox_input, sinogram,
  sigma_prox=None, weights=None, init_recon=None, do_initialization=True,
  stop_threshold_change_pct=0.2, max_iterations=3, first_iteration=0, ...)`
  (`mbirtorch/tomography_model.py` line 3284). The theory page describes the proximal-map
  prior form and its `sigma_prox` parameter (`docs/source/theory.rst`).
- Denoising: `QGGMRFDenoiser` computes the MAP for additive white Gaussian noise with a
  qGGMRF prior, with `sigma_noise` either supplied or estimated automatically, and a
  `sharpness` knob (default 0.0 for the denoiser)
  (`docs/source/usr_denoising.rst`; `mbirtorch/denoising.py`). Also `median_filter3d`, a
  3x3x3 median filter that can return neighborhood min and max
  (`mbirtorch/denoising.py` line 613).
- Neural-network denoisers: none inside the package. A DRUNet prior and a MACE loop exist
  only as research scripts under
  `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/experiments/drunet/`
  (`mace.py`, `agents.py`, `run_drunet_sweep.py`, `run_fusion_initial.py`,
  `run_fusion_sweep.py`, `run_qggmrf_gate.py`, `cone_beam_2d.py`, `cone_beam_3d.py`,
  `measure_3d_runtimes.py`, `sandbox.py`). They are not importable from the package,
  not in `__all__`, not documented, and not tested.
- Weights: `gen_weights(sinogram, weight_type)` with four options recorded in the theory
  page: `unweighted` (Lambda = 1), `transmission` (Lambda = exp(-y)), `transmission_root`
  (Lambda = exp(-y/2)), `emission` (Lambda = 1/(y + 0.1))
  (`docs/source/theory.rst`; `mbirtorch/vcd_utils.py`).
- Metal artifact reduction: `gen_weights_mar(sinogram, init_recon=None)` produces MAR
  weights (`docs/source/advanced_features.rst`), and
  `TomographyModel.recon_plastic_metal(sino, weights, num_BH_iterations=3, ...)` alternates
  adaptive plastic/metal beam-hardening correction with reconstruction
  (`mbirtorch/tomography_model.py` line 420). It "works on any geometry that provides
  `recon_direct` and `recon`" but "has been used mainly with cone beam models"
  (same docstring). Supporting code: `mbirtorch/preprocess/mar.py` (including
  `gen_huber_weights` and `correct_sino_plastic_metal`, which solves a constrained
  quadratic program via OSQP) and `mbirtorch/preprocess/segmentation.py`
  (`multi_threshold_otsu`, `segment_plastic_metal`).
- Beam hardening: `fit_beam_hardening_curve`, `apply_beam_hardening_curve`,
  `fit_inverse_beam_hardening_curve`, `apply_inverse_beam_hardening_curve`, and a
  `BH_correction` entry point (`mbirtorch/preprocess/utilities.py` lines 1250-1580;
  `docs/source/usr_preprocess.rst`).
- Stripe / ring removal: `remove_all_stripe` (sorting-based, combining small-stripe,
  large-stripe, and dead/fluctuating-stripe removal), `remove_stripe_fw` (wavelet-FFT),
  and `remove_sino_offset` (`mbirtorch/preprocess/stripe.py`;
  `docs/source/usr_preprocess.rst`). Also `interpolate_defective_pixels` and
  `correct_zinger_pixels` (`mbirtorch/preprocess/utilities.py` lines 174, 1582).
- Scatter correction: none found. Searches of `mbirtorch/*.py`,
  `mbirtorch/preprocess/*.py`, and `docs/source/*.rst` for scatter-correction machinery
  returned nothing, and no doc page mentions scatter. Not recorded.
- Positivity constraint: `positivity_flag`, default False (`mbirtorch/_utils.py`;
  `docs/source/usr_parameters.rst`).
- Region-of-interest / partial recon: `recon_shape` and `scale_recon_shape` set the
  reconstructed region; `use_ror_mask` restricts which voxel cylinders are updated;
  `recon_slice_offset` shifts the slab; `demo_3_parallel_roi.py` demonstrates ROI recon
  (`docs/source/usr_parameters.rst`; `docs/source/demos_and_faqs.rst`).
- Memory-limited large reconstruction: `recon_split_sino(sino, weights=None,
  half_overlap=5, ...)` splits the detector rows into overlapping bands, reconstructs one
  band at a time, and stitches. "A cone beam reconstruction splits into two halves; a
  parallel beam reconstruction splits into as many parts as the memory requires"
  (`docs/source/demos_and_faqs.rst`; `mbirtorch/tomography_model.py` line 372).
- Multi-resolution: there is no multi-resolution or coarse-to-fine grid option. What VCD
  varies coarse-to-fine is the pixel PARTITION granularity, not the voxel grid
  (`mbirtorch/_utils.py` `granularity`; `docs/source/theory.rst`).
- Hyperspectral neutron tomography: `mbirtorch/hsnt.py` provides `hyper_denoise`,
  `dehydrate`, `rehydrate`, `import_hsnt_data_hdf5`, `create_hsnt_metadata`,
  `export_hsnt_data_hdf5`, `generate_hyper_data`, citing Chowdhury et al., IEEE TCI,
  vol. 11, pp. 663-677, 2025 (`mbirtorch/hsnt.py`; `docs/source/usr_hsnt.rst`).
  It works on numpy arrays with the spectral axis last and uses NMF plus randomized SVD
  from scikit-learn; it is not a projector or reconstruction method.
- VCLS (view selection): `mbirtorch/vcls.py` provides automated view selection --
  `get_opt_views(ct_model, reference_object, num_selected_views, ...)`,
  `show_image_with_projection_rays`, plus helpers for view basis functions, covariance,
  and the greedy angle-subset search (`mbirtorch/vcls.py`; `docs/source/usr_vcls.rst`).
  A blue-noise sampling pattern ships as a 382 KB array literal in `mbirtorch/bn256.py`
  (`mbirtorch/__init__.py`).

---

## 5. Preprocessing

- Scanner readers, each with a one-call `get_sino_and_model(dataset_dir, ...)` that loads
  a scan, computes its sinogram, and returns a ready-to-reconstruct model
  (`docs/source/usr_preprocess.rst`):
  - NorthStar Instruments (NSI): `mbirtorch/preprocess/nsi.py`.
  - Zeiss Versa and Ultra: `mbirtorch/preprocess/zeiss.py`. The reader picks
    `ParallelBeamModel` for an Ultra scan and `ConeBeamModel` for a Versa scan
    (`docs/source/usr_preprocess.rst`). Zeiss/Xradia OLE containers are parsed by
    `mbirtorch/preprocess/_xradia_ole.py`; `.xrm` and `.txrm` files are read.
  - Zeiss translation tomography: `mbirtorch/preprocess/zeiss_tct.py`.
  - PYMBIR / ORNL HDF5: `mbirtorch/preprocess/pymbir.py`, with
    `create_proj_params_dict_ornl` and `load_projection_data_ornl`.
- No other vendor readers were found (no Bruker/Skyscan, no Nikon, no GE, no Thermo,
  no generic TIFF-plus-parameter-file reader beyond `read_tif_stack_dir`).
- Flat/dark correction: `compute_sino_transmission(obj_scan, blank_scan, dark_scan, ...)`
  and `scan_to_sino(...)` (`mbirtorch/preprocess/utilities.py` lines 51, 473).
- Defective pixel handling: `interpolate_defective_pixels(sino, defective_pixel_array=(),
  num_passes=3)` and `correct_zinger_pixels(sino, zinger_pixel_ratio=0.1, num_passes=3,
  ...)` (`mbirtorch/preprocess/utilities.py` lines 174, 1582).
- Background / offset: `correct_background_offset(sino, edge_width=9, option='global')`
  and `remove_sino_offset(sino)` (`mbirtorch/preprocess/utilities.py` line 295;
  `mbirtorch/preprocess/stripe.py` line 379).
- Geometry calibration and center-of-rotation estimation: `estimate_sino_view_offset(
  ct_model, sino, recon_direct)` and `align_sino_views(ct_model, sino, recon_direct)`
  (`mbirtorch/preprocess/utilities.py` lines 1072, 1205). View alignment uses an OpenCV
  ECC estimator (`pyproject.toml` comment: "opencv-python provides the view-alignment ECC
  estimator and Gaussian filtering"). Supporting: `sino_high_pass_filtering`
  (`mbirtorch/preprocess/utilities.py` line 1123).
  Note: this is a view-offset / view-alignment estimator; there is no full automated
  geometry calibration (no source position, tilt, or magnification solver).
- Detector rotation correction: `correct_det_rotation(sino, det_rotation=0.0,
  batch_size=30, devices=None)` (`mbirtorch/preprocess/utilities.py` line 265).
- Downsampling and cropping: `downsample_view_data`, `crop_view_data`,
  `detect_blank_margins`, `apply_detector_crop`, `apply_config_crop`, `finalize_model`
  with an `auto_crop` option (`mbirtorch/preprocess/utilities.py` lines 415-1071).
  `get_sino_and_model` takes `downsample_factor` and `subsample_view_factor`
  (`mbirtorch/preprocess/nsi.py` line 13; `mbirtorch/preprocess/zeiss.py` line 16).
- Masking: `apply_cylindrical_mask(recon, radial_margin=0, top_margin=0, bottom_margin=0)`
  (`mbirtorch/preprocess/utilities.py` line 800).
- Unit conversion: `to_alu(value, from_unit, alu_unit)`
  (`mbirtorch/preprocess/utilities.py` line 1235); readers set `alu_unit` and `alu_value`
  so reconstructions carry physical units (`docs/source/usr_parameters.rst`;
  `docs/source/unit_conversion.rst`).
- Preprocessing checkpoint I/O: `save_cone_preprocessing(file_path, sinogram,
  cone_beam_params, optional_params, weights=None)` and `load_cone_preprocessing`
  (`mbirtorch/preprocess/utilities.py` lines 1635, 1690).
- The preprocessing pipeline can run batched over views across devices:
  `mbirtorch/preprocess/pipeline.py` provides `permitted_devices` and
  `map_view_batches(array, kernel, batch_size, desc=None, devices=None)`.
- Example applications live in a separate repository, `mbirtorch_applications`
  (`docs/source/usr_preprocess.rst`).

---

## 6. Compute, devices, and multi-GPU

- Device resolution is automatic: "'auto' -> cuda if available, else mps, else cpu"
  (`mbirtorch/tomography_model.py` line 91). So CUDA, Apple MPS, and CPU are all
  supported. There is no `use_gpu` parameter; the docs record that mbirjax's `use_gpu` was
  deliberately not ported (`docs/source/usr_parameters.rst`, the REPLACED note).
- Multi-device sharding is automatic on CUDA with two or more visible devices, and works
  for all four geometries (`docs/source/usr_multi_gpu.rst`;
  `docs/source/dev_sharding_overview.rst`).
- What is sharded: the reconstruction volume by SLICE (each device holds a contiguous band
  of slices of the voxel cylinders) and the sinogram by VIEW (each device holds a block of
  views and produces all detector rows for them)
  (`docs/source/dev_sharding_overview.rst`).
- Projection between the two layouts is an all-to-all. The forward projection is a
  "cylinder transfer": each view-owner collects a batch of full-height voxel cylinders from
  every slice-owner. The back projection processes one slice-shard at a time and one band
  of slices within that shard (`docs/source/dev_sharding_overview.rst`).
- Uneven splits are allowed: "A device count need not divide the sinogram or the volume
  evenly; the shares then differ in size by at most one view or one slice"
  (`docs/source/usr_multi_gpu.rst`).
- Automatic device policy has two rules, applied in order: "Speed first" (each device
  count carries a measured speed floor in sinogram elements) and "Capacity always wins"
  (a count below its floor is set aside and tried only after every admitted count, so a
  reconstruction that needs the memory still gets it)
  (`docs/source/usr_multi_gpu.rst`; `mbirtorch/_widening_floors.py`).
- The memory ledger (`mbirtorch/_memory_ledger.py`, 82,996 bytes) estimates the memory
  each candidate layout would need before the first large allocation; if no layout fits,
  the run fails immediately naming the shortfall rather than dying mid-run with an OOM
  (`docs/source/usr_multi_gpu.rst`).
- Explicit control: `model.configure_devices(num_devices=n)`,
  `configure_devices(devices=["cuda:0", "cuda:2"])`, `configure_devices(like=other_model)`,
  and the environment variable `MBIRTORCH_NUM_DEVICES`. `configure_devices` skips the
  memory check; the environment variable does not. `MBIRTORCH_WIDENING_GUARD=0` disables
  the speed floors but keeps the automatic search
  (`docs/source/usr_multi_gpu.rst`; `mbirtorch/tomography_model.py` line 947).
- Multi-device CPU is supported for testing: `configure_devices(devices=['cpu']*n)`
  (`docs/source/usr_multi_gpu.rst`).
- One layout is refused: a device that would hold no real data on either axis
  (`docs/source/usr_multi_gpu.rst`; `docs/source/dev_sharding_overview.rst`).
- Multi-node execution is explicitly out of scope. "A reconstruction runs within a single
  process; multi-node execution is out of scope"
  (`docs/source/usr_multi_gpu.rst`; `docs/source/dev_sharding_overview.rst`).
- DTensor (torch's logically global sharded array) "was deliberately rejected as immature
  for the index-heavy kernels in MBIRTorch"; a plain `Shards` container is used instead
  (`docs/source/dev_sharding_overview.rst`).
- Memory levers: `back_project_slice_band` streams the back projection's slice axis in
  smaller pieces; `forward_project_pixel_batch` sets how many voxel cylinders move between
  devices at once (default `FORWARD_PIXEL_BATCH = 32768`, "the measured knee" from a
  2026-08-17 four-H100 sweep at the 2048-class cone and parallel cells)
  (`docs/source/usr_multi_gpu.rst`; `mbirtorch/tomography_model.py` lines ~46-59).
- Keep-on-device path: `prepare_sino_for_devices(sinogram, weights=None)` distributes a
  sinogram once, and `output_sharded=True` on `recon`, `recon_fbp`, `recon_fdk`,
  `prox_map`, and `denoise` returns the device form with no host gather
  (`docs/source/usr_multi_gpu.rst`).
- torch.compile: bodies are compiled and cached per (function, device index); a compile
  failure falls back to eager silently but recorded; compile events are serialized
  process-wide behind a lock because concurrent cold compiles crash the compiler stack
  (`docs/source/dev_projector_kernels.rst`; `mbirtorch/projectors.py`).
- The per-function recompile budget is raised to at least 64 before anything compiles
  (`_RECOMPILE_LIMIT_FLOOR = 64`), overridable by `MBIRTORCH_RECOMPILE_LIMIT`
  (`mbirtorch/projectors.py`; `docs/source/dev_projector_kernels.rst`).
- On-disk caches: `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` are pinned under
  `~/.mbirtorch/`, and `mbirtorch.clear_cache()` removes the whole directory. Recorded
  effect: "roughly 14 s down to 2 s for a first small reconstruction"
  (`docs/source/usr_utilities.rst`; `mbirtorch/__init__.py`). The cold Triton compile cost
  recorded in `__init__.py` is "about 1.2 s on one device and 4.8 s on four at the
  1024-class parallel cell".
- Per-device memory reporting: `mbirtorch.get_memory_stats()`
  (`mbirtorch/memory_stats.py`), which reads `torch.cuda` / `torch.mps` allocator stats and
  host RSS/USS via psutil.

---

## 7. Simulation, demos, visualization, I/O

### Phantoms and simulation
- `generate_demo_data(...)` builds a 3D phantom and its simulated sinogram for a chosen
  geometry. `object_type` is `'shepp-logan'` or `'cube'`; `model_type` is `'parallel'`,
  `'cone'`, or `'multiaxis'`; `target_max_attenuation` scales the phantom to a realistic
  peak attenuation (`docs/source/demos_and_faqs.rst`; `mbirtorch/utilities.py` line 1715).
- Other generators: `generate_3d_shepp_logan_low_dynamic_range`,
  `generate_3d_shepp_logan_reference`, `gen_cube_phantom`, `gen_translation_phantom`,
  `gen_dot_phantom`, `gen_text_phantom`, `gen_translation_vectors`, `add_ellipsoid`
  (`mbirtorch/utilities.py`).
- Noise simulation: there is NO built-in `add_noise` function. Demos add noise by hand,
  e.g. `noise_std = np.sqrt(np.exp(sinogram) / dosage)` then adding scaled Gaussian noise
  (`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/demo/demo_2_cone_beam.py`
  lines 31-40). `mbirtorch/hsnt.py` `generate_hyper_data` takes a `dosage_rate` and
  simulates its own noise for hyperspectral data.
- Demo datasets: `download_and_extract(download_url, save_dir)` fetches datasets, with
  gdown for Google Drive-hosted demo data (`mbirtorch/utilities.py` line 653;
  `pyproject.toml`).

### Demos
Nine demo scripts, all in `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/demo/`
(listed with their subjects in `docs/source/demos_and_faqs.rst`):
- `demo_1_parallel_basics.py` -- phantom, project, reconstruct, view.
- `demo_2_cone_beam.py` -- cone beam, simulated noise, noise weighting, saving results.
- `demo_3_parallel_roi.py` -- ROI recon when the object extends outside the field of view.
- `demo_4_cone_axial_fov.py` -- cone-beam artifacts from material above and below the
  field of view, and axial padding.
- `demo_5_direct_vs_mbir.py` -- FBP versus MBIR, including sparse views.
- `demo_6_helical.py` -- helical cone-beam scanning and reconstruction.
- `demo_7_multiaxis.py` -- multi-axis parallel geometry (laminography).
- `demo_8_units_and_voxels.py` -- ALUs, detector spacing, voxel shape,
  `auto_set_recon_geometry()`.
- `demo_9_denoiser.py` -- the qGGMRF denoiser on a noisy 3D image.
The demos are plain Python scripts, not Jupyter notebooks, although the overview page
still says "several demos as Jupyter notebooks and python scripts"
(`docs/source/overview.rst`) -- that sentence does not match the demo folder.

### Slice viewer
- `mbirtorch.slice_viewer(*datasets, ...)` is a matplotlib-based interactive viewer
  (`mbirtorch/view_utils.py`; `mbirtorch/viewer.py`, 103,726 bytes).
- Recorded features: multiple volumes side by side, synchronized slice navigation with
  proportional mapping across volumes of unequal depth, ROI statistics, difference images,
  axis transposition, file loading and saving, dynamic intensity range adjustment, a
  right-click context menu of per-image actions, a help overlay on 'h', and per-image data
  dicts rendered as text (`mbirtorch/view_utils.py`, slice_viewer docstring).
- The viewer accepts numpy arrays and torch tensors including CUDA and MPS tensors, and 2D
  arrays are promoted to 3D (`mbirtorch/view_utils.py`).
- The viewer module is package-independent by design: it imports only numpy, matplotlib,
  and lazily h5py (`mbirtorch/viewer.py` docstring).
- Viewer file I/O: reads `.npy`, `.npz`, and `.h5`/`.hdf5`, and writes `.h5` only; other extensions
  raise ValueError (`mbirtorch/viewer.py` lines 455-510).
- Matplotlib is not imported by a bare `import mbirtorch`; viewer names resolve lazily
  (`mbirtorch/__init__.py`).

### I/O
- HDF5: `save_data_hdf5`, `load_data_hdf5`, `export_recon_hdf5`, `import_recon_hdf5`,
  plus `TomographyModel.save_recon_hdf5(filepath, recon, recon_dict)` and the static
  `TomographyModel.load_recon_hdf5(filepath)` (`mbirtorch/utilities.py`;
  `mbirtorch/tomography_model.py` lines 3582, 3614).
- `export_recon_hdf5` has a `remove_flash` option with radial, top, and bottom margins
  (`mbirtorch/utilities.py` line 533).
- TIFF: `read_tif_img` and `read_tif_stack_dir` for reading scan data
  (`mbirtorch/preprocess/utilities.py` lines 642, 661). No TIFF WRITER was found.
- Model parameter save/load: there is no `to_file` / `from_file` pair (a search for those
  names across `mbirtorch/*.py` and `docs/source/*.rst` returned nothing). Parameters
  round-trip instead through `get_all_params()` plus `build_model(required_params,
  optional_params, regularization)` and `get_ct_model(geometry_type, sinogram_shape, ...)`
  (`mbirtorch/utilities.py` lines 611, 1257), and a model-parameter snapshot rides inside
  the recon HDF5 file (`mbirtorch/tomography_model.py`, `get_recon_dict` /
  `save_recon_hdf5`).
- `copy_ct_model(ct_model, new_angles=None, ...)` clones a model with changed view
  parameters or detector size (`mbirtorch/utilities.py` line 1001).
- `stitch_arrays(array_list, overlap, axis=2, ramp_overlap=None)` joins overlapping
  reconstruction bands (`mbirtorch/utilities.py` line 856).
- Unit conversion is documented on its own page (`docs/source/unit_conversion.rst`),
  covering the ALU convention and worked NSI, Zeiss, and emission-CT examples.

---

## 8. API ergonomics, documentation, tests, CI

- Minimal example, two lines:
  ```python
  import mbirtorch
  recon, recon_dict = mbirtorch.recon_simple_parallel(sinogram, angles)
  ```
  with `recon_simple_cone` the cone-beam counterpart
  (`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/README.md`;
  `docs/source/quick_start.rst`).
- Object-oriented path: `ct_model = mbirtorch.ParallelBeamModel(sinogram.shape, angles)`
  then `recon, recon_dict = ct_model.recon(sinogram)` (`docs/source/quick_start.rst`).
- Array conventions: sinogram is `(views, detector rows, detector channels)` in raster
  order; angles in radians; the reconstruction is `(rows, columns, slices)`
  (`docs/source/quick_start.rst`). The public API takes numpy arrays and returns numpy
  arrays by default; `output_sharded=True` returns the device tensor
  (`mbirtorch/__init__.py` module docstring).
- Parameter access is uniform: `set_params(**kwargs)`, `get_params(names)`,
  `print_params()`, `get_all_params()` (`mbirtorch/parameter_handler.py`;
  `docs/source/usr_tomography_model.rst`).
- Declared public surface: `__all__` in `mbirtorch/__init__.py` lists 40 names. The rule
  recorded in the docs is "documented if and only if declared"
  (`docs/source/_pending/README.rst`).
- Lazy imports: `preprocess`, `hsnt`, `vcls`, `bn256`, and the viewer names resolve on
  first attribute access, so a bare `import mbirtorch` does not pay for matplotlib, osqp,
  cv2, tifffile, or scikit-learn (`mbirtorch/__init__.py`).
- Documentation pages (all under
  `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/docs/source/`):
  index, overview, quick_start, advanced_features, theory, credits, install,
  unit_conversion, usr_api, usr_api_overview, usr_multi_gpu, demos_and_faqs,
  usr_geometry_models, usr_parallel_beam_model, usr_cone_beam_model,
  usr_translation_model, usr_multiaxis_parallel_beam_model, usr_tomography_model,
  usr_parameters, usr_preprocess, usr_utilities, usr_denoising, usr_autograd, usr_hsnt,
  usr_vcls, dev_performance_dashboard, dev_sharding_overview, dev_projector_kernels,
  dev_api, dev_maintenance. That is 30 pages plus a `_pending/README.rst`.
  Published at https://mbirtorch.readthedocs.io/ (`README.md`, `.readthedocs.yaml`).
- Tests: 769 test functions across 37 files in `tests/`, plus 13 in `ci/`, counted by
  `grep -c "^\s*def test_"` over
  `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/tests/*.py` and `ci/*.py`.
  Many use `pytest.mark.parametrize`, so the executed case count is higher than 769; the
  exact collected count was not measured (the test suite was not run, per instruction).
- Largest test files by test count: `test_device_policy.py` (93),
  `test_viewer_controller.py` (89), `test_memory_ledger.py` (87), `test_viewer_model.py`
  (60), `test_sharding.py` (50), `test_widening_floors.py` (30), `test_triton_parallel.py`
  (28), `test_triton_multiaxis.py` (28), `test_triton_cone.py` (24).
- Golden-value parity tests against mbirjax exist but are opt-in: `addopts = "-m 'not
  goldens'"`, and they "will retire once mbirjax freezes"
  (`pyproject.toml`; `tests/test_vs_goldens.py`; `tests/generate_goldens.py`).
- CI: GitHub Actions. Tests run on pull requests to `prerelease` and `main`, on
  ubuntu-latest with CPU torch, across Python 3.11, 3.12, 3.13, and 3.14; the docs job
  builds Sphinx with `-W` (warnings as errors) on Python 3.12
  (`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/.github/workflows/ci.yml`;
  `.github/python-versions.json`). There is also `dependency_watch.yml` and `release.yml`.
  Note: CI is CPU-only, so the CUDA and Triton paths are not covered by CI.
- Performance is tracked by a companion project with a published dashboard at
  https://gbuzzard.github.io/mbirjax_metrics/ , where mbirtorch appears on the
  `cpu-torch` and `gpu-torch` platforms next to mbirjax
  (`docs/source/dev_performance_dashboard.rst`).

---

## 9. Recorded performance carried inside the package and its docs

The measured time series from the metrics repositories is section 14. The numbers here are the ones
carried inside the mbirtorch package and its documentation.

### 9.1 Multi-GPU speed, from the user documentation
Measured on H100 for a parallel-beam model, over a warm 3-iteration reconstruction
(`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/docs/source/usr_multi_gpu.rst`):

| Volume | 1 device | 2 devices | 4 devices |
| --- | --- | --- | --- |
| 512 x 448 x 384 | 1.31 s | 1.29 s | 2.10 s |
| 1024 x 1008 x 992 | 21.3 s | 14.2 s | 10.8 s |

The same page states "The large volume improves by 1.5x at two devices and 2.0x at four."
No date or commit is attached to this table on the page.

### 9.2 Measured device-count speed floors carried in the package
Source for all of this subsection:
`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/mbirtorch/_widening_floors.py`.
Hardware: `MEASURED_GPU = 'NVIDIA H100 80GB HBM3 (4 per node)'`.
Protocol: `MEASURED_CONFIG = 'warm median of 3 seeded 3-iteration VCD recons, cold pass
discarded, package-default subset schedule, Triton kernels on, torch.compile auto'`.
All rows are `measured='2026-08-22', commit='c024ec9'` (mg56, job 15435735), except where
the note names another run.

Problem-size classes, as defined in that file:
- 384-class `(384, 336, 288)` = 37,158,912 elements
- 512-class `(512, 448, 384)` = 88,080,384 elements
- 768-class `(768, 672, 576)` = 297,271,296 elements
- 1024-class `(1024, 1008, 992)` = 1,023,934,464 elements

| Family | Count | Floor (sinogram elements) | Floor cell | Against | Losing speedup | Winning speedup | Largest tested |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cone | 2 | 88,080,384 | (512, 448, 384) | 1 | 0.88 at (384,336,288) | 1.31 at (512,448,384) | 297,271,296 |
| cone | 4 | 297,271,296 | (768, 672, 576) | 2 | none measured | 1.15 at (768,672,576) | 1,023,934,464 |
| parallel | 2 | 297,271,296 | (768, 672, 576) | 1 | 1.09 at (512,448,384) | 1.46 at (768,672,576) | 1,023,934,464 |
| parallel | 4 | 1,023,934,464 | (1024, 1008, 992) | 2 | 0.82 at (768,672,576) | 1.40 at (1024,1008,992) | 1,023,934,464 |
| multiaxis | 2 | 88,080,384 | (512, 448, 384) | 1 | 0.86 at (384,336,288) | 1.38 at (512,448,384) | 297,271,296 |
| multiaxis | 4 | 1,023,934,464 | (1024, 1008, 992) | 2 | 1.09 at (768,672,576) | 1.64 at (1024,1008,992) | 1,023,934,464 |
| translation | 2 | 364,800,000 | (256, 950, 1500) | 1 | 0.64 at (256,475,750) | 1.19 at (256,950,1500) | 1,459,200,000 |
| translation | 4 | 1,459,200,000 | (256, 1900, 3000) | 2 | 0.79 at (256,950,1500) | 1.15 at (256,1900,3000) | 1,459,200,000 |
| denoiser | 2 | None (no admission size measured) | None | 1 | 0.93 at (1024,1008,992) | none | 1,023,934,464 |
| denoiser | 4 | None (no admission size measured) | None | 1 | 0.90 at (1024,1008,992) | none | 1,023,934,464 |

Additional speedups quoted verbatim from the same file's notes:
- cone n=2: "0.88x at the 384-class, 1.31x at the 512-class, 1.65x at the 768-class".
- cone n=4: "the 768-class read 1.15x against two devices ... and the 1024-class read 1.64x".
- parallel n=2: "1.09x at the 512-class inside an 11.2 percent spread, which does not count
  as a win; 1.46x at the 768-class, 1.59x at the 1024-class".
- parallel n=4: "0.82x at the 768-class against two devices and 1.40x at the 1024-class".
- multiaxis n=2: "0.86x at the 384-class, 1.38x at the 512-class, 1.72x at the 768-class".
- multiaxis n=4: 4 devices beat 2 at "1.64x with 0.1 percent spread" at the 1024-class
  (mg55, job 15434826); mg56 read "0.70x at the 512-class and a thin 1.09x at the 768-class".
- translation n=2: "0.64x, 1.19x, 1.25x across the production-anchored cells".
- translation n=4: "0.79x at half scale against two devices and 1.15x at the production scan".
- denoiser n=2: "0.73x at the 512-class, 0.87x at the 768-class, 0.93x at the 1024-class"
  (image voxels, since the denoiser's sinogram shape is its image shape).
- denoiser n=4: "0.55x, 0.75x, 0.90x across the probes, measured against n=1".
- The multiaxis n=2 note records that when the Triton kernels landed "the one-device and
  two-device walls fell together (about four-fold; multigpu_findings.md sections 1.45 and
  1.46)".

### 9.3 Other measured numbers carried in the package
- torch.compile chain-level wins: "1.7-3.6x (CPU), 5-17x (MPS), and 2.6-22x (CUDA), with
  the fan chain's peak-memory transients collapsing 6-41x" (`mbirtorch/projectors.py`).
- Cone kernel width padding: "a cone band sweep read a 2.44x penalty across the
  divisibility boundary before the padding and 1.06x after it; the discarded lanes cost
  1.6 to 3 percent", measured 2026-08-18 (`docs/source/dev_projector_kernels.rst`).
- Multiaxis mask-bound padding: "a factor of 3.1 at every row count that was not a multiple
  of 16", measured 2026-08-24 (`mbirtorch/_utils.py`).
- Recompile budget: without the raise, remaining calls "ran at 5 to 11 times the compiled
  device time", measured 2026-08-19 on two H100s (job 15391547)
  (`mbirtorch/projectors.py`; `docs/source/dev_projector_kernels.rst`).
- Forward pixel-batch knee, measured 2026-08-17 on four H100s at the 2048-class cone and
  parallel cells: "forward busy time falling 11 percent from batch 8192 to 16384, 5 more to
  32768, and 2 to 3 more to 65536, with the transferred cylinders under 1.5 GiB at the
  largest batch" (`mbirtorch/tomography_model.py` lines ~48-57).
- Back-projection band memory lever: "a measured 2-device run at the 1024 class saved about
  0.5 GB of per-device peak for about 2 percent more time at the 252-slice band"
  (`docs/source/usr_multi_gpu.rst`).
- Compile-cache cold start: "roughly 14 s down to 2 s for a first small reconstruction"
  (`docs/source/usr_utilities.rst`); Triton cold compile "about 1.2 s on one device and
  4.8 s on four at the 1024-class parallel cell" (`mbirtorch/__init__.py`).
- Projector transient growth without pixel chunking: the single-view slab is
  "3.2 GB at 1024^3, 26 GB at 2048^3", and "past ~1400^3 the knob no longer protects at
  all" (`mbirtorch/projectors.py`, TODO(tuning) block, measured 2026-08-05 on MPS 256^3
  for the calibration, with the slab sizes stated as consequences of the formula).
- Memory rule of thumb in the FAQ: "a 2K x 2K x 2K reconstruction occupies 32GB of memory,
  not counting the sinogram or memory needed for processing"
  (`docs/source/demos_and_faqs.rst`).
- Horizontal fan weight accuracy: "The weights reproduce the golden reference to <= 1.6e-6
  rel-max" (`mbirtorch/horizontal_fan.py`).
- Device-count result difference: "measured to fall from 6.1e-3 at 3 iterations to 8.8e-4
  at 10" (`docs/source/usr_multi_gpu.rst`).

---

## 10. Known limitations and open issues, from the package and its documentation

(Plans-repository items are in section 12.)

- No multi-node execution. "A reconstruction runs within a single process; multi-node
  execution is out of scope"
  (`docs/source/usr_multi_gpu.rst`; `docs/source/dev_sharding_overview.rst`).
- Only one direct-reconstruction filter, "ramp"; anything else raises
  (`mbirtorch/tomography_utils.py`).
- No short-scan / Parker redundancy weighting; FDK "assumes equally spaced views over the
  full angular range" (`mbirtorch/cone_beam.py`). For nonuniform, limited-angle, or short
  scans, a standalone direct recon "is only approximate -- prefer `recon()`"
  (`mbirtorch/tomography_model.py`).
- Helical FDK "is approximate regardless" (`mbirtorch/cone_beam.py`).
- Float32 only; no bfloat16, float16, or float64 projector path (searched
  `mbirtorch/*.py`).
- Parallel beam refuses a non-unit voxel slice aspect ratio: "Setting voxel slice aspect
  ratio is not supported" (`mbirtorch/parallel_beam.py` line 223).
- The differentiable projector wrappers run on a single device only
  (`docs/source/usr_autograd.rst`).
- The projector drivers tile over views only. Above roughly 810^3 the view-batch knob
  forces a batch of 1 and "the SINGLE-view slab keeps growing as N^3 (3.2 GB at 1024^3,
  26 GB at 2048^3): past ~1400^3 the knob no longer protects at all -- needs pixel-axis
  chunking or the planned fused (Triton) kernels that never materialize the gather."
  The comment adds that with detector panels "heading to ~6K x 10K (2026 estimate) ...
  ONE view's slab against a 512-class pixel set is ~6 GB", so this is called near-term
  (`mbirtorch/projectors.py`, TODO(tuning) block).
- The translation geometry has a recorded scale limit: at production TCT detector shapes
  (~1900x3000 panels) "large pixel batches are memory-bound and the view batch shrinks
  accordingly. What would relieve it is a change to the projector drivers, not to this
  file ... tiling over the pixel axis as well ... Nothing here works around its absence"
  (`mbirtorch/translation_model.py` docstring).
- Translation has no hand-written Triton kernel; it runs on the compiled torch bodies
  (`docs/source/dev_projector_kernels.rst`).
- The multi-axis Triton kernels' speed is unmeasured: their tile constants were adopted
  from the cone kernels rather than swept, and "no composed measurement has been made for
  that geometry yet" (`docs/source/dev_projector_kernels.rst`;
  `mbirtorch/multiaxis_parallel.py`).
- Kernel performance is recorded on H100 only: "on a very different GPU the kernels are
  still safe to use but their speed is unmeasured"
  (`docs/source/dev_projector_kernels.rst`).
- Multi-device denoising never wins on speed at any measured size; it spreads only when a
  denoise cannot fit on one device (`docs/source/usr_multi_gpu.rst`;
  `mbirtorch/_widening_floors.py`, denoiser rows).
- Results are not bitwise reproducible across device counts, and partitions are drawn from
  numpy's global RNG (`mbirtorch/tomography_model.py`; `docs/source/usr_multi_gpu.rst`).
- The "sorted/CSR stream slot" is accepted and ignored in every Triton kernel wrapper --
  "not yet built" (`mbirtorch/triton_cone.py` lines 369, 671;
  `mbirtorch/triton_parallel.py` lines 283, 625; `mbirtorch/triton_multiaxis.py`
  lines 462, 805).
- Preprocessing TODOs still in the source:
  - "TODO: adjust detector offsets for asymmetric crops" (`mbirtorch/preprocess/nsi.py`
    line 100).
  - "TODO: Replace with more efficient code that doesn't use a nested loop"
    (`mbirtorch/preprocess/nsi.py` line 595).
  - "TODO: If there is no dark scan available, using an array of all 0s"
    (`mbirtorch/preprocess/zeiss_tct.py` line 161).
  - "TODO: It seems that we need to flip the scan to get the correct object orientation"
    (`mbirtorch/preprocess/zeiss_tct.py` line 168).
  - "TODO: Currently we assume that there is no dark scan for txrm file"
    (`mbirtorch/preprocess/zeiss.py` line 221).
  - "TODO: More metadata will be added in the future based on users' interest"
    (`mbirtorch/preprocess/zeiss.py` line 445).
  - "warnings.warn('TODO: Verify the direction of sinogram rotation.')"
    (`mbirtorch/preprocess/pymbir.py` line 85).
- The pending-documentation mechanism records pages ported from mbirjax whose modules are
  not yet ported. As of this reading the pending table is empty ("(none currently)"), and
  two entries are marked REPLACED rather than pending: `use_gpu` (replaced by
  `configure_devices`) and `device_summary` (replaced by `get_memory_stats`)
  (`docs/source/_pending/README.rst`).
- CI runs on CPU only, so the CUDA and Triton kernel paths are not exercised by continuous
  integration (`.github/workflows/ci.yml`).
- The GitHub issue tracker is enabled but empty, so there is no public list of user-reported
  defects (`gh issue list -R cabouman/mbirtorch --state all`).
- Documentation inconsistency: `docs/source/overview.rst` says demos are provided "as
  Jupyter notebooks and python scripts", but `demo/` contains only Python scripts.
- Documentation inconsistency: `docs/source/theory.rst` writes the qGGMRF shape constraint
  as `p < q = 2.0`, while `mbirtorch/_utils.py` defaults are `p = 2.0` and `q = 1.2`.

---

## 11. Open and in-progress work, from the plans repository

Plans repository: `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/`.

### 11.1 Where the forward plan lives now
- `plans/README.md` line 26 still points to `plans/current_plans.md` as "the EVOLVING forward
  plan (start here)", but that file no longer exists at that path. It was retired and moved
  to `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/torch_port/closed/current_plans.md`.
  The retirement is recorded at
  `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/torch_port/open_items_v5.md`
  line 634: "The current_plans.md migration (2026-08-19) -- Greg retired
  `plans/current_plans.md` in favor of this file."
- The live forward plan is
  `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/torch_port/open_items_v5.md`,
  header "Reorganized 2026-08-21" (line 3).

### 11.2 Live open items (open_items_v5.md)
Source for every row:
`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/torch_port/open_items_v5.md`.

| ID | Status label (verbatim) | Date | What it is | Line |
| --- | --- | --- | --- | --- |
| D7-translation | "The projection loop organization for translation is unexamined" (no status tag) | not recorded | Whether translation's forward and back loops scatter or gather the right way round was inherited without question | 54 |
| G1 | "The release-workflow remainder" | not recorded | Read the Docs stable default and token; `release.yml` unwritten; docs preview optional; wheel-check developer script unwritten | 66 |
| G3 | "Migrate to mbirtorch_metrics and retire the mbirjax dashboard" | not recorded | Make the new metrics repo and dashboard independent and torch-only | 72 |
| H1 | "MAR: cache the fitting matrix" | not recorded | Compute each column of the metal-artifact fit matrix once instead of O(num_cols^2) | 79 |
| H3 | "Multi-resolution reconstruction, back-burnered" -- "may be superseded by NN/INR approaches" | migrated 2026-08-19 | Binned-resolution reconstruction upsampled as the next level's initializer | 86 |
| H5 | "Archive old plans, back-burnered" | not recorded | Move old plan docs and scripts to archived storage | 94 |
| H6 | "Test-suite quality and cost" | not recorded | Demo-data gates sit 300-640x above measured noise floors; cone fixtures could be module-scoped (72 s serial) | 98 |
| H8-translation | "Translation kernels wait on measured triggers" | priced 2026-08-22 | Hand-written translation Triton kernels; three named triggers, including detector growth toward 6K x 10K panels | 105 |
| C3 | "PARKED 2026-08-21" | 2026-08-21 | The scan preprocessing pipeline's multi-device concurrency is correctness-gated only, never measured on a GPU node | 246 |
| E2 | "PARKED 2026-08-19, with a revisit trigger" | 2026-08-19 | The floors-refresh automation questions | 442 |

The file's own "Start here: the next session" section (line 34) names G1 and G3 as the
nearest actionable items, H1 and H6 as the self-contained code items, and says
H8-translation and D7-translation "wait on triggers, not on work".

### 11.3 Items recorded in the retired current_plans.md
Source: `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/torch_port/closed/current_plans.md`
("Rewritten 2026-08-07 ...; compressed 2026-08-09").
- Item 3, "Multi-GPU performance investigation -- MEASUREMENT COMPLETE; implementation in
  progress", state 2026-08-10 (lines 17, 77).
- Item 9, MAR H-matrix caching, "Reframed 2026-07-10" (lines 23, 175).
- Item 10, "LEAP/SVMBIR interfaces (back-burnered)", "State: Back-burnered per the port
  decision" (lines 24, 191).
- Item 11, "Miscellaneous / cleanup", with eight open bullets (line 201).
- Item 12, "Possible future direction: multi-resolution reconstruction
  (post-next-main)", added 2026-07-10 (line 229).
- Item 13, "Sorted-stream parallel forward -- SCHEDULED (after item 3)", chartered
  2026-08-07 (line 274).
- Item 15, "Sharded phantom generation -- COMPLETE", though the body text is stale
  (line 313).
Items 1, 2, 4, 5, 6, 7, 8 and 14 are all marked COMPLETE, dated 2026-08-07 to 2026-08-10.

### 11.4 Per-program status
| Program | Status label (verbatim) | Date | Source |
| --- | --- | --- | --- |
| nn_priors, MACE proof of concept | "Status: ACTIVE 2026-08-27"; all four increments "done 2026-08-27" | 2026-08-27 | `.../plans/nn_priors/mace_poc_plan.md` lines 3, 99 |
| nn_priors, multi-slice fusion | Header text is stale ("drafted 2026-08-27 and awaits review before work starts"); the increments table records all four increments "done 2026-08-27", plus an "Addendum 2026-08-27 -- first real-data run" | 2026-08-27 | `.../plans/nn_priors/multi_slice_fusion.md` lines 6, 83, 92 |
| flash_remediation | "Status (2026-07-12): investigation COMPLETE (Phases 1-2d); implementation COMPLETE and VALIDATED on real scans"; "Step E ... is in progress" | 2026-07-12 | `.../plans/flash_remediation/flash_remediation_plan.md` line 3 |
| flash_remediation, geometry-at-construction sub-plan | "Status: PLAN ONLY -- no code written yet." | not dated | `.../plans/flash_remediation/geometry_at_construction_plan.md` line 3 |
| partition_sequence | "Drafted 2026-07-04"; README calls it "(ACTIVE as of 2026-07)" | 2026-07-04 | `.../plans/partition_sequence/partition_sequence_plan.md` line 3; `.../plans/README.md` line 36 |
| projector_kernels | No program-level status label; per-document dates 2026-07-07 to 2026-07-20; `fwd_guard_sweep.md` reads "Status: SHIPPED (Greg approved 2026-07-13; commit 8ea8f7a)" | 2026-07 | `.../plans/projector_kernels/` |
| sharding | Program-level status not recorded in `sharding_status.md`; newest handoff heading "HANDOFF (2026-06-24) -- PRERELEASE PR #17 MERGED into prerelease"; README calls it "(COMPLETE, shipped 2026-07)" | 2026-06 / 2026-07 | `.../plans/sharding/sharding_status.md` lines 1, 24, 41; `.../plans/README.md` line 28 |
| torch_port | "Status: ACTIVE plan of record", reviewed 2026-08-04 | 2026-08-04 | `.../plans/torch_port/port_plan.md` line 3 |
| torch_port/active `multigpu_findings.md` | "Status: INCREMENT-1 CHECKPOINT"; header stale, last finding (section 1.51) dated 2026-08-24 | 2026-08-09 header, 2026-08-24 content | `.../plans/torch_port/active/multigpu_findings.md` line 3 |
| torch_port/active `multiaxis_kernel_plan.md` | "Ruled 2026-08-22 (Greg)"; "every increment of the campaign has landed" | 2026-08-22 | `.../plans/torch_port/active/multiaxis_kernel_plan.md` lines 3, 260 |
| torch_port/active `translation_kernel_memo.md` | No status field; ruling "The need test is not met today", "the recommendation is to hold" | written 2026-08-22 | `.../plans/torch_port/active/translation_kernel_memo.md` lines 3, 113, 130 |
| torch_port/active `pfwd_segmented_design.md` | "Status: the parallel library step SHIPPED as mbirtorch c761b24 (2026-08-18 ...), and the cone rework line (section 9) CLOSED by Greg's ruling 2026-08-19" | 2026-08-18 / 2026-08-19 | `.../plans/torch_port/active/pfwd_segmented_design.md` line 3 |
| torch_port/active `floors_refresh_automation.md` | "Status: DRAFT 2026-08-11, for the checkpoint to rule." No ruling recorded; all six increments unstarted | 2026-08-11 | `.../plans/torch_port/active/floors_refresh_automation.md` lines 3, 482, 518 |
| torch_port/active `greg_notes.md` | not recorded (no status line, no date); nine ranked open investigations | not recorded | `.../plans/torch_port/active/greg_notes.md` |
| plans/features/ | Empty. Zero files, zero git-tracked files, no git history; directory created on disk 2026-09-02 | 2026-09-02 (mtime only) | `.../plans/features/` |
| mbirtorch_metrics | "Status: increments 1 through 6 are built and pushed. The Mac is cut over. The cluster and the published page wait on two things only a repository admin or token owner can do." | 2026-08-23, last addendum 2026-08-24 | `.../plans/mbirtorch_metrics/mbirjax_port.md` line 3 |
| viewer, as-built | "Status: AS BUILT (2026-08-05)"; ThinLinc-verified 2026-08-07; "Decision (Greg, 2026-08-07): mbirjax keeps its current viewer; the retrofit is not planned." | 2026-08-05 / 2026-08-07 | `.../plans/viewer/mbirtorch_viewer_findings.md` line 3 |

### 11.5 Detail on the feature-relevant programs
- Neural-network priors (nn_priors). Both the MACE proof of concept and the multi-slice
  fusion campaign completed on 2026-08-27. Recorded result: multi-slice fusion beat every
  alternative at every strength on the grid, about 25 percent better than the best
  non-fusion method at convergence (0.0925 versus 0.1244) and about four times better than
  the standard qGGMRF recon (0.363)
  (`.../plans/nn_priors/multi_slice_fusion_findings.md`). A real-data run on an NSI Lilly
  Autoinjector scan is recorded in the addendum, with `sigma_scaled 0.04` the better
  operating point. Queued follow-ups: device-form (`Shards`) exchange between agents for
  volumes past single-device scale; `sigma_prox` and `rho` tuning or schedules;
  per-orientation strengths and a weight sweep; correlated-noise denoisers; more real data.
  Explicitly out of scope: production-scale volumes, denoiser training or fine-tuning,
  video/2.5D networks, time-resolved (4D) fusion, and posterior sampling or diffusion
  machinery (`.../plans/nn_priors/multi_slice_fusion.md` lines 145, 169).
  None of this work is in the shipped package; it lives in
  `mbirtorch/experiments/drunet/` as research scripts.
- Flash remediation (field-of-view truncation). Investigation and implementation are
  recorded COMPLETE and validated on real scans as of 2026-07-12; "Step E (re-baseline
  records + Lilly cache rebuild) is in progress". The forward queue in the release note
  (line 406) lists analogous per-end bounds for translation and multiaxis, a cone
  `scale_recon_shape` override warning, and an open z62 region-of-reconstruction boundary
  ring question (`.../plans/flash_remediation/flash_remediation_plan.md`).
- Partition sequence. Phases P0 to P4 are laid out at line 335. The plan's tail carries a
  recommendation to raise `max_iterations` from 15 into roughly the 25 to 50 range (exact
  value to be decided by the team), with `stop_threshold_change_pct` unchanged at 0.2
  (`.../plans/partition_sequence/partition_sequence_plan.md`). The package default is still
  `max_iterations=15` (`mbirtorch/tomography_model.py`).
- Translation Triton kernels are held, not scheduled: "The need test is not met today",
  "the recommendation is to hold"
  (`.../plans/torch_port/active/translation_kernel_memo.md` lines 113, 130).

---

## 12. Known limitations and open issues, from the plans repository

Provenance warning: the `sharding/` and `projector_kernels/` folders record a JAX-era
(mbirjax) program dated 2026-06 to 2026-07. mbirtorch has its own device policy and its own
kernels, so those items are background, not current mbirtorch behavior. They are labelled
below.

### 12.1 mbirtorch API-surface gaps
- Fourteen public entry points refuse the divided (`Shards`) device form rather than
  supporting it: `auto_set_sigma_y`, `get_forward_model_loss`, `get_forward_lin_quad`,
  `reshape_recon`, `recon`/`prox_map` on `init_recon`, `recon_simple_parallel`,
  `recon_simple_cone`, `gen_weights_mar`, `median_filter3d`, `qggmrf_loss`,
  `qggmrf_gradient_and_hessian_at_indices`, and the two differentiable projector wrappers
  (`.../plans/torch_port/active/multigpu_findings.md` section 1.41;
  `.../plans/torch_port/open_items_v5.md` line 332, item D4).
  `recon_split_sino`, `recon_plastic_metal`, `stitch_arrays` and the streaming preprocessing
  entries already refused it.
- Still open from the build-for-the-future rule: "accepting device-form (padded) shapes for
  init_recon/prox_input" (`.../plans/torch_port/port_plan.md` line 148).
- `compute_hessian_diagonal` with default indices back-projects the unmasked grid, which the
  direct plan under-prices (0.85x to 1.29x across 128- to 1024-class)
  (`.../plans/torch_port/open_items_v5.md` line 400).
- Per-instance loggers persist in Python's logger registry for the life of the process, a
  few KB per model (`.../plans/torch_port/open_items_v5.md` line 348).
- The lateral truncation warning fires even after its own fix is applied, because it reads
  the sinogram alone. "Noted for Greg, unscheduled."
  (`.../plans/torch_port/open_items_v5.md` line 554).
- Cone 1024 above one device is uncovered in the cross-framework value comparison.
  "Accepted 2026-08-10; a full re-run prices at eleven GPU-hours."
  (`.../plans/torch_port/open_items_v5.md` line 554).

### 12.2 Hardware and platform coverage
- AMD / ROCm is untested. The design is expected to carry over (HIP presents as
  `torch.cuda`, Triton has an AMD backend) but an AMD target needs hardware access, a
  vendor performance sweep, a `rocm-torch` harness column, and golden revalidation
  (`.../plans/torch_port/port_plan.md`).
- Apple MPS is float32 only, with immature compile support (the plan is eager there) and
  op-coverage gaps that fall back to CPU via `PYTORCH_ENABLE_MPS_FALLBACK`. MPS gating is
  "Informational only" (`.../plans/torch_port/port_plan.md` lines ~255, 464).
- Two-axis driver tiling is not implemented: mbirjax tiles pixels and views in its sparse
  drivers, the torch drivers tile views only -- "sufficient through the 1024 gate cells, but
  pixel chunking becomes first-class at ~1400^3-plus and for the 6K x 10K detector
  trajectory" (`.../plans/torch_port/port_plan.md` line 116).

### 12.3 Numerical and reproducibility risks (accepted)
- Compiled cross-count value differences of about 6e-4 at uneven splits stay; comparisons
  across device counts gate at 1e-3, not 1e-4
  (`.../plans/torch_port/open_items_v5.md` line 437, item E1, closed by ruling).
- The multiaxis forward's atomic adds are not bit-reproducible between launches; its repeat
  gate is a tolerance, and only the back projection is bit-exact
  (`.../plans/torch_port/active/multiaxis_kernel_plan.md` line 305).
- A bitwise repeat of a CPU reconstruction needs one torch thread, because float32
  reductions differ run to run at the default thread count
  (`.../plans/torch_port/open_items_v5.md` line 573, item J1).
- The kernel-versus-torch-body value gap grows with row count, 6.2e-06 at 113 rows to
  2.5e-05 at 449 rows; the multi-row-chunk test gates at 1e-4 while small cells hold 1e-5
  (`.../plans/torch_port/active/multigpu_findings.md` section 1.51;
  `.../plans/torch_port/active/multiaxis_kernel_plan.md` line 168).
- Eight semantic differences against jax are pre-registered as the standing checklist:
  out-of-bounds scatter/gather, rounding ties, autograd overhead, dtype promotion, index
  dtype (int64 doubles the scatter-centers array to about 0.5 to 1 GB at the 1024-class),
  determinism diagnostics, allocator fragmentation under shape churn, and RNG
  (`.../plans/torch_port/port_plan.md` line 407).
- The jax rounding-bug precondition stays monitor-only at six per-slice rounding sites
  (`.../plans/torch_port/open_items_v5.md` line 554). Note: the
  `bugs_and_artifacts/jax rounding bug/` documents that `plans/README.md` line 88
  references are not present in this repository.

### 12.4 Multi-GPU and kernel limitations
- The multiaxis back projection has a residual 1.17x non-dividing penalty that is
  "unexplained and left open"; cone pays a matching 1.14x. The forward half was fixed
  (3.19x, penalty now 1.03x) (`.../plans/torch_port/active/multigpu_findings.md`
  section 1.51, 2026-08-24; run detail in
  `.../plans/experiments/torch_port/mg63_multiaxis_row_bound.md` and `mg64_row_bound_verify.md`).
- No torch-body route can run the 2048-class multiaxis at any device count; the 2048 run
  completed on four devices and "its two-device arm's out-of-memory brackets the boundary"
  (`.../plans/torch_port/active/multigpu_findings.md` section 1.46;
  `.../plans/torch_port/active/multiaxis_kernel_plan.md` line 226).
- Translation and multiaxis have no widening floors of their own in the plans record; the
  execution overview says they inherit parallel's more permissive set and "Their own floors
  have never been measured" (`.../plans/torch_port/active/execution_overview.md` lines 136,
  735). Note that the shipped `mbirtorch/_widening_floors.py` DOES carry separate
  `multiaxis` and `translation` families measured 2026-08-22, so this plans note predates
  the shipped table.
- Denoiser floor rows are sentinels: "no admission size has ever been measured, so the
  automatic path never widens a denoise for speed". They are also misleading for
  `output_sharded=True` callers -- "The floors row says one device; it is answering a
  question about a call that gathers, and the loop never does"
  (`.../plans/torch_port/active/multigpu_plan_part_2.md` lines 129, 205;
  `.../plans/torch_port/active/multigpu_findings.md` sections 1.39, 1.47).
- A live floors-staleness note rides every nightly until someone re-measures or blesses the
  hashes; the shipped table reports every family stale, and "The staleness note is a true
  statement about the file hashes and a false alarm about cost"
  (`.../plans/torch_port/active/multigpu_findings.md` section 1.40;
  `.../plans/torch_port/active/multigpu_plan_part_2.md` line 191). Stale floors are an
  accepted risk because "stale floors are on the safe side"
  (`.../plans/torch_port/active/multigpu_findings.md` section 3.3).
- Triton cannot express the shared-memory atomic form the segmented forward originally named
  as its fallback; a scratch-atomic kernel "would need a CUDA- or pallas-level rewrite
  outside this campaign's tooling". Also recorded: `tl.dot` on float32 defaults to
  reduced-precision tensor-core mode, "which would fail the 1e-5 values gate", and "Triton
  can't scatter into a register tile by computed index"
  (`.../plans/torch_port/active/pfwd_segmented_design.md` lines 76, 83, 437).
- Recorded but not scheduled: the cone back kernel's register-pressure retune (bounded near
  1.4x on a kernel that is about a tenth of the four-device 2048-class wall); sort-ordering
  memoization (about 6 MB per subset per view batch); the multiaxis tile-constant tuning
  sweep; the denoiser ladder extension; the parallel-forward view-loop respecialization; a
  spill-versus-rematerialization diagnostic
  (`.../plans/torch_port/active/multigpu_findings.md` sections 1.24, 1.26, 1.30, 1.32,
  1.45, 1.47).
- The multiaxis vertical-fan gather divides by the per-view slope, which vanishes only at
  90-degree elevation; the model already warns above 45 degrees. Accepted: "the cost of
  extreme elevation is extra loop iterations, not wrong values"
  (`.../plans/torch_port/active/multiaxis_kernel_plan.md` lines 90, 164).
- Six open questions are listed at `.../plans/torch_port/active/execution_overview.md`
  section 7, all dated 2026-08-11 and all flagged "None is a finding", including that
  `usr_multi_gpu.rst` does not say which entry points spread across devices.

### 12.5 JAX-era (mbirjax) items carried in the sharding folder -- background only
Source: `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/sharding/`.
- Multi-node scaling is out of scope; all sharding is single-process
  (`sharding_status.md` line 2350).
- Detector-row sharding is a parked exploration: it caps parallelism at the slice count
  ("4 slices can't use 8 GPUs"), needs variable-footprint machinery, and is mutually
  exclusive with view sharding -- "You can't have both without a 2-D (slice x view) mesh"
  (`sinogram_sharding.md` lines 5, 8, 90, 113, 124, 129).
- Single-GPU is effectively capped near 1024^3; above that needs two or more devices. Cone
  2048^3 at four devices ran out of memory under plain `recon()` but fit under
  `split_sino_recon` (`sharding_status.md` lines 459, 56).
- A multi-GPU collective hang throws no exception, so it cannot be converted to a clean
  error; still open (`sharding_status.md` lines 64, 95).
- L40S hardware fault: device-resident cross-device transfer silently zeros the non-default
  shard (`sharding_status.md` line 2357). A matching unresolved L40S result corruption is at
  `sharding/parallel_performance/fbp_filter_parallelism_comparison.md` line 168 -- "The
  source of the discrepancy is not yet understood", H100 clean, and "path B is not safe for
  deployment on arbitrary GPU hardware".
- Halo-once-per-pass is an accepted approximation (NRMSE about 6e-5 to 3e-4, about 1000x
  below the recon-versus-phantom error) (`sharding_status.md` line 1678).
- CPU scaling: four devices is the sweet spot, eight regresses; memory-bandwidth-bound, not
  a defect (`sharding_status.md` line 2167).
- The "Phase tracker" at `sharding_status.md` line 2363 and `_file_index.md` line 3 are
  stale and contradict the handoffs.

### 12.6 JAX-era projector-kernel parked items -- background only
- CPU forward scatter is XLA-CPU-inherent; every alternative measured worse. Marked
  "[ACCEPTED/PARKED]" (`.../plans/projector_kernels/fwd_back_findings.md` line 144).
- Back tiling was swept and parked; defaults unchanged (same file, lines 44, 245).
- Vertical-fan per-slice rounds (Class V) remain documented accepted risk (same file,
  line 383).

### 12.7 Other programs
- MAR refactor Phase 3 is deferred (`.../plans/sharding/mar_refactor_plan.md` lines 8, 183).
- Preprocessing: "Biggest single duplication and NOT yet done (Greg deferred it): 9
  byte-identical OLE-reader helpers"
  (`.../plans/preprocessing/preprocessing_pipeline_refactor_plan.md` line 190).
- Viewer: three matplotlib 3.11 macosx problems are worked around (in-process Tk crashes
  under the macosx backend, a SIGBUS that cannot be caught; rubber-band zoom cancels on fast
  drags; the macosx blit path mishandles partial regions, so the fast path is whitelisted to
  Agg and TkAgg). Two feature deviations from mbirjax parity: exact-range entry uses a
  revised dialog, and h5 save is "parity except dict editing"
  (`.../plans/viewer/mbirtorch_viewer_findings.md` lines 74, 157).
- mbirtorch_metrics is blocked on human action: GitHub Pages needs admin on
  `cabouman/mbirtorch_metrics` (Greg has write only), and the cluster push token must be
  created by hand (`.../plans/mbirtorch_metrics/mbirjax_port.md` lines 433, 527, 537).

---

## 13. What the plans record says about LEAP specifically

This section exists because the caller's task is a comparison against LEAP. Everything here
is quoted from the plans repository; the throughput numbers are MBIRJAX numbers from
2026-07-12, before mbirtorch's Triton kernels existed, so they are a starting point for the
comparison and not a current mbirtorch measurement.

Source for this whole section:
`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/projector_kernels/headroom_appendices/appendix_ct_kernel_practice.md`,
a survey "Produced 2026-07-12 by a parallel research agent during the headroom-investigation
kickoff (five-agent workflow; this file is one agent's report, reproduced verbatim)". Its own
header warns that "quantitative traffic models are PRE-E0 estimates pending the HLO/ncu
verification pass", and that the normalized throughput numbers are the agent's own arithmetic
on published figures.

- How LEAP is characterized there: "Separable-Footprint (SF)-class model; parallelized over
  output samples in each direction (detector pixels for A, voxels for A^T) -- gather both
  ways, same weight model"; back projection uses "Same SF weights, voxel-parallel"; matched
  pairs are "the headline feature"; it uses "3D texture for the input array in each
  direction"; language "CUDA".
- So LEAP and mbirtorch use the same class of projector model (separable footprint) and both
  ship matched (exactly adjoint) forward/back pairs.
- The throughput table (the appendix's own arithmetic; "Updates/s per TB/s" is the
  bandwidth-normalized column):

| Source | Problem | GPU (approx bandwidth) | Time | Updates/s | Updates/s per TB/s |
| --- | --- | --- | --- | --- | --- |
| mbirjax | parallel fwd 1024^3 x 1024 | H100 (3.35 TB/s SXM assumed) | 8.19 s | 134 G/s | 40 G |
| mbirjax | parallel back, same | H100 | 10.92 s | 101 G/s | 30 G |
| mbirjax | cone fwd, same | H100 | 19.4 s | 57 G/s | 17 G |
| LEAP (Table 1) | parallel 1024^3 x 720 fwd | Tesla P100 (0.73 TB/s) | 11.5 s | 67 G/s | 92 G |
| LEAP | cone 1024^3 x 720 | P100 | 37.1 s | 21 G/s | 28 G |

- The appendix's calibration conclusion, quoted: "mbirjax parallel forward sits at 40 G
  (approximately 2.3x below SOTA), back at 30 G (approximately 3x), cone forward at 17 G
  (vs LEAP cone's 28 G, approximately 1.7x). If the H100 is PCIe (2 TB/s), the gaps shrink
  to ~1.4-2x."
- The appendix also records: "No published X-ray CT projector in Triton or Pallas found",
  and that CTorch, the PYRO-NN update, and LEAP "are all raw CUDA C". mbirtorch's Triton
  kernels are therefore unusual in the field, per this survey.
- On matchedness the appendix quotes LEAP's own documentation: unmatched projectors "may
  produce artifacts when used over enough iterations"; matched pairs "ensure convergence".
  It concludes "there is no literature supporting unmatched operators inside coordinate
  descent", so adjoint-breaking rewrites are "inadmissible for the VCD inner loop".
- LEAP/SVMBIR transition wrappers do not exist in mbirtorch and are not planned. The item is
  recorded as "H7. CLOSED IN FAVOR OF H2 2026-08-19: LEAP and SVMBIR interfaces"
  (`.../plans/torch_port/open_items_v5.md` line 535), and earlier as "back-burnered"
  (`.../plans/torch_port/closed/current_plans.md` line 191;
  `.../plans/torch_port/port_plan.md` line 175). The rationale recorded there: "LEAP presents
  as a PyTorch front end, so the wrapper is thinner on mbirtorch than it would have been on
  jax." Its replacement, the one-call functional interface `recon_simple_parallel` /
  `recon_simple_cone`, closed 2026-08-21.
- There is no feature-by-feature comparison against LEAP anywhere in the plans repository.
  The appendix above is a kernel-implementation survey, not a feature matrix.

---

## 14. Recorded performance: the measured time series (metrics repositories)

### 14.0 Where the numbers live and what the labels mean
- There are two metrics repositories. `mbirtorch_metrics` is a port of `mbirjax_metrics`
  at commit `e37bc93e`; the torch runs from 2026-08-05 onward were migrated out of
  `mbirjax_metrics/results/{cpu,gpu}-torch/` into `mbirtorch_metrics/results/{cpu,gpu}/`.
  The `*-torch` trees in `mbirjax_metrics` stop at 2026-08-22 and are superseded. The
  latest mbirtorch measurements are all in
  `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_metrics/results/`.
  The migration was verified as a pure key rename (only `mbirjax_version:` ->
  `mbirtorch_version:` and the platform label changed; no measured value moved).
- File schema, from
  `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_metrics/tooling/scaling_tests/performance_tracking.py`:
  `regression_<platform>_<timestamp>_<commit>.yaml` is the run; the `..._table.yaml`
  companion is the browsable dump organized as geometry -> op -> size -> `n=<devices>`;
  `records_<platform>.yaml` is a best-ever record book (min time, min memory, max speedup
  across all history), NOT the latest run. The tables below are from the latest-run files,
  not the record book.
- Size labels are SINOGRAM shapes `(n_views, n_rows, n_channels)`, not recon shapes. The
  recon shape is auto-derived, and pinned for cone via `CONE_RECON_SHAPE_PINS`. For
  example `512x448x384` pins the cone recon to `(384, 384, 448)` and `1024x1008x992` pins
  it to `(992, 992, 1008)`. The denoiser sizes are IMAGE shapes, not sinograms. The
  translation `15x2048x2048` label uses 15 as a placeholder; the real view count comes from
  the translation grid, and recon rows are 320.
- Operation meanings, from the same harness file:
  - `direct_filter` is the FILTER STEP ONLY of a direct reconstruction (`fbp_filter` for
    parallel and multiaxis, `fdk_filter` for cone and translation). A whole direct
    reconstruction is filter plus back projection. An end-to-end `recon_fbp` / `recon_fdk`
    wall time is NOT RECORDED anywhere in these repositories.
  - `forward` is `sparse_forward_project`; `back` is `sparse_back_project`.
  - `vcd_nonconst` is one full VCD reconstruction with nonconstant weights, at
    `vcd_iterations: 3`, with early stopping disabled.
  - `denoise` is `QGGMRFDenoiser.denoise` at 20 iterations with `sigma=0.1`.
- Rulers: time is the MINIMUM over `warmup=1` plus trials, with
  `trials_by_op = {direct_filter:3, forward:3, back:3, vcd_nonconst:1, denoise:1}` and
  `single_trial_sizes: ['1024x1008x992']`, so every 1024-class row is a single trial.
  GPU memory is `gpu_peak_per_device`, the max over the row's pinned devices of
  `torch.cuda.max_memory_allocated` (a per-device peak, not a sum). CPU memory is
  `cpu_rss`, whole-process resident set size, which is coarse and has a roughly 200 MB
  floor.

### 14.1 GPU (H100), latest run
Source: `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_metrics/results/gpu/prerelease/regression_gpu_20260827T175529Z_26bd0ea9_table.yaml`.
Branch `prerelease`, commit `26bd0ea9`, commit_date `2026-08-27T13:55:29-04:00`,
`date: '20260830'`, `measured_at: '2026-08-30T03:29:08-04:00'`,
`device: GPU (NVIDIA H100 80GB HBM3)`, `mbirtorch_version: mbirtorch 0.0.2`,
torch `2.13.0+cu130`, triton `3.7.1`, gate `pass`.

Parallel beam:

| op | size | n=1 time / mem | n=2 time / mem (speedup) | n=4 time / mem (speedup) |
| --- | --- | --- | --- | --- |
| direct_filter | 200x208x160 | 3.3 ms / 63.2 MB | not measured | not measured |
| direct_filter | 512x448x384 | 16.5 ms / 699.0 MB | 10.6 ms / 363.0 MB (1.55x) | 9.5 ms / 195.0 MB (1.72x) |
| direct_filter | 513x449x385 | 16.4 ms / 708.1 MB | not measured | not measured |
| direct_filter | 1024x1008x992 | 110.4 ms / 7,884.0 MB | 55.6 ms / 3,980.0 MB (1.99x) | 37.8 ms / 2,025.0 MB (2.92x) |
| forward | 200x208x160 | 4.1 ms / 161.6 MB | not measured | not measured |
| forward | 512x448x384 | 87.6 ms / 1,212.0 MB | 47.8 ms / 691.3 MB (1.83x) | 27.9 ms / 474.9 MB (3.13x) |
| forward | 513x449x385 | 91.0 ms / 1,429.2 MB | not measured | not measured |
| forward | 1024x1008x992 | 2,529.7 ms / 9,737.4 MB | 1,236.2 ms / 4,815.2 MB (2.05x) | 630.7 ms / 3,096.9 MB (4.01x) |
| back | 200x208x160 | 1.9 ms / 110.4 MB | not measured | not measured |
| back | 512x448x384 | 29.1 ms / 959.0 MB | 17.9 ms / 650.6 MB (1.62x) | 14.1 ms / 393.6 MB (2.06x) |
| back | 513x449x385 | 30.2 ms / 982.1 MB | not measured | not measured |
| back | 1024x1008x992 | 778.3 ms / 11,117.3 MB (throttled) | 414.5 ms / 7,488.5 MB (1.88x, throttled) | 245.2 ms / 4,140.9 MB (3.17x) |
| vcd_nonconst, 3 iterations | 200x208x160 | 227.4 ms / 235.6 MB | not measured | not measured |
| vcd_nonconst, 3 iterations | 512x448x384 | 1,041.4 ms / 2,145.7 MB | 916.1 ms / 1,163.0 MB (1.14x) | 1,837.9 ms / 716.5 MB (0.57x) |
| vcd_nonconst, 3 iterations | 513x449x385 | 1,034.0 ms / 2,370.3 MB | not measured | not measured |
| vcd_nonconst, 3 iterations | 1024x1008x992 | 18,976.3 ms / 23,418.0 MB (throttled) | 10,272.6 ms / 12,055.3 MB (1.85x, throttled) | 5,879.2 ms / 6,453.7 MB (3.23x, throttled) |

Cone beam:

| op | size | n=1 | n=2 (speedup) | n=4 (speedup) |
| --- | --- | --- | --- | --- |
| direct_filter | 200x208x160 | 5.6 ms / 64.0 MB | not measured | not measured |
| direct_filter | 512x448x384 | 26.7 ms / 701.7 MB | 18.6 ms / 365.7 MB (1.44x) | 18.6 ms / 197.7 MB (1.43x) |
| direct_filter | 1024x1008x992 | 142.2 ms / 7,891.8 MB | 83.9 ms / 3,987.8 MB (1.70x) | 86.7 ms / 2,032.8 MB (1.64x) |
| forward | 200x208x160 | 10.5 ms / 157.9 MB | not measured | not measured |
| forward | 512x448x384 | 307.4 ms / 1,268.3 MB | 154.5 ms / 691.3 MB (1.99x) | 78.3 ms / 474.9 MB (3.93x) |
| forward | 1024x1008x992 | 8,000.0 ms / 8,791.1 MB | 4,029.2 ms / 4,815.2 MB (1.99x) | 2,085.5 ms / 3,096.9 MB (3.84x) |
| back | 200x208x160 | 7.4 ms / 167.9 MB | not measured | not measured |
| back | 512x448x384 | 152.8 ms / 1,240.1 MB | 93.5 ms / 1,016.3 MB (1.63x) | 59.8 ms / 716.5 MB (2.56x) |
| back | 1024x1008x992 | 4,546.5 ms / 11,145.2 MB | 2,353.6 ms / 7,756.6 MB (1.93x) | 1,256.1 ms / 4,533.0 MB (3.62x) |
| vcd_nonconst, 3 iterations | 200x208x160 | 340.1 ms / 231.6 MB | not measured | not measured |
| vcd_nonconst, 3 iterations | 512x448x384 | 2,480.3 ms / 2,201.9 MB | 1,720.5 ms / 1,361.9 MB (1.44x) | 2,387.8 ms / 894.2 MB (1.04x) |
| vcd_nonconst, 3 iterations | 1024x1008x992 | 59,178.1 ms / 23,502.8 MB | 31,038.9 ms / 12,767.0 MB (1.91x) | 17,344.2 ms / 7,007.8 MB (3.41x) |

Translation (no VCD row is measured, by design):

| op | size | n=1 | n=2 | n=4 |
| --- | --- | --- | --- | --- |
| direct_filter | 15x65x65 | 1.1 ms / 5.1 MB | not measured | not measured |
| direct_filter | 15x257x257 | 2.2 ms / 29.9 MB | not measured | not measured |
| direct_filter | 15x2048x2048 | 81.7 ms / 648.0 MB | 76.4 ms / 424.0 MB (1.07x) | 75.5 ms / 296.0 MB (1.08x) |
| forward | 15x65x65 | 0.8 ms / 21.3 MB | not measured | not measured |
| forward | 15x257x257 | 3.1 ms / 316.5 MB | not measured | not measured |
| forward | 15x2048x2048 | 309.7 ms / 15,649.5 MB | 218.7 ms / 5,383.0 MB (1.42x) | 120.1 ms / 2,950.0 MB (2.58x) |
| back | 15x65x65 | 0.8 ms / 50.9 MB | not measured | not measured |
| back | 15x257x257 | 2.0 ms / 775.1 MB | not measured | not measured |
| back | 15x2048x2048 | 473.6 ms / 15,605.0 MB | 484.4 ms / 12,933.0 MB (0.98x) | 371.3 ms / 9,029.0 MB (1.28x) |

Multi-axis parallel (no VCD row is measured, by design):

| op | size | n=1 | n=2 | n=4 |
| --- | --- | --- | --- | --- |
| direct_filter | 256x224x192 | 4.6 ms / 97.5 MB | not measured | not measured |
| direct_filter | 512x448x384 | 15.9 ms / 699.0 MB | 11.2 ms / 363.0 MB (1.42x) | 9.8 ms / 195.0 MB (1.62x) |
| direct_filter | 1024x1008x992 | 111.6 ms / 7,884.0 MB | 56.4 ms / 3,980.0 MB (1.98x) | 38.5 ms / 2,025.0 MB (2.90x) |
| forward | 256x224x192 | 20.9 ms / 204.7 MB | not measured | not measured |
| forward | 512x448x384 | 309.7 ms / 987.1 MB | 161.1 ms / 643.3 MB (1.92x) | 84.0 ms / 426.9 MB (3.68x) |
| forward | 1024x1008x992 | 8,837.8 ms / 9,657.6 MB | 4,545.2 ms / 4,767.2 MB (1.94x) | 2,325.6 ms / 3,048.9 MB (3.80x) |
| back | 256x224x192 | 11.5 ms / 201.1 MB | not measured | not measured |
| back | 512x448x384 | 140.6 ms / 1,071.4 MB | 84.6 ms / 805.0 MB (1.66x) | 50.3 ms / 524.2 MB (2.80x) |
| back | 1024x1008x992 | 4,197.7 ms / 11,870.5 MB | 2,157.4 ms / 8,481.9 MB (1.95x) | 1,134.1 ms / 5,258.4 MB (3.70x) |

Denoiser (20 iterations; sizes are image shapes):

| size | n=1 | n=2 | n=4 |
| --- | --- | --- | --- |
| 225x241x257 | 128.5 ms / 180.1 MB | not measured | not measured |
| 512x448x384 | 274.7 ms / 1,093.8 MB | 661.7 ms / 590.6 MB (0.42x) | 1,233.1 ms / 297.6 MB (0.22x) |
| 1024x1008x992 | 2,456.7 ms / 12,458.5 MB | 3,476.6 ms / 6,850.8 MB (0.71x) | 3,479.0 ms / 3,441.6 MB (0.71x) |

Run-to-run spread: the same commit measured on branch `greg_dev` two days earlier
(`.../mbirtorch_metrics/results/gpu/greg_dev/regression_gpu_20260827T175529Z_26bd0ea9_table.yaml`,
`measured_at: '2026-08-28T03:32:48-04:00'`) reads parallel forward n=1 at 2,531.3 ms,
parallel back n=1 at 772.3 ms, parallel VCD n=1 at 18,921.8 ms, and cone VCD n=1 at
59,200.4 ms. Memory is identical between the two runs and times differ by well under
1 percent at the large cells.

### 14.2 CPU, latest run
Source: `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_metrics/results/cpu/prerelease/regression_cpu_20260827T175529Z_26bd0ea9_table.yaml`.
Branch `prerelease`, commit `26bd0ea9` (2026-08-27), `date: '20260829'`,
`measured_at: '2026-08-29T10:07:04-04:00'`, `device: CPU (arm)` (a local Mac), torch
`2.13.0`, python 3.12.13, `device_counts: [1]` only, gate `pass`. Memory is whole-process
resident set size.

| geometry | op | 128x112x96 | 129x113x97 | 200x208x160 |
| --- | --- | --- | --- | --- |
| parallel | direct_filter | 14.9 ms / 242.7 MB | 16.9 ms / 252.2 MB | 207.1 ms / 309.7 MB |
| parallel | forward | 84.8 ms / 431.8 MB | 81.6 ms / 426.3 MB | 658.8 ms / 562.5 MB |
| parallel | back | 45.9 ms / 444.2 MB | 45.6 ms / 423.4 MB | 372.4 ms / 549.0 MB |
| parallel | vcd_nonconst, 3 it | 1,132.4 ms / 615.6 MB | 1,182.0 ms / 617.0 MB | 8,304.2 ms / 899.2 MB |
| cone | direct_filter | 18.4 ms / 270.0 MB | 19.4 ms / 259.3 MB | 215.1 ms / 328.5 MB |
| cone | forward | 436.0 ms / 1,024.4 MB | 462.0 ms / 1,064.5 MB | 3,623.3 ms / 3,973.0 MB |
| cone | back | 388.1 ms / 2,141.6 MB | 455.7 ms / 2,222.8 MB | 5,070.2 ms / 9,979.4 MB |
| cone | vcd_nonconst, 3 it | 6,959.5 ms / 2,922.2 MB | 7,339.8 ms / 3,074.1 MB | 51,302.7 ms / 12,192.6 MB |

Other CPU rows from the same file: translation `direct_filter` 15x64x64 1.3 ms / 215.0 MB
and 15x65x65 5.3 ms / 216.2 MB; translation `forward` 15x64x64 9.1 ms / 398.0 MB;
translation `back` 15x64x64 4.4 ms / 417.4 MB; multiaxis `direct_filter` 128x112x96
15.1 ms / 253.2 MB; multiaxis `forward` 128x112x96 266.9 ms / 2,373.1 MB; multiaxis `back`
128x112x96 131.8 ms / 1,226.4 MB; denoiser 128x144x160 530.6 ms / 487.8 MB and
225x241x257 2,745.7 ms / 782.0 MB.

The same-commit `greg_dev` CPU run
(`.../mbirtorch_metrics/results/cpu/greg_dev/regression_cpu_20260827T175529Z_26bd0ea9_table.yaml`,
`measured_at: '2026-08-28T12:27:46-04:00'`) reads noticeably slower on the same machine,
for example parallel `direct_filter` 200x208x160 at 303.6 ms (flagged `soft: time +40.1%`).
The CPU readings are from a Mac and are noisy.

There are NO multi-device CPU rows; the CPU nightly runs `device_counts: [1]` only.

### 14.3 Composed VCD reconstruction walls (a different ruler)
`execution_overview.md` section 5 warns that the campaign ruler (warm seeded 3-iteration
VCD, median of three repeats) and the nightly ruler (the `vcd_nonconst` regression row)
"differ by up to 15 percent" on the same cell.

Source: `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/torch_port/active/execution_overview.md`
sections 5.1 to 5.3, from the mg27 re-run, job 15369703, 2026-08-19, four H100s (node
h018), commit `c761b24` on `f9fde0a`, warm median of three seeded 3-iteration
reconstructions after a discarded cold pass. Peaks are the busiest device.

| cell | geometry | n=1 | n=2 | n=4 |
| --- | --- | --- | --- | --- |
| 1024-class | parallel | 21.26 s / 22.87 GB | 14.24 s / 11.77 GB | 10.84 s / 6.30 GB |
| 1024-class | cone | 61.65 s / 22.95 GB | 35.06 s / 12.47 GB | 22.33 s / 6.84 GB |
| 512-class | parallel | 1.31 s / 2.10 GB | 1.29 s / 1.14 GB | 2.10 s / 0.70 GB |
| 512-class | cone | 2.75 s / 2.15 GB | 2.14 s / 1.33 GB | 2.80 s / 0.87 GB |

The user documentation table in section 9.1 above is the parallel subset of this run.

### 14.4 Capacity limits per GPU count (mg17)
Source: `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/experiments/torch_port/mg17_capacity_table.md`,
with the table itself in
`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/torch_port/closed/two_k_design.md`
section 2. Run of 2026-08-16, job 15307591, node h001, one H100, 27 s wall, tree synced to
`78b4f78`. IMPORTANT: "The results below are based on calculation rather than cluster
runs" -- these are ledger-modeled peaks, not measurements.

The reference problem, quoted from `two_k_design.md` section 2: "Its sinogram shape is
(2048, 2016, 1984) as (views, detector rows, channels), and the default reconstruction
shape is (1984, 1984, 2016). At four bytes per value the sinogram is 30.5 GiB, the
reconstruction volume 29.6 GiB, and one full-pixel-set cylinder stack 23.2 GiB."

Modeled peak per device, GiB, for the cone 2048-class cell ("fits" means 1.15 times the
modeled peak is at or below the 78.67 GiB idle budget measured on an H100):

| devices | today | band_knob | reduce_min | pre_stream | binding phase (today) | fits |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 181.5 | — | — | — | per-iteration statistics | NO |
| 2 | 97.9 | 97.9 | 97.9 | 97.9 | subset delta forward projection | NO |
| 3 | 66.7 | 66.7 | 66.7 | 69.2 | subset delta forward projection | yes (edge) |
| 4 | 51.1 | 51.1 | 51.1 | 57.8 | subset delta forward projection | yes |
| 5 | 41.8 | 41.8 | 41.8 | 51.0 | subset delta forward projection | yes |
| 6 | 35.6 | 35.6 | 35.6 | 46.4 | subset delta forward projection | yes |
| 7 | 31.1 | 31.1 | 31.1 | 43.1 | subset delta forward projection | yes |
| 8 | 27.8 | 27.8 | 27.8 | 40.7 | subset delta forward projection | yes |

Quoted verbatim from `mg17_capacity_table.md`:
- "Idle device budget read on h001: 78.67 GiB. The H100's total is 79.65 GiB".
- "Cone 2048-class, today variant: 181.5 / 97.9 / 66.7 / 51.1 GiB at one through four
  devices; 27.8 GiB at eight. Parallel reads 0.03 to 0.04 GiB lower everywhere."
- "Demand at three devices 76.7 GiB against the 78.67 GiB budget; demand at four devices
  58.8 GiB."
- "1024-class anchors: cone 22.8 / 12.9 / 7.1 GiB at one, two, and four devices."
- "Appendix, today variant, banded path: ma1024 68.0 / 51.8 / 46.3 GiB and tct2k 44.9 /
  39.5 / 34.0 GiB at one, two, and four devices."

So the recorded conclusion is that a 2048-class cone or parallel reconstruction needs at
least three H100s, and three is an edge fit with 1.9 GiB of slack.

A separate, larger modeled point for the multiaxis kernel route is in
`.../plans/torch_port/active/multigpu_findings.md` section 1.46: the 2048-class is modeled
"at 194 GB on one device and 103 GB per device on two".

### 14.5 2048-class (2K) measurements
Source: `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/experiments/torch_port/mg19_two_k_baselines.md`,
run of 2026-08-17, job 15314401, node h003, four H100s, 3 hours 18 minutes wall, tree at
commit `7cd32ed`. The tables are in
`.../plans/torch_port/active/multigpu_findings.md` section 1.20.

Memory validation, three-iteration reconstructions at the 2048-class cell:

| arm | modeled GiB | measured GiB |
| --- | --- | --- |
| cone n=3 | 66.7 | 56.5 |
| cone n=4 | 51.1 | 43.1 to 46.3 |
| parallel n=3 | 66.7 | 56.4 |
| parallel n=4 | 51.1 | 43.1 to 46.3 |

Component split of the three-iteration wall (a starred wall is the first arm at its call
shapes and carries compilation in "other"):

| geometry | devices | wall s | forward busy s | back busy s | other s | back share of wall |
| --- | --- | --- | --- | --- | --- | --- |
| cone | 3 | 459* | 203 | 137 | 119* | 0.30 |
| cone | 4 | 420 | 151 | 228 | 40 | 0.54 |
| parallel | 3 | 299* | 193 | 32 | 75* | 0.11 |
| parallel | 4 | 216 | 142 | 36 | 38 | 0.17 |

So a three-iteration 2048-class cone reconstruction took 420 s on four H100s and a
parallel one took 216 s. The mg19 file adds: "cone's first four-device wall is 485 s where
its repeat is 420 s"; "The generators staged a 29.6 GiB phantom and a 30.5 GiB sinogram per
geometry"; "Values legs: 4.0e-6 to 9.6e-6 against the staged references, gate 1e-4."

Back projection after the band padding fix (`multigpu_findings.md` section 1.23, jobs
15336959 and 15337015, 2026-08-18, three-iteration cone reconstructions, busiest device):

| arm | band | back busy s | mg19's reading |
| --- | --- | --- | --- |
| n=3 | 672, no pad | 137.1 | 136.8 |
| n=4 | 504 padded to 512 | 106.4 | 227.8 |
| n=4 repeat | 504 padded to 512 | 104.2 | 228.2 |

Multi-axis at the 2048-class (`multigpu_findings.md` section 1.46, mg55, 2026-08-22,
job 15434826), quoted verbatim: "Cell (2048, 2016, 1984), recon (1984, 1984, 2297),
9.0e9 voxels. The four-device arm built its own input in 90 s (one full forward projection
of the volume through the kernels took 51 s), then reconstructed: cold 355.95 s, warm
298.81 s at 0.1 percent spread, per-device peaks 50.59 / 50.27 / 50.27 / 48.61 GB... The
two-device arm ran out of memory."

Multi-axis 1024-class scaling on the kernel route, same section:

| devices | warm | speedup | busiest peak | peak ratio |
| --- | --- | --- | --- | --- |
| 1 | 67.63 s | 1.00x | 24.11 GB | 1.00x |
| 2 | 37.43 s | 1.81x | 12.95 GB | 0.54x |
| 4 | 22.75 s | 2.97x | 7.49 GB | 0.31x |

### 14.6 Per-iteration cost and cold-start cost
- The relation between 3-iteration and 15-iteration walls, quoted verbatim from
  `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans/plans/experiments/torch_port/mg41_production_compare.md`:
  "The 15-iteration times run about 10 percent above five times the 3-iteration times
  (117.09 against 5 x 21.26 on parallel torch). The package-default subset schedule changes
  across iterations, so per-iteration cost is not constant; the ratio is expected to be
  near five, not exactly five." A per-iteration second is not broken out anywhere.
- Cold versus warm cost of one reconstruction (`multigpu_findings.md` section 1.48, mg57,
  job 15449106, 2026-08-23, parallel (1024, 1008, 992), 3 iterations):

| | one device | four devices |
| --- | --- | --- |
| first run, both caches empty | 49.48 s | 41.11 s |
| new process, caches full | 26.02 s | 19.51 s |
| in-process warm | 21.17 s | 9.60 s |

  From the same section: dynamo costs "16.07 s at one device on the first run and 4.42 s in
  every later process; at four devices, 25.20 s and 8.00 s"; Triton costs "10 compiling
  launches costing 1.19 s at one device, 57 costing 4.82 s at four".
- Setup phase (sections 1.48 and 1.49, mg58 and mg60, job 15449436, 2026-08-23, same cell):
  `initialize_recon` fell "from the 2.61 s of section 1.48 to 0.855 s"; the sinogram check
  fell from 0.505 s to 0.061 s and the weights checks from 1.309 s to 0.063 s. "The saving
  is about 1.76 s per reconstruction, which is 8 percent of the one-device warm wall and
  18 percent of the four-device one."

### 14.7 Convergence facts
- Accuracy decay with iteration depth (`multigpu_findings.md` section 4.4, parallel 1024):
  the cross-framework residual is `6.112e-3 / 6.111e-3 / 6.084e-3` at 1/2/4 devices after
  3 iterations, and `8.768e-4 / 8.770e-4 / 8.733e-4` after 10, a decay of 6.97x at every
  device count. A whole-field norm-relative measure falls "from 8.03e-4 to 4.11e-5 over the
  same depth step, a 19.5x decay". The shipped docs state this as "measured to fall from
  6.1e-3 at 3 iterations to 8.8e-4 at 10" (`docs/source/usr_multi_gpu.rst`).
- A general "typical iterations to converge" figure for mbirtorch is NOT RECORDED. The only
  stop-iteration counts found are from a different study, the flash and truncation
  remediation work on SiC data, quoted verbatim from
  `.../plans/flash_remediation/flash_remediation_plan.md`: "the new slab crosses the 0.2%
  default stop at ~iteration 20; the old needs ~49. At +12-15% per-iteration cost the
  extension is about 2.2x faster wall-clock to the default stop, and under the shipped
  max_iterations=15 cap the old shape never approaches the stop while the new one nearly
  reaches it." A 1024-class replicate reads "stop ~20 vs >50".
- Note the tension with the shipped default: `max_iterations=15`
  (`mbirtorch/tomography_model.py`), while the partition-sequence plan recommends raising
  it into roughly the 25 to 50 range
  (`.../plans/partition_sequence/partition_sequence_plan.md`).

### 14.8 mbirtorch versus mbirjax
This matters for the LEAP comparison because the LEAP throughput comparison in section 13
was made against mbirjax, before mbirtorch's Triton kernels existed.

Controlled comparison, 3-iteration VCD, campaign ruler
(`.../plans/torch_port/active/execution_overview.md` sections 5.1 and 5.2; mbirtorch rows
from job 15369703 on 2026-08-19, mbirjax rows from mg1, job 15011662, on 2026-08-09; same
staged sinograms, md5-verified). 1024-class, sinogram (1024, 1008, 992):

| geometry | devices | mbirtorch time | mbirtorch peak | mbirjax time | mbirjax peak |
| --- | --- | --- | --- | --- | --- |
| parallel | 1 | 21.26 s | 22.87 GB | 25.80 s | 49.81 GB |
| parallel | 2 | 14.24 s | 11.77 GB | 14.33 s | 19.71 GB |
| parallel | 4 | 10.84 s | 6.30 GB | 11.52 s | 14.73 GB |
| cone | 1 | 61.65 s | 22.95 GB | 62.75 s | 48.45 GB |
| cone | 2 | 35.06 s | 12.47 GB | 43.37 s | 21.52 GB |
| cone | 4 | 22.33 s | 6.84 GB | 25.78 s | 12.23 GB |

Quoted verbatim: "mbirtorch now matches or beats mbirjax at EVERY cell of both geometries
... mbirtorch holds less memory at every cell, at 0.43x to 0.60x of mbirjax's peak."
And on the effect of the kernels: "parallel at one device read 94.0 s before the
hand-written kernels, 39.98 s on the per-tap kernel, and 21.26 s now."

Production-length reading, 15 iterations (mg41, job 15371081, 2026-08-19, node h001, one
H100, 1024-class, warm median of three after a discarded cold pass):

| geometry | library | cold s | warm median s | peak GB |
| --- | --- | --- | --- | --- |
| parallel | mbirtorch | 125.67 | 117.09 | 22.87 |
| parallel | mbirjax | 156.09 | 139.75 | 48.66 |
| cone | mbirtorch | 268.39 | 253.89 | 22.95 |
| cone | mbirjax | 292.79 | 271.52 | 48.66 |

Quoted verbatim: "mbirtorch is faster in both geometries at the production iteration count:
1.19x on parallel ... and 1.07x on cone ... mbirtorch holds 0.47x of mbirjax's device
memory in both geometries. The two libraries' answers agree at the 1e-7 class (worst gap
9.4e-7 relative)."

Kernel-free geometries (`multigpu_findings.md` section 1.43, mg52, job 15428371,
2026-08-22, one H100, warm medians of seeded 3-iteration reconstructions):

| cell | geometry | mbirtorch | mbirjax | jax/torch |
| --- | --- | --- | --- | --- |
| (256, 1900, 3000) | translation | 12.59 s | 16.26 s | 1.29x |
| (512, 448, 384) | multiaxis | 11.41 s | 11.06 s | 0.97x |
| (768, 672, 576) | multiaxis | 56.30 s | 60.06 s | 1.07x |
| (1024, 1008, 992) | multiaxis | 310.06 s | 431.07 s | 1.39x |

Instrument caveat, quoted verbatim from `execution_overview.md` sections 5.3 and 5.4:
"the torch counters reset after the cold pass, and jax's peak_bytes_in_use runs from
process start, so the jax peak also covers compile-time allocations."

---

## 15. Uncertain or unverified

- The test count of 769 test functions in `tests/` is a count of `def test_` lines, not the
  collected case count. Many tests are parametrized, so the number of executed cases is
  higher. I did not run the suite (the task forbade it), so the collected count is
  not verified.
- The `mbirtorch_metrics` GPU nightly reads `device_counts` of 1, 2 and 4 only. There is no
  recorded nightly measurement at 3, 5, 6, 7, or 8 devices; the 3-device and 8-device
  figures in section 14.4 are ledger models, not measurements.
- The mg17 capacity table (section 14.4) is explicitly modeled, not measured: "The results
  below are based on calculation rather than cluster runs". The mg19 measurements
  (section 14.5) came in about 15 percent BELOW the model at the same cells, so the model
  is conservative there.
- Every 1024-class metrics row is a single trial (`single_trial_sizes:
  ['1024x1008x992']`), so those readings carry no within-run repeat spread. Some carry a
  "throttled" note, which I did not investigate.
- Several GPU rows are marked "throttled" in the source table. I did not determine what
  the harness means by that flag or how much it biases the reading.
- The CPU numbers come from a local Mac (`device: CPU (arm)`), and the two runs of the same
  commit differ by up to about 40 percent on individual rows. Treat the CPU series as
  indicative only.
- An end-to-end direct reconstruction (`recon_fbp`, `recon_fdk`, `recon_direct`) wall time
  is NOT RECORDED at any size. Only the filter step is a measured row. A direct recon time
  would have to be estimated as filter plus back projection, which I did not do.
- A general "typical iteration count to converge" for mbirtorch is NOT RECORDED. The only
  stop-iteration numbers I found (~20 versus ~49) come from the flash-remediation study on
  SiC data and characterize a recon-shape change, not mbirtorch in general.
- Scatter correction: I found no scatter-correction code or documentation, but I searched
  by keyword rather than reading all 84,494 lines of `mbirtorch/preprocess/utilities.py`.
  A differently named implementation could exist.
- The LEAP throughput comparison in section 13 is (a) about MBIRJAX, not mbirtorch, (b)
  dated 2026-07-12, before mbirtorch's Triton kernels landed, (c) the survey agent's own
  arithmetic on published figures, and (d) self-labelled as "PRE-E0 estimates pending the
  HLO/ncu verification pass". Section 14.8 shows mbirtorch's parallel 1024-class one-device
  VCD wall fell from 94.0 s to 21.26 s across the kernel work, so the 2.3x-below-LEAP
  figure for the parallel forward is almost certainly stale in mbirtorch's favor. Nobody
  has re-normalized mbirtorch against LEAP; that arithmetic remains to be done.
- I did not verify LEAP's own current feature set or performance. That was outside this
  task by instruction.
- `plans/README.md` still points at a retired `current_plans.md` path, and several plan
  documents carry stale headers (noted inline in sections 11 and 12). Where a header and
  its body disagreed I recorded both rather than picking one.
- The plans note that translation and multiaxis "have no widening floors of their own"
  (`execution_overview.md`, 2026-08-11) is contradicted by the shipped
  `mbirtorch/_widening_floors.py`, which carries separate `multiaxis` and `translation`
  families measured 2026-08-22. I read the shipped file as current and the plans note as
  superseded, but I did not confirm that reading with anyone.
- The `plans/features/` directory is empty with no git history; its purpose is unknown.
- I did not execute `import mbirtorch`, so all API facts come from reading source and
  documentation rather than from introspection.
