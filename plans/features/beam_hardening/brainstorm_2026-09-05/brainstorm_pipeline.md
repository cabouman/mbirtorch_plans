# How a beam-hardening estimator by reconstruction quality would fit the code

Repository paths below are relative to `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch`
and, for plan documents, to `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch_plans`.

The short answer is that the search machinery transfers and the reduced problem does not.  A
beam-hardening search changes the sinogram, not the geometry, so it retraces nothing and reuses
`_search_minimum` as it is.  What it cannot reuse is the thin slab: in cone beam there is no finite
slab whose rays stay inside it, and the correction's columns are nonlinear, so detector binning
biases them.  The cheapest useful version is not a replacement for the fit but a search over the
four hyperparameters the fit already takes, scored on one reduced reconstruction per setting.

## 1. The reduced problem

**Binning is exact for the measurement and biased for the columns.**  `reduce_sinogram` averages
each `bin_factor` by `bin_factor` block of detector pixels (`mbirtorch/preprocess/geometry_calibration.py:446-447`).
Averaging is linear, so binning the measured sinogram is exact.  The columns of `H` are monomials
in the class projections `p` and `m` (`mbirtorch/preprocess/mar.py:286-320`), and the mean of a
monomial is not the monomial of the mean.

For a quadratic column the bias is exactly the within-bin variance: the mean of `x**2` over a bin
is `mean(x)**2 + var(x)`.  Model the variation inside a bin as a linear ramp of step `g` per pixel
in each detector direction.  The population variance of an arithmetic progression of `b` terms with
step `g` is `g**2 * (b**2 - 1) / 12`, and in two directions it is `(g_r**2 + g_c**2) * (b**2 - 1) / 12`.
That factor is `3/12 = 0.25` at `b = 2` and `15/12 = 1.25` at `b = 4`.  Relative to the squared bin
mean the bias is `0.25 * (g/m)**2` at `b = 2`.  A column that changes by 10 percent of its value per
detector pixel gives 0.25 percent at `b = 2` and 1.25 percent at `b = 4`.  At the silhouette edge of
a metal insert, `m` runs from zero to full value across one or two pixels, so `g/m` is about 1 and
the bias is 25 percent at `b = 2` and 125 percent at `b = 4`.  The cubic column is worse: the mean
of `x**3` is `m**3 + 3*m*var + skew`, a relative bias of `3*var/m**2`, which is 75 percent at
`g/m = 1` and `b = 2`.  Those edge pixels are the ones the correction acts on.

Two consequences follow.  First, the fit is linear in theta (`mbirtorch/preprocess/mar.py:501-513`,
`776-779`), so binning each column after it is built leaves the least-squares problem in theta a
correctly binned linear problem; building the columns from binned `p` and `m` does not.  Second, a
view stride is exact, because `reduce_sinogram` selects views and averages nothing
(`mbirtorch/preprocess/geometry_calibration.py:417-419`).  The cheap reduction for a
beam-hardening search is therefore the view stride, with `bin_factor` left at 1 on the channel axis,
which is the fallback the geometry plan already writes for its coarse level
(`plans/features/geometric_calibration/estimate_by_recon_plan.md`, sub-increment 1.3).

**In cone beam no finite slab is self-contained.**  `build_reduced_problem` crops the detector to
the rows `_slab_row_window` returns (`geometry_calibration.py:328-330`), and its own docstring
states the cost: rays through the slab also cross material outside it
(`geometry_calibration.py:224-226`).  Here is the size of that.

A voxel at in-plane depth `y` projects with magnification `sdd / (sid - y)`
(`mbirtorch/cone_beam.py:572-582`), so a voxel at axial position `z` lands at detector height
`v = z * sdd / (sid - y)`.  Inverting, a ray fixed at height `v` crosses axial positions
`z = v * (sid - y) / sdd`, which over `y` in `[-r, r]` spans a length `2*r*v/sdd`.  The window's top
row is the largest `v` whose ray still touches the slab top `z = h`, which is `v = h*sdd/(sid - r)`.
That ray reaches up to `z = v*(sid + r)/sdd = h*(sid + r)/(sid - r)`.  It therefore leaves the slab
by `h * 2*r/(sid - r)` above it.

Numbers, on the geometry the existing rotation experiment used: `sid = 400` ALU, `sdd = 800` ALU,
half fan angle 5.7 degrees (`plans/experiments/features/geometric_calibration/rotation_zero_point_synthetic.md:61-63`).
Then `r = 400 * sin(5.7 deg) = 400 * 0.0993 = 39.7` ALU, and `2*r/(sid - r) = 79.4/360.3 = 0.220`.
An 8-slice slab has `h = 4` slices, so the overreach is `4 * 0.220 = 0.88` slices at each end.  At a
15 degree half fan, `r/sid = 0.2588` and `2*r/(sid - r) = 0.5176/0.7412 = 0.698`, so the overreach is
`4 * 0.698 = 2.8` slices at each end, 35 percent of the slab height.

A taller slab does not close the gap.  Thickening the slab to `h*k` with `k = (sid + r)/(sid - r)`
widens the row window in the same proportion, and the new window's rays reach `h*k**2`.  Here
`k = 439.7/360.3 = 1.220` in the synthetic geometry and `1.2588/0.7412 = 1.698` at a 15 degree half
fan.  Both exceed 1, so the recursion diverges until the window is the whole detector.  The columns
for the cropped rows must therefore be forward projections of the full-extent class masks, read out
on the window rows.

There is a cheap way to read exactly those rows.  Build a copy of the model whose detector is
cropped to the window and whose `det_row_offset` is compensated, which is the arithmetic
`build_reduced_problem` already performs (`geometry_calibration.py:340-345`), then forward project
the full-extent masks through it.  The output is views by window rows by channels rather than the
whole sinogram.  The volume traversal is not reduced; only the sinogram-side memory is.
`forward_project` itself takes no row window and always builds the whole sinogram
(`mbirtorch/tomography_model.py:1906-1938`).  The simplest correct alternative is what
`check_rotation_direction` already does: pass `num_slab_slices=None` and keep the whole axial
extent (`geometry_calibration.py:644`), taking the savings from the view stride and the in-plane
binning.

## 2. Cost and memory of basis images

`_generate_metal_exponent_list(K, order)` enumerates the `K`-tuples of nonnegative exponents whose
total degree lies between 1 and `order` (`mbirtorch/preprocess/mar.py:206-235`), so it has
`C(K + order, K) - 1` entries.  `H` has `1 + cross(order - 1) + metal(order)` columns
(`mbirtorch/preprocess/mar.py:765-779`), that is `C(K + order - 1, K) + C(K + order, K) - 1`.

| metals K | order 2 | order 3 | order 4 |
|---|---|---|---|
| 1 | 4 | 6 | 8 |
| 2 | 8 | 15 | 24 |
| 3 | 13 | 29 | 54 |

The 6 and the 15 match the two counts given in the charge.

Direct reconstruction is linear in the sinogram, and the correction's numerator is affine in the
metal-only block of theta once the plastic block is fixed
(`mbirtorch/preprocess/mar.py:677-682`).  So the basis stack needs one reconstruction of the
measured sinogram plus one per metal-only column, which is `C(K + order, K)` images: 3, 4, 5 for
one metal at orders 2, 3, 4; 6, 10, 15 for two metals; 10, 20, 35 for three.  The denominator
`Sp` enters through a division (`mbirtorch/preprocess/mar.py:671-675`, `715-718`), so varying the
plastic block needs a new reconstruction per candidate and is outside the stack.  The outer scale
`plastic_sino_corrected_scale` (`mbirtorch/preprocess/mar.py:814`) is one global number, and a
normalized gradient-energy score is insensitive to it, so it need not be recomputed per candidate.

Image memory is small.  At the reduced scale, 8 slices of 500 by 500 in float32 is
`500 * 500 * 8 * 4 = 8.0e6` bytes, 8 MB per image: 32 MB for one metal at order 3, 80 MB for two
metals at order 3, 280 MB for three metals at order 4.  At full resolution, 3 slices of 2000 by
2000 is `2000 * 2000 * 3 * 4 = 4.8e7` bytes, 48 MB per image: 192 MB, 480 MB, and 1.68 GB for the
same three cases.  Every column rather than the metal-only block costs 2.64 GB in the largest of
those.  All are affordable.

Sinogram memory is not.  A full sinogram of 1800 by 2000 by 2000 in float32 is
`1800 * 4e6 * 4 = 2.88e10` bytes, 28.8 GB.  `_est_plastic_metal_sinos_from_recon` returns `p` and
the `m` list as full-size device sinograms (`mbirtorch/preprocess/mar.py:271-283`), and
`correct_sino_plastic_metal` holds them together with the placed measured sinogram for the whole
call (`mbirtorch/preprocess/mar.py:782`, `785-824`): `3 * 28.8 = 86.4` GB across the devices for one
metal.  Holding all six columns of `H` would add `6 * 28.8 = 173` GB, which is why
`_compute_entry_for_OSQP` rebuilds each column inside its loops and discards it
(`mbirtorch/preprocess/mar.py:501-513`).  That loop builds 6 + 21 = 27 columns for `H` of 6 columns
and holds two at a time, so its live peak is `5 * 28.8 = 144` GB.  Recomputing per view batch is the
only version that fits.  At the reduced scale with the whole axial extent, view stride 4 and bin 2,
a column is `450 * 1000 * 1000 * 4 = 1.8e9` bytes, 1.8 GB, still too large to hold six.  With the
row window it is `450 * 48 * 1000 * 4 = 8.6e7` bytes, 86 MB, and six columns are 0.5 GB.  The row
window is what makes holding columns possible.

## 3. Applying theta to the full sinogram

The pattern exists.  `BH_correction` builds a per-pixel kernel that closes over host constants and
hands it to `pipeline.map_view_batches` (`mbirtorch/preprocess/mar.py:196-203`), which pre-allocates
one host output and writes each batch's view slice in place
(`mbirtorch/preprocess/pipeline.py:140-150`), so the peak is the input plus the one output.

The plastic-metal correction cannot use that pattern as written, because its kernel needs `p` and
`m` for the same views.  Those are full-size sinograms and are produced whole today:
`_est_plastic_metal_sinos_from_recon` shards the recon once
(`mbirtorch/preprocess/mar.py:257`), segments it (`mbirtorch/preprocess/mar.py:264-266`) and
forward projects `1 + K` masks with `output_sharded=True`
(`mbirtorch/preprocess/mar.py:271-281`).  Nothing there is batched over views.

To go batchwise, each batch needs the mask projections for that batch's views only.  Neither
`forward_project` (`mbirtorch/tomography_model.py:1906-1938`) nor `sparse_forward_project`
(`mbirtorch/tomography_model.py:583-594`) takes a view subset.  The available route is a per-batch
model copy through `copy_ct_model(ct_model, new_angles=angles[batch])`
(`mbirtorch/utilities.py:1001-1013`).  The cost is `1 + K` whole-volume traversals per batch, so the
projector work is multiplied by the batch count, and the masks must stay resident: a 2000-cubed
float32 mask is `8e9 * 4 = 32` GB, a packed boolean one 8 GB.  The choice between recomputation and
residency should be measured rather than assumed.

Once `p` and `m` exist in some form, the rest is easy.  All of `_correct_plastic_sinogram`
(`mbirtorch/preprocess/mar.py:632-722`) is per-pixel except the mean of `Sp`
(`mbirtorch/preprocess/mar.py:699-703`), and `_estimate_plastic_scaling` needs two inner products
(`mbirtorch/preprocess/mar.py:724-739`).  Both are scalar reductions that a two-pass view-batch loop
computes exactly, so no second full-size sinogram is needed for the correction itself.

## 4. Integration with recon_plastic_metal

The driver alternates correction at `mbirtorch/tomography_model.py:546` and reconstruction at
`mbirtorch/tomography_model.py:557`, for `num_BH_iterations` passes
(`mbirtorch/tomography_model.py:420-580`).  Three ways to join it:

**Replace the theta fit.**  The estimator would stand in for `_estimate_BH_model_params`
(`mbirtorch/preprocess/mar.py:534-629`) at its call site
(`mbirtorch/preprocess/mar.py:807`).  What changes: the fit becomes a search over an image score,
which needs a reduced model, a reduced sinogram, and a basis stack, all built from the `p` and `m`
the correction already holds; the OSQP constraint machinery is dropped.  What stays: the driver,
the correction, the alternation, the segmentation.  The risk is dimension.  Theta has 6 to 54 free
parameters, and the geometry estimator's own record says an image score separated candidates half a
channel apart and no closer (`plans/features/geometric_calibration/estimate_by_recon.md:33-37`).  A
search in six or more dimensions on such a score is a different problem from the one- and
two-parameter searches that plan gates.

**Choose the hyperparameters.**  Search `order`, `alpha`, `beta`, and `gamma`, which are four
scalars with small ranges and untested defaults (`mbirtorch/tomography_model.py:420-424`,
`mbirtorch/preprocess/mar.py:741-742`).  What changes: a new function loops over settings, calls the
existing fit and correction for each, reconstructs the slab, and scores it; the chosen values are
then passed to `recon_plastic_metal`, which already takes them.  What stays: everything inside
`correct_sino_plastic_metal`.  The basis-image trick does not apply, because changing `order`
changes the columns and changing `beta` changes theta nonlinearly, so each setting costs one fit
plus one reduced reconstruction.  This is the cheapest path, and it is the one the
sweep-do-not-guess rule points at.

**Serve as a check.**  Score the slab reconstruction after each pass and warn when the score gets
worse.  What changes: one score call in the driver loop after
`mbirtorch/tomography_model.py:557`.  What stays: everything else.  This is the smallest change and
it catches the failures the code already anticipates: a non-solved OSQP status keeping the previous
theta (`mbirtorch/preprocess/mar.py:620-627`) and a negative mean of `Sp`
(`mbirtorch/preprocess/mar.py:707-708`).  Neither is checked against the image today.

## 5. An API sketch

Mirror `CalibrationResult` (`mbirtorch/preprocess/geometry_calibration.py:69-88`) and add what a
vector-valued answer needs:

- `parameter`, the name searched, for example `'bh_beta'`, and `value`, the scalar at the minimum,
  or NaN when undecided, as the geometry plan's API section specifies;
- `theta`, the coefficient vector at that value, and `exponents`, the `H` exponent list, so theta is
  interpretable without rebuilding it;
- `hyperparameters`, a dict of `num_metal`, `order`, `alpha`, `beta`, `gamma`;
- `classes`, the Otsu thresholds `segment_plastic_metal` used
  (`mbirtorch/preprocess/segmentation.py:365`, `370-371`) and the class scales, so the segmentation
  is recorded rather than recomputed;
- `candidates`, `scores`, `score`, `method='recon'`, `notes`, `undecided`;
- `reduction`, the record from `build_reduced_problem` plus the scored slices and the blur width in
  ALU.

A signature in the module's style:

```python
def estimate_bh_from_recon(ct_model, sino, recon, *, parameters='beta', bounds=None,
                           num_coarse=7, num_metal=1, order=3, alpha=1.0, gamma=0.1,
                           num_slices=4, slice_indices=None, blur=None,
                           view_stride=4, bin_factor=1) -> dict[str, BHCalibrationResult]
```

The default `bin_factor=1` follows section 1.  The function belongs in
`mbirtorch/preprocess/mar.py` rather than `geometry_calibration.py`, because it needs the
segmentation, the exponent list, and the correction, and because `geometry_calibration.py` fixes its
own place in the preprocessing order before any reconstruction exists
(`mbirtorch/preprocess/geometry_calibration.py:14-21`).

## 6. Recompile and retrace hazards

`set_params` rebuilds the projectors when any changed parameter carries `recompile_flag`
(`mbirtorch/parameter_handler.py:297-318`, `343-344`), which runs `refresh_device_bindings` and
`create_projectors` (`mbirtorch/tomography_model.py:903-918`).  `parameter_sweep` documents the
cost: the first changed detector offset costs one retrace of the compiled bodies, and later values
do not (`mbirtorch/preprocess/geometry_calibration.py:468-471`).

A beam-hardening search touches none of that.  A candidate theta changes sinogram values only; the
geometry, the shapes, and the devices are fixed, so the search calls `set_params` zero times.  A
search over `order` also retraces nothing, because the order changes only the number of host-side
column builds.

What does cost traces is building the reduced model.  `build_reduced_problem` calls
`copy_ct_model`, `auto_set_recon_geometry`, and `configure_devices`
(`mbirtorch/preprocess/geometry_calibration.py:304-351`), and `configure_devices` rebuilds the
projectors when they already exist (`mbirtorch/tomography_model.py:1100`).  Build the reduced model
once and reuse it for every candidate, as `parameter_sweep` does
(`mbirtorch/preprocess/geometry_calibration.py:521-537`).

The basis stack itself adds no shape.  Compiled callables are cached per function and per device
instance, and torch.compile specializes per input shape within a raised recompile budget
(`mbirtorch/projectors.py:29-38`, `50-66`, `123-162`); `N` basis images are `N` reconstructions at
one shape.  Two things would add shapes: a coarse level and a fine level with different recon
shapes, and a per-view-batch model copy for the mask projections whose last batch has a different
view count.  Making every view batch the same size removes the second.

## 7. Gates

**The existing MAR golden carries no beam hardening.**
`tests/generate_preprocess_goldens.py:166-169` builds a two-level block phantom, plastic at 0.02 and
metal at 0.2, and forward projects it linearly.  The hardening goldens at
`tests/generate_preprocess_goldens.py:107-115` are a one-dimensional curve fit on synthetic samples,
unconnected to that sinogram.  So the existing golden cannot test recovery of a known theta, and a
new synthetic case is needed.

**The pieces for one exist.**  `apply_beam_hardening_curve`
(`mbirtorch/preprocess/utilities.py:1340-1362`) applies a fitted hardening curve elementwise to any
array, so a known nonlinearity can be injected.  `generate_demo_data`
(`mbirtorch/utilities.py:1715-1736`) and `generate_3d_shepp_logan_low_dynamic_range`
(`mbirtorch/utilities.py:263`) build the object and the cone geometry that
`tests/test_geometry_calibration.py:64-67` and `116-132` already use.  The golden's two-level
phantom is the smallest object with both classes.  The gate: project that phantom, apply a known
hardening to the metal class projection, fit and correct, and require the estimator to score the
true setting best and recover the injected coefficients within a stated tolerance.  A second gate
mirrors the geometry plan's binning gate and directly tests section 1's bound: the ranking must be
unchanged at a detector binning of 2.  The current MAR goldens stay as a no-regression gate,
including `test_bh_correction` (`tests/test_preprocess_mar.py:64-68`).

**A real-scan gate without ground truth.**  The NSI phantom was scanned with and without the metal
insert, so the plastic region is the same object in both.  Comparing the corrected metal scan's
reconstruction with the no-metal scan's reconstruction over the plastic region is the one real-data
comparison with a defensible reference.  Two weaker measures back it: the standard deviation inside
the plastic mask (`mbirtorch/preprocess/segmentation.py:388`), which streaks and cupping raise, and
the normalized high-pass residual `_direct_residual_score` already computes
(`mbirtorch/preprocess/geometry_calibration.py:544-583`), which says whether the corrected sinogram
is more consistent with a linear model than the raw one.
