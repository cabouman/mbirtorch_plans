# Estimating geometry from reconstructions: an outline of `estimate_geometry_from_recon`

Date: 2026-09-05.  Status: the design behind `estimate_by_recon_plan.md`, which Greg's panel
review shaped; this outline itself was not separately panel-reviewed.  Greg chose the function
name `estimate_geometry_from_recon` on 2026-09-05, and this file keeps its earlier short name.
Every number cited here was read in this session from the record named beside it, in this
directory or in `plans/experiments/features/geometric_calibration/`; records of the closed
campaign now live under `closed/` in each of those two directories.

`estimate_geometry_from_recon` estimates scan geometry by reconstructing slices at candidate values and
scoring their quality.  A wrong offset doubles edges and draws rings, and a wrong rotation blurs
the slices far from the central plane, so the reconstruction itself is the measurement.  The
principle is proven on the failing case: direct reconstructions at four rotations ranked the
vendor's 0.167 degrees first on every far slice of both NSI scans
(`real_scan_rotation_recon.md`), on the same object where the conjugate-view band estimator
reads 0.047 (`real_scan_validation.md`).  It is a standalone,
optional estimator that needs no opposite views, no vendor value, and no model fit.

## Two modes, and when to use each

The estimator has a rotation-only mode and a joint mode:

- Rotation only.  The channel offset is taken as given, from the conjugate-view estimator or a
  vendor value, and only the rotation is searched.  This is the default companion to the existing
  offset estimator on full rotations.  Use it when the reader supplies no vendor tilt, or to check
  one, because the band estimator's zero point (the angle that minimizes the score
  over candidate angles) depends on the object and was wrong by 0.12 degrees
  on a real scan (`closed/increment_6_findings.md`).
- Joint.  Both `det_channel_offset` and the rotation are searched, by block coordinate descent
  over the same machinery.  This is the fallback for scans the conjugate method refuses: the short
  scan `z62`, and any scan where an independent cross-check of both values is wanted.

The precision differs by mode, and that decides which mode serves which scan.  The conjugate offset
estimator resolves about 0.01 channels and passed every real-scan gate
(`real_scan_validation.md`).  An image-quality metric separated offset candidates half a channel
apart on `z62` and could not separate candidates 0.02 to 0.07 channels apart on any scan
(`real_scan_followup.md`, `real_scan_validation.md`).  So the joint mode replaces nothing where
the conjugate method works; it serves the scans where nothing else works.  The rotation mode's
resolution is about 0.02 degrees, from candidates 0.023 degrees apart that the far-slice measures
separated by a few percent (`real_scan_rotation_recon.md`).

## The pipeline

The estimator runs five steps:

1. Scout.  One cheap direct reconstruction at reduced resolution finds the scoring slices: for
   the rotation, three to five slices far from the central plane with high cross-row content; for
   the offset, central slices with strong in-plane structure.  Slices near dense features are
   avoided.
2. Reduce.  Each scoring slice needs only the detector rows that feed it, which
   `parameter_sweep`'s row crop already computes.  The coarse level adds a view stride of about 4
   and a detector binning of 2 to 4 through `reduce_sinogram`.
3. Search, at two levels.  A coarse grid at the reduced resolution localizes the minimum, and a
   golden-section polish at full resolution on the cropped rows refines it.  The joint mode
   alternates the two parameters for two rounds, which the measured coupling of about 0.1
   channels per 0.1 degrees allows (`real_scan_rotation_check.md`).  A candidate rotation is
   applied by resampling, as today.
4. Score.  Each candidate reconstructs the scoring slices directly, blurs them at a width fixed
   in ALU, and takes the normalized gradient energy of each, combined as a trimmed mean over
   slices.
5. Check.  The winner is verified two ways: the per-slice minima must agree, and a high-passed
   normalized correlation between the reprojected winner and the measured sinogram must prefer it.

## The metric, and the two rules it must follow

The primary metric is blurred, normalized gradient energy, because it is the one that worked.
The unblurred versions of the same measures ranked the unresampled candidate first everywhere,
reading the interpolation instead of the geometry, and the blur is what removed that
(`real_scan_rotation_recon.md`).  The reprojection residual is the check rather than the target,
because on real data its minimum was 1.5 percent deep where the image measures moved by tens of
percent (`real_scan_followup.md`).  Correlation rather than mean squared error makes the check
insensitive to gain and smooth-bias mismatch.

Two rules follow from measured failures.  Every scale, the blur width and any filter width, is
expressed in ALU rather than pixels, because fixed pixel widths changed what the direction check
measured when the binning changed (`real_scan_followup.md`).  And every comparison is made at a
matched resampling count, which the blur enforces, so no candidate wins by being the unresampled
one.

## Uncertainty and the undecided verdict

The estimator returns its evidence, in the shape `CalibrationResult` already has: the candidates,
the scores, and the value.  Three numbers qualify the value:

- a half width, from a quadratic fit to the score curve at a stated fractional rise;
- the spread of the per-slice minima, as a trimmed standard deviation;
- the agreement between the coarse and fine levels, in coarse grid steps.

The estimator reports undecided rather than a number when the per-slice minima disagree beyond a
tolerance, when the coarse and fine levels disagree by more than one coarse step, or when the
correlation check prefers a different candidate.  An undecided answer with the curves attached is
the designed outcome on an object that carries no signal.  The band estimator's failure was to
return a confident number in exactly that situation.

## Cost

On the H100, one slice at four candidates took 3 to 20 seconds at full resolution in the
far-slice job (`real_scan_rotation_recon.md`), so a search of about twenty evaluations over three
slices is a few minutes, and the coarse level is 16 to 64 times cheaper than that.  The two-level
scheme is what makes a CPU run practical.  Rough estimate for the build: two to three days, with
a synthetic gate and a real gate.

## Risks, and the measurements that settle them

- Beam hardening.  Cupping is smooth and the gradient metric barely sees it.  Streak behavior
  under candidate changes is unmeasured; the encouraging fact is that the metal scan's far
  slices, which carry hardening and an insert, still ranked the true rotation first
  (`real_scan_rotation_recon.md`).  The gate: inject the validation harness's quadratic
  hardening over the whole sinogram and require the candidate ordering to hold.
- Short scans.  `recon_direct` applies no redundancy weighting, so every candidate's slice
  carries the limited-angle artifact (`real_scan_followup.md`).  The weighting, about a day, is
  a precondition for using either mode on `z62`.
- The coarse level on weak signal.  The no-metal scan is the weakest case measured, so the gate
  requires the coarse argmin to land within one grid step of the fine answer there, else the
  coarse level keeps full channels and bins rows only.
- The offset resolution limit.  The joint mode's offset is a fallback figure, about half a
  channel, which is the separation the real sweeps resolved, and the docstring says so.

## What this builds toward

Two later steps are kept open by the design.  The metric path stays in torch with no numpy
detours, so gradients through the resampler and the back projection are available without
redesign when the parameter count grows past two.  And a `det_rotation` inside the projectors, the change the plan defers,
would remove the resampling from the loop entirely and enable the full iterative version, in
which the geometry is one more parameter of the reconstruction.  The out-of-plane lean the config
vectors record, 0.079 degrees, stays outside every mode here by design: it does not bias the
in-plane estimate (`leap_axis_tilt.md`), and the in-plane correction is what reconstruction
quality wants (`real_scan_rotation_recon.md`).

## Relation to the open decisions

This estimator is the far-slice remedy of the rotation zero-point decision, generalized to both
parameters (`closed/status_2026-09-05.md`, decision 1).  The reach job has since run: a band of
hundreds of rows recovers the conjugate estimate to 0.130 degrees against the vendor's 0.167, at
21 to 43 minutes and up to 139 GB of host memory per estimate (`real_scan_band_reach.md`).  The
tall band is therefore a partial and expensive repair, this estimator is the remedy, and its
first increment settles which of the two values reconstruction quality prefers.  The driver of
decision 4 would call the rotation mode when the reader supplies no vendor tilt.
