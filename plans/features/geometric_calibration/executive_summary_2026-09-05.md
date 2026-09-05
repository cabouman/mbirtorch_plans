# Geometric calibration: executive summary, 2026-09-05

The geometry calibration module estimates two scanner parameters from the scan data itself, and
it checks a third setting.  Correct values matter because a geometry error blurs the
reconstruction and adds rings.  The three functions work as follows:

- `estimate_det_channel_offset` estimates the detector channel offset.  In a scan over a full
  rotation every ray is measured twice, once from each side.  The function pairs each view with
  its opposite view, mirrors the opposite along the channel axis, and finds the channel shift
  that best aligns the pairs.  Half of that shift is the offset.  In cone beam the opposite
  view of each channel is chosen by its fan angle.
- `estimate_det_rotation` estimates the in-plane detector rotation, using the same view pairs.
  It rotates a band of rows around the central plane by each candidate angle and returns the
  angle at which the pairs agree best.
- `check_rotation_direction` checks the sign of the rotation direction.  It reconstructs a
  reduced problem twice, once with the view angles as given and once with them negated, forward
  projects each reconstruction, and keeps the direction whose data residual is lower.

This page is the short form, updated late on 2026-09-05 after the follow-up experiments.  The
plan of record is `geometric_calibration_plan_v2.md` in this directory, and the new estimator's
plan is `estimate_by_recon_plan.md`.  The evidence behind the morning's status, with sources, is
in `closed/status_2026-09-05.md`; the follow-up records are in
`plans/experiments/features/geometric_calibration/` and its `closed/`.

## Where the work stands

The three functions were run on four real scans from two scanners, and how well each one agrees
with the scanners' own recorded values differs by function:

- `estimate_det_channel_offset` agrees with the vendors' recorded offsets to within a tenth of
  a channel, and it tracks a known integer roll of the data to a twentieth of a channel.  It
  passed every accuracy gate the plan set.  Three of the four scans are one object, so the real
  evidence covers two objects and two scanners.
- `estimate_det_rotation` disagrees with the vendor's recorded rotation on one object.  Its
  zero point is the angle it returns on the data as given, which is the argmin of its score
  over candidate angles.  On this object that argmin sits at 0.047 degrees, where
  reconstructions show the vendor's 0.167 degrees to be the best of the values tested, so the
  argmin is displaced by roughly 0.12 degrees.  The displacement is a constant: adding a known
  rotation to the data moves the estimate one for one.  The docs now tell users to prefer a
  vendor value.  The cause and the remedy are settled below.
- `check_rotation_direction` agrees with the readers' geometry on three of four scans.  Its
  warning fired on the one wrong answer and also on a correct answer, so the warning marks
  uncertainty rather than error.  Widening the filter raised the margin on one scan and lowered
  it on the other, so the check needs agreement across settings rather than one default
  (`closed/real_scan_followup.md`).

Two limits apply to all three functions.  They run on the host in numpy, with about 50 GB of
peak memory on the largest real scan, and they refuse a sinogram already divided across
devices.  Helical and multiaxis scans have no automatic estimator; the manual path through
`parameter_sweep` serves them.

The first plan's increments stand as follows; the new plan renumbers the remaining work:

- Increments 1 and 2 are committed.  They add the module, `parameter_sweep`, and the two
  conjugate-view estimators.
- Increment 6 is staged and uncommitted.  It adds the docs, the Zeiss `det_rotation` argument, and the demo.
  The full test suite has not been run on the staged state, because another session may be
  running it.
- Of the first plan's remaining increments, none is built: Increment 3, the short-scan
  estimates, Increment 4, the residual method, and Increment 5, the one-call driver.
  Increments 7 and 8, the parameter-system work, are untouched.

One new fact came from the real scans: a real short scan exists.  The Zeiss scan `z62` covers
218 degrees.  On `z62` the manual sweep works, and no automatic estimate does.

## The answer to the synthetic-data question

Greg asked whether the synthetic examples were made by rotating the sinogram with the same code
the correction uses.  The answer splits by parameter:

- For the offset, no.  The true offsets entered through the projector.  The estimator and the
  real-scan tests use two other mechanisms, so nothing is compared with itself.  The offset
  results stand.
- For the rotation, yes.  Every synthetic rotation was applied with the module's own bilinear
  kernel, because the projectors have no rotation parameter.  The tests were built so that
  this sharing favors no interpolation kernel.  A new measurement, recorded in
  `rotation_kernel_conventions.md`, shows they would have caught a sign or center error.

The synthetic rotation results therefore check consistency, not correctness.  The independent
check came from reconstructing real scans at candidate rotations.  That check confirmed the
correction code, and it exposed the estimator's zero-point error.

## What the follow-up experiments settled

Four measurements ran after the morning's status.  They settled the cause of the zero-point
error and left its exact value open:

- The cause is the object, and two independent experiments showed it.  The band estimator
  needs in-plane structure across its band, and a phantom weak in that structure made it
  under-read a known angle, on data from mbirtorch's projector and again on data from LEAP's,
  which shares no code with mbirtorch (`rotation_zero_point_synthetic.md`,
  `leap_axis_tilt.md`).
- The rotation axis's out-of-plane lean is real and is not the cause.  The vendor's config
  vectors split the misalignment into 0.167 degrees in plane and 0.079 out of plane
  (`closed/real_scan_band_height.md`).  Simulating the lean in LEAP moved the in-plane estimate
  by at most 0.011 degrees, and the metal scan carries the same lean while the band estimator's
  answers there stayed near their 0.15-to-0.19 bracket (`leap_axis_tilt.md`,
  `closed/real_scan_band_height.md`).
- A taller comparison band recovers only part of the error, at a real cost.  The no-metal
  estimate moves from 0.047 to 0.130 degrees by a 501-row band, which took 21 minutes, and a
  1001-row band returns the same value at 43 minutes and a 139 GB peak
  (`closed/real_scan_band_reach.md`).  Whether reconstruction quality prefers 0.130 or the
  vendor's 0.167 is the open question the new estimator's first sub-increment settles.
- The 200-view and 1800-view scans are one acquisition: matched frames differ by exactly zero,
  so their identical estimates were guaranteed (`closed/real_scan_band_height.md`).

The remedy is a new estimator, `estimate_geometry_from_recon`.  It scores reconstructed
slices directly, which is the measurement that ranked the vendor's value first among four
candidates when the band estimator could not.  Its design and its panel-reviewed plan are
`estimate_by_recon.md` and `estimate_by_recon_plan.md`.

## The comparison with LEAP

LEAP is the reference package this feature is measured against, and the two packages fail in
different places:

- On the offset the two packages agree on the `bga` scan to a thousandth of a channel.  On the
  NSI scans LEAP reads 0.21 to 0.98 channels higher than the module, while the module stays
  within 0.08 channels of the vendor.  The cause of LEAP's difference has not been tested.
- On synthetic data the module's offset is the more accurate.  It is within 0.005 channels
  where LEAP errs by up to 0.024.  LEAP is four to six times faster.
- On the rotation neither package measured the NSI value.  LEAP's `estimate_tilt` has no
  usable minimum on these scans and returns its search bound or a drift, with a row band or
  with the full detector height.  The module has a clear minimum at the wrong angle.  The two
  failures are different, and neither package offers a fix for the other.

## The remaining work, in the plan's order

The plan of record is `geometric_calibration_plan_v2.md`, and its increments are these:

- Increment 1 builds `estimate_geometry_from_recon`, in five sub-increments with their own
  gates.  Six days.  Sub-increment 1.1 is one cluster job that settles the 0.130-versus-0.167
  question.
- Increment 2 adds short-scan redundancy weighting to the direct reconstruction, which makes
  `z62` usable by the sweep and by the new estimator's joint mode.  A day.
- Increment 3 makes the direction check run at several filter widths and report a direction
  only when the widths that clear its margin agree.  A day.
- Increment 4 builds the driver, `calibrate_geometry`, which runs the calibration sequence in
  one call.  A day.
- Increment 5 measures the joint mode on `z62` and records the retirements that result: the
  short-scan conjugate method and the residual method may retire, on the criterion the plan
  states.  A day.
- Increment 6 is the parameter-system work of the first plan, carried over unchanged; it can
  run at any point.  Six days.

Four items stay parked, and none blocks the work above.  The first is the unread angular ranges
of the depot's remaining Zeiss scans.  The second is LEAP's unexplained offset difference on the
NSI scans.  The third is a `det_rotation` parameter inside the projectors, which would make the
new estimator differentiable and allow estimating geometry inside an iterative reconstruction.
The fourth is the rotation axis's out-of-plane lean, which needs no correction on the evidence.
