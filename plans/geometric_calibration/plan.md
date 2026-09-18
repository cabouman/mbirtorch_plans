# Geometric calibration, plan of record v2

Date: 2026-09-05.  Status: APPROVED by Greg on 2026-09-05.  A panel of three reviewed the
first draft, and this version applies their findings.  This plan supersedes
`closed/geometric_calibration_plan.md`, called v1 below.  The estimator this plan builds first
has its own detailed plan, `estimate_by_recon_plan.md` in this directory, whose increments are
numbered 1.1 to 1.5 here.  The current evidence is summarized in
`executive_summary_2026-09-05.md`.

Citations.  Pages of this directory's closed campaign are cited as `closed/NAME.md`.  Experiment
records are cited by bare name and live in `plans/experiments/features/geometric_calibration/`,
the closed campaign's in the `closed/` subdirectory there.  mbirtorch file paths are given from
the package directory, so `mbirtorch/cone_beam.py` means `mbirtorch/mbirtorch/cone_beam.py` in
the sibling repository.  Measured numbers were read in this session from the cited records.

## Terms and scans

The terms of `estimate_by_recon_plan.md`'s "Units and terms" section apply here unchanged.  In
particular: the band estimator is `estimate_det_rotation`, its zero point is the angle it
returns on the data as given, the central plane is the plane through the source perpendicular to
the rotation axis, and a short scan covers 180 degrees plus the fan angle.

An offset scan displaces the detector laterally by hundreds of channels, so each view covers
about half the object and a full rotation covers all of it; the displacement enlarges the field
of view.  Four real scans carry the evidence.  Three are one NSI artifact phantom: `nsi_small` at 200
views, `nsi_no_metal` at 1800, and `nsi_metal` at 1800 with a metal insert; the 200-view file
holds every ninth frame of the 1800-view acquisition, exactly
(`closed/real_scan_band_height.md`).  The fourth is `bga`, a Zeiss scan of a ball grid array.  A
fifth scan, `z62`, is a Zeiss short scan of 218 degrees that the conjugate-view method refuses.

The calibration sequence is the recommended order of calls: `estimate_det_channel_offset`; then
the rotation, from a caller-supplied vendor value when one exists and from
`estimate_geometry_from_recon` when none does; then `estimate_det_rotation` as a cross-check,
recorded and not applied; then `estimate_det_channel_offset` again at the chosen rotation.  One
function, `calibrate_geometry`, runs the sequence.  To retire a method means to record, in this
plan's corrections list and in the docs, that it will not be built.

## Executive summary

The feature lets mbirtorch estimate its own scan geometry from the sinogram.  The first release
is built and validated on real scans.  v1's Increments 1 and 2 are committed as `550b5d3` and
`4781600`, and v1's Increment 6 set, the docs, the Zeiss `det_rotation` argument, and the demo,
is implemented, staged, and uncommitted, with its gates recorded in
`closed/increment_6_findings.md`.

The direction changes in one way.  Geometry estimation no longer rests on view-pair comparisons
alone, and reconstruction quality becomes the primary evidence for the rotation.  Three
measurements forced the change.  The band estimator returned a wrong angle, with a deep score
minimum and no warning, on an object whose in-plane structure across the band is weak
(`closed/real_scan_validation.md`, `closed/real_scan_rotation_recon.md`).  Widening its band
moved the estimate from 0.047 only to 0.130 degrees against the vendor's 0.167, at 21 to 43
minutes and up to 139 GB per estimate (`closed/real_scan_band_reach.md`).  Reconstructing far
slices at candidate angles ranked the vendor's value first among four candidates on the same
object (`closed/real_scan_rotation_recon.md`).  The new estimator,
`estimate_geometry_from_recon`, makes that reconstruction scoring automatic.  It also serves
helical and multiaxis scans, and, once short-scan weighting exists, short scans.

## The work

Increments 1 to 5 run in order, with a review stop at the end of each; Increment 1 carries five
sub-increments with their own stops, so the work to Increment 5 has nine stops.  Increment 6 is
an independent line of work that may start at any time, with its own stops.  A gate that needs
a GPU runs as a cluster batch job, and jobs on the real scans request two GPUs for host memory.
The estimator work runs first because its first sub-increment is the measurement that the
0.130-versus-0.167 question, the staged warning texts, and the retirement decisions all wait on.

**Increment 1.  `estimate_geometry_from_recon`.  Rough estimate 6 days.**  This increment is the
plan of `estimate_by_recon_plan.md`, whose sub-increments 1.1 to 1.5 name their own files,
tests, and gates.  Sub-increment 1.1 settles whether reconstruction quality on `nsi_no_metal`
prefers 0.130 or 0.167 degrees, and its deliverables include revising the four staged texts that
today assert the vendor's value as right, in `docs/source/usr_preprocess.rst` and the module's
docstrings and warning, so that what is committed matches the measurement.

**Increment 2.  Redundancy weighting for the direct reconstruction.  Rough estimate 1.5
days.**  This increment adds angle-based redundancy weighting to the direct reconstruction of
the circular geometries, in `mbirtorch/cone_beam.py` and `mbirtorch/parallel_beam.py` over the
shared entry in `mbirtorch/tomography_model.py`, with helpers in `mbirtorch/tomography_utils.py`;
helical and translation geometries are out of its scope.  Tests go in
`tests/test_recon_simple.py`.  Two assumptions of today's `recon_direct` are removed rather than
worked around, because the gate scan itself violates one of them: `recon_fdk` assumes equally
spaced views over a full rotation (`mbirtorch/cone_beam.py:806`), and the textbook short-scan
weights assume an extent of exactly 180 degrees plus the fan angle, while `z62` covers 218.0
degrees against a short-scan extent of 207.9 (`closed/real_scan_validation.md`).  The weighting
is therefore the general form: for each measured ray, a smooth weight normalized over all of
that ray's measured conjugate copies within the scan's actual angles, and the backprojection
takes a per-view angular width, so irregular spacing, over-scans, and multi-turn scans are
served by one formula.  An extent below 180 degrees is accepted and weighted, and the
limited-angle caution stays, because missing rays cannot be weighted into existence; nothing is
claimed about estimator usability at such an extent.  It has five gates.  For every ray with at
least one measured copy, the weights of its copies sum to one, tested directly on the weight
array over the extents 120, 207.9, 218, 360, and 720 degrees.  On an equally spaced full
rotation the weights are identically one and the weighted path returns the unweighted result
exactly, which is a constructed identity rather than a computed-float comparison.  A full
rotation with every view duplicated reconstructs to the single-copy result within 1e-6
relative, which tests the multiplicity normalization and the per-view width together.  On a
synthetic short scan the weighted direct reconstruction's error against the full-rotation
reconstruction is at most half the unweighted error.  On `z62` the direct-residual score's rise
two channels from its minimum at least doubles the recorded 1.5 to 1.8 percent, written to a
new findings page (`closed/real_scan_followup.md`).

**Increment 3.  The direction-check rule.  Rough estimate 1 day.**  This increment makes
`check_rotation_direction` in `mbirtorch/preprocess/geometry_calibration.py` score both
directions at several filter widths and return a direction only when every width that clears its
margin gives the same answer, and undecided otherwise, with tests in
`tests/test_geometry_calibration.py`.  The filter width is the varied setting rather than the
bin factor, because changing the bin factor changed the 200-view scan's answer while width
changes moved only the margin (`closed/real_scan_followup.md`); the widths are already arguments
of `sino_high_pass_filtering`.  The check's docstring and warning drop the advice to raise the
bin factor, which the findings called unsound (`closed/increment_6_findings.md`).  It has three
gates.  On `nsi_small` the check returns undecided or the readers' direction, not the wrong
direction.  On the other three measured scans the answers are unchanged.  The cost is recorded,
and it stays below one default run on `nsi_small` and below four default runs on `bga`, the
bounds the measured width costs imply (`closed/real_scan_followup.md`).

**Increment 4.  The driver.  Rough estimate 1 day.**  This increment adds `calibrate_geometry`
to `mbirtorch/preprocess/geometry_calibration.py`, running the calibration sequence read-only,
with tests in `tests/test_geometry_calibration.py`.  Its signature takes `det_rotation=None` for
a caller-supplied vendor value, because no reader can hand one over: the NSI reader applies its
tilt inside `scan_to_sino` and does not return it, and the Zeiss readers have none
(`mbirtorch/preprocess/nsi.py:130-136`).  The second offset pass calls
`estimate_det_channel_offset` with its `det_rotation` argument, which resamples only the
comparison band, so the driver allocates no sinogram-sized array.  It has three gates.  On a
synthetic joint case at 512 channels the driver returns exactly the values the estimators return
when called by hand in the same order, and it records how far the second offset pass moved the
estimate.  A cluster job on `nsi_no_metal`, `nsi_metal`, and `bga` shows the same equality
against by-hand sequences.  A refusal in any step propagates with that step's message.

**Increment 5.  The `z62` measurement and the retirements.  Rough estimate 1 day.**  This
increment runs the joint mode on `z62` as a cluster job, after Increment 2 has lifted the
short-scan refusal, and records what the result decides.  It has two gates.  The joint mode's
offset on `z62` agrees with the two independent estimates already recorded, LEAP's -0.802 and
the direct-residual score's -0.806 channels, within the mode's documented resolution of about
half a channel, with the vendor's -0.928 recorded as a sanity comparison only
(`closed/real_scan_followup.md`).  The rotation result's curve passes the estimator's own
undecided rules.  If both gates pass, the short-scan conjugate method of v1's Increment 3 is
retired, resting on this one scan because the retirement is reversible and the depot inventory
stays parked; the residual method of v1's Increment 4 is retired on coverage grounds, because
the new estimator serves the helical and multiaxis scans it existed for and none is in hand,
and either retirement reverses if a scan defeats the new estimator.  If a gate fails, this plan
gains an increment chosen on that evidence.

**Increment 6.  The parameter work, carried from v1.  Rough estimate 6 days.**  This increment
is v1's Increments 7 and 8, unchanged: the runtime detector offsets, and the parameter-table
review with `ParamDict`, the flag rename, and the name-coordination test.  Their full
statements, files, tests, and gates are v1's increment blocks
(`closed/geometric_calibration_plan.md`, "Increments", Increments 7 and 8), with the design in
that plan's "Parameter system" section and the prototype in
`../runtime_offsets/runtime_offsets_findings.md`.

## Where each v1 item went

- v1 Increments 1 and 2: built and committed.  v1 Increment 6: built and staged.
- v1 Increment 3, the short-scan conjugate method: retirement decided by Increment 5 here.
- v1 Increment 4, the residual method: retirement decided by Increment 5 here.
- v1 Increment 5, the driver: Increment 4 here, without the joint five-by-five grid, which is
  dropped; the one hand-run joint case ended within the search tolerance
  (`closed/increment_3_evaluation.md`).
- The derivative-filter method: dropped by Greg's decision of 2026-09-04, recorded in v1's
  corrections list.
- The `method='auto'` short-scan fallback of v1: superseded by the refusal message and, after
  Increment 2, by the new estimator's joint mode.
- `estimate_det_row_offset` and `estimate_magnification`: still deferred, for v1's stated
  reasons.
- Offset scans, whose detector is displaced by hundreds of channels: still not served, as the
  module docstring says; v1's v3 change list said otherwise and lapses with v1.
- Decision 5's four working-copy edits (`closed/status_2026-09-05.md`): lapse with v1, except
  the offset-scan line, which the bullet above restates.
- v1 Increments 7 and 8: Increment 6 here.

## Scope limits that stand

Three statements bound the scope.

Device use follows the split the preprocessing already uses.  The data-reduction steps and every
candidate reconstruction of the new estimator run on the devices, through `reduce_sinogram`'s
batch pipeline and the model, while the pairing, scoring, and search arithmetic stays on the
host, where its measured cost is seconds against the reconstruction's minutes
(`closed/calibration_512_gautschi.md`).  Every entry point refuses a sinogram in the divided
device form the multi-GPU projectors produce, so a caller holding one gathers it explicitly.
The refusal keeps that full-size host allocation visible instead of hiding it inside an
estimator, and calibration runs before reconstruction in the workflow, where the sinogram is a
host array in any case (`closed/increment_3_evaluation.md`).

Offset scans are not served.  An offset scan's view and its opposite overlap only in part, and
the conjugate comparison is not built for partial overlap; the direct reconstruction also lacks
the lateral redundancy weighting such a scan needs, which the LEAP comparison lists as its own
feature, so the new estimator cannot score one either
(`closed/geometric_calibration_plan.md`, "Ordering, scope, and deferred work").

The band estimator keeps its current form behind its warning: the failure detector and the
widened-band remedy of decision 1 were not built, because the new estimator covers their
purpose, and the band estimator's role narrows to the recorded cross-check in the calibration
sequence.

## Parked

Each parked item names the evidence that would activate it.

- The depot's remaining Zeiss scans have unread angular ranges; reading them would say how
  common short scans are in the collection.
- LEAP's `find_centerCol` reads 0.21 to 0.98 channels above the module on the NSI scans
  (`closed/real_scan_validation.md`); the two ablations of decision 7 fit inside any future
  cluster job (`closed/status_2026-09-05.md`).
- A `det_rotation` inside the projectors would remove candidate resampling, make the
  reconstruction score differentiable, and allow the version in which geometry is one more
  parameter of an iterative reconstruction.  A scan the new estimator cannot serve would
  activate it.
- The rotation axis's out-of-plane lean, 0.079 degrees on the NSI scans, needs no correction on
  the evidence: simulating it moved the in-plane estimate by at most 0.011 degrees, and the
  metal scan carries the same lean while the band estimator's answer there stayed near its
  bracket (`leap_axis_tilt.md`, `closed/real_scan_band_height.md`).

## Constraints carried forward

The constraints of v1 stay in force.  Increments run with the review stops stated above.  A
sinogram correction must not allocate a second full-size sinogram.  Code comments carry no plan
or increment references.  Geometry arithmetic stays in the model classes.  Experiment scripts go
in `plans/experiments/features/geometric_calibration/`.  Each script sets its run parameters at
the top and takes no command-line arguments.  Findings pages go in this directory.  A measured
number appears in a durable record only after it was read from its source in the same session.
Work happens on the `geometric_calibration` branch, staged and not committed without
authorization.

## Validation posture

Three rules from the closed campaign govern new evidence.  A synthetic rotation is injected
either at four times the detector resolution or through LEAP's modular-beam projector, never by
resampling at the detector's own resolution with the kernel under test
(`closed/rotation_kernel_conventions.md`, `leap_axis_tilt.md`).  Real-scan gates use quantities
that need no ground truth where possible: rolls, rotations added in place, agreement between
independent methods, and reconstruction quality.  A deep score minimum is never taken as a right
answer without one of those checks, which is the lesson of the band estimator's zero point.

## Corrections after acceptance

None yet.
