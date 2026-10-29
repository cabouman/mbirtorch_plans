# Decisions: Time-resolved 4D reconstruction

One entry per decision, newest last.  Each entry gives the date, the
blueprint, the question, what was decided, and any alternative that was
rejected.  Rejected alternatives live here, not on the blueprints.

## 2026-10-22, Project scope

- **Question.** How should a 4D reconstruction handle objects that move
  faster than the windowed frames of MACE4D can follow?
- **Decided.** Model each view at its own acquisition time, with the
  frames as the coefficients of a temporal basis.  Use a
  motion-compensated temporal prior.  (Greg)
- **Rejected.** Combining windowed reconstructions that use different
  window spacings.  The rotation sets the period of the modulation, so the
  period in time is the same for every spacing.  All the reconstructions
  also come from the same sinogram.
- **Rejected.** Treating the windowed frames as measurements, with a model
  of their aliasing as the forward model.  The frames are a lossy summary
  of the sinogram, and their noise is correlated across overlapping
  windows.
- **Rejected.** A temporal low-pass filter on the windowed frames as the
  only remedy.  The filter removes the modulation only when the temporal
  bandwidth of the object is below one cycle per rotation.

## 2026-10-22, Front page

- **Question.** Which blueprints does the project need?
- **Decided.** The four core blueprints and Testing and Verification.
  The Hardware Mapping blueprint also compares compressed representations
  of the motion field with a dense field.  (Greg)
- **Rejected.** Pre and Post Processing, Documentation, and Release and
  Maintenance.  These parts work as they do for MACE4D now.

## 2026-10-22, Theory, motion model

- **Question.** Which motion model does the first version use?
- **Decided.** Rigid parts plus springs.  The object has a static part,
  moving parts that translate, and springs between them that stretch or
  compress.  (Greg)
- **Rejected for now.** A general displacement field with three numbers
  for every voxel.

## 2026-10-22, Theory, temporal basis

- **Question.** Should the data term interpolate between frames with hat
  functions or with motion-compensated interpolation?
- **Decided.** Hat functions, for now.  (Greg)
- **Rejected for now.** Motion-compensated interpolation between frames.
  It would allow fewer frames, but it makes the data-fit agents depend on
  the motion.

## 2026-10-22, Theory, temporal potential

- **Question.** Should the potential of the temporal prior be the qGGMRF
  potential or a quadratic?
- **Decided.** The qGGMRF potential.  (Greg)
- **Rejected.** A quadratic potential.  Its proximal map is linear and
  faster, but it does not tolerate errors of the motion model.

## 2026-10-22, Theory, spatial denoisers

- **Question.** Should a 3D qGGMRF denoiser on each frame replace the
  three denoisers of MACE4D?
- **Decided.** Yes.  (Greg)
- **Rejected.** The three qGGMRF denoisers of MACE4D, which act on
  (x, y, t), (y, z, t), and (x, z, t).  They compare the same voxel in
  neighboring frames, which works against the motion compensation.

## 2026-10-22, Theory, labels

- **Question.** Should the user supply the label volume, or should the
  code segment it from a first reconstruction?
- **Decided.** The user supplies it, for now.  (Greg)
- **Rejected for now.** Automatic segmentation of the labels from a first
  reconstruction.

## 2026-10-22, Theory, hard constraint

- **Question.** Should the design offer a mode that reconstructs one
  volume plus the motion, with every frame a warped copy of frame 0?
- **Decided.** Not in this version.  (Greg)
- **Rejected for this version.** The mode with one volume plus the motion.

## 2026-10-22, Theory, start of the outer loop

- **Question.** Should the outer loop start from d = 0, or from a
  registration of the MACE4D reconstruction?
- **Decided.** Start d from a registration of the MACE4D reconstruction
  on which the labels are drawn, raise b_t over the outer iterations, and
  stop when d settles.  (Greg)
- **Rejected.** Starting from d = 0 with the final b_t.  The temporal
  prior then pulls the frames toward a static average, so the registration
  underestimates the motion and the loop can stall near d = 0.

## 2026-10-22, Theory, registration target

- **Question.** Should each moving part be registered to frame 0, or to
  the mean of all frames moved back to frame 0?
- **Decided.** The mean of all frames moved back to frame 0.  (Greg)
- **Rejected.** Frame 0 alone.  It holds the views of one interval only,
  half as many as the other frames hold.

## 2026-10-22, Theory, interpolation

- **Question.** Should the motion operator and the registration use
  trilinear interpolation or cubic B-splines?
- **Decided.** Cubic B-splines, unless they add much to the total
  reconstruction time.  In that case the question returns for discussion.
  (Greg)
- **Rejected.** Trilinear interpolation.  It blurs at sub-voxel shifts, so
  the temporal prior would penalize sharp moving edges.

## 2026-10-22, Theory, boundary band

- **Question.** Should the temporal prior skip a band one voxel wide
  around each moving part and spring?
- **Decided.** Yes.  (Greg)
- **Rejected.** Using every voxel at the boundaries.  Label errors and
  partial-volume voxels there would leave trailing ghosts.

## 2026-10-22, Theory, critique items not adopted

- **Question.** Should the design adopt three further suggestions from the
  independent critique of the Theory blueprint?
- **Decided.** None of them for now.  (Greg)
- **Rejected for now.** Motion-compensated interpolation in the data term
  now, which would make the number of frames independent of the speed of
  the fastest part.  The Hardware Mapping blueprint sizes the memory
  instead.
- **Rejected for now.** A translation of the static part, which would
  absorb stage drift and center-of-rotation errors.  Geometry errors are
  left to calibration.
- **Rejected for now.** Keeping the temporal filter of MACE4D.  The Testing
  blueprint measures any periodic component that remains.

## 2026-10-22, Theory, temporal agents

- **Question.** Should the temporal prior be split into two agents, one
  for the even pairs of frames and one for the odd pairs?
- **Decided.** Yes.  Each agent then solves independent two-frame
  problems.  (Greg)
- **Rejected.** One agent for the whole temporal prior.  Its proximal map
  links all frames, so it would need its own inner solver.
- **Rejected.** Putting each temporal pair into the data-fit agent of its
  interval.  That would mix the motion operator into the CT solver.

## 2026-10-22, Software Architecture, new class

- **Question.** Should the time-resolved reconstruction be a new class in
  a new module, or a mode of `MACE4DModel`?
- **Decided.** A new class in a new module.  (Greg)
- **Rejected.** A mode of `MACE4DModel`.  Its inputs, outputs, and agents
  differ from those of the time-resolved reconstruction.

## 2026-10-22, Software Architecture, two-frame data-fit proximal map

- **Question.** Should the two-frame proximal map extend the VCD code of
  `TomographyModel`, or be a separate implementation?
- **Decided.** A new implementation in its own file, `two_frame_prox.py`.
  It calls the sparse projections, the Hessian diagonal, the pixel
  partitions, and the partition iterator of `TomographyModel`, and
  `tomography_model.py` does not change.  (Greg)
- **Rejected.** Adding the two-frame case to `tomography_model.py`.

## 2026-10-22, Software Architecture, spatial denoiser

- **Question.** Should `HyperplaneAgent` accept axis 0, or should a new
  agent denoise the frames?
- **Decided.** `HyperplaneAgent` accepts axis 0 and rejects a temporal
  filter with axis 0.  (Greg)
- **Rejected.** A new agent that denoises the frames.

## 2026-10-22, Software Architecture, labels

- **Question.** Should the labels be one `uint8` volume with a dict that
  maps each label to `'part'` or `'spring'`?
- **Decided.** Yes.  Label 0 is the static part.  (Greg)
- **Rejected.** No alternative was proposed.

## 2026-10-22, Software Architecture, starting point

- **Question.** Should `recon` take the MACE4D reconstruction together
  with the `MACE4DModel` that made it?
- **Decided.** Yes, so that the times of the starting frames are known.
  (Greg)
- **Rejected.** No alternative was proposed.

## 2026-10-22, Software Architecture, home of the two-frame proximal map

- **Question.** Should the two-frame proximal map have its own file, or
  should it live in the new module next to the agent that calls it?
- **Decided.** It is a private function in `time_resolved.py`, next to
  `_IntervalDataFitAgent`, which is the only code that calls it.  This
  replaces the earlier decision for `two_frame_prox.py`.  (Greg)
- **Rejected.** Its own file, `two_frame_prox.py`.  The function serves
  only the time-resolved reconstruction.
- **Rejected.** A public method of the time-resolved model.  Users never
  call it.

## 2026-10-22, Software Architecture, one MACE object

- **Question.** Should `recon` keep one MACE object for the whole run and
  replace only the two temporal agents between passes?
- **Decided.** Yes.  (Greg)
- **Rejected.** A new MACE object in each pass.  It would reset every
  agent input, and its new threads would build and compile the denoisers
  again.

## 2026-10-22, Software Architecture, labels revised

- **Question.** Should label 0 be the background, with `part_kinds`
  naming the static labels?
- **Decided.** Yes.  `part_kinds` maps every label other than 0 to
  `'static'`, `'part'`, or `'spring'`.  Each spring touches exactly one
  static label and one part label.  This revises the earlier labels
  decision.  (Greg)
- **Rejected.** Label 0 as the whole static part.  It includes the air, so
  the contact of a spring with the static part is its whole surface, and
  the ends of the spring cannot be found.

## 2026-10-22, Software Architecture, last interval

- **Question.** Should the views after the last full interval be dropped?
- **Decided.** Yes, as `construct_time_frame_models` drops the views that
  cannot fill a frame.  The formula for K in the Theory blueprint now uses
  the floor, and the Theory blueprint records this as Q11.  (Greg)
- **Rejected.** The ceiling in the formula for K.  It can create a last
  interval with zero or one view.
