# Beam hardening from reconstruction quality: brainstorm of 2026-09-05

This directory holds one brainstorming session's records.  The page to read is
`../survey.md`.  Everything else is the material it was built from.

The synthesis:

- `../survey.md`: the page.  Executive summary, answer, the existing routines,
  the proposed modifications with their experiments, the alternative techniques with their
  experiments, the run order, and the open questions.

The six agent reports, one per charge:

- `brainstorm_score.md`: which image-domain measures see hardening artifacts, and what makes a
  score safe to search on.
- `brainstorm_search.md`: how to make a search over 6 to 15 coefficients tractable, and the
  candidate parametrizations.
- `brainstorm_physics.md`: the polychromatic forward model, the accuracy of the polynomial and
  the mixture-of-exponentials families, and what one scan identifies.
- `brainstorm_skeptic.md`: the case against the premise, with the experiments that would settle
  each objection.
- `brainstorm_literature.md`: the published methods, with verified citations.
- `brainstorm_pipeline.md`: how an estimator would fit the mbirtorch code, with file and line
  pointers.

The three panel reviews of the page's first draft, whose findings the page applies:

- `review_accuracy.md`, `review_reasoning.md`, `review_style.md`.

The scripts whose outputs the page cites.  They are records of the agents' runs and keep the
command-line arguments those runs used:

- `counts_and_binning.py`: the column counts of the polynomial model and the binning bias of a
  quadratic fit on a disk.  Run it with no arguments in the mbirtorch environment.
- `bh_physics_sim.py`: the physics simulation.  It needs the `xraydb` package, which was
  installed into a session-local directory during the brainstorm, and it takes that directory's
  path as its one argument: `python bh_physics_sim.py <directory holding xraydb>`.  Its output
  is `bh_physics_sim_output.txt`.
- `bh_physics_extra.py`: three follow-up calculations that reuse the simulation's helper
  functions.  It takes two arguments, the `xraydb` directory and the path of
  `bh_physics_sim.py`: `python bh_physics_extra.py <directory holding xraydb> bh_physics_sim.py`.
  Its output is `bh_physics_extra_output.txt`.

One note of process:

- `own_view_before_reports.md`: the session's own view, written before any agent report was
  read, so that the reports could be judged against it rather than absorbed.
