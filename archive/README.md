# archive/ — closed plans and programs

These folders are records of finished work.  Each keeps its plan of record, its findings
pages, and the run records that were written beside its scripts.  The scripts themselves
were removed on 2026-09-18, when the repository moved to its present layout; they are in
the git history at commit `cd4aac3`, for example
`git show cd4aac3:plans/experiments/archive/torch_port/README.md`.  Paths cited inside these
documents are the paths of their time, before the moves of 2026-09-12 and 2026-09-18.

Three more files left the archive on 2026-09-19 and are in the history at commit `8d2bb50`:
`flash_remediation/publish_pages.sh`, the script that published the phase pages to the depot;
`sharding/_file_index.md`, an index whose first entries were the deleted v1 and v2 sharding
plans; and `sharding/preprocessing_pipeline_refactor_plan.md`, the 2026-06-29 version of the
plan whose record is `preprocessing/preprocessing_pipeline_refactor_plan.md`.

| Folder | What it was | When |
|---|---|---|
| `center_slice_noise/` | The center-slice noise investigation and the preconditioner notes from the mbirjax era.  The per-slice DC damping in mbirtorch's `cone_beam.py` is the fix of the kind the notes propose. | 2026 (undated) |
| `cylinder_subsets/` | The slice-parity study of cylinder subsets: plan, findings page, and the published comparison pages. | 2026-06 |
| `flash_remediation/` | The field-of-view truncation ("flash") remediation program: sinogram edge tapering against reconstruction-support padding, characterized synthetically and validated on real scans.  `flash_remediation_plan.md` is the plan of record; the phase reports are self-contained HTML pages. | 2026-07 |
| `mbirtorch_metrics/` | The port of the nightly regression engine and dashboard to mbirtorch, the cluster probe plan, and the torch version policy. | 2026-08 |
| `partition_sequence/` | The VCD partition-sequence convergence study. | 2026-07 |
| `preprocessing/` | The scanner-reader API refactor (readers return a ready model) and the migration of `mbirjax_applications` to it. | 2026-07 |
| `projector_batching/` | The projector-batching characterization and the retired v2 batching refactor. | 2026-07 |
| `projector_kernels/` | The projector-kernel campaign (`fwd_back_findings.md` is the record) and the GPU headroom study with its appendices. | 2026-07 |
| `sharding/` | The multi-device sharding program of mbirjax, shipped 2026-07: `sharding_status.md` is the end state, `sharding_implementation_plan_v3.md` the final plan, and `back_projection_overview.md` and `sinogram_sharding.md` the architecture records. | 2026-06 to 2026-07 |
| `torch_port/` | The PyTorch port of mbirjax that became mbirtorch: `port_plan.md` is the plan of record and `README.md` maps the subfolders (the six phases, the closed campaigns, the reviews, and `closed/current_plans.md`, the running plan of that period). | 2026-08 to 2026-09 |
| `viewer/` | The slice-viewer evaluation and the build of the mbirtorch viewer. | 2026-08 |
