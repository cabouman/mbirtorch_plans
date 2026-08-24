We're continuing work on the `mbirtorch` repo, which is a port of the parallel checkout
`mbirjax`; mbirjax is READ-ONLY reference. `mbirtorch_plans` is parallel to
both and contains plans related to mbirtorch.  `mbirtorch_metrics` is the nightly
regression engine and dashboard and is also parallel to both.  There are older
versions of plans and metrics for `mbirjax`, but these are also for reference only.  

The task for this session is to investigate this observation:
```
The multiaxis forward projector is 3.28 times slower at the non-dividing 
size than at the dividing one. It reads 305.9 ms at 512x448x384 and 1004.6 ms 
at 513x449x385, on one H100.

Every other geometry pays far less for the same step. Parallel pays 1.04 
times on forward, cone pays 1.03 times, and multiaxis itself pays only 1.14 
times on its filter and 1.17 times on its back projection. So the penalty is 
specific to the multiaxis forward projector at the non-dividing size.
 ```

This is quite possibly similar to the behavior on other geometries that was 
remedied in greg_dev with commit 64dedb8732, which introduced rounding up to the 
next multiple of 16 for the triton kernels.  

**IMPORTANT — workflow protocol:** stage only (`git add` by explicit file
name), never `git commit` unless Greg directs it (he commits from
PyCharm).  Shared checkouts — never `git add -A`; verify staged-file
lists at report time.  No plan notation in
code or tests.  Cluster jobs are pre-authorized during the agreed
investigation.  Durable records and summary status reports in Alley style — reread
`.claude/writing_style.md` before drafting; plan entries and chat
summaries stay short and plain, with run detail in script comments or a
companion `.md` beside the script.  Have opus carry out well-defined
plans, then review.

Read for orientation (code and measured results over recollection or .md files):
1. `.claude/claude_prompt.md`, `.claude/lessons.md` (§2, §5, §6),
   `.claude/cluster_use.md`.
2. `plans/open_items_v4.md` — the task source, with the Start-here
   order this session follows and per-item status labels.
3. `plans/API_specification.md` for reference.

The nightly dashboard is live and seeding history — its rows are regression
protection for this campaign's tuning, not its instrument; campaign
measurements use your own gated harnesses.

## Standing context

- Cluster: gautschi (ssh BatchMode; accepted key `~/.ssh/id_rsa` — if key
  files are unreadable in your environment, ask Greg to run
  `ssh-add ~/.ssh/id_rsa` once).  sbatch on partition `ai`, account
  `bouman`, --cpus-per-task=14 per GPU, --gpus-per-node=2 or 4 for the
  multi-device cells.  mbirjax scratch checkout:
  `/scratch/gautschi/buzzard/torch_p3/mbirjax_src`;
  TORCHPY=`/scratch/gautschi/buzzard/torch_p0/env/bin/python`; results in
  `/scratch/gautschi/buzzard/torch_p3/results/`.  SYNC RULE: per-file scp +
  md5 verify of every changed file — and the scratch tree lags the
  repository, so sync it to the current tip before the first job.  Slurm
  `--export` splits on commas — pass env via the submission shell.  The
  torch_p3 sbatch files pip-install into the shared environment at
  start, so never run two such jobs at once; chain with
  `--dependency=afterany:<jobid>`.
- Scripts to `plans/experiments/torch_port/` (suggested prefix `mg*_`;
  mg1 through mg24 are used, so new scripts start at mg25); findings to
  `plans/torch_port/active/`, with measured rows filed under
  `plans/experiments/torch_port/rows/`.
- Concurrent sessions may be active.  Terminology: "variants"
  (never arms/cells for variant sets); the multi-device forward's
  mechanism is the "cylinder transfer" — pre-2026-08-17 records call it
  the "column gather".  The kernel width rule: hand-written kernels
  round width-class arguments up to the next multiple of 16
  (`mbirtorch/_utils.padded_kernel_width`, landed 2026-08-18), so
  records from before that date describe the unpadded kernels.
