We're continuing work on the `mbirtorch` repo. `mbirtorch_plans` is parallel to
`mbirtorch` and contains plans related to mbirtorch.  `mbirtorch_metrics` is the nightly
regression engine and dashboard and is also parallel to both.   

The task for this session is to evaluate and refine a plan and implementation for 
mace4d in mbirtorch.  An existing version is in mbirjax, parallel to mbirtorch.  

The initial plan in /Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch/plans/mace4d_migration_plan.md
provides one possible approach.  However, I'd like to explore if there are approaches
more suited to torch rather than trying to mimic jax directly.  

**IMPORTANT — workflow protocol:** stage only (`git add` by explicit file
name), never `git commit` unless the user directs it (the user commits from
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
- Concurrent sessions may be active.  Terminology: "variants"
  (never arms/cells for variant sets); the multi-device forward's
  mechanism is the "cylinder transfer" — pre-2026-08-17 records call it
  the "column gather".  The kernel width rule: hand-written kernels
  round width-class arguments up to the next multiple of 16
  (`mbirtorch/_utils.padded_kernel_width`, landed 2026-08-18), so
  records from before that date describe the unpadded kernels.
