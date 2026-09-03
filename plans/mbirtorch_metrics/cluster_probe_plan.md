# Weekly cluster probe — plan

Status: DRAFT 2026-09-03 (rev 2, after a fact-check on the cluster), for an Opus plan review
before implementation.
Code lands in `mbirtorch_metrics/tooling/regression/` beside the nightly; this file is the
record.  Runs from Greg's gautschi scrontab.

## Why

The nightly proves the library.  Nothing watches the cluster underneath it, and the
2026-09-03 pass through `.claude/cluster_use.md` found every one of these by hand:

| failure mode | how it surfaces without a probe | evidence |
|---|---|---|
| a nightly dies silently: scrontab entry auto-disabled (`#DISABLED:`), job never submits | nothing — no run, no FAIL mail | the Mac jax series lost 51 nights (lessons.md §9) |
| driver or module change breaks the wheels' CUDA | nightly aborts; interactive envs fail at the next session | `nvidia-smi` CUDA 13.2 vs pip-bundled 13.x today |
| scratch purge hollows a conda env (60 days unaccessed) | cryptic ImportError / `DirectoryNotACondaEnvironmentError` | the hollow `mbirjax` env found 2026-09-03 |
| home quota creep (logs, pip/triton caches) | a job dies mid-write, no message | 25 GB quota, `myquota` |
| GPU-hour balance drains | submissions start failing | metered account; balance moved 2,716 → 21,857 between 08-06 and 09-03 |
| cluster preamble drifts from the repo example | exactly this week's finding | md5 mismatch found 2026-09-03 |
| a publish leaves bad perms or non-HTML on the public www root | 403s, or a leak | sphinx `_static` files already under `www/mbirjax/` |
| version tables in the guide rot | stale advice | every VERIFIED date in cluster_use.md |

## What

One SLURM job per week (Monday 06:00, account `bouman`, 1 GPU, `-t 0:15:00`, QoS `normal`)
under Greg's scrontab, installed and removed the way the nightly is (managed block,
markers `# mbirtorch-probe-BEGIN/END`).  The job:

1. **Gathers facts** on a compute node (a GPU is needed for `nvidia-smi` and a CUDA init per
   env), as `key=value` lines in a fixed order.
2. **Classifies** them.  *Identity* facts are compared with a committed baseline; any change
   is a finding.  *Threshold* facts are compared with knobs; a crossing is a finding.
3. **Writes** the facts and the verdict to a group-readable depot directory, keeps the
   previous facts file (for burn-rate), and appends a dated copy to `history/`.
4. **Exits 1 when there is any finding**, so `--mail-type=FAIL` mails the report to
   `NOTIFY`; exits 0 otherwise.  The report lists each finding with the old/new value.
5. When identity facts changed, **prints a ready-to-commit baseline** so accepting a change
   is one copy plus one commit, not a hand edit.

The probe never changes anything on the cluster: it reads, computes, and writes its own
output files.  It does not touch `run_regression.sh`.

## Facts

Identity (baseline-diffed):

```
node                         (informational, excluded from the diff)
driver                       nvidia-smi driver version
cuda_max                     nvidia-smi "CUDA Version"
module.conda / module.cuda / module.cudnn     versions loaded after sourcing ~/load_conda_cuda.sh
cuda_default                 the marked (D) cuda module
partition.ai.DefaultTime / MaxTime / DefMemPerCPU / MaxMemPerCPU
env.<name>.python / torch / torch_cuda / triton / jax     per PROBE_ENVS, from pip list (no imports)
env.<name>.gpu_ok            1 if a tiny CUDA workload ran in that env (torch matmul / jax reduce)
preamble.matches_example     1 if md5(~/load_conda_cuda.sh) == md5(<metrics checkout>/tooling/regression/cluster_preamble.sh.example)
```

Threshold (knob-checked):

```
hollow_envs                  dirs under ~/.conda/envs that `conda env list` does not list      == 0
env.<name>.exposed           files with BOTH atime and mtime older than PROBE_STALE_DAYS (45)   informational only (see below)
home_pct                     myquota, `home` row, Use column (72.9% on 2026-09-03)              < PROBE_HOME_PCT_MAX (80)
depot_pct                    myquota, `depot bouman` row (66.6%); df shows the whole NFS, not the share   < PROBE_DEPOT_PCT_MAX (90)
slist_balance                gautschi GPU-hours                                                 > PROBE_BALANCE_MIN (2000)
burn_per_week                previous balance − this balance, scaled to 7 days                  < PROBE_BURN_MAX (1000)
scrontab.blocks              names of managed blocks present                                    contains mbirtorch-nightly and mbirtorch-probe
scrontab.disabled            count of `#DISABLED:` lines                                        == 0
nightly_log_age_h            age of the newest nightly-*.log                                    < 48
www.unreadable               under www/mbirtorch, www/mbirjax, www/pcdrecon: files without o=r, dirs without o=rx   == 0
www.world_writable           anything under /depot/bouman/www with o=w (one such entry exists today)   == 0
www.dangerous                data/archive/secret types (npy npz h5 hdf5 tgz tar gz zip pkl pt pth env pem key) under the three project roots   == 0
```

Why these forms and not the earlier ones (fact-check 2026-09-03): an exact `-perm 644/755` test flags
every setgid directory (`2755`) and the frozen, read-only mbirjax tree, 317 false findings today; an
"only HTML" rule flags the 99 asset files of the sphinx export under `www/mbirjax/preprocessing/`;
and `www/cisym/` legitimately serves `.zip` templates, so the dangerous-type rule is scoped to the
three project roots.  Purge exposure: atimes do move on the Lustre scratch (verified), but an in-use
env always has thousands of never-imported files older than any window, so a count is not a finding;
it is logged so a rebuild can be planned.  The findings that mean damage are `hollow_envs` and
`env.<name>.gpu_ok`.  (ctime is useless here: every env file's ctime is 8 days old today, some
metadata pass touched them all.)

## Files

- `tooling/regression/cluster_probe.sh` — the job script (bash; python one-liners only inside
  the batch job, never on a login node).
- `tooling/regression/cluster_baseline.txt` — the committed identity facts.
- `tooling/regression/enable_probe.sh`, `disable_probe.sh` — the managed scrontab block,
  copied from `enable_nightly.sh`.
- `tooling/regression/status_nightly.sh` — three more lines: probe block present, age of the
  last facts file, last verdict.
- `tooling/regression/regression.env` — `PROBE_SCHEDULE`, `PROBE_WALLTIME`, `PROBE_ENVS`,
  `PROBE_STATUS_DIR`, and the thresholds above.  Note home is at 72.9% today, so the 80%
  threshold will fire within weeks unless something is moved; that is the intended first alert.
- `tooling/regression/README.md` — a section.
- depot: `/depot/bouman/data/cluster_status/{gautschi_facts.txt, gautschi_facts.prev.txt,
  probe_status.txt, history/}` (group `bouman-data`, readable by every member).
- `mbirtorch_plans/.claude/cluster_use.md` — a "live facts" pointer at the depot file, and
  the instruction that a Claude session reads `probe_status.txt` first and treats a facts
  file older than 8 days as "the probe is dead".

## What is deliberately not in it

- No change to `run_regression.sh` and no second GPU job on nightly cadence.
- No prevention of the scratch purge (Greg accepts rebuilds); the probe only warns early.
- No dashboard; the facts file and the mail are the interface.
- No gilbreth: its module set is unverified since the jax era and it is not metered.

## Slurm mechanics the script must respect

- A scrontab job runs with a clean environment ("the user environment variables are ignored",
  Slurm 26.05 manual) and in `$HOME`: the script sources `/etc/profile` and the preamble itself,
  uses absolute paths, and reads its knobs from `regression.env` by absolute path.
- Cancelling a scron job comments its entry out (`#DISABLED:` prefix); so does a submission
  failure.  The check is a grep for that prefix in `scrontab -l`, which works from inside a job.
- `srun` inside the job would eat the script's stdin; the probe uses none.
- The nightly's log directory is `$HOME/.mbirtorch/regression` (readable: same account).

## Self-death

The probe cannot report its own missed run.  Backstops: `status_nightly.sh` shows the facts
file's age; the cluster guide tells every session to check it; the mbirtorch nightly's
`#DISABLED:` scan is done by the probe and the probe's by the reader of the facts file.
That is the accepted gap.

## Cost

One GPU for three to five minutes per week, under half a GPU-hour a month.

## Verification before scheduling

1. `sbatch cluster_probe.sh` by hand; read the facts file and the report.
2. Commit the printed baseline; a second run must pass with no findings.
3. Break one knob on purpose (`PROBE_HOME_PCT_MAX=1` via `--export`) and confirm exit 1 and
   a report listing that finding only.
4. `enable_probe.sh`; `scrontab -l` shows the block; `status_nightly.sh` shows it.

## Questions for the review

1. Baseline-diff for identity facts versus thresholds only: is the diff worth its baseline
   maintenance, or should identity facts merely be logged?
2. Weekly: is any row worth a daily check, given the nightly already runs daily?
3. Anything missing from the fact list, or anything there that is noise?
4. The self-death gap: is there a cheap closure?
5. The facts file is group-readable on depot: anything in it that should not be?
