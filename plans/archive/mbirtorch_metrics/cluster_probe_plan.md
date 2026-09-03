# Weekly cluster probe — plan and record

Status: IMPLEMENTED 2026-09-03 (rev 3: the design after review, and the verification log at
the end).  Code: `mbirtorch_metrics/tooling/regression/` (`cluster_probe.sh`, `lib_scron.sh`,
`enable_probe.sh`, `disable_probe.sh`, the `PROBE_*` block in `regression.env`, three lines in
`status_nightly.sh`, a README section).  Runs from Greg's gautschi scrontab.

## Why

The nightly proves the library.  Nothing watched the cluster underneath it, and the
2026-09-03 pass through `../../../.claude/cluster_use.md` found every one of these by hand:

| failure mode | how it surfaces without a probe | evidence |
|---|---|---|
| a nightly dies silently: scrontab entry auto-disabled (`#DISABLED:`), job never submits, log cut off mid-write, results never pushed | nothing — no run, no FAIL mail | the Mac jax series lost 51 nights (lessons.md §9); push failure is non-fatal by design |
| driver or partition change; an env whose torch / CUDA build moved | nightly aborts; interactive envs fail at the next session | `nvidia-smi` CUDA 13.2 vs pip-bundled 13.x today |
| scratch purge hollows a conda env (60 days unaccessed) | cryptic ImportError / `DirectoryNotACondaEnvironmentError` | the hollow `mbirjax` env found 2026-09-03 |
| home quota creep (logs, pip/triton caches) | a job dies mid-write, no message | 25 GB quota, 72.9% used on 2026-09-03 |
| GPU-hour balance drains | submissions start failing | metered account; balance moved 2,716 → 21,857 between 08-06 and 09-03 |
| cluster preamble drifts from the repo example | exactly this week's finding | mismatch found 2026-09-03 |
| a publish leaves bad perms or data/source on the public www root | 403s, or a leak | sphinx `_static` files already under `www/mbirjax/` |
| version tables in the guide rot | stale advice | every VERIFIED date in cluster_use.md |

## What was built

One SLURM job per week (Monday 08:00, account `bouman`, partition `ai`, QoS `normal`, 1 GPU,
`-n 14`, `-t 0:15:00`) under Greg's scrontab, installed and removed the way the nightly is: a
managed block with markers `# mbirtorch-probe-BEGIN/END`, written by `lib_scron.sh`.  The job:

1. **Gathers facts** on a compute node as `key=value` lines in a fixed order.  Every fact is
   tri-state: `key=value` or `key=UNKNOWN:<reason>`.  Every external command runs under
   `timeout`; a `find` that fails or times out yields UNKNOWN, never 0.
2. **Classifies.**  Every UNKNOWN is a finding.  *Threshold* facts pass only if the value
   parses as a number and satisfies the rule.  *Identity* facts are compared with the
   **previous run's facts file**, which then advances — one mail per change, no committed
   baseline.
3. **Writes** the facts, the previous facts, a one-line status, and dated history copies to
   `PROBE_STATUS_DIR` (`/depot/bouman/data/cluster_status`, readable by the whole group;
   atomic tmp + `mv`).  When that directory is not writable it writes to `~/.mbirtorch/probe/`
   and that is itself a finding.
4. **Mails the report every run** (`sendmail -t`, the nightly's own idiom) — PASS or findings —
   so a Monday without a `[cluster-probe]` mail is the signal that the probe died.  The scron
   options carry `--mail-type=FAIL,TIME_LIMIT`, so a script that dies before line 1 still
   produces a Slurm mail.
5. **Exits 1 on any finding.**  An EXIT trap catches a death before the verdict, writes an
   UNKNOWN status, mails, and exits 1.

Nothing here changes the cluster: it reads, computes, and writes its own output files.

## Facts

Identity (diffed against the previous run):

```
driver, cuda_max                                 nvidia-smi driver and "CUDA Version"
partition.ai.{DefaultTime,MaxTime,DefMemPerCPU,MaxMemPerCPU}
env.<e>.{python,torch,torch_cuda,triton}         one python one-liner per env in PROBE_ENVS
```

Threshold / rule (each a finding when violated):

```
env.<e>.gpu_ok           1 only if that env printed a token after a real GPU matmul     == 1
preamble.ok              ~/load_conda_cuda.sh == cluster_preamble.sh.example, comments and blank lines stripped   == 1
hollow_envs              dirs under ~/.conda/envs with no bin/python or conda-meta/history (no conda needed)   == 0
home_pct / scratch_files_pct / depot_pct     myquota rows                              < 80 / < 80 / < 90
slist_balance            gautschi GPU-hours                                          > 2000
scrontab.disabled        `#DISABLED:` lines                                          == 0
scrontab.nightly         the mbirtorch-nightly block is present                      == 1
nightly.log_age_h        newest nightly-*.log (skipped while the nightly is RUNNING)  < 48
nightly.log_complete     its last line is one of the wrapper's exit messages          == 1
nightly.unpushed         `git rev-list --count @{u}..HEAD` in the nightly's clone     == 0
www.unreadable / www.world_writable / www.dangerous / www.escaping_symlinks           == 0
depot_writable           PROBE_STATUS_DIR accepted a write                            == 1
```

Informational (logged, never a finding): `module.conda/cuda/cudnn`, `cuda_default`
(verified 2026-09-03 not to matter), `burn_per_week` (negative after a top-up),
`nightly.last_line`, `www.symlinks`, `home_used`, `scratch_pct`, node, job, date.

Rules that came out of the fact-check and the review: the www permission audit tests "readable
by others" and "world-writable", skipping symlinks, because an exact `-perm 644/755` test flags
setgid directories, the frozen read-only mbirjax tree, and the always-777 symlink; the
dangerous-type rule (`npy npz h5 hdf5 tgz tar gz zip pkl pt pth env pem key py sh ipynb`) is
scoped to the three project roots and skips sphinx `_static/_sources/_downloads`; symlinks
under the roots must resolve inside the web root.  Purge exposure is not measured: an in-use
env always has thousands of never-imported files older than any window, `find` over them was
the slowest step, and RCAC mails a week ahead; `hollow_envs` and `gpu_ok` are the checks that
mean damage.

## Slurm mechanics the script respects

- A scron job runs with a clean environment and in `$HOME`; the script sources `/etc/profile`
  and the preamble itself and resolves everything by absolute path.  `set -o pipefail`
  without `-e` (a failing gatherer must not abort before the verdict), `set -u` only after the
  profile sourcing.
- Cancelling a scron job, or a submission failure, comments the entry out with `#DISABLED:`.
- `scrontab -` re-registers every entry, so installing the probe gives the nightly a new job
  id and log filename.  Harmless.
- A hand `sbatch` with `--export=ALL` would mask environment problems; trials use
  `--export=PROBE_MAIL=0` (only that variable), and the last step is a real scheduled firing.

## Review outcome (2026-09-03)

Four Opus launches failed on server overload (HTTP 529); the review was done by the default
model instead.  Verdict: implement with the must-fix items.  Adopted, all of them:

1. Slurm's FAIL mail is subject-only → the probe mails its own report (the nightly's
   `sendmail -t` block), every run, and keeps `--mail-type=FAIL,TIME_LIMIT` as the backstop.
2. Trials must not export the login environment → `--export=PROBE_MAIL=0` plus one real
   scheduled firing.
3. A raw md5 of the preamble fires on every comment edit → comment- and blank-stripped
   comparison against the checkout's own example.
4. Monday 06:00 can overlap a measuring nightly → 08:00, and the log checks are skipped while
   the nightly is RUNNING; `-n 14` for one GPU.
5. Burn rate is negative after a top-up and undefined on the first run → informational.
6. Add `nightly.log_complete` (mid-write death) and `nightly.unpushed` (push failures are
   non-fatal by design).
7. Drop the purge-exposure scan; make the module facts informational; drop the tautological
   "probe block present" check.
8. Diff identity facts against the previous run and auto-advance; no committed baseline.
9. Mail every week; a missing mail closes the self-death gap more reliably than a file age.
10. Tri-state facts, pass-only-if tests, `timeout` everywhere, an EXIT trap → no false PASS.
11. Depot unwritable → home fallback plus a finding; atomic writes.
12. `gpu_ok` needs a printed token, not an exit code; `hollow_envs` without conda.
13. One python one-liner per env instead of `pip list` parsing; knobs sourced from
    `regression.env` with per-run overrides.
14. Separate enable/disable pair, sharing `lib_scron.sh` (the nightly's scripts are unchanged).

## Cost

One GPU for two to three minutes per week (trial 1: 97 s); well under half a GPU-hour a month.

## Verification log

- Trial 1 (job 15856399, h001, `--export=PROBE_MAIL=0`): 97 s.  Every check worked except
  `myquota`, which prints nothing on a compute node; the three quota facts came back
  UNKNOWN and the run ended `FINDINGS(3)` — the tri-state rule doing its job.  Driver
  595.71.05 / CUDA 13.2, all three envs `gpu_ok=1` on torch 2.13.0+cu130, preamble matches,
  nightly log 8 h old and complete, nothing unpushed, www clean, depot writable.
- Cause of the `myquota` failure: it fetches its table from the cluster's internal aux
  server over HTTPS, and the preamble's `HTTPS_PROXY` sends that request to squid, which
  cannot reach it (confirmed on a login node: with the proxy set it prints only the header).
  Fix: the probe runs `myquota` with the proxy variables unset and logs its raw output when
  the home row is missing.  It also records the job's real log path from `scontrol`.
- Trials 2–4 (jobs 15856565–7, submitted together with `--export=PROBE_MAIL=0,...`, ~60 s
  each): trial 2 `PASS`, exit 0, quotas parsed (home 72.9%, scratch inodes 2.8%, depot
  66.6%), identity facts unchanged against trial 1's file.  Trial 3 with
  `PROBE_HOME_PCT_MAX=1`: `FINDINGS(1)`, exactly `home_pct=72.9 (must be < 1)`, exit 1.
  Trial 4 with `PROBE_STATUS_DIR=/depot/nonexistent/...`: `FINDINGS(1)`,
  `depot_writable=0`, output in `~/.mbirtorch/probe/`, exit 1.
- Live scheduled firing (job 15856600): the block was installed at 11:55 with the one-off
  schedule `58 11 * * *`; Slurm released the job at 11:58, it waited two minutes for a GPU,
  ran at 12:00 on h001, `PASS`, and mailed the report to buzzard@purdue.edu (`mail sent`).
  `enable_probe.sh` was then re-run with the default `0 8 * * 1`, so the first regular run
  is Monday 2026-09-07 08:00.
- Deployment note: the cluster checkout `~/PycharmProjects/mbirtorch_metrics` carries the
  probe files as untracked/modified copies (identical to what is staged locally).  After
  these are committed and pushed, update the cluster with
  `git stash -u && git pull --ff-only && git stash drop` in that checkout.
- Expected first real finding: home is at 72.9% of its 25 GB, so `home_pct` will cross the
  80% threshold within weeks unless something moves off home.
