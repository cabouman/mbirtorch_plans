# Using the Purdue RCAC clusters (for the group's Claude sessions)

Two clusters, one shared group storage.  **gautschi = H100, gilbreth = A100.**  Slurm
account `bouman` on both.  This guide is for a Claude session working on behalf of **any
member of the group** who has the `bouman` queue on gautschi.  Conventions:

- **mbirtorch is the supported library.  mbirjax (JAX) is legacy** — maintenance is
  expected to end after 2026.  Material that exists only for mbirjax is marked *(legacy)*
  and kept so old results can still be read; the
  [sunset checklist](#legacy-mbirjax-sunset-checklist) at the end lists everything that
  still points at it.
- `<user>` is your Purdue career account; on the cluster the same thing is `$USER`.
  Home (`~`) and scratch (`/scratch/<cluster>/<user>`) are **per account and mode 700** —
  nobody else in the group can read them.  Anything meant for another member goes through
  depot.
- Greg's paths (`/home/buzzard/…`, `/scratch/gautschi/buzzard/…`) appear only where they
  are the real location of something shared: his two nightlies and some staged data.
- The account is shared.  **Coordinate before heavy or long batch use** — everyone's
  interactive sessions, the nightlies and every batch job draw on one queue and, on
  gautschi, one metered GPU-hour balance.

## Contents

- [First-time setup (once per person)](#first-time-setup-once-per-person)
- [The workflows that matter](#the-workflows-that-matter)
- [SSH access](#ssh-access)
- [Watching a job](#watching-a-job)
- [Remote GUI windows (slice_viewer, PyCharm) — VERIFIED 2026-07-25](#remote-gui-windows-slice_viewer-pycharm-verified-2026-07-25)
- [gautschi — H100 (the nightly-regression cluster)](#gautschi-h100-the-nightly-regression-cluster)
  - [The node preamble, and which CUDA version actually matters](#the-node-preamble-and-which-cuda-version-actually-matters)
  - [The nightly regression system (runs on gautschi)](#the-nightly-regression-system-runs-on-gautschi)
- [gilbreth — A100 (lightly used by our group)](#gilbreth-a100-lightly-used-by-our-group)
- [Scratch — fast, big, PURGE-ELIGIBLE](#scratch-fast-big-purge-eligible)
- [Home — a 25 GB quota that fails SILENTLY](#home-a-25-gb-quota-that-fails-silently)
- [Depot — durable, group-shared (mounted on BOTH clusters)](#depot-durable-group-shared-mounted-on-both-clusters)
- [Moving data on and off](#moving-data-on-and-off)
- [Nested ssh — run on a worker node without waiting in the queue](#nested-ssh-run-on-a-worker-node-without-waiting-in-the-queue)
- [Repos, data, and resources](#repos-data-and-resources)
- [Running a specific library state](#running-a-specific-library-state)
- [Job preflight — two lines that catch the two worst failures](#job-preflight-two-lines-that-catch-the-two-worst-failures)
- [Failure signatures → what they actually mean](#failure-signatures-what-they-actually-mean)
- [Don't](#dont)
- [Legacy mbirjax — sunset checklist](#legacy-mbirjax-sunset-checklist)

## First-time setup (once per person)

Everything below assumes these are done.  The allocation itself (what was bought, how
access is granted, GPU-hour accounting) is described in `.claude/gpu-resources.md`.

1. **Get onto the queue.**  Request `bouman` on gautschi (and on gilbreth) at
   <https://www.rcac.purdue.edu/account/request>, or have the PI add you from the group's
   Members tab on the RCAC account portal.  Propagation takes up to a day.  Done when
   `slist` on a login node lists `bouman` (on gautschi it also shows the GPU-hour balance).
2. **SSH keys**, so Claude can run commands unattended.  RCAC accepts either Duo MFA or a
   key in `~/.ssh/authorized_keys` on the cluster
   (<https://docs.rcac.purdue.edu/userguides/gautschi/accounts/>).  For `BatchMode` the
   key must work without a prompt — passphrase-less, or loaded in `ssh-agent`.
   ```bash
   ssh-keygen -f ~/.ssh/id_rcac -N ''                                  # skip if you already have a key
   ssh-copy-id -i ~/.ssh/id_rcac.pub <user>@gautschi.rcac.purdue.edu   # password + one Duo push
   ssh-copy-id -i ~/.ssh/id_rcac.pub <user>@gilbreth.rcac.purdue.edu   # the same key serves both
   ssh -o BatchMode=yes <user>@gautschi.rcac.purdue.edu hostname       # must print loginNN…
   ```
   RCAC's own example uses an RSA key (`~/.ssh/id_rsa.pub`); if a newer key type is
   refused, regenerate with `-t rsa -b 4096`.  A `~/.ssh/config` stanza is worth having:
   it pins the key and makes the short names used later in this guide work.
   ```
   Host gautschi
       HostName gautschi.rcac.purdue.edu
       User <user>
       IdentityFile ~/.ssh/id_rcac
       IdentitiesOnly yes
   Host gilbreth
       HostName gilbreth.rcac.purdue.edu
       User <user>
       IdentityFile ~/.ssh/id_rcac
       IdentitiesOnly yes
   ```
   With several keys and no stanza, ssh offers them in turn and a key that is not in
   `authorized_keys` is a confusing failure — Greg's account hit exactly this (his default
   `~/.ssh/id_rsa` works, his `id_rsa_gau` is rejected), so on that account pass no `-i`.
3. **ThinLinc desktop** — needed only for workflows with a window (1 and 3 below).  Web
   client at <https://desktop.gautschi.rcac.purdue.edu/>, or the native client from
   <https://www.cendio.com/thinlinc/download> pointed at the same host; Purdue password +
   Duo.  A session is pinned to the login node it started on and survives disconnects.
4. **`~/load_conda_cuda.sh` on the cluster** — the node preamble every job sources.  It is
   **per account**: create it from the listing in
   [the preamble section](#the-node-preamble-and-which-cuda-version-actually-matters), or
   copy `tooling/regression/cluster_preamble.sh.example` from `mbirtorch_metrics` — the two
   are identical from 2026-09-03 on (the example's header carries that date; an older clone
   lacks the non-login bootstrap and pins modules that do not matter).
5. **Conda envs, and where they live.**  `conda create` puts envs under `~/.conda/envs/`,
   i.e. inside the **25 GB home quota**, and one torch+CUDA env is several GB.  Greg's
   practice, recommended: point `~/.conda` at scratch **before creating any env**:
   ```bash
   mkdir -p /scratch/gautschi/$USER/.conda && ln -s /scratch/gautschi/$USER/.conda ~/.conda
   ```
   The trade: scratch is **purged** — files not accessed or modified for 60 days are
   removed, with an email one week ahead
   (<https://www.rcac.purdue.edu/policies/scratchpurge>) — so an env you stop using is
   hollowed out and has to be rebuilt with the project's `dev_scripts/clean_install_all.sh`.
   That failure is cryptic but cheap, and it forces a genuinely fresh env now and then.  The
   alternative failure, home full, is equally cryptic and far more invasive: it kills jobs
   mid-write with no message at all (see Home).  What a purged env looks like:
   `conda env list` no longer shows it, yet `~/.conda/envs/<name>/` still exists holding
   only `conda-meta/` (Greg's retired `mbirjax` env was in exactly that state on
   2026-09-03, since removed); `conda create -n <name>` or `conda remove -n <name>` then trips over the
   leftover directory (`DirectoryNotACondaEnvironmentError`) — `rm -rf` it first.  An env
   that is still in use but partly purged should show up as an `ImportError` for one
   rarely-imported module while everything else works; treat that as "rebuild", not as a
   packaging bug.
6. **Helper scripts, only if you use the ThinLinc recipes.**  Copy
   `plans/experiments/remote_cluster/` to a directory of your own on scratch and run the
   scripts from there: they locate each other through their own directory and take the
   conda env from `CONDA_ENV` (default `mbirtorch`), so nothing inside needs editing.  The
   viewer demo they run is a small mbirtorch cone-beam recon (ported 2026-09-03, not yet run
   end to end on a display — the 2026-07-25 verification was of the mbirjax original).

## The workflows that matter

1. **Shared interactive session (the usual one).**  Either the user or Claude starts a
   terminal in the user's ThinLinc desktop holding a GPU allocation; both work on that
   node — the user can type in it and start PyCharm, Claude can run in it with
   `srun --overlap --jobid=<id>`.  Ends only on `exit`.
   → `remote_cluster/tl_gpu_session.sh`, then `remote_cluster/tl_node_terminal.sh`.
2. **Batch jobs for data collection (Claude, unattended).**  `sbatch` a self-contained
   script, results to scratch, poll the log.  The default for sweeps, benchmarks and
   anything long — no GUI, no held allocation.
3. **A dedicated GUI session Claude starts and the user watches** — e.g. mbirtorch's
   `slice_viewer` on a GPU node rendering into ThinLinc.  → `remote_cluster/tl_slice_viewer.sh`.
4. **Batch on the cluster, look at it locally.**  Compute remotely, write the volume or
   PNGs to scratch, copy down, view on the laptop.  Best when the result fits on a laptop
   (a 128³ float32 recon is ~8 MB) — no network in the interaction loop.

Rules of thumb: **prefer 2** unless a human needs to see something live; **prefer 4 over
3** when the data can travel; use **1** when the user wants to drive.

## SSH access

```bash
ssh -o BatchMode=yes <user>@gautschi.rcac.purdue.edu   '<cmd>'
ssh -o BatchMode=yes <user>@gilbreth.rcac.purdue.edu   '<cmd>'
```

- Key auth (setup step 2).  With the `~/.ssh/config` stanza, `ssh gautschi '<cmd>'` is
  the same thing with the key pinned.  Without it, pass **no** `-i` and let ssh pick the
  default key; never force a key that is not in `authorized_keys`.
- `-o BatchMode=yes` so a failure errors instead of hanging on a Duo/password prompt.
- gilbreth's handshake is **slow (~90 s)** — give ssh/scp to it a generous timeout and
  prefer `run_in_background: true`.  gautschi is fast.
- Login nodes have **no GPU**.  GPU work goes through `sbatch`/`srun`, or directly on the
  worker node of an existing interactive session (see "Nested ssh" below).
- To check the job queue: `squeue -u $USER`.  Verify a run by reading its **log file**,
  not by `pgrep` (a `pgrep -f` self-matches the ssh command that launched it).
- Feeding a script to `ssh host 'bash -s'` on stdin is convenient, but an `srun` inside
  that script **eats the rest of the script from stdin**.  Redirect it: `srun … </dev/null`.

## Watching a job

```bash
squeue -u $USER                        # your queued + running jobs
squeue -A bouman                       # everyone's — check before a big submission
squeue -j <id> -o "%.10i %.8T %.6M %.6L %N"   # state, elapsed, LEFT, node
squeue -h -j <id> -o "%L"              # just the time remaining
sacct -j <id> --format=JobID,State,ExitCode,Elapsed   # after it ends
scancel <id>                           # stop it
```

- **Read the LOG FILE, not the process list.**  `pgrep -f <pattern>` run over ssh matches
  its own ssh command line and reports a false positive.
- `sbatch -o`/`-e` decide where output lands — always point them at scratch.
- Watch a long run with a `tail -f` filtered to the lines that matter, and make the filter
  cover FAILURE signatures too (`Traceback|Error|FAILED|OOM`), not just progress: a filter
  that only matches success is silent during a crash, and silence looks like "still running".
- VCD prints `Error sino RMSE` every iteration — grepping bare `Error` for failures gives
  a hit on every healthy run.

## Remote GUI windows (slice_viewer, PyCharm) — VERIFIED 2026-07-25

Claude *can* put a live GUI window (mbirtorch's `slice_viewer`, any matplotlib figure) on
the user's screen from a GPU compute node.  Two routes, both tested end to end on Greg's
account on 2026-07-25 with the mbirjax viewer; the display plumbing is framework-independent,
and the scripts carry no account-specific paths (setup step 6).
**Prefer route B (ThinLinc): route A is noticeably laggy on sliders, route B has no
perceptible lag.**

For headless checks neither is needed — save PNG/HTML to scratch and send the file.
For anything that fits on the laptop, copying the volume down and viewing locally beats
both (a 128³ float32 recon is ~8 MB).

**Route A — `ssh -Y` to the Mac's X server.**  Needs **XQuartz 2.8.6+** on the Mac
(universal/arm64-native), installed from <https://www.xquartz.org/>, followed by a
**log out and back in** so the launchd `DISPLAY` socket registers.  Verify with
`xdpyinfo` before blaming anything downstream.  If XQuartz refuses to start with
"Cannot establish any listening sockets", delete the stale `/tmp/.X0-lock` and retry.

```bash
ssh -Y <user>@gautschi.rcac.purdue.edu 'srun --x11 -A bouman -p ai -N1 --gpus-per-node=1 \
    --cpus-per-task=14 -t 01:00:00 python my_viewer_script.py'
```
The window dies if that ssh drops.

**Route B — into the user's ThinLinc desktop (preferred).**  The session is persistent on
one login node (whichever it started on — login01 in the 2026-07-25 tests) and survives
disconnects.  Constraints that shape the recipe:

* ThinLinc's Xvnc runs `-nolisten tcp -localhost`, so **only processes on that same login
  node** can draw into it.  A compute node cannot reach it directly — `srun --x11` bridges
  it (verified: login01 `:2` → h002 `localhost:42.0`).
* **The display number is NOT stable** — a restarted session moves `:1` → `:2` → …
  Discover it from the live Xvnc process; never hardcode.
* `login01.gautschi.rcac.purdue.edu` fails host-key verification directly; hop via the
  round-robin address instead.

```bash
# discover (run ON the session's login node):
XVNC=$(ps -u $USER -o args= | grep "[X]vnc :" | head -1)
export DISPLAY=$(printf '%s' "$XVNC" | grep -oE 'Xvnc :[0-9]+' | grep -oE ':[0-9]+')
export XAUTHORITY=$(printf '%s' "$XVNC" | sed -n 's/.*-auth \([^ ]*\).*/\1/p')
# then launch; nohup + & so it outlives the ssh that started it:
nohup srun --x11 -A bouman -p ai -N1 --gpus-per-node=1 --cpus-per-task=14 \
      -t 04:00:00 python my_viewer_script.py > /tmp/viewer.log 2>&1 &
```
Working example: `plans/experiments/remote_cluster/tl_slice_viewer.sh`
(+ `x11_slice_viewer_demo.py`, a small mbirtorch GPU recon + viewer) — auto-discovers the
display, sanity-checks it, then submits.

**Use the DESKTOP'S terminal, not bare `xterm`.**  ThinLinc here runs **XFCE**, and
`xfce4-terminal` is installed — the terminal the desktop itself uses.  It has the
File/Edit/View/Terminal/Tabs/Help menu bar (**Edit → Copy/Paste** is the one people reach
for) and inherits the user's saved profile, so the font matches.  Bare `xterm` has
neither: small fixed font, no menus, no copy/paste.  Launch with `--disable-server` so it
is a private process rather than attaching to a terminal server (the nohup/detach pattern
needs that), and `--hold` so errors stay readable.  A harmless `Failed to connect to
session manager` warning appears when launching outside the desktop session manager.

**Allocation lifetime — the gotcha.**  `srun <cmd>` allocates the node to run *that one
command*: when the viewer window closes, the command exits and **the whole allocation
ends**.  That surprises anyone whose own workflow is `sinteractive` → a login *shell*
holds the allocation → PyCharm started from it → closing GUI windows changes nothing →
the session ends only on `exit`.  So:

* want the node to persist across closing windows → hold it with a **shell** (`sinteractive`
  / `salloc`), not with the app;
* want to run more work in an allocation that already exists →
  **`srun --overlap --jobid=<id> <cmd>`** (verified working — this is how to inspect a node
  Claude already holds, without re-queuing);
* `salloc --no-shell` (allocate with no controlling process, then ssh in) did not show up in
  this slurm's `--help` — **unverified, needs testing** if a detached persistent allocation
  is wanted.

Working example of the shell-held form: `plans/experiments/remote_cluster/tl_gpu_session.sh`
(discovers the ThinLinc display, opens `xfce4-terminal`, runs `sinteractive --x11` in it).
Verified 2026-07-25: terminal on login01 → shell on h008, released only on `exit`.

Two things that bite when sharing that allocation:

* **`--x11` is set at ALLOCATION time, not per step** — and steps inherit it.  Passing
  `--x11` to a step inside an existing job is refused:
  `srun: error: Ignoring --x11 option for a job step within an existing job.  Set x11
  options at job allocation time.`  Harmless (the step still runs), but the lesson is that
  the allocation must have been created with X11 — `sinteractive --x11` — after which every
  `srun --overlap --jobid=<id>` step gets `DISPLAY` for free (verified: `localhost:75.0`
  with its own `/tmp/.Xauthority-*` on the compute node).  If the allocation was created
  WITHOUT `--x11`, no step can display and it cannot be retrofitted — start a new one.
* **`--overlap` SHARES the GPU, it does not partition it.**  Fine for inspection
  (`nvidia-smi`, `ps`, `module list`).  Two torch processes on one card each take what they
  allocate and collide only at OOM; *(legacy)* two JAX processes collide at once, because
  JAX preallocates ~75% of the card — take turns, set `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  or use separate allocations.  Do not start a CUDA context inside someone's *running* job
  step: even an idle context takes a few hundred MB and can push a job at its memory limit
  into OOM.

**Match the user's login shell.**  On a stock account the prompt is
`(<env>) <user>@login01.gautschi:[<dir>] $` -- the `user@host:[dir]` part comes from the
SYSTEM profile (`/etc/profile.d/000_rcac_prompt.sh`), and the `(<env>)` prefix comes from
conda activation, which `conda init` does NOT do at shell start (it installs the hook only).
So a fresh shell is missing the prefix.  **Caveat: that is only the default.**  A member
whose `~/.bashrc`, `~/.bash_profile` or `~/.profile` sets `PS1` or `PROMPT_COMMAND`, or who
has `conda config --set changeps1 false`, gets a different prompt — and `claude_bashrc`
reproduces it only as far as it sources `~/.bashrc`: customizations that live in
`~/.bash_profile`/`~/.profile` are read by a login shell but not by `bash --rcfile`.  Check
with `grep -n 'PS1\|PROMPT_COMMAND' ~/.bashrc ~/.bash_profile ~/.profile` before treating
the prompt as a signal.  (Greg's account: only `~/.bashrc` exists and it sets no PS1 —
VERIFIED 2026-09-03.)
`remote_cluster/claude_bashrc` does what a LOGIN shell does: source **`/etc/profile`**
(which runs all of `/etc/profile.d/*.sh`), then `~/.bashrc`, then `conda activate <env>`,
then print a session banner (job, node, walltime, **time remaining**, end time).  Sourcing
`/etc/profile` — rather than cherry-picking the prompt file — is essential: that directory
also defines the **`module`** command (`modules.sh`, `00-modulepath.sh`,
`z01_default_module.sh`), so a shell without it cannot `module load conda/cuda` at all.
Terminals get it with `bash --rcfile <path> -i`; `sinteractive` needs
`env SHELL=remote_cluster/claude_shell` instead, because it runs `$SHELL -l` and a LOGIN
shell ignores `--rcfile` (the wrapper drops the `-l`).  Note `xfce4-terminal --command`
parses argv directly, so use `env VAR=val cmd`, never a bare `VAR=val cmd` prefix.
`claude_bashrc` activates `$CONDA_ENV` (default `mbirtorch`) and prints a warning when that
env does not exist — an earlier version activated a since-retired env with `|| true`, so the
only symptom was a missing `(<env>)` prefix.  Export `CONDA_ENV=<name>` before running the
`tl_*` scripts to use another env.

**A terminal ON the compute node** (so work runs where the GPU is, and PyCharm/viewers can
be started from it): `plans/experiments/remote_cluster/tl_node_terminal.sh` — run it on the
login node with `JOBID=<id>`; it opens `xfce4-terminal` on the allocated node via
`srun --overlap`, optionally running a command first and then dropping to a shell.  Verified
2026-07-25: terminal on h008 inside job 14201524, viewer displayed in ThinLinc.  Each such
terminal is just a job STEP — closing it leaves the allocation alone, so open as many as
needed.

**Backend note.**  mbirtorch's viewer (`mbirtorch/viewer.py`) imports pyplot lazily, on
first use, and takes whatever backend matplotlib picks: TkAgg when `DISPLAY` is set (its
partial-redraw fast path is verified for TkAgg and Agg), Agg when it is not — in which case
`show()` warns and returns without drawing.  So a batch job that reaches the viewer is a
no-op, not an error.  *(legacy)* mbirjax's `viewer.py` instead does `matplotlib.use('TkAgg')`
at import, which is the "TkAgg not available" warning in every old mbirjax nightly log — not
a missing dependency.

**Cost.**  Torch allocates lazily, so an idle mbirtorch viewer holds only what it drew — but
on gautschi an idle *allocation* still burns the group's GPU-hours at the full rate, so
release it when done.  *(legacy)* JAX preallocates ~75% of the GPU at startup, so an idle
mbirjax viewer squats on ~77 GB of an H100 at 0% utilization;
`XLA_PYTHON_CLIENT_PREALLOCATE=false` avoids it, at some performance cost.

## gautschi — H100 (the nightly-regression cluster)

- GPUs: **H100 80GB HBM3**, `gpu:h100:8` per node.  Partition `ai`, account `bouman`,
  QoS `normal`.  **`--cpus-per-task=14` per GPU is required.**
- **Walltime:** `DefaultTime=00:30:00`, `MaxTime=14-00:00:00` (`scontrol show partition
  ai`, VERIFIED 2026-09-03).  Always pass `-t`; a job without it gets 30 minutes.
- **GPU-hours are metered** here (gilbreth is not): `slist` shows the account balance
  (21,857 H100 GPU-hours on 2026-09-03).  QoS `normal` burns one GPU-hour per GPU per hour;
  `-q preemptible` burns 0.25× but the job can be preempted.  Purchase history and
  top-ups: `.claude/gpu-resources.md`.
- **The `ai` partition REFUSES `--mem`** — do not pass it.  Host memory is strictly
  proportional to CPUs, and CPUs to GPUs:

  | slurm setting (`scontrol show partition ai`) | value |
  |---|---|
  | `DefCpuPerGPU` | 14 |
  | `DefMemPerCPU` = `MaxMemPerCPU` | 9200 MB |

  Def == Max is exactly why a `--mem` request has nowhere to land.  So **host RAM per GPU is
  fixed at 9200 MB x 14 ≈ 126 GB**, and the only way to get more host memory is to **request
  more GPUs** (`--gpus-per-node=2` → ~252 GB, and so on).  The node is provisioned to match:
  h-nodes have 112 CPUs / 8 GPUs / 1,031,500 MB, and 9200 x 112 = 1,030,400 MB.
  (gilbreth is different — its `sinteractive` line below does take `--mem`.)
- **Driver:** 595.71.05, **CUDA 13.2** (`nvidia-smi` on h000, VERIFIED 2026-09-03).  This
  is the number that gates which torch/jax wheels work — see the preamble section.
- **Repos: clone your own** under your home (all public on GitHub — see "Repos, data, and
  resources").  Greg's live checkouts sit under his `~/PycharmProjects/`, which is mode
  700: nobody else can read or run them, and a Claude session on his account must not edit
  them — an interactive env there may hold an EDITABLE install pointing at one, so a change
  in the checkout changes what runs.
- **Conda envs are per account — list them, do not assume:**
  `source ~/load_conda_cuda.sh && conda env list`.  Convention: an env named
  `*_regression` belongs to a nightly and is never used interactively; the rest are
  interactive.  Greg's roster on 2026-09-03: `mbirtorch`, `pcdrecon`,
  `mbirtorch_regression`, and `mbirjax_regression` *(legacy)*.
- **Node preamble `~/load_conda_cuda.sh`** (per account, setup step 4) — sourced first in
  every job; contents and what it really loads are in the next subsection.
  `sbatch --wrap "source ~/load_conda_cuda.sh && ..."` needs no other preamble line
  (VERIFIED 2026-09-01, job 15746730, `BATCH_OK`).
- **Greg's nightly** runs from *his* scrontab: `mbirtorch-nightly` at 03:00 (4 GPUs /
  56 cores / up to 4 h).  `scrontab -l` shows only your own entries, so from any other
  account it is invisible — but it still takes 4 GPUs of one node whenever a tracked branch
  has moved, and draws on the shared balance.  Its log is readable only by Greg (path in the
  nightly subsection).  The mbirjax nightly *(legacy)* was disabled on 2026-09-03.

Example batch header (1 GPU):

```bash
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 04:00:00
```

### The node preamble, and which CUDA version actually matters

`~/load_conda_cuda.sh` is sourced first by every batch job, interactive shell and both
nightlies.  Greg's working copy as of 2026-09-03 — create yours with the same contents:

```bash
# ~/load_conda_cuda.sh — node preamble.  Sourced, never executed; must not `exit`.
#
# Non-login bootstrap: `module` is a shell function defined by /etc/profile.d/*.sh, which
# only a LOGIN shell sources.  `sbatch --wrap "source ~/load_conda_cuda.sh && ..."` and
# `ssh host '<cmd>'` are not login shells, so source /etc/profile when the function is
# missing.  /etc/profile reads unset variables, so clear -e/-u across it (the regression
# runner sources this file under `set -uo pipefail`) and restore them afterwards.
if ! command -v module >/dev/null 2>&1 && [ -r /etc/profile ]; then
    __lcc_opts=$-
    set +eu
    . /etc/profile
    case $__lcc_opts in *e*) set -e ;; esac
    case $__lcc_opts in *u*) set -u ;; esac
    unset __lcc_opts
fi
module load conda          # -> conda/2026.03 (the marked default)
module load modtree/gpu    # the GPU module tree; its toolchain brings cuda/12.6.1 with it
module load cuda           # a no-op today: modtree/gpu already loaded cuda/12.6.1 (pin cuda/13.3.0 explicitly if nvcc 13 is wanted)
export HTTPS_PROXY=squid.rcac.purdue.edu:3128   # compute nodes reach github/PyPI only via squid
export HTTP_PROXY=squid.rcac.purdue.edu:3128
# module use /depot/bouman/apps/modules          # group tree (cuda/12.9.0, 13.0.0; cudnn/9.11.0, 9.12.0) — unused
module load cudnn          # -> cudnn/9.2.0.82-12; harmless and unused (below)
```

**What the modules resolve to** (VERIFIED 2026-09-03 on a login node and on h000):

| module | loaded by the preamble | marked default `(D)` | also available |
|---|---|---|---|
| `conda` | 2026.03 | 2026.03 | 2024.09, 2025.02, 2025.09; `anaconda/2025.12-py313` |
| `cuda` | **12.6.1**, pulled in by `modtree/gpu` | 13.3.0 | 12.8.0, 12.9.0, 13.2.0 |
| `cudnn` | 9.2.0.82-12 | 9.2.0.82-12 | 9.12.0.46-12 (both are CUDA-12 builds) |

The 12.6.1 comes from modtree/gpu's toolchain (gcc 11.4.1 / openmpi 5.0.5), not from the
`module load cuda` line, which finds cuda already loaded and does nothing.  An explicit
`module load cuda/13.3.0` or `module unload cuda` does take effect inside a job (VERIFIED
2026-09-03, job 15855388; one login-node trial saw the pin report the change and then show
12.6.1 again, so check `module list` if it matters).  If RCAC moves the toolchain, the
loaded version changes silently — which is why it matters that nothing below depends on it.

**What the envs actually run on** (`pip list`, 2026-09-03):

| env | python | framework | GPU runtime it bundles (pip `nvidia-*`) |
|---|---|---|---|
| `mbirtorch` | 3.11 | torch 2.13.0 (default PyPI wheel), triton 3.7.1, mbirtorch 0.0.2 | CUDA 13.0 runtime, cuDNN 9.20 (`-cu13`) |
| `pcdrecon` | 3.12 | torch 2.13.0, mbirtorch 0.0.2, pcdrecon 0.1.0 | same |
| `mbirtorch_regression` | 3.12 | torch 2.13.0+cu130 (from the cu130 wheel index) | same |
| `mbirjax_regression` *(legacy)* | 3.11 | jax 0.10.1 + `jax-cuda13-plugin`, mbirjax 0.7.2 | CUDA 13.3 runtime, cuDNN 9.25, `nvidia-cuda-nvcc` (its own ptxas) |

So **the CUDA and cuDNN modules are not what torch or jax run on.**  Every env carries its
own CUDA-13 runtime and cuDNN as pip packages and loads them from there; the module's
libraries and its `nvcc`/`ptxas` on PATH go unused (triton and jax ship their own `ptxas`,
and nothing in mbirtorch, pcdrecon or mbirjax compiles CUDA).  The binding constraint is
the **driver**: its CUDA version (13.2 today) must be at least the major the wheels bundle
(13).  **VERIFIED 2026-09-03, job 15855388 on h006** (script and log in
`/scratch/gautschi/buzzard/cluster_use_check/`): with the preamble as it is (cuda 12.6.1 +
cudnn 9.2), with **no cuda or cudnn module at all** (no `ptxas` on PATH), and with
`cuda/13.3.0` pinned (ptxas 13.3 on PATH), the `mbirtorch`, `pcdrecon` and
`mbirjax_regression` envs all initialised CUDA on the H100, ran a GPU matmul (torch) or
reduction (jax), and imported triton and mbirtorch — identical output in all three
configurations.  The nightlies say the same: mbirtorch measured and passed on 2026-08-24,
mbirjax on 2026-08-08, with the module at 12.x and the wheels at 13.x.  The two module
lines are therefore optional; they stay in the listing because they are harmless and give
an `nvcc` on PATH for anything that ever needs one.

**If something changes:**

- *`conda` module bump* — nothing to do: envs carry their own python, and
  `clean_install_all.sh` passes `PYTHON_VERSION` explicitly.  Only a `conda create` with no
  python pin would notice.
- *`cuda`/`cudnn` module change* (RCAC moving the modtree/gpu toolchain) — nothing for the
  torch envs.  For legacy jax, re-check the nightly's `INSTALL_EXTRAS_gpu` against the
  **driver**, not the module — the comments in both metrics repos' `action_scripts/run_configs.env`
  say so since 2026-09-03 (they used to tie the choice to the module).
- *Driver update* — the only change that can break the wheels.  Compare `nvidia-smi`'s
  `CUDA Version` with the bundled major before anything else; a driver that drops below it
  is the one case where the cu12 wheel index (torch) or the `cuda12` extra (mbirjax) would
  come back.
- *torch bump* — mbirtorch pins `torch>=2.13` ("re-test on each torch bump via the
  metrics"); the regression pre-installs from `TORCH_INDEX_URL_gpu=…/whl/cu130`, the
  interactive envs take PyPI's default wheel.  When torch's default CUDA build moves on,
  change that index only if the driver supports the new major.
- *pcdrecon* — depends on `torch` + `mbirtorch` from default PyPI with no CUDA extra, so it
  simply follows torch.  Its `dev_scripts/clean_install_all.sh` was simplified on 2026-09-03
  to the same plain editable install on every host (it had carried a late-2024 gilbreth jax
  workaround and a `[cuda12]` extra its pyproject no longer defines).
- *mbirjax (legacy)* — `INSTALL_EXTRAS_gpu="cuda13,test"`; the `cuda12` extra exists for a
  driver that cannot do 13.  mbirjax also pins `jax>=0.10,!=0.10.2,!=0.11.0` around an XLA
  codegen regression, so new jax releases are the risk there, not CUDA.

### The nightly regression system (runs on gautschi)

Automated performance + correctness tracking: one system, two instances (the mbirjax one
disabled on 2026-09-03), both under Greg's account (anyone can read the code and the
dashboards; only he can read the logs or edit the scrontab entries).

| | mbirtorch (supported) | mbirjax *(legacy)* |
|---|---|---|
| repo | <https://github.com/cabouman/mbirtorch_metrics> | <https://github.com/gbuzzard/mbirjax_metrics> |
| scrontab | `0 3 * * *`, job `mbirtorch-nightly`, 4 GPUs / 56 cores / 4 h | **disabled 2026-09-03** (`disable_nightly.sh`); was `0 2 * * *`, 4 GPUs / 56 cores / 6 h |
| env | `mbirtorch_regression`; torch from `TORCH_INDEX_URL_gpu` (cu130) | `mbirjax_regression`; `INSTALL_EXTRAS_gpu="cuda13,test"` |
| tracked branches (`action_scripts/run_configs.env`) | `main`, `prerelease`, `greg_dev` | `prerelease`, `main` |
| log | `/home/buzzard/.mbirtorch/regression/nightly-<jobid>.log` | `/home/buzzard/.mbirjax/regression/nightly-<jobid>.log` |
| dashboard | published from the repo (see its README) | <https://gbuzzard.github.io/mbirjax_metrics/> |
| last measurement | 2026-08-24, `GATE: PASS` | 2026-08-08, `GATE: WARN`; every night since: `unchanged — skip` |

- **It FRESH-CLONES both repos from origin each run.**  So **uncommitted or unpushed local
  changes have no effect on it** — a fix must be committed and pushed to be picked up.
  This is the single most surprising thing about it.
- **Fire-on-change:** each tracked branch is measured only when its remote tip moves;
  unchanged branches log `unchanged — skip` and the job ends in seconds.  A gap in a
  branch's data usually means it simply did not move, not that anything failed.
- **Logs:** **scron reuses the job id, so the log file is OVERWRITTEN every night** — there
  is no history; capture anything you need before the next run.
- **Alerting:** `--mail-type=FAIL` plus an explicit notify email; the run exits 1 when a gate
  trips, e.g. `main: GATE FAIL (perf regression) — REGRESSION DETECTED`.  That is the system
  working.  What it could NOT catch before 2026-07-25 was a run that silently measured on the
  wrong platform — the mbirtorch engine's platform guard now hard-aborts on that; see the
  failure table.
- **CUDA choice:** the wheel index (torch) or extra (mbirjax) must match the **driver** —
  see the preamble section.  *(legacy)* mbirjax's `INSTALL_EXTRAS_gpu` is independent of the
  library's own `dev_scripts/clean_install_all.sh`; changing one does not change the other.

## gilbreth — A100 (lightly used by our group)

- Account `bouman` shows **4× A100-40GB total** (`slist`).  Greg believes there may be a
  cap of ~2 concurrent GPUs but has not found the command that expresses it —
  **treat the concurrent-GPU limit as unconfirmed; needs investigation.**  No GPU-hour
  metering here: the subscription caps concurrency, not total use.  The slurm account is
  `bouman` here too (`slist`, `sacctmgr`, 2026-09-03); the `bouman-g`/`bouman-n` names in
  `gpu-resources.md` are the portal's purchase queues, used only when requesting access.
- Partition `a100-40gb` **silently mixes two hardware classes** — A100-**SXM4** (features
  `N`/`nvlink`, 4 GPU/node, 400 W) and A100-**PCIe** (`G`, 2 GPU/node, 250 W).  Their
  clocks differ, so **wall times are not comparable across them** — pin one class with
  `--constraint=N` (SXM4) or `--constraint=G` (PCIe) on any timing job.
- Same layout as gautschi: your own clone and conda env, scratch at
  `/scratch/gilbreth/<user>/`, depot mounted at the same path.  Its module set and driver
  have not been checked since the jax era (pcdrecon's install script still carries a
  late-2024 gilbreth workaround) — verify `nvidia-smi` before assuming the CUDA-13 wheels
  work there.
- Interactive session (Greg's usual invocation, known to work):

```bash
sinteractive -N1 -n20 --gpus-per-node=1 --account=bouman \
             --partition=a100-40gb --mem=40G --time=04:00:00
```

- Batch header mirrors gautschi but with `-p a100-40gb --constraint=N`.  **Whether
  gilbreth enforces a per-GPU `--cpus-per-task` rule (gautschi requires 14) is
  unconfirmed — needs investigation;** the `sinteractive` above uses `-n20` for 1 GPU and
  works, so a plain core request is at least accepted.

## Scratch — fast, big, PURGE-ELIGIBLE

- gautschi: `/scratch/gautschi/<user>/`   gilbreth: `/scratch/gilbreth/<user>/` — created
  with the account, mode 700 (so not a way to hand files to another member; use depot).
- Multi-TB, fast, but **purged**: files not accessed or modified for **60 days** are
  removed, after an email warning one week ahead; renaming or `chmod` does not protect a
  file (<https://www.rcac.purdue.edu/policies/scratchpurge>).  Use it for job outputs,
  staging, any large intermediate (`.npy`/`.npz`, traces, compile caches) — and, by Greg's
  practice, conda envs (setup step 5).  Not for anything that must persist.

## Home — a 25 GB quota that fails SILENTLY

- **Never write large artifacts under `~`.**  A job that fills home dies mid-write with
  `sacct` `FAILED 1:0` and **no traceback, no shell echo**, leaving a truncated file.
  Quota accounting can also lag ~one retry after freeing space.  Use symlinks into depot
  or scratch for local access to large files — including `~/.conda` itself (setup step 5).
- Diagnose with `myquota`.  If a job fails with exit 1 and an empty-looking log, check
  `myquota` **before** debugging the code.  A home path that must hold big data can be a
  symlink into scratch.

## Depot — durable, group-shared (mounted on BOTH clusters)

`/depot/bouman/` (≈7 TB of 10 TB used).  Group policy: **long-term = depot, temporary =
scratch.**  It is also the only store other members can read, so it is how files move
between accounts.

- **Access is by unix group** (VERIFIED 2026-09-03): `/depot/bouman` itself is group
  `bouman`, `data/` is group `bouman-data`, `www/` is group `bouman-www`, `apps/` is
  `bouman-apps`; all setgid and group-writable.  A `Permission denied` writing there means
  you are missing the group — ask Greg to add you, then log out and back in.
- **Primary data:** `/depot/bouman/data/` — scan datasets (`.txrm`, etc.), converged
  reference recons, durable results.  Group-shared, so mind others' files.
- **Group module tree:** `/depot/bouman/apps/modules` (`module use` it) carries
  `cuda/12.9.0`, `cuda/13.0.0`, `cudnn/9.11.0`, `cudnn/9.12.0` — user-local builds from
  2025.  Not needed by any current env (the preamble section); mentioned so the line in
  the metrics repos' preamble example makes sense.
- **Public web pages:** `/depot/bouman/www/` is the **web root**, served publicly at
  **`https://www.datadepot.rcac.purdue.edu/bouman/`** (the `www` is dropped in the URL).
  Its landing page is mostly a data repository (its CT-data section links both packages).
  Project report areas are subdirectories:

  | filesystem | public URL | status |
  |---|---|---|
  | `/depot/bouman/www/mbirtorch/<area>/` | `…/bouman/mbirtorch/<area>/` | the root for new pages, created 2026-09-03 with an index (source: `plans/www/mbirtorch_index.html` — add an entry there and redeploy when an area is added) |
  | `/depot/bouman/www/pcdrecon/` | `…/bouman/pcdrecon/` | exists (Aug 2026) |
  | `/depot/bouman/www/mbirjax/<area>/` | `…/bouman/mbirjax/<area>/` | *(legacy)* six mbirjax-era report areas from July 2026, **frozen 2026-09-03**: tree made read-only, index carries a banner pointing at the mbirtorch root; may be deleted later (`chmod -R u+w` first) |

  Publish only finished, shareable **HTML** here (no source, no data) — the destination
  is on the open internet.  Files need `chmod 644`, directories `chmod 755`.  The publish
  idiom is an `rsync` of `*.html` to the depot www dir; see
  `plans/flash_remediation/publish_pages.sh` for the idiom (its `DEST` is the frozen
  `mbirjax/` tree, so as written it now fails by design — copy it and change `DEST`).

## Moving data on and off

```bash
# to the cluster (scp uses the same key as ssh; no -i when the config stanza pins it)
scp -o BatchMode=yes myscript.py <user>@gautschi.rcac.purdue.edu:/scratch/gautschi/<user>/<dir>/
# back to the laptop
scp -o BatchMode=yes <user>@gautschi.rcac.purdue.edu:/scratch/gautschi/<user>/<dir>/result.npz .
```

- **Staging code:** put run scripts in a scratch `scripts/` dir and `scp` updates as you
  iterate.  Do NOT edit inside the user's own checkouts (Greg's are `~/PycharmProjects/*`)
  — those are their working trees, and an interactive env may hold an EDITABLE install
  pointing at one, so a change there changes what runs.
- **Which store:** scratch for job output and anything regenerable; **depot for anything that
  must survive** (scratch is purge-eligible).  `/depot/bouman/` is mounted on **both**
  clusters, so it is the natural way to move results between gautschi and gilbreth without
  going through the laptop — and the only way to hand results to another member.
- **Bringing results home for local viewing** (workflow 4): a 128³ float32 recon is ~8 MB —
  copy it down and use `slice_viewer` locally at full speed rather than over X11.
- Big transfers: prefer `rsync -av` (resumable, skips unchanged) over repeated `scp`.

## Nested ssh — run on a worker node without waiting in the queue

`pam_slurm_adopt` admits ssh to a node currently running one of **your own** jobs (the
session dies with that allocation); it cannot reach a node another member holds.  From the
login node, hop to the worker:

```bash
# find the node running the user's interactive session (job name 'interact'):
ssh -o BatchMode=yes <user>@gautschi.rcac.purdue.edu 'squeue -u $USER'
# then hop to it (double-hop through the login node):
ssh -o BatchMode=yes <user>@gautschi.rcac.purdue.edu 'ssh -o BatchMode=yes h001 "<cmd>"'
```

Keep nested commands **simple** (`;`-joined) — complex quoting through the double hop
silently mangles.  If that job is the user's live interactive session, coordinate before
running anything heavy in it: it shares the GPU (see `--overlap` above).

## Repos, data, and resources

- **Repos — all public on GitHub; clone your own on the cluster:**
  - <https://github.com/cabouman/mbirtorch> — the supported library (torch + triton
    kernels; `mbirtorch/viewer.py` is the slice viewer).
  - <https://github.com/cabouman/mbirtorch_applications> — demos, application/workflow
    scripts, and larger worked examples built on it.
  - <https://github.com/cabouman/mbirtorch_metrics> — the nightly + dashboard; also holds
    `tooling/regression/cluster_preamble.sh.example` (the preamble listing above, since
    2026-09-03).
  - <https://github.com/cabouman/mbirtorch_plans> (this repo) — plans, findings, and the
    experiment scripts, including `plans/experiments/remote_cluster/`.
  - `pcdrecon` (`~/PycharmProjects/pcdrecon` on Greg's account; torch + mbirtorch) — the
    photon-counting-detector reconstruction project; its pages live under
    `/depot/bouman/www/pcdrecon/`.
  - *(legacy)* <https://github.com/cabouman/mbirjax> (docs <https://mbirjax.readthedocs.io>),
    <https://github.com/cabouman/mbirjax_applications>,
    <https://github.com/gbuzzard/mbirjax_metrics> (dashboard
    <https://gbuzzard.github.io/mbirjax_metrics/>), <https://github.com/gbuzzard/mbirjax_plans>.
    Read for old results; do not start new work there.
- **Lilly flash-remediation data** *(legacy analysis; the scans themselves stay useful)*:
  `/scratch/gautschi/buzzard/flash_lilly/` — Greg's scratch, so purge-eligible and not
  readable by other accounts; the durable copies live under depot.
- **Project web pages** (internal findings, published HTML): under
  <https://www.datadepot.rcac.purdue.edu/bouman/> — see the Depot section.

## Running a specific library state

To measure or debug a particular commit WITHOUT disturbing your working checkout or envs:

```bash
# 1. a worktree of the commit -- never touches the working checkout
git -C <your mbirtorch clone> worktree add --detach ~/mbirtorch_main_wt <commit>
# 2. a venv layered over an existing env (no copy of torch/cuda deps)
python -m venv --system-site-packages ~/venvs/mbirtorch_main
# 3. the worktree, deps already satisfied by the parent env
~/venvs/mbirtorch_main/bin/pip install -e ~/mbirtorch_main_wt --no-deps
```

Three properties worth knowing:

* it **wins over any editable install** already in the parent env, so it is a reliable
  override rather than a hope;
* it **never modifies your envs** — the parent conda env is untouched, so the nightly and
  interactive work carry on unaffected;
* **venvs ignore `~/.local` user-site packages**, which is a real advantage: a stray
  `pip install --user` shadowing the library is a distinct failure mode from the mount flap,
  and this layout is immune to it.

Working instance on Greg's gautschi account *(legacy)*: `~/venvs/sharpness_main` over
`~/mbirjax_main_wt`.  Remove a worktree with `git worktree remove <path>` (the
branch/commit is untouched).

## Job preflight — two lines that catch the two worst failures

Put this at the top of every sbatch/srun python entry point:

```python
import torch, mbirtorch
assert torch.cuda.is_available(), "NOT ON GPU"
print("library under test:", mbirtorch.__file__, flush=True)
```

*(legacy)* for mbirjax:

```python
import jax, mbirjax
assert jax.devices()[0].platform == 'gpu', f"NOT ON GPU: {jax.devices()}"
print("library under test:", mbirjax.__file__, flush=True)
```

The assert catches a **silent CPU fallback** (a broken CUDA setup does not raise — jax just
uses the CPU, and torch tensors quietly land on it; the run looks fine while measuring the
wrong thing, which cost three nights of GPU data in July 2026).  The print catches the
**wrong library state** — editable installs, worktrees, venv layering and `~/.local`
shadowing all fail the same way, by running code you did not think you were running.

## Failure signatures → what they actually mean

| symptom | cause / fix |
|---|---|
| `sbatch: error: ... Invalid account or account/partition combination specified`, or `slist` does not list `bouman` | you are not on the queue yet, or the change has not propagated (up to a day, plus a log-out/log-in).  Setup step 1; `.claude/gpu-resources.md`. |
| `Permission denied` creating a file under `/depot/bouman/data/` or `/depot/bouman/www/` | missing unix group (`bouman-data` / `bouman-www`).  Ask Greg to add you; log out and back in. |
| `conda env list` no longer shows an env, its directory holds only `conda-meta/`, and `conda create`/`conda remove` on that name fail with `DirectoryNotACondaEnvironmentError` | the scratch purge hollowed out an env under a scratch-linked `~/.conda` (60 days unused).  `rm -rf ~/.conda/envs/<name>`, then rebuild with the project's `clean_install_all.sh`. |
| an env that worked last month raises `ImportError`/`ModuleNotFoundError` for one rarely-used module, everything else fine, same result on every node | the same purge, partial: files of that env not touched in 60 days are gone.  Rebuild; do not chase a packaging bug. |
| `module list` shows `cuda/12.6.1` although you loaded nothing of the kind, or a `module load cuda/13.3.0` on a login node reports the change and then shows 12.6.1 again | modtree/gpu's toolchain brings cuda 12.6.1 along; inside a job an explicit pin or unload holds (job 15855388).  Harmless either way: torch and jax use their pip-bundled CUDA, not the module (preamble section). |
| `ls: Cannot send after transport endpoint shutdown`, or an intermittent `ModuleNotFoundError` for numpy/stdlib internals that **differs run to run and hits every env** | the LOGIN NODE's home mount is flapping.  The files are fine and compute nodes are unaffected — retry, or move the work to a node.  Do not go hunting for a broken install.  (Bit three times on 2026-07-25.)  Distinguish from the purge row above: this one varies run to run. |
| one specific file returns `Input/output error` on EVERY read/write, across connections, while its siblings are fine | an INTERRUPTED transfer (scp killed by a mount flap) left a corrupt Lustre file — this is persistent damage, not the transient flap above.  `rm -f` the file and rewrite it; then verify by md5, and do the verification **in the batch job on the compute node**, not on a login node (2026-08-08, nt1 staging). |
| sbatch/srun on gautschi `ai` rejected for a memory request | that partition refuses `--mem` (`DefMemPerCPU == MaxMemPerCPU == 9200`).  Drop `--mem`; ask for more GPUs if you need more host RAM. |
| job killed at exactly 30 minutes | no `-t` was passed; the `ai` partition's `DefaultTime` is 00:30:00. |
| job exits 1, log looks empty or truncated mid-write | **home quota full** (25 GB, fails SILENTLY).  `myquota`; write to scratch instead. |
| `Access denied by pam_slurm_adopt: you have no active jobs on this node` | ssh to a compute node is only allowed while you hold a job there.  Get an allocation first, or `srun --overlap --jobid=<id>` instead. |
| `Host key verification failed` on `loginNN.gautschi…` | the per-node name is not in known_hosts.  Hop through the round-robin address: `ssh gautschi 'ssh login01 "…"'`. |
| a script fed to `ssh host 'bash -s'` stops silently right after an `srun` line | `srun` consumed the rest of the script from stdin.  `srun … </dev/null`. |
| a job dies in seconds with `pip` `OSError: No such file or directory: ...__editable__*.pth` | two jobs ran `pip install -e` into the SAME env at once and one read the torn `.pth` (any sbatch script that installs on startup can do this).  Don't launch them simultaneously — chain with `sbatch --dependency=afterany:<jobid>`, which also protects a RUNNING job whose subprocesses re-import the package. |
| a worker subprocess raises `ImportError: cannot import name ...` from a package that is definitely current | the package tree was scp-SYNCED while the job was running, and the worker imported a half-updated pair of modules (new `__init__` against an old sibling).  Same class as the pip race: never mutate a staged tree or env while a job runs from it — sync first, then submit, or chain the sync's consumer with `--dependency`. |
| GPU charts stop updating; a nightly aborts with `PLATFORM MISMATCH` / `DEVICE PIN FAILED` (torch) or `Jax plugin configuration error` *(legacy)*, or *(legacy)* results land as `regression_cpu_*` under `results/gpu/` | the framework fell back to CPU: the wheels' CUDA major is not supported by the driver (check `nvidia-smi`), or the env was rebuilt against the wrong index/extra.  The torch engine hard-aborts on this; before 2026-07-25 the jax one measured the whole sweep on CPU. |
| `srun: error: Ignoring --x11 option for a job step within an existing job` | harmless.  X11 is set at ALLOCATION time; steps inherit it.  If the allocation lacks `--x11` it cannot be retrofitted — start a new one. |
| prompt shows `bash-5.1$`, **or `module: command not found`** | non-login shell: `/etc/profile` (hence all of `/etc/profile.d/*.sh`) was not sourced.  That directory supplies BOTH the prompt and the `module` function.  The preamble listed above sources `/etc/profile` itself when `module` is missing, so this no longer comes from `~/load_conda_cuda.sh`; any OTHER script that calls `module` in a non-login shell still shows it.  Use `remote_cluster/claude_bashrc`, or copy the guard from the top of the preamble. |
| XQuartz: "Cannot establish any listening sockets" | stale `/tmp/.X0-lock` from a failed start — delete it and retry. |
| a GUI window vanished when its app closed | the allocation was `srun <cmd>`, which ends with the command.  Hold it with a shell instead. |
| *(legacy)* tests pass but prove nothing about the GPU kernels | `tests/test_pallas_kernels.py` silently runs in interpret mode when `_pallas_kernels.availability()` is False.  Assert availability first. |

## Don't

- **No compute on login nodes at all** — not even a short python analysis script, not even
  `import torch`.  They are shared, have no GPU, and their home mount flaps (see the failure
  table).  There is a one-liner substitute with no excuse not to use it:
  ```bash
  sbatch -A bouman -p ai -N1 --gpus-per-node=1 --cpus-per-task=14 -t 0:20:00 \
         --wrap "source ~/load_conda_cuda.sh && conda activate <env> && python -u script.py"
  ```
- **Never write large artifacts under `~`** — the 25 GB quota kills jobs with no traceback.
- **Nothing but finished HTML to `/depot/bouman/www/`** — it is served on the open internet.
  No data, no source, no drafts.
- **Don't edit the user's cluster checkouts** (Greg's are `~/PycharmProjects/*`) — stage
  your own scripts in scratch.
- **Don't hold a GPU you are not using** — on gautschi an idle allocation burns the group's
  metered GPU-hours at the full rate.  `exit` interactive sessions; give batch jobs a
  realistic `-t`.
- **Coordinate before heavy gautschi use** — the group's interactive sessions and Greg's
  02:00 and 03:00 nightlies share the account, the queue and the GPU-hour balance.
  gilbreth is lightly used by the group; submit freely there.
- **Don't assume an uncommitted fix reaches the nightly** — it fresh-clones from origin.
- **Don't start new work on mbirjax** — it is legacy; port to mbirtorch.

## Legacy mbirjax — sunset checklist

What still points at mbirjax, and what was already done.

**Done 2026-09-03** (working-tree changes are staged, not committed; the metrics-repo changes
reach the nightly only once pushed):

- `www/mbirjax/` frozen (tree made read-only) with a banner at the top of its index linking
  to the new `www/mbirtorch/` root; the root data page now names both packages.  Unfreeze
  with `chmod -R u+w /depot/bouman/www/mbirjax`.
- `mbirjax-nightly` removed from the scrontab (`disable_nightly.sh`); the Mac agent
  `com.mbirjax.regression` was already disabled.  The `mbirjax_regression` env is kept.
- `cluster_preamble.sh.example` in both metrics repos replaced with the working preamble;
  the CUDA comments in both `run_configs.env` reworded to the driver rule.
- pcdrecon `clean_install_all.sh`: jax-era branches removed.
- `plans/experiments/remote_cluster/`: account paths and env name parameterized, viewer
  demo ported to mbirtorch, `cu_check.sh` added.
- Greg's account: the legacy venv + worktree and the hollow `mbirjax` env directory removed.

**Remaining**

1. *Decide:* delete `www/mbirjax/` (and the *(legacy)* rows and the mbirjax column of the
   nightly table in this guide) once nobody needs the archive; until then it stays frozen.
2. *Decide:* delete `~/.conda/envs/mbirjax_regression` and the cluster's `mbirjax_metrics`
   checkout once mbirjax is fully frozen (no goldens regeneration is planned).
3. Push the metrics-repo and pcdrecon changes; the nightly fresh-clones from origin, and a
   new clone of `mbirtorch_metrics` gets the corrected preamble example only after that.
4. Run the ported viewer demo once through `tl_slice_viewer.sh` to re-verify route B with
   mbirtorch (the 2026-07-25 verification was of the mbirjax original).
5. Leave alone: `cabouman/mbirjax`, `mbirjax_applications`, `mbirjax_plans`, readthedocs,
   the metrics dashboard — archives.  The mbirjax landing page's source in
   `mbirjax_plans/plans/slice_parity/` no longer matches the deployed, bannered copy, which
   is fine while the tree is frozen.
