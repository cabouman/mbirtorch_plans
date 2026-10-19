---
name: queues
description: Refresh the GPU queue page (Gilbreth, Gautschi, Perlmutter, Delta) and republish it so the user can pick the best machine and queue for a job. Use when they ask for the queue page, queue status, wait times, or "which machine".
---

# GPU queue page

`collect.py` (in this skill's directory) reads every queue listed in `machines.json` over ssh
and writes `queues.html` and `queues.json`.  Nothing is submitted to any cluster: the start
estimate comes from `sbatch --test-only`.

Per-laptop settings live in `~/.config/gpu-queues/config.json`:

- `artifact_url`: the page's Artifact URL.  An Artifact belongs to one Claude account, so each
  account (each laptop) has its own.
- `out_dir`: where the HTML and JSON go (default `~/.cache/gpu-queues`).

`tools/clusters/install.sh` in mbirtorch_plans creates that file and links this skill into
`~/.claude/skills/queues`.

## Steps

1. Run the collector (1 to 2 minutes):

       python3 ~/.claude/skills/queues/collect.py

   To refresh only some machines, name them: `collect.py delta perlmutter`.  Machines that
   cannot be reached keep their last numbers and are shown as stale.

2. Publish `<out_dir>/queues.html` with the Artifact tool, passing `url: <artifact_url>` so
   the link stays the same.  The collector prints both paths on its last two lines.  If
   `artifact_url` is not set, publish without `url` (icon `chart`), then write the new URL
   into `artifact_url` in the config file.

3. Tell the user, in at most three lines: the page is updated, which machines are stale, and
   the one login that would refresh each stale machine (the collector prints it, e.g.
   `gpu-login nersc`).  Never ask them to log in before running; run first, then report.

## What the columns mean

- GPUs free: total GPUs minus GPUs in use, over nodes that are up and not reserved.
- Waiting and Running: jobs in that queue right now, from everyone.
- Max job length: the smaller of the partition limit and the QoS limit.
- Slurm estimate: start time of a typical test job (`sbatch --test-only`) minus now.  Slurm
  assumes every running job uses its full time limit, so this is usually pessimistic.
- My recent waits: actual waits of the user's own jobs in that queue over the last 30 days.

## Logins

Run `gpu-login status` to see which logins are live.  Gilbreth and Gautschi use keys and are
always reachable.  Perlmutter needs the 24-hour key from `gpu-login nersc` and Delta the
4-hour shared connection from `gpu-login delta`; only the user can run those (they type the
NERSC code or approve the Duo push).

## Adding a queue

Add an entry to `machines.json`: `partition`, `qos` (for the time limit), `job_qos` (what jobs
in the queue are tagged with, for counting; empty means do not filter), `gpu` (a label), and
`test` (the sbatch options of a typical job on that queue).  `-A $ACC` in `test` uses the
account auto-detected on a machine whose `account` is `"auto"`.  Accounts here are the
group's; a member whose accounts differ edits the `test` lines.
