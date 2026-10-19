# tools/clusters: logins, queue page and GPU check for the group's clusters

The guide is [`.claude/cluster_use.md`](../../.claude/cluster_use.md).  This folder holds the
tools it refers to.  They are generic: everything per person lives on the laptop (ssh user
names, Keychain items, the queue page's URL), so any group member can use them.

| file | what it is |
|---|---|
| `gpu-login` | `gpu-login status` shows which logins are live (any Claude session can run it); `gpu-login nersc` / `delta` / `all` renew them (the human runs these). |
| `install.sh` | one-time setup of a laptop; safe to re-run |
| `ssh_config.example` | the `gautschi`, `gilbreth`, `perlmutter`, `delta` stanzas for `~/.ssh/config` |
| `queues/` | the `queues` Claude skill: reads every GPU queue (nothing is submitted) and publishes a page that names the best queue for each job length |
| `gpu_test/` | a 2-minute PyTorch GPU check, with job scripts for Perlmutter and Delta |

## Setting up a laptop (macOS)

The person must already have accounts on the machines: see "Accounts and access" in the guide.

1. Clone this repo (or `git pull`), then run
   `tools/clusters/install.sh --purdue-user <u> --nersc-user <u> --ncsa-user <u>`.
   It links `gpu-login` into `~/.local/bin` and the skill into `~/.claude/skills/queues`, adds
   the missing ssh stanzas (after a dated backup of `~/.ssh/config`), and checks for
   `sshproxy` and `expect`.  Leave out a site you have no account on.
2. Install NERSC's `sshproxy` if step 1 says it is missing:
   `sshproxy-<version>-macos-universal.pkg` from <https://portal.nersc.gov/cfs/mfa/>.
3. Store the two passwords in the Keychain.  The person types these in their own terminal;
   each command prompts for the password, and Claude never sees or types it:

       security add-generic-password -U -s nersc-password -a <nersc user> -w
       security add-generic-password -U -s delta-password -a <ncsa user> -w

4. Run `ssh delta` once by hand to accept Delta's host key (and `ssh gilbreth`,
   `ssh gautschi` if they ask), then `gpu-login all`: type the 6-digit NERSC code, and tap
   Approve on the phone for Delta.
5. Check with `gpu-login status`.  The first time the queue page is published (ask Claude for
   "the queue page"), its URL is saved in `~/.config/gpu-queues/config.json`.
6. So that every Claude session on the laptop knows about the clusters, add a section like this
   to `~/.claude/CLAUDE.md`, with your own user names:

       # GPU clusters
       Gautschi and Gilbreth (Purdue RCAC), Perlmutter (NERSC), Delta (NCSA).  User names: ...
       - Before any cluster work, run `gpu-login status`.  It never prompts.  If a login is down,
         tell me the one command to run (`gpu-login nersc` or `gpu-login delta`).  Never type,
         store or ask for a password or one-time code.
       - The guide is `.claude/cluster_use.md` in mbirtorch_plans.  Read its first part, through
         "Delta (NCSA)", before using a cluster.
       - For "which machine" or queue waits, use the `queues` skill.

## Day to day

- **NERSC:** `gpu-login nersc` once every 24 hours.  It reads the password from the Keychain
  and asks only for the 6-digit code, so NERSC stays two-factor: the code comes from the phone.
- **Delta:** `gpu-login delta` once every 4 hours of no use.  It reads the password from the
  Keychain and asks Duo for a push; you tap Approve.
- **RCAC** (Gautschi, Gilbreth): keys, nothing to renew.
- `gpu-login all` renews whatever needs it; `gpu-login status` exits 1 when something needs you.

## Notes

- The skill and `gpu-login` are links into this clone, so `git pull` updates them.  Switching
  this clone to a branch that lacks them breaks the links until you switch back.
- Cloud Claude sessions cannot use any of this: they have no keys and cannot answer Duo.
- Off macOS there is no Keychain: `gpu-login` falls back to plain `sshproxy` and `ssh -fN delta`
  prompts, and `install.sh` works except for the Keychain step.
- The queue page started as Charlie's `queues` skill (October 2026); this copy reads its
  output folder and Artifact URL from `~/.config/gpu-queues/config.json` instead of fixed paths.
