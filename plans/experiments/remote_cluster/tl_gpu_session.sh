#!/bin/bash
# Open a terminal INSIDE the ThinLinc desktop that holds an interactive GPU allocation.
#
# This reproduces the usual interactive workflow, started by Claude:
#     terminal on the login node -> sinteractive -> shell on a GPU node -> run GUI apps
# and it has the lifetime the user expects: the allocation is held by the SHELL in the
# terminal, so closing a viewer window changes nothing.  It ends only on `exit` (or closing
# the terminal, or the walltime expiring).
#
# Contrast with `srun --x11 <app>`, which allocates the node to run ONE command: when the
# app exits, the allocation dies with it.
#
# `sinteractive --x11` forwards X from the compute node back to this login node's display,
# which is the ThinLinc session -- so GUI apps launched in that shell appear in ThinLinc.
#
# Claude can run further work in the SAME allocation without re-queuing:
#     srun --overlap --jobid=<id> <cmd>
# `--overlap` is required: since Slurm 20.11 job steps are exclusive, and sinteractive's
# own `srun --pty $SHELL` step already holds the resources (see `sinteractive --help`).
#
# Run ON the login node hosting the ThinLinc session, from a copy of this directory on
# scratch.  The scripts find each other through SCRIPTS_DIR, which defaults to the directory
# this file lives in; nothing account-specific is hard-coded.
#     nohup bash tl_gpu_session.sh > /tmp/tl_gpu_session.log 2>&1 &
# Env: CONDA_ENV (default mbirtorch) is the env claude_bashrc activates in the new shell.
set -u

SCRIPTS_DIR=${SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
export CONDA_ENV=${CONDA_ENV:-mbirtorch}

# ── allocation request (edit here; no command-line arguments by project convention) ──
ACCOUNT=bouman
PARTITION=ai
GPUS=1
CPUS=14                 # gautschi requires 14 cores per GPU
WALLTIME=04:00:00
TITLE="GPU interactive session (exit here to release the node)"
# Use the SAME terminal as the desktop (ThinLinc here runs XFCE), not bare xterm:
# xfce4-terminal has the File/Edit/View/Terminal/Tabs/Help menu bar -- Edit > Copy/Paste is
# the one people reach for -- and it inherits the user's saved profile, so the font matches.
# Plain xterm has neither.  --disable-server forces a private process rather than attaching
# to an existing terminal server, which the nohup/detach pattern needs.
TERM_CMD=xfce4-terminal

# ── discover the live ThinLinc session on this node ───────────────────────────
XVNC_ARGS=$(ps -u "$USER" -o args= 2>/dev/null | grep "[X]vnc :" | head -1)
if [ -z "$XVNC_ARGS" ]; then
    echo "FATAL: no live Xvnc for $USER on $(hostname) -- wrong login node?"
    exit 2
fi
DISPLAY=$(printf '%s\n' "$XVNC_ARGS" | grep -oE 'Xvnc :[0-9]+' | grep -oE ':[0-9]+')
XAUTHORITY=$(printf '%s\n' "$XVNC_ARGS" | sed -n 's/.*-auth \([^ ]*\).*/\1/p')
export DISPLAY XAUTHORITY
echo "host=$(hostname)  DISPLAY=$DISPLAY  XAUTHORITY=$XAUTHORITY  scripts=$SCRIPTS_DIR  env=$CONDA_ENV"

# ── the TERMINAL holds the allocation: sinteractive is its foreground process ─
# --hold keeps the window up after the shell exits so any error stays readable.
# `env SHELL=...` (not a bare VAR=val prefix): xfce4-terminal --command parses argv
# directly rather than via a shell, so a bare prefix would be taken as the program name.
# SHELL= makes sinteractive (which runs `$SHELL -l`) start our wrapper instead, so the
# shell comes up with the conda env active -- the usual
#     (<env>) <user>@host:[dir] $
# prompt -- and prints a banner with the job id, node and TIME REMAINING.  CONDA_ENV is
# exported above and travels with the job's environment.
ALLOC="env SHELL=$SCRIPTS_DIR/claude_shell sinteractive --x11 -A $ACCOUNT -p $PARTITION -N1 \
       --gpus-per-node=$GPUS --cpus-per-task=$CPUS -t $WALLTIME"

if command -v "$TERM_CMD" >/dev/null 2>&1; then
    exec "$TERM_CMD" --disable-server --hold --geometry=110x32+60+60 \
         --title="$TITLE" --command="$ALLOC"
fi

echo "WARNING: $TERM_CMD not found; falling back to xterm (no menu bar, no copy/paste)."
exec xterm -geometry 100x30+60+60 -title "$TITLE" -bg black -fg lightgray -hold \
     -e $ALLOC
