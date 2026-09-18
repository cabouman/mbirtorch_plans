#!/bin/bash
# Open a terminal ON THE COMPUTE NODE of an existing interactive allocation, with X
# forwarded to the ThinLinc desktop, and (optionally) run a command in it.
#
# This is the "second window" half of the workflow: tl_gpu_session.sh gets the node and
# holds it from a login-node terminal; this puts a shell ON that node so work runs where
# the GPU is, and GUI apps land in ThinLinc.
#
# Run ON the login node hosting the ThinLinc session, from a copy of this directory on
# scratch:
#     JOBID=<allocation id> bash tl_node_terminal.sh             # (squeue -u $USER)
#     JOBID=<id> RUN_CMD='<command>' bash tl_node_terminal.sh    # run something first
# Env: CONDA_ENV (default mbirtorch) picks the env for both the demo and the shell;
#      SCRIPTS_DIR defaults to the directory this file lives in.
set -u

SCRIPTS_DIR=${SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
export CONDA_ENV=${CONDA_ENV:-mbirtorch}
JOBID=${JOBID:?set JOBID=<allocation id> (see: squeue -u $USER)}
# Command run in the new terminal before it drops to an interactive shell.  `exec bash`
# afterwards keeps the window alive when the command (or its viewer window) exits.
RUN_CMD=${RUN_CMD:-"\$HOME/.conda/envs/$CONDA_ENV/bin/python -u $SCRIPTS_DIR/x11_slice_viewer_demo.py"}
RCFILE=$SCRIPTS_DIR/claude_bashrc

XVNC_ARGS=$(ps -u "$USER" -o args= 2>/dev/null | grep "[X]vnc :" | head -1)
[ -n "$XVNC_ARGS" ] || { echo "FATAL: no live Xvnc for $USER on $(hostname)"; exit 2; }
DISPLAY=$(printf '%s\n' "$XVNC_ARGS" | grep -oE 'Xvnc :[0-9]+' | grep -oE ':[0-9]+')
XAUTHORITY=$(printf '%s\n' "$XVNC_ARGS" | sed -n 's/.*-auth \([^ ]*\).*/\1/p')
export DISPLAY XAUTHORITY
echo "login node=$(hostname) DISPLAY=$DISPLAY jobid=$JOBID env=$CONDA_ENV scripts=$SCRIPTS_DIR"

# --overlap: share the allocation with the sinteractive shell already holding it
#            (job steps are exclusive since Slurm 20.11 -- without this it would hang).
# --x11    : X11 is set at ALLOCATION time (sinteractive --x11) and steps inherit it; inside
#            such a job this flag draws a harmless "Ignoring --x11 option" and DISPLAY is set
#            anyway.  If the allocation was created WITHOUT --x11, no step can display.
exec srun --overlap --jobid="$JOBID" --x11 \
     xfce4-terminal --disable-server --geometry=110x34+120+120 \
       --title="h-node shell (job $JOBID) -- GPU work runs here" \
       --command="bash --rcfile $RCFILE -i -c '$RUN_CMD; echo; echo \"[command finished -- shell follows]\"; exec bash --rcfile $RCFILE -i'"
