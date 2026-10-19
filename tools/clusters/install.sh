#!/bin/bash
# install.sh -- set up this laptop for the group's GPU clusters (see README.md beside it).
# Safe to re-run: it adds only what is missing and never edits an existing ssh stanza.
#
#   tools/clusters/install.sh [--purdue-user U] [--nersc-user U] [--ncsa-user U]
#
# A user name that is not given is asked for when there is a terminal; without one, the ssh
# stanza for that site is skipped.  What it does:
#   ~/.local/bin/gpu-login        -> link to gpu-login in this clone
#   ~/.claude/skills/queues       -> link to queues/ in this clone (the queue-page skill)
#   ~/.config/gpu-queues/config.json   created if missing (artifact_url, out_dir)
#   ~/.ssh/cm                     created (Delta's shared-connection sockets)
#   ~/.ssh/config                 missing gautschi/gilbreth/perlmutter/delta stanzas appended
#                                 from ssh_config.example, after a dated backup
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd -P)

PURDUE_USER='' NERSC_USER='' NCSA_USER=''
while [ $# -gt 0 ]; do
    case $1 in
        --purdue-user) PURDUE_USER=$2; shift 2 ;;
        --nersc-user)  NERSC_USER=$2;  shift 2 ;;
        --ncsa-user)   NCSA_USER=$2;   shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# link SRC DST: make DST a link to SRC; leave a real file or directory at DST alone.
link() {
    if [ -L "$2" ] || [ ! -e "$2" ]; then
        ln -sfn "$1" "$2" && echo "linked   $2 -> $1"
    else
        echo "SKIPPED  $2 exists and is not a link; move it aside and re-run"
    fi
}

echo "== links"
mkdir -p ~/.local/bin ~/.claude/skills
link "$HERE/gpu-login" ~/.local/bin/gpu-login
link "$HERE/queues" ~/.claude/skills/queues

echo "== queue-page config"
CFG=~/.config/gpu-queues/config.json
if [ -e "$CFG" ]; then
    echo "exists   $CFG"
else
    mkdir -p "$(dirname "$CFG")"
    printf '{\n  "artifact_url": "",\n  "out_dir": "~/.cache/gpu-queues"\n}\n' > "$CFG"
    echo "created  $CFG (artifact_url is filled in the first time the page is published)"
fi

echo "== ssh"
mkdir -p -m 700 ~/.ssh ~/.ssh/cm
touch ~/.ssh/config && chmod 600 ~/.ssh/config
has_host() { grep -qE "^[[:space:]]*Host[[:space:]]+(.*[[:space:]])?$1([[:space:]]|\$)" ~/.ssh/config; }
ask() {  # ask VAR PROMPT: fill VAR from the terminal when it is empty and there is one
    if [ -z "${!1}" ] && [ -t 0 ]; then read -r -p "$2: " "$1"; fi
}
BACKED_UP=''
add_host() {  # add_host HOST PLACEHOLDER USER
    if has_host "$1"; then echo "exists   Host $1"; return; fi
    if [ -z "$3" ]; then echo "SKIPPED  Host $1 (no user name given)"; return; fi
    if [ -z "$BACKED_UP" ]; then
        BACKED_UP=~/.ssh/config.bak-$(date +%Y%m%d-%H%M%S)
        cp ~/.ssh/config "$BACKED_UP" && echo "backup   $BACKED_UP"
    fi
    { echo; echo "# Added by mbirtorch_plans/tools/clusters/install.sh on $(date +%F)."
      awk -v h="$1" '$1 == "Host" && $2 == h {on = 1} on && /^$/ {exit} on' "$HERE/ssh_config.example" \
        | sed "s/<$2>/$3/"; } >> ~/.ssh/config
    echo "added    Host $1 (User $3)"
}
if ! has_host gautschi || ! has_host gilbreth; then ask PURDUE_USER "Purdue career account"; fi
if ! has_host perlmutter; then ask NERSC_USER "NERSC user name"; fi
if ! has_host delta; then ask NCSA_USER "NCSA login name (from NCSA's welcome email)"; fi
add_host gautschi purdue_user "$PURDUE_USER"
add_host gilbreth purdue_user "$PURDUE_USER"
add_host perlmutter nersc_user "$NERSC_USER"
add_host delta ncsa_user "$NCSA_USER"

echo "== checks"
command -v python3 >/dev/null && echo "ok       python3" || echo "MISSING  python3"
command -v expect >/dev/null && echo "ok       expect" || echo "MISSING  expect (ships with macOS; on Linux install it)"
command -v sshproxy >/dev/null && echo "ok       sshproxy" \
    || echo "MISSING  sshproxy: install sshproxy-<version>-macos-universal.pkg from https://portal.nersc.gov/cfs/mfa/"
case ":$PATH:" in *":$HOME/.local/bin:"*) echo "ok       ~/.local/bin is on PATH" ;;
    *) echo "MISSING  ~/.local/bin is not on PATH: add  export PATH=\"\$HOME/.local/bin:\$PATH\"  to ~/.zshrc" ;; esac

user_of() { ssh -G "$1" 2>/dev/null | awk '$1 == "user" {print $2; exit}'; }
cat <<EOF

== next steps (you, in your own terminal; Claude never handles passwords)
1. Store the two passwords in the Keychain (each command prompts for the password):
     security add-generic-password -U -s nersc-password -a $(user_of perlmutter) -w
     security add-generic-password -U -s delta-password -a $(user_of delta) -w
2. The first time on this laptop, run  ssh delta  once by hand to accept Delta's host key
   (then type exit), and  ssh gilbreth  /  ssh gautschi  likewise if they ask.
3. Log in:  gpu-login all   (you type the NERSC code and tap Approve for Delta)
4. Check:   gpu-login status
EOF
