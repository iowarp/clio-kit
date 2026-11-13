#!/usr/bin/env bash
# AQ39 support: what is running on each machine. Run it during AQ38. It
# prints the host's ssh clients and every container's process tree, then
# the user processes on login that are not sshd: AQ39 wants none of ours.
# Exit 0 when login has none, 1 otherwise.
set -uo pipefail
"$CLIO_PV_CLUSTER" ps host node08 login
extra="$("$CLIO_PV_CLUSTER" exec login ps -u researcher -o pid,args --no-headers \
    | grep -v -E 'sshd: |ps -u ' || true)"
if [ -n "$extra" ]; then
    echo "login runs processes of ours:"; echo "$extra"; exit 1
fi
echo "login: nothing of ours"
