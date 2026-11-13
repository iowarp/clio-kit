#!/usr/bin/env bash
# AQ43 support, at the SSH level: a BatchMode connection to `locked` fails at
# once with an authentication error, and leaves no ssh process behind.
set -uo pipefail
start=$(date +%s%N)
out="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$CLIO_PV_CLUSTER_LOCKED" true 2>&1)"
status=$?
ms=$(( ($(date +%s%N) - start) / 1000000 ))
echo "ssh -o BatchMode=yes $CLIO_PV_CLUSTER_LOCKED true -> exit $status in ${ms}ms: $out"
left="$(pgrep -af "ssh .*$CLIO_PV_CLUSTER_LOCKED" || true)"
[ -z "$left" ] || { echo "ssh processes left: $left"; exit 1; }
[ "$status" -ne 0 ] && [ "$ms" -lt 5000 ] && printf '%s' "$out" | grep -q 'Permission denied'
