#!/usr/bin/env bash
# AQ50 support: the node half's port can be reached only through the
# person's SSH connection. Give the port as CLIO_PV_NODE_PORT, or it is
# found as the one TCP port a researcher process listens on in node08
# (other than the fixture pvserver). Exit 2 (skip) when no node half runs.
#   from login:            node08:PORT -> refused (it binds loopback)
#   on node08, no token:   127.0.0.1:PORT answers 401/403, or closes
#   from this machine:     node08 does not resolve; nothing is published
set -uo pipefail
C="$CLIO_PV_CLUSTER"
port="${CLIO_PV_NODE_PORT:-}"
if [ -z "$port" ]; then
    # As researcher, ss names the process only for researcher's own sockets.
    port="$("$C" exec node08 ss -ltnpH 2>/dev/null | grep 'users:(' \
        | grep -v ":$CLIO_PV_CLUSTER_PVSERVER_PORT " | awk '{print $4}' | sed 's/.*://' | head -1)"
fi
[ -n "$port" ] || { echo "no node half listening on node08; SKIP"; exit 2; }
echo "node half port: $port"
"$C" exec node08 ss -ltnpH | grep ":$port " || true
fail=0
bind="$("$C" exec node08 ss -ltnH "sport = :$port" | awk '{print $4}')"
echo "bound to: $bind"
case "$bind" in 127.0.0.1:*|\[::1\]:*) ;; *) echo "FAIL: not loopback"; fail=1 ;; esac
from_login="$("$C" exec login curl -sS -m 5 -o /dev/null -w '%{http_code}' "http://node08:$port/" 2>&1)"
echo "from login, no token: $from_login"
case "$from_login" in *"Connection refused"*|*"Failed to connect"*|401|403) ;; *) fail=1 ;; esac
on_node="$("$C" exec node08 curl -sS -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/" 2>&1)"
echo "on node08, no token: $on_node"
case "$on_node" in 401|403|*"Empty reply"*|*"reset"*) ;; *) fail=1 ;; esac
direct="$(curl -sS -m 5 -o /dev/null -w '%{http_code}' --noproxy '*' "http://node08:$port/" 2>&1)"
echo "from this machine: $direct"
case "$direct" in *"resolve"*|*"Failed to connect"*|*"refused"*) ;; *) fail=1 ;; esac
exit "$fail"
