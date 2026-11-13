#!/usr/bin/env bash
# The cluster fixture's self-checks. Each line is: PASS|FAIL, the check, the
# command, its exit status, how long it took and its output (first lines).
# Run with `cluster.sh check` (or `cluster.sh scenarios selfcheck`).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$("$HERE/cluster.sh" env)"
failures=0
P="$CLIO_PV_CLUSTER_PREPARE"

# check NAME EXPECT CMD: EXPECT is "ok" (exit 0), "fail" (non-zero) or a
# regular expression the output must match (any exit status).
check() {
    local name="$1" expect="$2" cmd="$3" out status start ms verdict=FAIL
    start=$(date +%s%N)
    out="$(timeout 120 bash -c "$cmd" 2>&1)"
    status=$?
    ms=$(( ($(date +%s%N) - start) / 1000000 ))
    case "$expect" in
        ok) [ "$status" -eq 0 ] && verdict=PASS ;;
        fail) [ "$status" -ne 0 ] && verdict=PASS ;;
        *) printf '%s' "$out" | grep -Eq -- "$expect" && verdict=PASS ;;
    esac
    [ "$verdict" = PASS ] || failures=$((failures + 1))
    printf '%s | %s | `%s` | exit %s | %sms | %s\n' "$verdict" "$name" "$cmd" "$status" "$ms" \
        "$(printf '%s' "$out" | head -4 | tr '\n' ' ' | cut -c1-300)"
}

check "login answers over SSH" ok "ssh login true"
check "ares is the same host" "^login$" "ssh ares hostname"
check "node08 has ParaView 6 behind the prepare line" "paraview version 6\." \
    "ssh node08 '$P; pvpython --version'"
check "node08's pvpython is Python 3.12" "^3\.12" \
    "ssh node08 '$P; pvpython -c \"import sys; print(sys.version)\"'"
check "node08 has no ParaView without the prepare line" fail "ssh node08 'command -v pvpython'"
check "login has no ParaView" fail "ssh login '$P; command -v pvpython'"
check "node08 renders headless (OSMesa)" "rendered .* [0-9]+ bytes" \
    "ssh node08 '$P; pvpython /opt/clio-pv-fixture/probe.py render'"
check "node08 cannot reach the package index" fail \
    "ssh node08 'curl -sS -m 8 -o /dev/null https://pypi.org/simple/'"
check "login reaches the package index" "^200$" \
    "ssh login 'curl -sS -m 20 -o /dev/null -w %{http_code} https://pypi.org/simple/trame/'"
check "login can pip-download for Python 3.12" "Saved|already" \
    "ssh login 'rm -rf /tmp/pipcheck; python3 -m pip download -q --no-deps -d /tmp/pipcheck \
--only-binary=:all: --python-version 3.12 --platform manylinux2014_x86_64 \
--implementation cp trame-vtk && ls /tmp/pipcheck && echo Saved; rm -rf /tmp/pipcheck'"
check "node08 is not reachable without login" fail \
    "ssh -o ProxyJump=none -o ConnectTimeout=5 node08 true"
check "ares-compute-08 needs the via value" fail \
    "ssh -o ConnectTimeout=5 ares-compute-08 true"
check "ares-compute-08 through -J ares" "^node08$" "ssh -J ares ares-compute-08 hostname"
check "login reaches node08 inside the cluster" "^node08$" "ssh login 'ssh node08 hostname'"
check "login and node08 share one home" "^shared-[0-9]+$" \
    "n=shared-\$\$; ssh login \"echo \$n > ~/.clio-pv-share-check\"; \
ssh node08 'cat ~/.clio-pv-share-check; rm -f ~/.clio-pv-share-check'"
check "datasets are in node08's home" "pulse_series.pvd.*wavelet.vti|wavelet.vti" \
    "ssh node08 'ls ~/data'"
check "locked refuses BatchMode at once" "Permission denied \(password\)" \
    "ssh -o BatchMode=yes -o ConnectTimeout=5 locked true"
check "locked refusal is non-zero" fail "ssh -o BatchMode=yes -o ConnectTimeout=5 locked true"

if [ "$CLIO_PV_CLUSTER_VARIANT" = pvserver ]; then
    check "pvserver listens on node08:11111" "LISTEN" \
        "ssh node08 'ss -ltn \"sport = :$CLIO_PV_CLUSTER_PVSERVER_PORT\"'"
    check "the pvserver session already holds the dataset" "pulse_series.pvd" \
        "ssh node08 '$P; pvpython /opt/clio-pv-fixture/probe.py sources $CLIO_PV_CLUSTER_PVSERVER_PORT'"
fi

echo "selfcheck: $failures failed"
[ "$failures" -eq 0 ]
