#!/usr/bin/env bash
# The ParaView cluster fixture: containers standing in for a site's nodes,
# for testing the ParaView connector's remote attach.
#
#   login   (alias ares)             SSH on 127.0.0.1:$LOGIN_PORT; reaches the
#                                    package index; no ParaView.
#   node08  (alias ares-compute-08)  SSH only from login (internal network);
#                                    ParaView 6 after `module load paraview`;
#                                    no route to the package index.
#   locked                           SSH on 127.0.0.1:$LOCKED_PORT; password only.
#
# login and node08 share one home volume, one user and one key pair.
# `up` writes a dedicated SSH config and includes it from ~/.ssh/config;
# `down` removes both. Run `cluster.sh help` for the commands.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${CLIO_PV_CLUSTER_PREFIX:-clio-pv}"
IMAGE="${CLIO_PV_CLUSTER_IMAGE:-clio-pv-cluster}"
LOGIN_PORT="${CLIO_PV_CLUSTER_LOGIN_PORT:-19922}"
LOCKED_PORT="${CLIO_PV_CLUSTER_LOCKED_PORT:-19923}"
CLUSTER_USER=researcher
REMOTE_HOME="/home/$CLUSTER_USER"
STATE_DIR="${CLIO_PV_CLUSTER_STATE:-$HOME/.ssh/clio-pv-cluster}"
USER_SSH_CONFIG="$HOME/.ssh/config"
INCLUDE_LINE="Include $STATE_DIR/config"
INCLUDE_MARK="# clio-pv-cluster: added by mcp-servers/paraview/cluster/cluster.sh up; removed by down"
CA_FILE="${CLIO_PV_CLUSTER_CA:-${SSL_CERT_FILE:-}}"
HTTPS_PROXY_FOR_LOGIN="${CLIO_PV_CLUSTER_HTTPS_PROXY:-}"
NET_EDGE="$PREFIX-edge"
NET_CLUSTER="$PREFIX-cluster"
VOLUME="$PREFIX-home"
C_LOGIN="$PREFIX-login"
C_NODE="$PREFIX-node08"
C_LOCKED="$PREFIX-locked"
LABEL="org.iowarp.clio-pv-cluster=1"
PREPARE_LINE="module load paraview"
# The cluster network has fixed addresses and every node has them in
# /etc/hosts, as sites do: sshd's lookups through Docker's embedded DNS
# failed now and then ("Temporary failure in name resolution") under load.
SUBNET_PREFIX="${CLIO_PV_CLUSTER_SUBNET_PREFIX:-10.208.8}"
LOGIN_IP="$SUBNET_PREFIX.10"
NODE_IP="$SUBNET_PREFIX.8"

say() { printf '%s\n' "$*" >&2; }
die() { say "cluster.sh: $*"; exit 1; }

usage() {
    cat <<EOF
Usage: cluster.sh COMMAND [ARGS]

  build [--no-cache]     Build the login, node08 and locked images from scratch
                         (ParaView 6 from conda-forge inside node08).
  up [--pvserver]        Start the cluster and write the SSH config. With
                         --pvserver, node08 starts with a pvserver on 11111
                         that already has a dataset loaded.
  down                   Remove containers, networks, the home volume, the SSH
                         config and its Include line. Images are kept.
  variant plain|pvserver Replace node08 by the other form (home is kept).
  status                 Containers, variant, ports and the SSH config.
  ps [ROLE...]           Processes per container (login node08 locked host).
  check                  The self-checks; exits non-zero if one fails.
  env                    Shell exports describing the cluster for scenarios.
  scenarios [NAME...]    Run scenarios/*.sh (or the named ones) with that env.
  run -- CMD...          Run any command with that env (the hook for Phase 3:
                         e.g. a UI step file driving CLIO against node08).
  grow start|stop|status [--interval S] [--steps N]
                         The growing time series on node08 (~/data/growing).
  exec ROLE CMD...       Run a command as $CLUSTER_USER in a container.
  logs ROLE              Container logs.

Ports: login 127.0.0.1:$LOGIN_PORT, locked 127.0.0.1:$LOCKED_PORT
SSH config: $STATE_DIR/config, included from $USER_SSH_CONFIG
EOF
}

container_of() {
    case "$1" in
        login) echo "$C_LOGIN" ;;
        node08) echo "$C_NODE" ;;
        locked) echo "$C_LOCKED" ;;
        *) die "unknown role '$1' (login, node08, locked)" ;;
    esac
}

running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = "true" ]; }

variant() {
    local v
    v="$(docker inspect -f '{{index .Config.Labels "org.iowarp.clio-pv-cluster.variant"}}' \
        "$C_NODE" 2>/dev/null || true)"
    echo "${v:-none}"
}

# --- build -------------------------------------------------------------------

cmd_build() {
    local extra=() target
    [ "${1:-}" = "--no-cache" ] && extra+=(--no-cache --pull)
    if [ -n "$CA_FILE" ] && [ -s "$CA_FILE" ]; then
        extra+=(--secret "id=extra_ca,src=$CA_FILE")
    fi
    for target in login locked node08; do
        say "== building $IMAGE:$target"
        docker build "${extra[@]}" --target "$target" -t "$IMAGE:$target" "$HERE"
    done
    docker image ls "$IMAGE"
}

# --- up / down ---------------------------------------------------------------

ensure_images() {
    local t
    for t in login locked node08; do
        docker image inspect "$IMAGE:$t" >/dev/null 2>&1 \
            || die "image $IMAGE:$t is missing; run: cluster.sh build"
    done
}

ensure_key() {
    mkdir -p "$STATE_DIR"
    chmod 700 "$STATE_DIR"
    [ -f "$STATE_DIR/id_ed25519" ] \
        || ssh-keygen -q -t ed25519 -N '' -C "clio-pv-cluster fixture" -f "$STATE_DIR/id_ed25519"
}

run_node08() {
    local form="$1" env_args=()
    [ "$form" = "pvserver" ] && env_args+=(-e CLUSTER_START_PVSERVER=1)
    docker run -d --init --name "$C_NODE" --hostname node08 \
        --label "$LABEL" --label "org.iowarp.clio-pv-cluster.variant=$form" \
        --network "$NET_CLUSTER" --ip "$NODE_IP" \
        --network-alias node08 --network-alias ares-compute-08 \
        --add-host "login:$LOGIN_IP" --add-host "ares:$LOGIN_IP" \
        -v "$VOLUME:$REMOTE_HOME" "${env_args[@]}" "$IMAGE:node08" >/dev/null
}

host_key() { docker exec "$1" cat /etc/ssh/ssh_host_ed25519_key.pub | cut -d' ' -f1,2; }

write_ssh_files() {
    {
        echo "clio-pv-login $(host_key "$C_LOGIN")"
        echo "node08,ares-compute-08 $(host_key "$C_NODE")"
        echo "clio-pv-locked $(host_key "$C_LOCKED")"
    } > "$STATE_DIR/known_hosts"
    cat > "$STATE_DIR/config" <<EOF
# The ParaView cluster fixture (mcp-servers/paraview/cluster/cluster.sh).
# Written by 'cluster.sh up' and removed by 'cluster.sh down'.
Host login ares
    HostName 127.0.0.1
    Port $LOGIN_PORT
    HostKeyAlias clio-pv-login

# node08 carries its jump host here; ares-compute-08 has none, so it needs
# the connector's "via" value (ssh -J login ares-compute-08).
Host node08
    HostName node08
    ProxyJump login

Host ares-compute-08
    HostName ares-compute-08

Host locked
    HostName 127.0.0.1
    Port $LOCKED_PORT
    HostKeyAlias clio-pv-locked

Host login ares node08 ares-compute-08 locked
    User $CLUSTER_USER
    IdentityFile $STATE_DIR/id_ed25519
    IdentitiesOnly yes
    UserKnownHostsFile $STATE_DIR/known_hosts
    StrictHostKeyChecking yes
EOF
    chmod 600 "$STATE_DIR/config" "$STATE_DIR/known_hosts"
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"
    if [ ! -e "$USER_SSH_CONFIG" ]; then
        : > "$USER_SSH_CONFIG"
        chmod 600 "$USER_SSH_CONFIG"
        touch "$STATE_DIR/created-user-config"
    fi
    if ! grep -qxF "$INCLUDE_LINE" "$USER_SSH_CONFIG"; then
        # First in the file, so it applies whatever Host blocks follow.
        { echo "$INCLUDE_MARK"; echo "$INCLUDE_LINE"; cat "$USER_SSH_CONFIG"; } \
            > "$USER_SSH_CONFIG.clio-pv.tmp"
        cat "$USER_SSH_CONFIG.clio-pv.tmp" > "$USER_SSH_CONFIG"
        rm -f "$USER_SSH_CONFIG.clio-pv.tmp"
    fi
}

seed_home() {
    # The shared home: the user's key pair, authorized_keys, the cluster's
    # host keys, and the datasets (copied from node08's /opt/datasets).
    docker exec -i -u "$CLUSTER_USER" "$C_NODE" sh -c \
        'umask 077; mkdir -p ~/.ssh; cat > ~/.ssh/id_ed25519' < "$STATE_DIR/id_ed25519"
    docker exec -i -u "$CLUSTER_USER" "$C_NODE" sh -c \
        'umask 077; t=$(cat); echo "$t" > ~/.ssh/id_ed25519.pub; echo "$t" > ~/.ssh/authorized_keys' \
        < "$STATE_DIR/id_ed25519.pub"
    {
        echo "login,ares $(host_key "$C_LOGIN")"
        echo "node08,ares-compute-08 $(host_key "$C_NODE")"
    } | docker exec -i -u "$CLUSTER_USER" "$C_NODE" sh -c 'umask 077; cat > ~/.ssh/known_hosts'
    docker exec -u "$CLUSTER_USER" "$C_NODE" sh -c \
        'mkdir -p ~/data && cp -rn /opt/datasets/. ~/data/'
}

wait_ssh() {
    local host="$1" i
    for i in $(seq 1 40); do
        ssh -o BatchMode=yes -o ConnectTimeout=3 "$host" true 2>/dev/null && return 0
        sleep 0.5
    done
    die "ssh $host did not answer"
}

cmd_up() {
    local form=plain
    [ "${1:-}" = "--pvserver" ] && form=pvserver
    ensure_images
    docker inspect "$C_LOGIN" >/dev/null 2>&1 && die "already up; run: cluster.sh down"
    ensure_key
    docker network create --label "$LABEL" "$NET_EDGE" >/dev/null
    docker network create --internal --label "$LABEL" \
        --subnet "$SUBNET_PREFIX.0/24" "$NET_CLUSTER" >/dev/null
    docker volume create --label "$LABEL" "$VOLUME" >/dev/null
    local login_args=()
    if [ -n "$CA_FILE" ] && [ -s "$CA_FILE" ]; then
        login_args+=(-v "$CA_FILE:/etc/clio-pv/extra-ca.crt:ro")
    fi
    [ -n "$HTTPS_PROXY_FOR_LOGIN" ] && login_args+=(-e "CLUSTER_HTTPS_PROXY=$HTTPS_PROXY_FOR_LOGIN")
    docker run -d --init --name "$C_LOGIN" --hostname login --label "$LABEL" \
        --network "$NET_EDGE" -p "127.0.0.1:$LOGIN_PORT:22" \
        --add-host "node08:$NODE_IP" --add-host "ares-compute-08:$NODE_IP" \
        -v "$VOLUME:$REMOTE_HOME" "${login_args[@]}" "$IMAGE:login" >/dev/null
    docker network connect --ip "$LOGIN_IP" --alias login --alias ares "$NET_CLUSTER" "$C_LOGIN"
    run_node08 "$form"
    docker run -d --init --name "$C_LOCKED" --hostname locked --label "$LABEL" \
        --network "$NET_EDGE" -p "127.0.0.1:$LOCKED_PORT:22" "$IMAGE:locked" >/dev/null
    sleep 1
    seed_home
    write_ssh_files
    wait_ssh login
    wait_ssh node08
    wait_index
    [ "$form" = "pvserver" ] && wait_pvserver
    say "cluster up (node08: $form). SSH hosts: login, node08, locked"
}

wait_index() {
    # Name lookups from a just-started login failed for a few seconds in the
    # cloud session; wait until login resolves the package index.
    local i host="${CLIO_PV_CLUSTER_INDEX_HOST:-pypi.org}"
    for i in $(seq 1 30); do
        docker exec "$C_LOGIN" getent hosts "$host" >/dev/null 2>&1 && return 0
        sleep 1
    done
    say "warning: login cannot resolve $host; the package index is unreachable from login"
}

wait_pvserver() {
    local i
    for i in $(seq 1 120); do
        docker exec "$C_NODE" test -s /tmp/clio-pv-fixture/holder.ready && {
            say "pvserver on node08:11111 holds: $(docker exec "$C_NODE" \
                cat /tmp/clio-pv-fixture/holder.ready | tr '\n' ' ')"
            return 0
        }
        sleep 1
    done
    docker exec "$C_NODE" sh -c 'tail -20 /tmp/clio-pv-fixture/*.log' >&2 || true
    die "the pvserver on node08 did not load its dataset"
}

cmd_variant() {
    local form="${1:-}"
    case "$form" in plain|pvserver) ;; *) die "variant plain|pvserver" ;; esac
    running "$C_LOGIN" || die "the cluster is not up"
    docker rm -f "$C_NODE" >/dev/null
    run_node08 "$form"
    sleep 1
    write_ssh_files   # node08's host key is per image, so unchanged; rewrite anyway
    wait_ssh node08
    [ "$form" = "pvserver" ] && wait_pvserver
    say "node08 is now: $form"
}

cmd_down() {
    local c
    for c in "$C_LOGIN" "$C_NODE" "$C_LOCKED"; do
        docker rm -f "$c" >/dev/null 2>&1 && say "removed container $c" || true
    done
    for c in "$NET_EDGE" "$NET_CLUSTER"; do
        docker network rm "$c" >/dev/null 2>&1 && say "removed network $c" || true
    done
    docker volume rm "$VOLUME" >/dev/null 2>&1 && say "removed volume $VOLUME" || true
    if [ -f "$USER_SSH_CONFIG" ]; then
        grep -vxF -e "$INCLUDE_LINE" -e "$INCLUDE_MARK" "$USER_SSH_CONFIG" \
            > "$USER_SSH_CONFIG.clio-pv.tmp" || true
        cat "$USER_SSH_CONFIG.clio-pv.tmp" > "$USER_SSH_CONFIG"
        rm -f "$USER_SSH_CONFIG.clio-pv.tmp"
        if [ -f "$STATE_DIR/created-user-config" ] && [ ! -s "$USER_SSH_CONFIG" ]; then
            rm -f "$USER_SSH_CONFIG"
            say "removed $USER_SSH_CONFIG (created by up)"
        else
            say "removed the Include line from $USER_SSH_CONFIG"
        fi
    fi
    if [ -d "$STATE_DIR" ]; then
        rm -rf "$STATE_DIR"
        say "removed $STATE_DIR"
    fi
}

# --- inspection --------------------------------------------------------------

cmd_status() {
    docker ps -a --filter "label=$LABEL" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}'
    echo "node08 form: $(variant)"
    docker image ls "$IMAGE" --format '{{.Repository}}:{{.Tag}}  {{.Size}}'
    if [ -f "$USER_SSH_CONFIG" ] && grep -qxF "$INCLUDE_LINE" "$USER_SSH_CONFIG"; then
        echo "ssh config: $USER_SSH_CONFIG includes $STATE_DIR/config"
    else
        echo "ssh config: not installed"
    fi
}

cmd_ps() {
    local roles=("$@") role c
    [ ${#roles[@]} -eq 0 ] && roles=(host login node08 locked)
    for role in "${roles[@]}"; do
        if [ "$role" = "host" ]; then
            echo "== host: ssh clients"
            ps -C ssh -o pid,ppid,etime,args 2>/dev/null || echo "(none)"
            continue
        fi
        c="$(container_of "$role")"
        echo "== $role ($c)"
        if ! running "$c"; then echo "(not running)"; continue; fi
        docker exec "$c" ps -eo user,pid,ppid,etime,args --forest
        echo "-- $role: $CLUSTER_USER processes other than sshd"
        docker exec "$c" ps -u "$CLUSTER_USER" -o pid,etime,args --no-headers \
            | grep -v -E 'sshd: |ps -u ' || echo "(none)"
    done
}

cmd_env() {
    cat <<EOF
export CLIO_PV_CLUSTER="$HERE/cluster.sh"
export CLIO_PV_CLUSTER_SSH_CONFIG="$STATE_DIR/config"
export CLIO_PV_CLUSTER_ADDRESS=node08
export CLIO_PV_CLUSTER_VIA=login
export CLIO_PV_CLUSTER_PREPARE="$PREPARE_LINE"
export CLIO_PV_CLUSTER_ALT_ADDRESS=ares-compute-08
export CLIO_PV_CLUSTER_ALT_VIA=ares
export CLIO_PV_CLUSTER_LOCKED=locked
export CLIO_PV_CLUSTER_HOME="$REMOTE_HOME"
export CLIO_PV_CLUSTER_DATASET="$REMOTE_HOME/data/wavelet.vti"
export CLIO_PV_CLUSTER_DATASET_FIELD=RTData
export CLIO_PV_CLUSTER_SERIES="$REMOTE_HOME/data/pulse_series.pvd"
export CLIO_PV_CLUSTER_SERIES_FIELD=pressure
export CLIO_PV_CLUSTER_GROWING="$REMOTE_HOME/data/growing/growing.vti.series"
export CLIO_PV_CLUSTER_PVSERVER_PORT=11111
export CLIO_PV_CLUSTER_VARIANT="$(variant)"
EOF
}

cmd_scenarios() {
    local scripts=() name s failed=0
    if [ $# -eq 0 ]; then
        scripts=("$HERE"/scenarios/[0-9]*.sh)
    else
        for name in "$@"; do
            s=("$HERE"/scenarios/*"$name"*.sh)
            [ -e "${s[0]}" ] || die "no scenario matches '$name'"
            scripts+=("${s[@]}")
        done
    fi
    eval "$(cmd_env)"
    for s in "${scripts[@]}"; do
        say "== scenario $(basename "$s")"
        local rc=0
        bash "$s" || rc=$?
        case "$rc" in
            0) say "-- $(basename "$s"): PASS" ;;
            2) say "-- $(basename "$s"): SKIP" ;;
            *) say "-- $(basename "$s"): FAIL"; failed=1 ;;
        esac
    done
    return "$failed"
}

cmd_grow() {
    local action="${1:-status}"; shift || true
    local interval=5 steps=60
    while [ $# -gt 0 ]; do
        case "$1" in
            --interval) interval="$2"; shift 2 ;;
            --steps) steps="$2"; shift 2 ;;
            *) die "grow: unknown option $1" ;;
        esac
    done
    case "$action" in
        start)
            docker exec -d -u "$CLUSTER_USER" -w "$REMOTE_HOME" "$C_NODE" bash -c \
                ". /etc/clio-pv/site-bashrc.sh; $PREPARE_LINE; exec python \
                /opt/clio-pv-fixture/grow_series.py ~/data/growing --interval $interval \
                --steps $steps > /tmp/clio-pv-fixture-grow.log 2>&1"
            say "growing ~/data/growing on node08 every ${interval}s up to step $steps" ;;
        stop)
            docker exec "$C_NODE" pkill -f grow_series.py && say "stopped" || say "not running" ;;
        status)
            docker exec "$C_NODE" pgrep -af grow_series.py || echo "not running"
            docker exec "$C_NODE" sh -c "tail -2 /tmp/clio-pv-fixture-grow.log 2>/dev/null" || true ;;
        *) die "grow start|stop|status" ;;
    esac
}

main() {
    local cmd="${1:-help}"; shift || true
    case "$cmd" in
        build) cmd_build "$@" ;;
        up) cmd_up "$@" ;;
        down) cmd_down ;;
        variant) cmd_variant "$@" ;;
        status) cmd_status ;;
        ps) cmd_ps "$@" ;;
        check) bash "$HERE/scenarios/00-selfcheck.sh" ;;
        env) cmd_env ;;
        scenarios) cmd_scenarios "$@" ;;
        grow) cmd_grow "$@" ;;
        run) [ "${1:-}" = "--" ] && shift
             [ $# -ge 1 ] || die "run -- CMD..."
             running "$C_LOGIN" || die "the cluster is not up; run: cluster.sh up"
             eval "$(cmd_env)"; exec "$@" ;;
        exec) [ $# -ge 2 ] || die "exec ROLE CMD..."
              local c; c="$(container_of "$1")"; shift
              docker exec -i -u "$CLUSTER_USER" -w "$REMOTE_HOME" "$c" "$@" ;;
        logs) docker logs "$(container_of "${1:-login}")" ;;
        help|-h|--help) usage ;;
        *) usage; exit 2 ;;
    esac
}

main "$@"
